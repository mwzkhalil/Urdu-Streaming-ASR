import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import hf_hub_download
import k2
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler, K2SpeechRecognitionDataset, OnTheFlyFeatures
from lhotse.features import Fbank, FbankConfig
import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
ICEFALL_ROOT = ROOT / "third_party" / "icefall"
RECIPE_STREAM = ICEFALL_ROOT / "egs" / "librispeech" / "ASR" / "pruned_transducer_stateless7_streaming"
RECIPE_BASE = ICEFALL_ROOT / "egs" / "librispeech" / "ASR" / "pruned_transducer_stateless7"

for p in (RECIPE_BASE, ICEFALL_ROOT, RECIPE_STREAM, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from decoder import Decoder
from joiner import Joiner
from model import Transducer
from optim import Eden, ScaledAdam
from zipformer import Zipformer
from icefall.checkpoint import load_checkpoint, save_checkpoint as save_checkpoint_impl
from icefall.utils import AttributeDict, create_grad_scaler, torch_autocast

from urdu_streaming_asr.adapters import parallel_download_hf_file
from urdu_streaming_asr.checkpoint_remap import adapt_pretrained_checkpoint, sha256_file


PRETRAINED_REPO = "desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small"
PRETRAINED_REV = "be162ecc09bade73063a671fad9d18220149d25b"


def build_transducer(vocab_size: int = 1024, blank_id: int = 0, context_size: int = 2) -> Transducer:
    encoder = Zipformer(
        num_features=80,
        output_downsampling_factor=2,
        zipformer_downsampling_factors=(1, 2, 4, 8, 2),
        encoder_dims=(256, 256, 256, 256, 256),
        attention_dim=(192, 192, 192, 192, 192),
        encoder_unmasked_dims=(192, 192, 192, 192, 192),
        nhead=(8, 8, 8, 8, 8),
        feedforward_dim=(768, 768, 768, 768, 768),
        cnn_module_kernels=(31, 31, 31, 31, 31),
        num_encoder_layers=(2, 2, 2, 2, 2),
        num_left_chunks=4,
        short_chunk_size=50,
        decode_chunk_size=16,
    )
    decoder = Decoder(
        vocab_size=vocab_size,
        decoder_dim=512,
        blank_id=blank_id,
        context_size=context_size,
    )
    joiner = Joiner(
        encoder_dim=256,
        decoder_dim=512,
        joiner_dim=512,
        vocab_size=vocab_size,
    )
    return Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=256,
        decoder_dim=512,
        joiner_dim=512,
        vocab_size=vocab_size,
    )


def set_batch_count(model: nn.Module, batch_count: float) -> None:
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


@torch.no_grad()
def streaming_greedy_decode_waveform(
    model: Transducer,
    sp: spm.SentencePieceProcessor,
    waveform_16k: np.ndarray,
    device: torch.device,
    decode_chunk_frames: int = 32,
    num_left_chunks: int = 4,
) -> Dict[str, Any]:
    import math
    import torch.nn.functional as F

    model.eval()
    extractor = Fbank(FbankConfig(num_mel_bins=80))
    feats_np = extractor.extract(waveform_16k, sampling_rate=16000)
    feats = torch.from_numpy(feats_np).to(device=device, dtype=torch.float32)

    pad_length = 7
    tail_pad_len = decode_chunk_frames
    log_eps = math.log(1e-10)
    feats_padded = F.pad(feats, (0, 0, 0, pad_length + tail_pad_len), mode="constant", value=log_eps)
    num_frames = feats_padded.size(0)
    num_processed = 0

    states = model.encoder.get_init_state(device=device)
    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    hyp: List[int] = [-1] * (context_size - 1) + [blank_id]
    decoder_input = torch.tensor([hyp[-context_size:]], dtype=torch.int64, device=device)
    decoder_out = model.joiner.decoder_proj(model.decoder(decoder_input, need_pad=False))

    partials: List[Dict[str, Any]] = []
    chunk_idx = 0

    while num_processed < num_frames:
        chunk_length = decode_chunk_frames + pad_length
        ret_len = min(num_frames - num_processed, chunk_length)
        chunk_x = feats_padded[num_processed : num_processed + ret_len].unsqueeze(0)
        num_processed += decode_chunk_frames
        if chunk_x.size(1) < 23:
            pad_more = 23 - chunk_x.size(1)
            chunk_x = F.pad(chunk_x, (0, 0, 0, pad_more), mode="constant", value=log_eps)
            ret_len = 23
        chunk_len = torch.tensor([ret_len], dtype=torch.int64, device=device)
        enc_out, enc_out_lens, states = model.encoder.streaming_forward(
            x=chunk_x,
            x_lens=chunk_len,
            states=states,
        )
        enc_out = model.joiner.encoder_proj(enc_out)
        valid_t = int(enc_out_lens[0].item())
        for t in range(valid_t):
            cur_enc = enc_out[:, t : t + 1, :]
            logits = model.joiner(
                cur_enc.unsqueeze(2),
                decoder_out.unsqueeze(1),
                project_input=False,
            ).squeeze(1).squeeze(1)
            y_id = int(logits.argmax(dim=-1).item())
            if y_id != blank_id:
                hyp.append(y_id)
                decoder_input = torch.tensor([hyp[-context_size:]], dtype=torch.int64, device=device)
                decoder_out = model.joiner.decoder_proj(model.decoder(decoder_input, need_pad=False))
        cur_tokens = hyp[context_size:]
        cur_text = sp.decode(cur_tokens) if cur_tokens else ""
        partials.append({"chunk_index": chunk_idx, "frames_processed": num_processed, "text": cur_text})
        chunk_idx += 1

    final_tokens = hyp[context_size:]
    final_text = sp.decode(final_tokens) if final_tokens else ""
    return {
        "final_text": final_text,
        "token_ids": final_tokens,
        "num_chunks": chunk_idx,
        "partials": partials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pretrained English model, adapt to Urdu, and run smoke checks")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--exp-dir", type=Path, default=ROOT / "exp")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    pretrained_dir = args.data_dir / "pretrained_en_20m"
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA GPU is required for training smoke checks!"

    print("[1/5] Downloading pretrained English streaming checkpoint and reference assets...")
    en_files = {}
    for fname in [
        "exp/pretrained.pt",
        "data/lang_bpe_500/bpe.model",
        "data/lang_bpe_500/tokens.txt",
        "test_wavs/1089-134686-0001.wav",
        "test_wavs/1221-135766-0001.wav",
        "test_wavs/trans.txt",
    ]:
        p = parallel_download_hf_file(
            repo_id=PRETRAINED_REPO,
            filename=fname,
            repo_type="model",
            out_path=pretrained_dir / fname,
            revision=PRETRAINED_REV,
            workers=16,
        )
        en_files[fname] = p

    print("[2/5] Verifying English pretrained model load and reference streaming decode...")
    en_sp = spm.SentencePieceProcessor()
    en_sp.load(str(en_files["data/lang_bpe_500/bpe.model"]))
    en_model = build_transducer(vocab_size=500, blank_id=0, context_size=2).to(device)
    en_ckpt = torch.load(str(en_files["exp/pretrained.pt"]), map_location="cpu", weights_only=False)
    en_state = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in en_ckpt["model"].items()
    }
    en_model.load_state_dict(en_state, strict=True)

    wav1, sr1 = sf.read(str(en_files["test_wavs/1089-134686-0001.wav"]), dtype="float32")
    en_decode_res = streaming_greedy_decode_waveform(en_model, en_sp, wav1, device)
    print("  English reference streaming decode:", en_decode_res["final_text"])
    assert "AFTER EARLY NIGHTFALL" in en_decode_res["final_text"].upper(), f"Unexpected English decode: {en_decode_res}"
    del en_model
    torch.cuda.empty_cache()

    print("[3/5] Adapting pretrained checkpoint to Urdu SentencePiece vocabulary (1024)...")
    ur_bpe_path = args.data_dir / "lang_bpe_1024" / "bpe.model"
    ur_sp = spm.SentencePieceProcessor()
    ur_sp.load(str(ur_bpe_path))
    ur_vocab_size = ur_sp.get_piece_size()

    ur_model = build_transducer(vocab_size=ur_vocab_size, blank_id=0, context_size=2)
    adapted_ckpt_path = args.exp_dir / "pretrained_urdu_init.pt"
    lineage_report = adapt_pretrained_checkpoint(
        src_ckpt_path=en_files["exp/pretrained.pt"],
        target_model=ur_model,
        out_ckpt_path=adapted_ckpt_path,
        src_bpe_path=en_files["data/lang_bpe_500/bpe.model"],
        dst_bpe_path=ur_bpe_path,
        reuse_matching_tokens=False,
        seed=42,
    )
    lineage_report["english_reference_decode"] = en_decode_res
    lineage_report["pretrained_repo"] = PRETRAINED_REPO
    lineage_report["pretrained_revision"] = PRETRAINED_REV
    (args.reports_dir / "checkpoint_lineage.json").write_text(
        json.dumps(lineage_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[4/5] Running CUDA mixed-precision RNN-T forward/backward & 24-clip overfit smoke check...")
    ur_model = ur_model.to(device)
    cuts_train = CutSet.from_jsonl(str(args.data_dir / "manifests" / "cuts_train.jsonl.gz"))
    tiny_cuts = cuts_train.filter(lambda c: 1.2 <= c.duration <= 4.5).subset(first=24).to_eager()

    dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
        return_cuts=True,
    )
    batch = dataset[tiny_cuts]
    feats = batch["inputs"].to(device)
    feat_lens = batch["supervisions"]["num_frames"].to(device)
    texts = batch["supervisions"]["text"]
    y_ids = ur_sp.encode(texts, out_type=int)
    y_ragged = k2.RaggedTensor(y_ids).to(device)

    enc_params = [p for n, p in ur_model.named_parameters() if n.startswith("encoder.")]
    enc_names = [n for n, _ in ur_model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in ur_model.named_parameters() if not n.startswith("encoder.")]
    head_names = [n for n, _ in ur_model.named_parameters() if not n.startswith("encoder.")]
    optimizer = ScaledAdam(
        [
            {"params": enc_params, "lr": 0.005},
            {"params": head_params, "lr": 0.035},
        ],
        parameters_names=[enc_names, head_names],
        lr=0.035,
        clipping_scale=2.0,
    )
    scheduler = Eden(optimizer, lr_batches=500, lr_epochs=2)
    scaler = create_grad_scaler(enabled=True)

    torch.cuda.reset_peak_memory_stats(device)
    overfit_losses: List[Dict[str, float]] = []

    for step in range(1, 51):
        ur_model.train()
        set_batch_count(ur_model, 100000.0 + step)
        optimizer.zero_grad()
        with torch_autocast(dtype=torch.float16, enabled=True):
            simple_loss, pruned_loss = ur_model(
                x=feats,
                x_lens=feat_lens,
                y=y_ragged,
                prune_range=5,
                am_scale=0.0,
                lm_scale=0.25,
            )
            loss = 0.5 * simple_loss + pruned_loss
        assert torch.isfinite(loss).item(), f"Non-finite loss at smoke step {step}: {loss.item()}"
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step_batch(step)
        if step == 1 or step % 5 == 0:
            overfit_losses.append(
                {
                    "step": step,
                    "total_loss": round(float(loss.item()), 4),
                    "simple_loss": round(float(simple_loss.item()), 4),
                    "pruned_loss": round(float(pruned_loss.item()), 4),
                }
            )

    peak_alloc_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
    peak_res_mb = round(torch.cuda.max_memory_reserved(device) / (1024 ** 2), 2)
    loss_start = overfit_losses[0]["total_loss"]
    loss_end = overfit_losses[-1]["total_loss"]
    print(f"  24-clip overfit loss decreased from {loss_start} -> {loss_end} (peak VRAM allocated={peak_alloc_mb} MiB)")
    assert loss_end < loss_start * 0.72, f"Insufficient loss decrease on tiny subset: {loss_start} -> {loss_end}"

    sample_cut = tiny_cuts[0]
    sample_wav = sample_cut.load_audio()[0]
    tiny_decode_after = streaming_greedy_decode_waveform(ur_model, ur_sp, sample_wav, device)

    print("[5/5] Verifying checkpoint save/resume fidelity and streaming state isolation...")
    smoke_ckpt = args.exp_dir / "smoke_resume_test.pt"
    dummy_params = AttributeDict(
        {
            "best_train_loss": loss_end,
            "best_valid_loss": loss_end,
            "best_train_epoch": 1,
            "best_valid_epoch": 1,
            "batch_idx_train": 50,
            "cur_epoch": 1,
        }
    )
    model_avg = build_transducer(vocab_size=ur_vocab_size, blank_id=0, context_size=2)
    save_checkpoint_impl(
        filename=smoke_ckpt,
        model=ur_model,
        model_avg=model_avg,
        params=dummy_params,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=None,
        rank=0,
    )

    reloaded_model = build_transducer(vocab_size=ur_vocab_size, blank_id=0, context_size=2).to(device)
    reloaded_avg = build_transducer(vocab_size=ur_vocab_size, blank_id=0, context_size=2)
    reloaded_opt = ScaledAdam(
        [
            {"params": [p for n, p in reloaded_model.named_parameters() if n.startswith("encoder.")], "lr": 0.005},
            {"params": [p for n, p in reloaded_model.named_parameters() if not n.startswith("encoder.")], "lr": 0.035},
        ],
        parameters_names=[enc_names, head_names],
        lr=0.035,
        clipping_scale=2.0,
    )
    reloaded_sched = Eden(reloaded_opt, lr_batches=500, lr_epochs=2)
    reloaded_scaler = create_grad_scaler(enabled=True)
    saved_meta = load_checkpoint(
        smoke_ckpt,
        model=reloaded_model,
        model_avg=reloaded_avg,
        optimizer=reloaded_opt,
        scheduler=reloaded_sched,
        scaler=reloaded_scaler,
    )
    for (k1, p1), (k2_name, p2) in zip(ur_model.named_parameters(), reloaded_model.named_parameters()):
        assert k1 == k2_name and torch.allclose(p1, p2), f"Resume mismatch on parameter {k1}"
    assert saved_meta["batch_idx_train"] == 50
    smoke_ckpt.unlink(missing_ok=True)

    decode_run_1 = streaming_greedy_decode_waveform(reloaded_model, ur_sp, sample_wav, device)
    decode_run_2 = streaming_greedy_decode_waveform(reloaded_model, ur_sp, sample_wav, device)
    assert decode_run_1["token_ids"] == decode_run_2["token_ids"], "Streaming state was not isolated between runs!"

    smoke_report = {
        "status": "PASSED",
        "cuda_device": torch.cuda.get_device_name(device),
        "english_reference_decode_verified": True,
        "english_reference_text": en_decode_res["final_text"],
        "vocab_size": ur_vocab_size,
        "overfit_24_clips": {
            "loss_step_1": loss_start,
            "loss_step_25": loss_end,
            "trajectory": overfit_losses,
            "sample_ref_text": sample_cut.supervisions[0].text,
            "sample_decoded_after_25_steps": tiny_decode_after["final_text"],
        },
        "vram_calibration_24_clips": {
            "peak_allocated_mib": peak_alloc_mb,
            "peak_reserved_mib": peak_res_mb,
        },
        "checkpoint_resume_verified": True,
        "streaming_state_isolation_verified": True,
    }
    (args.reports_dir / "smoke_checks_report.json").write_text(
        json.dumps(smoke_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("All smoke checks PASSED!")
    print(json.dumps(smoke_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
