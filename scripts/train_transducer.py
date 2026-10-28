import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from shutil import copyfile
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import jiwer
import k2
from lhotse import CutSet
from lhotse.dataset import (
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    OnTheFlyFeatures,
    SpecAugment,
)
from lhotse.features import Fbank, FbankConfig
from lhotse.utils import fix_random_seed
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
from icefall.checkpoint import (
    load_checkpoint,
    remove_checkpoints,
    save_checkpoint as save_checkpoint_impl,
    update_averaged_model,
)
from icefall.utils import AttributeDict, MetricsTracker, add_sos, create_grad_scaler, setup_logger, torch_autocast
from urdu_streaming_asr.normalizer import normalize_urdu_text


def build_transducer(vocab_size: int, blank_id: int = 0, context_size: int = 2) -> Transducer:
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
    encoder.short_chunk_threshold = 2.0
    decoder = Decoder(vocab_size=vocab_size, decoder_dim=512, blank_id=blank_id, context_size=context_size)
    joiner = Joiner(encoder_dim=256, decoder_dim=512, joiner_dim=512, vocab_size=vocab_size)
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
    decode_chunk_len: int = 32,
) -> Tuple[str, List[int]]:
    model.eval()
    extractor = Fbank(FbankConfig(num_mel_bins=80))
    feats_np = extractor.extract(waveform_16k, sampling_rate=16000)
    feats = torch.from_numpy(feats_np).to(device=device, dtype=torch.float32)

    pad_length = 7
    tail_pad_len = decode_chunk_len
    log_eps = math.log(1e-10)
    feats_padded = F.pad(feats, (0, 0, 0, pad_length + tail_pad_len), mode="constant", value=log_eps)
    num_frames = feats_padded.size(0)
    num_processed = 0

    states = model.encoder.get_init_state(device=device)
    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    hyp: List[int] = [-1] * (context_size - 1) + [blank_id]
    decoder_input = torch.tensor([hyp[-context_size:]], device=device, dtype=torch.int64)
    decoder_out = model.joiner.decoder_proj(model.decoder(decoder_input, need_pad=False))

    while num_processed < num_frames:
        chunk_length = decode_chunk_len + pad_length
        ret_len = min(num_frames - num_processed, chunk_length)
        chunk_x = feats_padded[num_processed : num_processed + ret_len].unsqueeze(0)
        num_processed += decode_chunk_len
        if chunk_x.size(1) < 23:
            pad_more = 23 - chunk_x.size(1)
            chunk_x = F.pad(chunk_x, (0, 0, 0, pad_more), mode="constant", value=log_eps)
            ret_len = 23
        x_lens = torch.tensor([ret_len], device=device, dtype=torch.int64)
        enc_out, enc_out_lens, states = model.encoder.streaming_forward(x=chunk_x, x_lens=x_lens, states=states)
        enc_out = model.joiner.encoder_proj(enc_out)
        valid_t = int(enc_out_lens[0].item())
        for t in range(valid_t):
            cur_enc = enc_out[:, t : t + 1, :]
            logits = model.joiner(cur_enc.unsqueeze(2), decoder_out.unsqueeze(1), project_input=False).squeeze(1).squeeze(1)
            y_id = int(logits.argmax(dim=-1).item())
            if y_id != blank_id:
                hyp.append(y_id)
                decoder_input = torch.tensor([hyp[-context_size:]], device=device, dtype=torch.int64)
                decoder_out = model.joiner.decoder_proj(model.decoder(decoder_input, need_pad=False))

    tokens = hyp[context_size:]
    return (sp.decode(tokens) if tokens else ""), tokens


def compute_loss(
    params: AttributeDict,
    model: Transducer,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, MetricsTracker]:
    feature = batch["inputs"].to(device)
    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)
    texts = supervisions["text"]
    y = k2.RaggedTensor(sp.encode(texts, out_type=int)).to(device)
    row_splits = y.shape.row_splits(1)
    y_lens = row_splits[1:] - row_splits[:-1]
    y_padded = y.pad(mode="constant", padding_value=0).to(torch.int64)

    with torch.set_grad_enabled(is_training):
        encoder_out, x_lens = model.encoder(feature, feature_lens)
        sos_y = add_sos(y, sos_id=0)
        sos_y_padded = sos_y.pad(mode="constant", padding_value=0)
        decoder_out = model.decoder(sos_y_padded)
        boundary = torch.zeros((feature.size(0), 4), dtype=torch.int64, device=device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens

        lm = model.simple_lm_proj(decoder_out)
        am = model.simple_am_proj(encoder_out)

        with torch_autocast(dtype=torch.float16, enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=0,
                lm_only_scale=params.lm_scale,
                am_only_scale=params.am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )
            ctc_log_probs = am.float().log_softmax(dim=-1).permute(1, 0, 2)
            ctc_loss = F.ctc_loss(
                ctc_log_probs,
                y_padded,
                x_lens,
                y_lens,
                blank=0,
                reduction="sum",
                zero_infinity=True,
            )

        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=params.prune_range,
        )
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=model.joiner.encoder_proj(encoder_out),
            lm=model.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )
        logits = model.joiner(am_pruned, lm_pruned, project_input=False)
        with torch_autocast(dtype=torch.float16, enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=0,
                boundary=boundary,
                reduction="sum",
            )

        loss = params.simple_loss_scale * simple_loss + pruned_loss + params.ctc_loss_scale * ctc_loss

    assert loss.requires_grad == is_training
    info = MetricsTracker()
    num_frames = int((feature_lens // 4).sum().item())
    info["frames"] = num_frames
    info["utterances"] = feature.size(0)
    info["loss"] = loss.detach().cpu().item()
    info["simple_loss"] = simple_loss.detach().cpu().item()
    info["pruned_loss"] = pruned_loss.detach().cpu().item()
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    return loss, info


@torch.no_grad()
def compute_validation_metrics(
    params: AttributeDict,
    model: Transducer,
    sp: spm.SentencePieceProcessor,
    valid_dl: DataLoader,
    dev_decode_cuts: List[Any],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    tot_loss = MetricsTracker()
    for batch in valid_dl:
        with torch_autocast(dtype=torch.float16, enabled=params.use_fp16):
            loss, loss_info = compute_loss(
                params=params,
                model=model,
                sp=sp,
                batch=batch,
                is_training=False,
                device=device,
            )
        assert torch.isfinite(loss).item(), f"Non-finite validation loss: {loss}"
        tot_loss = tot_loss + loss_info

    frames = max(1, tot_loss["frames"])
    refs, hyps = [], []
    for c in dev_decode_cuts:
        wav = c.load_audio()[0]
        hyp_s, _ = streaming_greedy_decode_waveform(model, sp, wav, device)
        refs.append(normalize_urdu_text(c.supervisions[0].text))
        hyps.append(normalize_urdu_text(hyp_s))

    dev_wer = round(float(jiwer.wer(refs, hyps)) * 100.0, 2)
    dev_cer = round(float(jiwer.cer(refs, hyps)) * 100.0, 2)
    return {
        "valid_loss": round(float(tot_loss["loss"] / frames), 4),
        "valid_simple_loss": round(float(tot_loss["simple_loss"] / frames), 4),
        "valid_pruned_loss": round(float(tot_loss["pruned_loss"] / frames), 4),
        "valid_ctc_loss": round(float(tot_loss["ctc_loss"] / frames), 4),
        "dev_subset_wer": dev_wer,
        "dev_subset_cer": dev_cer,
        "sample_ref": refs[0] if refs else "",
        "sample_hyp": hyps[0] if hyps else "",
    }


def write_live_status(status_path: Path, docs_status_path: Path, payload: Dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md = (
        f"# Project Execution Status\n\n"
        f"- **Updated UTC**: `{payload.get('timestamp_utc')}`\n"
        f"- **Job State**: `{payload.get('state')}`\n"
        f"- **PID**: `{payload.get('pid')}`\n"
        f"- **Current Epoch / Total**: `{payload.get('cur_epoch')} / {payload.get('num_epochs')}`\n"
        f"- **Optimizer Updates (`batch_idx_train`)**: `{payload.get('batch_idx_train')}`\n"
        f"- **Audio Hours Seen**: `{payload.get('audio_hours_seen')}`\n"
        f"- **Latest Train Total Loss**: `{payload.get('train_loss')}`\n"
        f"- **Best Valid Loss**: `{payload.get('best_valid_loss')}` (epoch `{payload.get('best_valid_epoch')}`)\n"
        f"- **Best Dev Subset CER / WER**: `{payload.get('best_dev_cer')}% / {payload.get('best_dev_wer')}%`\n"
        f"- **Peak VRAM Allocated / Reserved (MiB)**: `{payload.get('peak_allocated_mib')} / {payload.get('peak_reserved_mib')}`\n"
        f"- **Throughput (audio sec / wall sec)**: `{payload.get('audio_sec_per_wall_sec')}`\n"
        f"- **Estimated Remaining Minutes**: `{payload.get('eta_minutes')}`\n"
        f"- **Latest Checkpoint**: `{payload.get('latest_checkpoint')}`\n"
    )
    docs_status_path.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Urdu Streaming Zipformer Transducer")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--exp-dir", type=Path, default=ROOT / "exp" / "urdu_zipformer_20m_v2")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--init-ckpt", type=Path, default=ROOT / "exp" / "pretrained_urdu_init.pt")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in exp-dir")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--base-lr", type=float, default=0.025)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.32)
    parser.add_argument("--decoder-lr-scale", type=float, default=0.16)
    parser.add_argument("--lr-batches", type=float, default=1500.0)
    parser.add_argument("--lr-epochs", type=float, default=4.5)
    parser.add_argument("--am-scale", type=float, default=0.45)
    parser.add_argument("--lm-scale", type=float, default=0.05)
    parser.add_argument("--ctc-loss-scale", type=float, default=0.50)
    parser.add_argument("--max-duration", type=float, default=150.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--save-every-n", type=int, default=300)
    parser.add_argument("--valid-interval", type=int, default=300)
    parser.add_argument("--average-period", type=int, default=25)
    parser.add_argument("--keep-last-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-fp16", type=int, default=1)
    args = parser.parse_args()

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    lock_file = args.exp_dir / ".train.lock"

    if lock_file.exists():
        old_pid_str = lock_file.read_text().strip()
        if old_pid_str.isdigit() and Path(f"/proc/{old_pid_str}").exists():
            raise RuntimeError(f"Training lock active for live PID {old_pid_str} in {lock_file}")
    lock_file.write_text(str(os.getpid()), encoding="utf-8")

    setup_logger(f"{args.exp_dir}/log/log-train")
    fix_random_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA GPU required"

    bpe_path = args.data_dir / "lang_bpe_1024" / "bpe.model"
    sp = spm.SentencePieceProcessor()
    sp.load(str(bpe_path))
    vocab_size = sp.get_piece_size()

    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_dev_cer": 100.0,
            "best_dev_wer": 100.0,
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "cur_epoch": args.start_epoch,
            "num_epochs": args.num_epochs,
            "start_epoch": args.start_epoch,
            "use_fp16": bool(args.use_fp16),
            "base_lr": args.base_lr,
            "encoder_lr_scale": args.encoder_lr_scale,
            "decoder_lr_scale": args.decoder_lr_scale,
            "lr_batches": args.lr_batches,
            "lr_epochs": args.lr_epochs,
            "prune_range": 5,
            "lm_scale": args.lm_scale,
            "am_scale": args.am_scale,
            "simple_loss_scale": 0.5,
            "ctc_loss_scale": args.ctc_loss_scale,
            "max_duration": args.max_duration,
            "grad_accum_steps": args.grad_accum_steps,
            "average_period": args.average_period,
            "save_every_n": args.save_every_n,
            "valid_interval": args.valid_interval,
            "keep_last_k": args.keep_last_k,
            "exp_dir": args.exp_dir,
            "vocab_size": vocab_size,
            "blank_id": 0,
            "context_size": 2,
        }
    )

    model = build_transducer(vocab_size=vocab_size, blank_id=0, context_size=2).to(device)
    model_avg: Optional[nn.Module] = copy.deepcopy(model).to("cpu")

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    enc_names = [n for n, _ in model.named_parameters() if n.startswith("encoder.")]
    am_join_params = [p for n, p in model.named_parameters() if n.startswith("simple_am_proj.") or n.startswith("joiner.")]
    am_join_names = [n for n, _ in model.named_parameters() if n.startswith("simple_am_proj.") or n.startswith("joiner.")]
    dec_lm_params = [p for n, p in model.named_parameters() if n.startswith("decoder.") or n.startswith("simple_lm_proj.")]
    dec_lm_names = [n for n, _ in model.named_parameters() if n.startswith("decoder.") or n.startswith("simple_lm_proj.")]

    optimizer = ScaledAdam(
        [
            {"params": enc_params, "lr": args.base_lr * args.encoder_lr_scale},
            {"params": am_join_params, "lr": args.base_lr},
            {"params": dec_lm_params, "lr": args.base_lr * args.decoder_lr_scale},
        ],
        parameters_names=[enc_names, am_join_names, dec_lm_names],
        lr=args.base_lr,
        clipping_scale=2.0,
    )
    scheduler = Eden(optimizer, lr_batches=args.lr_batches, lr_epochs=args.lr_epochs)
    scaler = create_grad_scaler(enabled=params.use_fp16)

    latest_ckpt = args.exp_dir / "checkpoint-latest.pt"
    if args.resume and latest_ckpt.exists():
        logging.info(f"Resuming from {latest_ckpt}")
        saved = load_checkpoint(
            latest_ckpt,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        params.batch_idx_train = saved.get("batch_idx_train", 0)
        params.best_train_loss = saved.get("best_train_loss", float("inf"))
        params.best_valid_loss = saved.get("best_valid_loss", float("inf"))
        params.best_dev_cer = saved.get("best_dev_cer", 100.0)
        params.best_dev_wer = saved.get("best_dev_wer", 100.0)
        params.best_train_epoch = saved.get("best_train_epoch", -1)
        params.best_valid_epoch = saved.get("best_valid_epoch", -1)
        params.start_epoch = saved.get("cur_epoch", 1)
    else:
        logging.info(f"Initializing from adapted checkpoint {args.init_ckpt}")
        init_obj = torch.load(str(args.init_ckpt), map_location="cpu", weights_only=False)
        model.load_state_dict(init_obj["model"], strict=True)
        model_avg.load_state_dict(init_obj["model"], strict=True)

    cuts_train = CutSet.from_jsonl(str(args.data_dir / "manifests" / "cuts_train.jsonl.gz"))
    cuts_dev = CutSet.from_jsonl(str(args.data_dir / "manifests" / "cuts_dev.jsonl.gz"))
    dev_decode_cuts = list(cuts_dev)[:45]

    input_transforms = [
        SpecAugment(
            time_warp_factor=80,
            num_frame_masks=6,
            features_mask_size=27,
            num_feature_masks=2,
            frames_mask_size=80,
        )
    ]

    train_dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
        input_transforms=input_transforms,
        return_cuts=True,
    )
    valid_dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
        return_cuts=True,
    )

    train_sampler = DynamicBucketingSampler(
        cuts_train,
        max_duration=args.max_duration,
        shuffle=True,
        num_buckets=20,
        drop_last=False,
        seed=args.seed,
    )
    valid_sampler = DynamicBucketingSampler(
        cuts_dev,
        max_duration=200.0,
        shuffle=False,
    )

    train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, num_workers=1, persistent_workers=True)
    valid_dl = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=None, num_workers=0)

    total_train_audio_sec = sum(float(c.duration) for c in cuts_train)
    est_batches_per_epoch = max(1, int(total_train_audio_sec / max(20.0, args.max_duration * 0.75)))
    train_start_wall = time.time()
    total_audio_seconds_seen = 0.0
    status_path = args.reports_dir / "training_live_status.json"
    docs_status_path = ROOT / "docs" / "STATUS.md"
    last_ckpt_written = str(args.init_ckpt)

    try:
        for epoch in range(params.start_epoch, params.num_epochs + 1):
            params.cur_epoch = epoch
            fix_random_seed(args.seed + epoch)
            train_sampler.set_epoch(epoch)
            scheduler.step_epoch(epoch - 1)

            model.train()
            tot_loss = MetricsTracker()
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_dl):
                cuts_in_batch = batch["supervisions"]["cut"]
                batch_audio_sec = sum(float(c.duration) for c in cuts_in_batch)
                total_audio_seconds_seen += batch_audio_sec

                set_batch_count(model, 100000.0 + params.batch_idx_train)

                with torch_autocast(dtype=torch.float16, enabled=params.use_fp16):
                    loss, loss_info = compute_loss(
                        params=params,
                        model=model,
                        sp=sp,
                        batch=batch,
                        is_training=True,
                        device=device,
                    )
                    scaled_loss = loss / args.grad_accum_steps

                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Non-finite training loss at epoch={epoch} batch={batch_idx}: {loss.item()}")

                scaler.scale(scaled_loss).backward()
                tot_loss = tot_loss + loss_info

                is_update_step = ((batch_idx + 1) % args.grad_accum_steps == 0)
                if is_update_step:
                    params.batch_idx_train += 1
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step_batch(params.batch_idx_train)

                    if params.batch_idx_train % params.average_period == 0:
                        update_averaged_model(
                            params=params,
                            model_cur=model,
                            model_avg=model_avg,
                        )

                    if params.batch_idx_train % 25 == 0 or params.batch_idx_train == 1:
                        elapsed = max(1.0, time.time() - train_start_wall)
                        frame_loss = tot_loss["loss"] / max(1, tot_loss["frames"])
                        pruned_f = tot_loss["pruned_loss"] / max(1, tot_loss["frames"])
                        ctc_f = tot_loss["ctc_loss"] / max(1, tot_loss["frames"])
                        peak_alloc = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
                        peak_res = round(torch.cuda.max_memory_reserved(device) / (1024 ** 2), 2)
                        throughput = round(total_audio_seconds_seen / elapsed, 2)
                        epoch_frac = (epoch - params.start_epoch) + min(0.99, (batch_idx + 1) / est_batches_per_epoch)
                        rem_epochs = max(0.0, (params.num_epochs - params.start_epoch + 1) - epoch_frac)
                        sec_per_epoch = elapsed / max(0.05, epoch_frac)
                        eta_min = round((rem_epochs * sec_per_epoch) / 60.0, 1)

                        logging.info(
                            f"Epoch {epoch}/{params.num_epochs} | update {params.batch_idx_train} | "
                            f"loss={frame_loss:.4f} (pruned={pruned_f:.4f}, ctc={ctc_f:.4f}) | "
                            f"audio_hrs={total_audio_seconds_seen/3600.0:.2f} | "
                            f"peak_vram={peak_alloc}MiB | speed={throughput}x RT | eta={eta_min}m"
                        )
                        write_live_status(
                            status_path,
                            docs_status_path,
                            {
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "state": "running",
                                "pid": os.getpid(),
                                "cur_epoch": epoch,
                                "num_epochs": params.num_epochs,
                                "batch_idx_train": params.batch_idx_train,
                                "audio_hours_seen": round(total_audio_seconds_seen / 3600.0, 3),
                                "train_loss": round(frame_loss, 4),
                                "best_valid_loss": round(params.best_valid_loss, 4) if params.best_valid_loss < 1e8 else None,
                                "best_valid_epoch": params.best_valid_epoch,
                                "best_dev_cer": params.best_dev_cer,
                                "best_dev_wer": params.best_dev_wer,
                                "peak_allocated_mib": peak_alloc,
                                "peak_reserved_mib": peak_res,
                                "audio_sec_per_wall_sec": throughput,
                                "eta_minutes": eta_min,
                                "latest_checkpoint": last_ckpt_written,
                            },
                        )

                    if params.batch_idx_train % params.save_every_n == 0:
                        save_checkpoint_impl(
                            filename=latest_ckpt,
                            model=model,
                            model_avg=model_avg,
                            params=params,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            sampler=train_sampler,
                            rank=0,
                        )
                        last_ckpt_written = str(latest_ckpt)

            train_epoch_loss = tot_loss["loss"] / max(1, tot_loss["frames"])
            if train_epoch_loss < params.best_train_loss:
                params.best_train_loss = train_epoch_loss
                params.best_train_epoch = epoch

            val_metrics = compute_validation_metrics(params, model, sp, valid_dl, dev_decode_cuts, device)
            val_loss = val_metrics["valid_loss"]
            dev_wer = val_metrics["dev_subset_wer"]
            dev_cer = val_metrics["dev_subset_cer"]
            logging.info(
                f"End of Epoch {epoch}: train_loss={train_epoch_loss:.4f}, valid_loss={val_loss:.4f} "
                f"(pruned={val_metrics['valid_pruned_loss']:.4f}, ctc={val_metrics['valid_ctc_loss']:.4f}) | "
                f"DEV WER={dev_wer:.2f}% CER={dev_cer:.2f}%"
            )
            logging.info(f"  DEV Sample REF: {val_metrics['sample_ref']}")
            logging.info(f"  DEV Sample HYP: {val_metrics['sample_hyp']}")

            epoch_ckpt = args.exp_dir / f"epoch-{epoch}.pt"
            save_checkpoint_impl(
                filename=epoch_ckpt,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                sampler=train_sampler,
                rank=0,
            )
            copyfile(src=epoch_ckpt, dst=latest_ckpt)
            last_ckpt_written = str(epoch_ckpt)

            if dev_cer < params.best_dev_cer:
                params.best_dev_cer = dev_cer
                params.best_dev_wer = dev_wer
                copyfile(src=epoch_ckpt, dst=args.exp_dir / "best-dev-cer.pt")

            if val_loss < params.best_valid_loss:
                params.best_valid_loss = val_loss
                params.best_valid_epoch = epoch
                copyfile(src=epoch_ckpt, dst=args.exp_dir / "best-valid-loss.pt")

            remove_checkpoints(
                out_dir=args.exp_dir,
                topk=params.keep_last_k,
                rank=0,
            )

        elapsed_total = max(1.0, time.time() - train_start_wall)
        write_live_status(
            status_path,
            docs_status_path,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "state": "completed",
                "pid": os.getpid(),
                "cur_epoch": params.num_epochs,
                "num_epochs": params.num_epochs,
                "batch_idx_train": params.batch_idx_train,
                "audio_hours_seen": round(total_audio_seconds_seen / 3600.0, 3),
                "train_loss": round(params.best_train_loss, 4),
                "best_valid_loss": round(params.best_valid_loss, 4),
                "best_valid_epoch": params.best_valid_epoch,
                "best_dev_cer": params.best_dev_cer,
                "best_dev_wer": params.best_dev_wer,
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / (1024 ** 2), 2),
                "audio_sec_per_wall_sec": round(total_audio_seconds_seen / elapsed_total, 2),
                "eta_minutes": 0.0,
                "latest_checkpoint": last_ckpt_written,
            },
        )
        logging.info("Training completed successfully!")
    finally:
        lock_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
