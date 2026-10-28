import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import sentencepiece as spm
import torch
import torch.nn as nn


VOCAB_DEPENDENT_KEYS = {
    "decoder.embedding.weight",
    "joiner.output_linear.weight",
    "joiner.output_linear.bias",
    "simple_am_proj.weight",
    "simple_am_proj.bias",
    "simple_lm_proj.weight",
    "simple_lm_proj.bias",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_token_overlap_mapping(
    src_bpe_path: Optional[Path],
    dst_bpe_path: Optional[Path],
) -> Dict[int, int]:
    mapping: Dict[int, int] = {0: 0, 1: 1, 2: 2}
    if src_bpe_path and dst_bpe_path and src_bpe_path.exists() and dst_bpe_path.exists():
        src_sp = spm.SentencePieceProcessor()
        dst_sp = spm.SentencePieceProcessor()
        src_sp.load(str(src_bpe_path))
        dst_sp.load(str(dst_bpe_path))
        src_piece_to_id = {src_sp.id_to_piece(i): i for i in range(3, src_sp.get_piece_size())}
        for dst_id in range(3, dst_sp.get_piece_size()):
            piece = dst_sp.id_to_piece(dst_id)
            if piece in src_piece_to_id:
                mapping[dst_id] = src_piece_to_id[piece]
    return mapping


def adapt_pretrained_checkpoint(
    src_ckpt_path: Path,
    target_model: nn.Module,
    out_ckpt_path: Path,
    src_bpe_path: Optional[Path] = None,
    dst_bpe_path: Optional[Path] = None,
    reuse_matching_tokens: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    raw_ckpt = torch.load(str(src_ckpt_path), map_location="cpu", weights_only=False)
    src_state: Dict[str, torch.Tensor] = raw_ckpt.get("model", raw_ckpt)

    cleaned_src_state: Dict[str, torch.Tensor] = {}
    for k, v in src_state.items():
        clean_k = k[len("module.") :] if k.startswith("module.") else k
        cleaned_src_state[clean_k] = v

    dst_state = target_model.state_dict()
    token_map = build_token_overlap_mapping(src_bpe_path, dst_bpe_path) if reuse_matching_tokens else {}

    retained_keys: List[str] = []
    reinitialized_vocab_keys: List[str] = []
    missing_in_src: List[str] = []
    unexpected_in_src: List[str] = [k for k in cleaned_src_state if k not in dst_state]
    reused_token_rows = len(token_map)

    new_state_dict: Dict[str, torch.Tensor] = {}

    for k, dst_tensor in dst_state.items():
        if k not in cleaned_src_state:
            missing_in_src.append(k)
            new_state_dict[k] = dst_tensor.clone()
            continue

        src_tensor = cleaned_src_state[k]
        if k in VOCAB_DEPENDENT_KEYS:
            reinitialized_vocab_keys.append(k)
            adapted = dst_tensor.clone()
            with torch.no_grad():
                if k == "decoder.embedding.weight":
                    emb_std = float(src_tensor[1:].std().item()) if src_tensor.shape[0] > 1 else 1.0
                    nn.init.normal_(adapted, mean=0.0, std=emb_std)
                    adapted[0].zero_()
                elif k == "joiner.output_linear.weight":
                    join_std = float(src_tensor.std().item())
                    nn.init.normal_(adapted, mean=0.0, std=join_std)
                elif k == "simple_am_proj.weight":
                    am_std = float(src_tensor.std().item())
                    nn.init.normal_(adapted, mean=0.0, std=am_std)
                elif k == "simple_lm_proj.weight":
                    nn.init.normal_(adapted, mean=0.0, std=0.09)
                elif k.endswith(".bias"):
                    adapted.zero_()
                if reuse_matching_tokens:
                    for dst_tok_id, src_tok_id in token_map.items():
                        if dst_tok_id < adapted.shape[0] and src_tok_id < src_tensor.shape[0]:
                            adapted[dst_tok_id] = src_tensor[src_tok_id].to(adapted.dtype)
            new_state_dict[k] = adapted
        else:
            if tuple(src_tensor.shape) != tuple(dst_tensor.shape):
                raise ValueError(
                    f"Shape mismatch on non-vocabulary parameter {k}: "
                    f"src={tuple(src_tensor.shape)} vs dst={tuple(dst_tensor.shape)}"
                )
            retained_keys.append(k)
            new_state_dict[k] = src_tensor.to(dst_tensor.dtype).clone()

    if missing_in_src:
        raise ValueError(f"Target model has missing keys not found in source checkpoint: {missing_in_src}")

    target_model.load_state_dict(new_state_dict, strict=True)

    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": target_model.state_dict()}, str(out_ckpt_path))

    total_params = sum(p.numel() for p in target_model.parameters())
    encoder_params = sum(p.numel() for p in target_model.encoder.parameters())
    decoder_params = sum(p.numel() for p in target_model.decoder.parameters())
    joiner_params = sum(p.numel() for p in target_model.joiner.parameters())
    deployed_params = encoder_params + decoder_params + joiner_params

    report = {
        "source_checkpoint_path": str(src_ckpt_path),
        "source_checkpoint_sha256": sha256_file(src_ckpt_path),
        "adapted_checkpoint_path": str(out_ckpt_path),
        "adapted_checkpoint_sha256": sha256_file(out_ckpt_path),
        "retained_tensor_count": len(retained_keys),
        "reinitialized_vocab_tensor_count": len(reinitialized_vocab_keys),
        "missing_tensor_count": len(missing_in_src),
        "unexpected_tensor_count": len(unexpected_in_src),
        "reinitialized_vocab_keys": reinitialized_vocab_keys,
        "unexpected_keys": unexpected_in_src,
        "reused_token_row_count": reused_token_rows,
        "blank_id_preserved": getattr(target_model.decoder, "blank_id", 0) == 0,
        "parameter_counts": {
            "total_training_params": total_params,
            "deployed_asr_params": deployed_params,
            "encoder_params": encoder_params,
            "decoder_params": decoder_params,
            "joiner_params": joiner_params,
        },
    }
    return report
