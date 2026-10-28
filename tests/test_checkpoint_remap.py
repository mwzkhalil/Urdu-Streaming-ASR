from pathlib import Path
import torch
import torch.nn as nn

from urdu_streaming_asr.checkpoint_remap import adapt_pretrained_checkpoint


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)


class DummyDecoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.conv = nn.Conv1d(16, 16, kernel_size=2, groups=4, bias=False)


class DummyJoiner(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.encoder_proj = nn.Linear(16, 16)
        self.decoder_proj = nn.Linear(16, 16)
        self.output_linear = nn.Linear(16, vocab_size)


class DummyTransducer(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.encoder = DummyEncoder()
        self.decoder = DummyDecoder(vocab_size)
        self.joiner = DummyJoiner(vocab_size)
        self.simple_am_proj = nn.Linear(16, vocab_size)
        self.simple_lm_proj = nn.Linear(16, vocab_size)


def test_adapt_pretrained_checkpoint(tmp_path: Path):
    src_model = DummyTransducer(vocab_size=500)
    with torch.no_grad():
        src_model.encoder.linear.weight.fill_(0.42)
        src_model.decoder.conv.weight.fill_(0.15)
        src_model.decoder.embedding.weight[0].fill_(0.99)
        src_model.joiner.output_linear.weight[0].fill_(-0.77)

    src_ckpt = tmp_path / "pretrained.pt"
    out_ckpt = tmp_path / "adapted.pt"
    torch.save({"model": src_model.state_dict()}, str(src_ckpt))

    dst_model = DummyTransducer(vocab_size=1024)
    report = adapt_pretrained_checkpoint(src_ckpt, dst_model, out_ckpt, reuse_matching_tokens=False)

    assert report["blank_id_preserved"] is True
    assert report["missing_tensor_count"] == 0
    assert report["reinitialized_vocab_tensor_count"] == 7
    assert torch.allclose(dst_model.encoder.linear.weight, torch.full_like(dst_model.encoder.linear.weight, 0.42))
    assert torch.allclose(dst_model.decoder.conv.weight, torch.full_like(dst_model.decoder.conv.weight, 0.15))
    assert torch.allclose(dst_model.decoder.embedding.weight[0], torch.zeros(16))
