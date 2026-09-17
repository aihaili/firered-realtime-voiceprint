# -*- coding: utf-8 -*-
"""定位 graph 解码与官方解码的差异点：逐 token 对比（含 beam=1 对照）。"""
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, "FireRedASR2S")
sys.path.insert(0, REPO)
sys.path.insert(0, BASE)
os.environ.setdefault("USERNAME", "00")

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config  # noqa: E402
import firede_graph                                                   # noqa: E402

torch.set_grad_enabled(False)


def load(beam):
    return FireRedAsr2.from_pretrained(
        "aed", os.path.join(REPO, "pretrained_models", "FireRedASR2-AED"),
        FireRedAsr2Config(use_gpu=True, use_half=False, beam_size=beam, nbest=1,
                          return_timestamp=False))


def raw_ids(asr, seg, sr):
    """直接拿解码器的 yseq（绕开文本层）。"""
    feats, lengths, durs, _, _ = asr.feat_extractor([(sr, seg)], ["x"])
    feats, lengths = feats.cuda(), lengths.cuda()
    eo, el, em = asr.model.encoder(feats, lengths)
    hyps = asr.model.decoder.batch_beam_search(eo, em, asr.config.beam_size, 1, 0,
                                               asr.config.softmax_smoothing,
                                               asr.config.aed_length_penalty,
                                               asr.config.eos_penalty, None, 0.0)
    best = max(hyps[0], key=lambda h: float(h["confidence"]))
    return [int(v) for v in best["yseq"].cpu()], hyps[0]


wav, sr = sf.read(os.path.join(BASE, "test_call.wav"), dtype="float32")
wav = wav * 32768.0
seg = wav[35 * sr:65 * sr]

asr = load(3)
off_ids, off_hyps = raw_ids(asr, seg, sr)
print(f"官方 beam=3: {len(off_ids)} tokens, conf={float(off_hyps[0]['confidence']):.4f}")

gbs = firede_graph.patch_graph_decode(asr, verbose=False)
feats, lengths, durs, _, _ = asr.feat_extractor([(sr, seg)], ["x"])
feats, lengths = feats.cuda(), lengths.cuda()
eo, el, em, ck, cv = gbs._encode(feats, lengths)
g = gbs._get_graph(eo.size(1))
ys, sc, cf, steps = g.run(eo, ck, cv, 300, 4)
yl = (ys != asr.model.decoder.eos_id).sum(dim=-1).float()
pen = torch.pow((5 + yl) / 6.0, asr.config.aed_length_penalty)
best_i = int(torch.argmax(sc.view(-1) / pen).item())
gr_ids = [int(v) for v in ys[best_i][1:int(yl[best_i].item())].cpu()]
print(f"graph beam=3: {len(gr_ids)} tokens, steps={steps}, all beams len="
      f"{[int(x) for x in yl.tolist()]}")

d = next((i for i in range(min(len(off_ids), len(gr_ids))) if off_ids[i] != gr_ids[i]), None)
print(f"\n首个差异位置: {d}")
if d is not None:
    lo = max(0, d - 6)
    print(f"官方 [{lo}:{d+6}]: {off_ids[lo:d+6]}")
    print(f"graph[{lo}:{d+6}]: {gr_ids[lo:d+6]}")
    tok = asr.tokenizer
    print(f"官方文本片段: {tok.detokenize(off_ids[lo:d+6])}")
    print(f"graph文本片段: {tok.detokenize(gr_ids[lo:d+6])}")
print(f"\n官方尾部: {off_ids[-8:]}")
print(f"graph尾部: {gr_ids[-8:]}")
