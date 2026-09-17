# -*- coding: utf-8 -*-
"""解码速度还能压多少：fp32 vs fp16（CUDA Graph 下），并测 encoder/每步耗时。

用法: python bench_precision.py fp32|fp16
"""
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

PREC = sys.argv[1] if len(sys.argv) > 1 else "fp32"
if PREC == "fp16":                      # 逼出 .half()（默认会走 bf16，而 bf16 会破坏 forced_align）
    torch.cuda.is_bf16_supported = lambda *a, **k: False

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config  # noqa: E402
import firede_graph                                                   # noqa: E402

torch.set_grad_enabled(False)
asr = FireRedAsr2.from_pretrained(
    "aed", os.path.join(REPO, "pretrained_models", "FireRedASR2-AED"),
    FireRedAsr2Config(use_gpu=True, use_half=(PREC == "fp16"), beam_size=3, nbest=1,
                      return_timestamp=True))
print(f"=== {PREC} (model dtype={next(asr.model.parameters()).dtype}, "
      f"VRAM {torch.cuda.memory_allocated()/2**30:.2f}G) ===", flush=True)

wav, sr = sf.read(os.path.join(BASE, "test_call.wav"), dtype="float32")
wav *= 32768.0
segs = {"10s": wav[20 * sr:30 * sr], "20s": wav[20 * sr:40 * sr]}


def bench(fn, n=3):
    r = fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.time()
        r = fn()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    return min(ts), r


# --- 编码器单测 ---
seg = segs["20s"]
feats, lengths, durs, _, _ = asr.feat_extractor([(sr, seg)], ["x"])
feats, lengths = feats.cuda(), lengths.cuda()
if PREC == "fp16":
    feats = feats.half()
t, (eo, el, em) = bench(lambda: asr.model.encoder(feats, lengths))
n_frames = feats.shape[1]
print(f"encoder: {t*1000:6.1f} ms  (T={n_frames} 帧, {len(seg)/sr:.0f}s)", flush=True)
del eo, el, em
torch.cuda.empty_cache()

# --- 官方路径（无时间戳，隔离解码成本）---
asr.config.return_timestamp = False
t_off, _ = bench(lambda: asr.transcribe(["x"], [(sr, seg)]), n=3)
print(f"官方解码 transcribe(无时间戳): {t_off*1000:6.1f} ms  RTF={t_off/(len(seg)/sr):.3f}", flush=True)

# --- Graph 路径 ---
firede_graph.patch_graph_decode(asr, verbose=False)
asr.transcribe(["x"], [(sr, seg)])                      # 触发捕获
t_g, r = bench(lambda: asr.transcribe(["x"], [(sr, seg)]), n=3)
print(f"graph 解码 transcribe(无时间戳): {t_g*1000:6.1f} ms  RTF={t_g/(len(seg)/sr):.3f}", flush=True)

asr.config.return_timestamp = True
t_gt, r2 = bench(lambda: asr.transcribe(["x"], [(sr, seg)]), n=3)
print(f"graph 解码 transcribe(带时间戳): {t_gt*1000:6.1f} ms  RTF={t_gt/(len(seg)/sr):.3f}", flush=True)
n_tok = len(r2.get("timestamp") or [])
print(f"  输出 {n_tok} token → 每步约 {(t_gt - t)*1000/max(n_tok,1):.2f} ms"
      f"（扣除编码器 {t*1000:.1f}ms）", flush=True)
print(f"  文本: {r2.get('text','')[:50]}", flush=True)
