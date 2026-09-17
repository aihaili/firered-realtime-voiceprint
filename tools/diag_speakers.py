# -*- coding: utf-8 -*-
"""声纹相似度诊断：把录音切成 5s 窗口，算 campplus 嵌入，看跨说话人的真实相似度。

用法: python diag_speakers.py [wav] [窗口秒数]
"""
import os
import sys

import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("USERNAME", "00")
WAV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "mic_debug.wav")
WIN = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

from modelscope.pipelines import pipeline  # noqa: E402

sv = pipeline(task="speaker-verification", model="iic/speech_campplus_sv_zh-cn_16k-common")
wav, sr = sf.read(WAV, dtype="float32")
if wav.ndim > 1:
    wav = wav.mean(axis=1)
print(f"{WAV}: {len(wav)/sr:.1f}s @ {sr}Hz, 窗口 {WIN}s", flush=True)


def emb(a):
    r = sv([a], output_emb=True)
    e = np.array(r["embs"][0], dtype=np.float32)
    n = float(np.linalg.norm(e))
    return e / n if n > 0 else None


embs, tags = [], []
n = int(WIN * sr)
for i in range(0, len(wav) - n + 1, n):
    seg = wav[i:i + n]
    rms = float(np.sqrt(np.mean(seg ** 2)))
    e = emb(seg) if rms > 0.005 else None
    embs.append(e)
    tags.append(f"{i/sr:.0f}s")
    print(f"  [{i/sr:6.1f}s] rms={rms:.4f} emb={'ok' if e is not None else '低能量跳过'}",
          flush=True)

valid = [(t, e) for t, e in zip(tags, embs) if e is not None]
print(f"\n有效窗口 {len(valid)}/{len(tags)}", flush=True)
if len(valid) < 2:
    sys.exit(0)

M = np.stack([e for _, e in valid])
S = M @ M.T
print("\n=== 相似度矩阵（行=窗口）===")
hdr = "        " + " ".join(f"{t:>6s}" for t, _ in valid)
print(hdr, flush=True)
for i, (t, _) in enumerate(valid):
    row = " ".join(f"{S[i, j]:6.2f}" for j in range(len(valid)))
    print(f"{t:>6s}  {row}", flush=True)

# 以第一个窗口为基准（若它是 A），看后续窗口与它的相似度
print("\n=== 与「前 3 个窗口平均」的相似度（近似第一个说话人质心）===", flush=True)
c0 = M[:3].mean(axis=0)
c0 /= np.linalg.norm(c0)
for (t, _), e in zip(valid, M):
    print(f"  {t:>6s}: {float(e @ c0):.3f}", flush=True)
