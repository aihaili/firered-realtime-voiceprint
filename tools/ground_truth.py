# -*- coding: utf-8 -*-
"""建立录音真值：每 20s 一块，打印文本 + 与前一块的声纹相似度，看有几个人、在什么时间。"""
import os
import sys

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
from modelscope.pipelines import pipeline                             # noqa: E402

torch.set_grad_enabled(False)
asr = FireRedAsr2.from_pretrained(
    "aed", os.path.join(REPO, "pretrained_models", "FireRedASR2-AED"),
    FireRedAsr2Config(use_gpu=True, use_half=False, beam_size=3, nbest=1, return_timestamp=False))
firede_graph.patch_graph_decode(asr, verbose=False)
sv = pipeline(task="speaker-verification", model="iic/speech_campplus_sv_zh-cn_16k-common")

wav, sr = sf.read(os.path.join(BASE, "mic_debug.wav"), dtype="float32")
if wav.ndim > 1:
    wav = wav.mean(axis=1)
dur = len(wav) / sr
print(f"总长 {dur:.1f}s\n", flush=True)

CH = 10.0
embs = []
for i in range(int(dur // CH) + 1):
    a = wav[int(i * CH * sr):int((i + 1) * CH * sr)]
    if len(a) < sr:
        break
    rms = float(np.sqrt(np.mean(a ** 2)))
    txt = asr.transcribe(["x"], [(sr, (a * 32768.0))])[0].get("text", "") if rms > 0.005 else ""
    r = sv([a], output_emb=True)
    e = np.array(r["embs"][0], dtype=np.float32)
    e /= np.linalg.norm(e)
    embs.append(e)
    sims = " ".join(f"{float(e @ p):.2f}" for p in embs[:-1]) or "-"
    print(f"[{i*CH:5.1f}-{(i+1)*CH:5.1f}s] rms={rms:.4f} 与前面各块相似度: {sims}", flush=True)
    print(f"        {txt[:70]}", flush=True)

print("\n=== 相邻块相似度矩阵（判断换人点）===", flush=True)
M = np.stack(embs)
for i in range(len(M)):
    print(f"  block{i} [{i*CH:.0f}s]: " + " ".join(f"{float(M[i] @ M[j]):5.2f}" for j in range(len(M))),
          flush=True)
