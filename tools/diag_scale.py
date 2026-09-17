# -*- coding: utf-8 -*-
"""标度诊断：kaldiio 加载 vs soundfile 加载（[-1,1]）vs soundfile*32768，看 AED 输出差异。"""
import os
import sys

import numpy as np
import soundfile as sf
import torch
import kaldiio

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, "FireRedASR2S")
sys.path.insert(0, REPO)
if not os.environ.get("USERNAME"):
    os.environ["USERNAME"] = "dsh"

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config  # noqa: E402

asr = FireRedAsr2.from_pretrained(
    "aed", os.path.join(REPO, "pretrained_models", "FireRedASR2-AED"),
    FireRedAsr2Config(use_gpu=True, use_half=False, beam_size=3, nbest=1,
                      return_timestamp=True))
print("AED ready\n", flush=True)

for name, sec in [("FireRedASR2S/assets/hello_zh.wav", 3),
                  ("test_audio.flac", 11),
                  ("mic_debug.wav", 10)]:
    path = os.path.join(BASE, name.replace("/", os.sep))
    if not os.path.exists(path):
        print(f"{name}: NOT FOUND", flush=True)
        continue
    sr_k, arr_k = kaldiio.load_mat(path)
    wav_sf, sr_sf = sf.read(path, dtype="float32")
    if wav_sf.ndim > 1:
        wav_sf = wav_sf.mean(axis=1)
    n = int(sec * sr_k)
    print(f"=== {name}", flush=True)
    print(f"  kaldiio: sr={sr_k} dtype={arr_k.dtype} min={arr_k.min():.4f} max={arr_k.max():.4f}", flush=True)
    print(f"  soundfile: sr={sr_sf} min={wav_sf.min():.4f} max={wav_sf.max():.4f}", flush=True)

    a_k = np.asarray(arr_k[:n], dtype=np.float32)
    a_sf = wav_sf[:n].astype(np.float32)
    a_sc = a_sf * 32768.0

    for tag, a in [("kaldiio", a_k), ("sf[-1,1]", a_sf), ("sf*32768", a_sc)]:
        r = asr.transcribe(["x"], [(16000, a)])[0]
        print(f"  [{tag:10s}] text=[{r.get('text')}] conf={r.get('confidence')}", flush=True)
    print(flush=True)
