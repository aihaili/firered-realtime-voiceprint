# -*- coding: utf-8 -*-
"""合成 4 说话人测试音频：通话(A/B) + 小说朗读(C) + 英文演讲(D)，中间留 1s 静音。"""
import os

import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.abspath(__file__))


def load16k(path, a=0.0, b=None):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != 16000:
        idx = np.arange(0, len(w), sr / 16000.0)
        w = np.interp(idx, np.arange(len(w)), w).astype(np.float32)
        sr = 16000
    return w[int(a * sr):int(b * sr) if b else len(w)]


mic = os.path.join(BASE, "mic_debug.wav")
segs = [
    ("A/B 通话 0-30s", load16k(mic, 0, 30)),
    ("C 朗读 70-95s", load16k(mic, 70, 95)),
    ("D 英文演讲", load16k(os.path.join(BASE, "test_audio.flac"), 0, 12)),
    ("B 通话 30-45s", load16k(mic, 30, 45)),   # 再回到通话，检验能否回到原说话人
]
gap = np.zeros(int(1.0 * 16000), dtype=np.float32)
out = []
for name, w in segs:
    print(f"  + {name}: {len(w)/16000:.1f}s", flush=True)
    out.extend([w, gap])
audio = np.concatenate(out)
sf.write(os.path.join(BASE, "test_4spk.wav"), audio, 16000)
print(f"写出 test_4spk.wav: {len(audio)/16000:.1f}s", flush=True)
