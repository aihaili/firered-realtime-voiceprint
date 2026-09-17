# -*- coding: utf-8 -*-
"""检查编码器输出 mask 是否含 0（决定 graph 路径是否会走 fallback）。"""
import os
import sys

import numpy as np
import soundfile as sf
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, "FireRedASR2S")
sys.path.insert(0, REPO)
os.environ.setdefault("USERNAME", "00")

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config  # noqa: E402

torch.set_grad_enabled(False)
asr = FireRedAsr2.from_pretrained(
    "aed", os.path.join(REPO, "pretrained_models", "FireRedASR2-AED"),
    FireRedAsr2Config(use_gpu=True, use_half=False, beam_size=3, nbest=1, return_timestamp=True))

wav, sr = sf.read(os.path.join(BASE, "test_call.wav"), dtype="float32")
wav *= 32768.0
for sec in [3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 20.7]:
    n = int(sec * sr)
    seg = wav[20 * sr:20 * sr + n]
    feats, lengths, durs, _, _ = asr.feat_extractor([(sr, seg)], ["x"])
    eo, el, em = asr.model.encoder(feats.cuda(), lengths.cuda())
    zeros = int(em.eq(0).sum().item())
    print(f"{sec:5.1f}s: T_in={feats.shape[1]:4d} -> T_enc={eo.shape[1]:4d} "
          f"enc_len={int(el.item()):4d} mask_zeros={zeros}", flush=True)
