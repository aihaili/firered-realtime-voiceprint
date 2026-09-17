# -*- coding: utf-8 -*-
"""打印 AED checkpoint 的 args，确认解码器规模与是否有 NAR/CTC 相关配置。"""
import os
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
ckpt = os.path.join(BASE, "FireRedASR2S", "pretrained_models", "FireRedASR2-AED", "model.pth.tar")
pkg = torch.load(ckpt, map_location="cpu", weights_only=False)
print("keys:", list(pkg.keys()))
args = pkg["args"]
for k, v in sorted(vars(args).items()):
    if not k.startswith("_"):
        print(f"  {k} = {v}")
