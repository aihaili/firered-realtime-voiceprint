# -*- coding: utf-8 -*-
"""查看声纹库内容：每个说话人多少条嵌入、最后一句是什么。"""
import json
import os
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "speakers_firered.json"
d = json.load(open(path, encoding="utf-8"))
s = d["speakers"]
print(f"{path}: {len(s)} speakers")
for k, v in sorted(s.items(), key=lambda x: int(x[0].split("_")[1]) if "_" in x[0] else 0):
    txt = v["samples"][-1]["text"][:38] if v["samples"] else ""
    print(f"  {k:8s} embs={len(v['embeddings']):2d}  name={v.get('name','')!r:8s} {txt}")
