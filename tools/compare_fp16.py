# -*- coding: utf-8 -*-
"""比对 fp32 / fp16 两次验证结果：文本、时间戳、置信度、耗时。"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
a = json.load(open(os.path.join(BASE, "verify_fp16_fp32.json"), encoding="utf-8"))
b = json.load(open(os.path.join(BASE, "verify_fp16_fp16.json"), encoding="utf-8"))

same = 0
print(f"{'用例':16s} {'fp32(s)':>8s} {'fp16(s)':>8s} {'加速':>6s}  文本  时间戳  dconf")
for k in a:
    va, vb = a[k], b[k]
    txt_same = va["text"] == vb["text"]
    ts_same = va["ts_head"] == vb["ts_head"] and va["n_ts"] == vb["n_ts"]
    dconf = abs((va["conf"] or 0) - (vb["conf"] or 0))
    same += txt_same
    print(f"{k:16s} {va['time']:8.3f} {vb['time']:8.3f} "
          f"{va['time']/vb['time']:5.2f}x  {'一致' if txt_same else '**不同**'}  "
          f"{'一致' if ts_same else '差异'}  {dconf:.4f}")
    if not txt_same:
        print(f"    fp32: {va['text']}")
        print(f"    fp16: {vb['text']}")

print(f"\n文本一致: {same}/{len(a)}")
tot_a = sum(a[k]["time"] for k in a)
tot_b = sum(b[k]["time"] for k in b)
print(f"总耗时 fp32 {tot_a:.2f}s → fp16 {tot_b:.2f}s = {tot_a/tot_b:.2f}x")
