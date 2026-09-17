# -*- coding: utf-8 -*-
"""微基准：本机 CUDA 算子派发开销 & CUDA Graph 回放收益（决定是否值得上 graph）。"""
import time

import torch

dev = "cuda"
torch.set_grad_enabled(False)

# 模拟解码器一步：240 个小型算子（gemv + layernorm + softmax + gather 混合）
NB, H, D = 3, 1280, 1280
x = torch.randn(NB, 1, D, device=dev)
w = torch.randn(D, D, device=dev) * 0.01


def step():
    z = x
    for _ in range(10):  # 24 组 × 10 = 240 算子
        z = z @ w
        z = torch.nn.functional.layer_norm(z, (D,))
        z = torch.softmax(z, dim=-1)
        z = z.view(NB, 1, 16, 80).transpose(1, 2).contiguous().view(NB, 1, D)
        z = z + x
        z = torch.nn.functional.gelu(z)
        z = z * 0.5 + x * 0.5
        z = z.view(NB, 1, 20, 64).transpose(1, 2).reshape(NB, 1, D)
        z = torch.nn.functional.layer_norm(z, (D,))
        z = torch.gather(z, 2, torch.zeros(NB, 1, D, dtype=torch.long, device=dev))
        z = z / 8.0
        z = z.view(NB, 1, D)
        z = z - 0.001
        z = z + 0.001
        z = torch.relu(z)
        z = z * 1.0001
        z = torch.sigmoid(z)
        z = z + x
        z = torch.nn.functional.layer_norm(z, (D,))
        z = z @ w
        z = torch.nn.functional.gelu(z)
        z = z + x
        z = torch.clamp(z, -10, 10)
    return z


for _ in range(3):
    step()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    step()
torch.cuda.synchronize()
eager_cpu = (time.time() - t0) / 20

# GPU 纯执行时间（用 profiler 隔离）
from torch.profiler import profile, ProfilerActivity  # noqa: E402
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        step()
    torch.cuda.synchronize()
gpu_ms = sum(getattr(e, "self_device_time_total", 0) for e in prof.key_averages()) / 5 / 1000

# CUDA Graph
g = torch.cuda.CUDAGraph()
side = torch.empty_like(step())
for _ in range(3):
    step()
torch.cuda.synchronize()
with torch.cuda.graph(g):
    out = step()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    g.replay()
torch.cuda.synchronize()
graph_ms = (time.time() - t0) / 20

print(f"eager  每步 {eager_cpu*1000:7.2f} ms")
print(f"  └ GPU kernel 实际执行 {gpu_ms:7.2f} ms  → 派发/CPU 开销 {eager_cpu*1000-gpu_ms:.2f} ms")
print(f"graph  每步 {graph_ms*1000:7.2f} ms   →  相对 eager 加速 {eager_cpu/graph_ms:.2f}x")
