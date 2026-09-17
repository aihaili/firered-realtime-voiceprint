# 推理加速与实时语音转写 —— 优化结论（可复用参考）

> 场景：Windows 10/11 + RTX 3090(24G) + PyTorch 2.8.0+cu128 + Python 3.12
> 对象：实时语音转写（FireRedASR2S = FireRedASR2-AED + FireRedPunc + campplus 声纹）
> 写成"结论 + 判据 + 坑"，不依赖本项目上下文，其它程序可直接照搬方法。
> 原始项目细节见 `FIRERED_NOTES.md`；本文只讲**可迁移的部分**。

---

## 0. 一页结论（TL;DR）

| 优化方向 | 预期 | **实测结果** | 结论 |
|---|---|---|---|
| **CUDA Graph 整步回放** | 消除派发开销 | 解码 **4.1~4.4x**（20s 句子 2.50s→0.53s） | ✅ **首选** |
| **fp16（.half()，不是 bf16）** | 显存减半/带宽减半 | **1.23~1.33x** + 显存 4.47→2.18G，8/8 输出逐字一致 | ✅ 已默认开启 |
| **输入长度分桶（bucketize）** | 少重捕获 | 图捕获 24+ → **8 次**，消除每次 0.5~1s 卡顿 | ✅ 必做（配合 Graph） |
| **live 预览换 CTC 贪心** | 省 GPU | final 0.89→**0.44s**（2x），live 刷新 33→43 次 | ⚠️ 预览会漏字，可选 |
| **增量解码**（cross-attn K/V 只算一次 + self-attn KV 缓存） | 少算重复量 | **1.5x**，输出**逐字一致** | ✅ 保守场景 |
| 输入采样值标度（int16 vs [-1,1]） | — | 不修则**整段输出空文本** | ✅ **必做**（正确性） |
| `beam_size=1` | 解码更快 | **0~5%** | ❌ 排除 |
| `torch.compile(编码器)` | 融合加速 | 收益 <3%（编码器只占 2.5%） | ❌ 排除 |
| `use_half=bf16`（官方默认路径） | 算力翻倍 | 破坏词级时间戳（见坑清单） | ❌ 改用 fp16 |
| TensorRT / Triton | 官方称 12.7x | 官方 runtime 仅 Linux/Docker | ⏸ Windows 不可用 |
| 减少管线冗余（live 全量重转写） | 省 GPU | 重复量 = D²/2T，20s 句子烧 167s 音频算力 | ⚠️ 见下方专项 |

**一句话**：在 Windows/PyTorch 上，小算子密集的自回归解码，**瓶颈几乎从来不是算力，而是算子派发开销**；
CUDA Graph 干掉派发开销后，**瓶颈才轮到权重带宽**——这时 fp16 才真正开始有效
（在派发瓶颈阶段 fp16 实测 0 收益，别提前下结论）。
先量「GPU 实跑时间 : 墙钟时间」，再决定往哪个方向优化。

---

## 1. 诊断方法论（三步定位，套用到任何"推理慢"）

### Step 1 · 组件级 profiling：先找到瓶颈在哪一层
把链路拆成可独立计时的段（特征提取 / 编码器 / 解码器 / 后处理），分别计时。
**本案例的反直觉结果**（20s 音频）：

| 环节 | 耗时 | 占比 |
|---|---|---|
| fbank 特征提取 | 27 ms | 1% |
| **编码器（16 层 Conformer）** | **62 ms** | 2.5% |
| **解码器 beam search** | **2400 ms** | **95%** |
| CTC 时间戳对齐 | 5 ms | 0.2% |

→ 结论：**别优化编码器**（fp16/torch.compile 都作用在这，全是白干）。

> ⚠️ 写 profiling 脚本时**必须包 `with torch.no_grad()`**。我第一版直接调 `model.encoder(...)`，
> autograd 保留了 16 层全部中间激活 → 24G 卡报 "44.5 GiB allocated" 的**假 OOM**，白白排查半天。

### Step 2 · kernel 级 profiling：区分"算力瓶颈"与"派发瓶颈"
```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    run_one_thing()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```
本案例结果：解码一步 **~240 个 CUDA 算子**，`Self CUDA time` 合计 450ms/86 步（≈5.2ms/步），
而 **CPU 侧 `Self CPU time` 856ms**（≈10ms/步）。

**判据**：若「wall time ≫ GPU 实跑时间」，且算子数量巨大、每个算子都很小 → **派发/启动开销瓶颈**，
此时 **fp16/量化/换算法基本无救，只有两条路：减少算子数（融合）或绕过 Python（CUDA Graph）**。

### Step 3 · 微基准：量化每算子开销与 CUDA Graph 上限
写一个约 240 个小算子的函数，分别测：eager 墙钟 / GPU 实跑 / **CUDA Graph 回放**。

```python
g = torch.cuda.CUDAGraph()
for _ in range(3): step()                     # 预热
torch.cuda.synchronize()
with torch.cuda.graph(g):
    out = step()                              # 捕获
torch.cuda.synchronize()
# 之后 g.replay() 即整段回放
```

本机实测（RTX 3090 / Windows / WDDM）：

| 方式 | 每步耗时 |
|---|---|
| eager | 11.04 ms |
| └ 其中 GPU 实际执行 | **0.81 ms** |
| **CUDA Graph 回放** | **0.80 ms**（**13.8x**） |

→ 反推本机**每个小算子派发开销约 42µs**（正常 Linux 上一般 5~10µs；Windows/WDDM 明显更高）。
**这个微基准 5 分钟就能跑完，是决定要不要上 CUDA Graph 的关键证据。**

---

## 2. 有效优化①：增量解码（数学等价，零风险）

### 官方实现的浪费
自回归 beam search 里，每一步、每一层都对**整段编码器输出**重算 cross-attention 的 K/V 投影：
`16 层 × 1998 帧 × 1280 维 ≈ 105 GFLOP / 步`；而同一句的 3 个 beam 用的是**完全相同**的编码器输出。

### 做法
1. **cross-attn K/V 每句只投影一次**；不同 beam 之间用 `(N,B)` 广播算 attention，不要 `repeat` 出 `N*B` 份。
2. **self-attn 用增量 KV 缓存**：预分配 `(NB, heads, maxlen, d_k)`，每步只算新 token 的 q/k/v，`index_copy_` 写入。
3. 丢掉只用于取最后一帧的前缀输出缓存（很多实现里有 `cat(cache, x)` 但最后只取 `[-1]`）。

### 结果
- 20s 句子 2.50s → **1.67s（1.5x）**，RTF 0.125 → 0.084
- **输出与官方逐字一致**（文本、词级时间戳、置信度全部相同，6/6 用例）
- 实现：`firede_fast.py`（约 200 行，用 monkeypatch 挂载，不改官方源码）

---

## 3. 有效优化②：CUDA Graph 整步回放（4x，默认方案）

### 核心思想
把**一整步解码**（16 层 Transformer + beam 剪枝 + KV 重排）捕获成一张 CUDA Graph，
之后每步只 `graph.replay()`，Python 开销归零。

### 实现要点（缺一不可）
1. **所有跨步状态都是预先分配的静态缓冲，图内原地更新**。
2. **变长输入用「静态填充 + 加性 mask」换静态形状**：
   KV cache 预分配到 maxlen，用加性 mask（-1e10）屏蔽未用位置；位置索引用 device 上的 step tensor 在**图内自增**，
   host 每步零交互。
3. **cross-attn K/V 不进图内重排**（它们不含 beam 维度），用 `(N,1)` 广播到 beam。
4. **图按输入长度缓存**（dict + LRU）：本机每图约 180MB（含私有内存池），24 张时显存稳定不涨（实测 43 次捕获无泄漏）。
5. **结束条件**（所有 beam 出 EOS）需要 host 同步 → 每 4 步查一次即可；多跑的步**对结果无害**
   （已 finished 的 beam 只会追加 EOS，不计入长度）。

### 结果
- 20s 句子 2.50s → **0.53s（4.1~4.4x）**，RTF 0.125 → **0.027**
- 输出：多数与官方**逐字一致**；少数句子差 1~2 字（见下方"数值等价性"）
- 实现：`firede_graph.py`（约 250 行）

### ⚠️ 数值等价性：数学等价 ≠ 逐位等价
为了静态形状而把 KV cache 填充到 maxlen 后，softmax 的**归约分组**与"精确短前缀"不同
（多出来的项是精确 0，但树形归约的分组变了）→ 极少数 **beam 打平**的句子会差 1~2 个字。
- 判定方法：拿官方与优化版在多样本上**逐 token 对比**，并交叉验证（换第三方模型、换上下文长度看哪个更合理）
- 本案例：7 用例中 5 例逐字一致，2 例差 1~2 字，交叉验证显示优化版**不更差**
- 需要严格复现官方时，退回 `fast`（逐字一致）或 `official`

---

## 4. 坑清单（症状 → 根因 → 修法）速查

| 症状 | 根因 | 修法 |
|---|---|---|
| 转写整段返回空文本，只有一个 `<sil>`；但官方示例音频仍能出字（confidence 忽高忽低） | 采样值标度错：模型 fbank 期望 **int16 量级**，传了 [-1,1] | 传模型前 `×32768`（注意别的模型如声纹可能仍要 [-1,1]，**同一进程里约定可能相反**） |
| 词级时间戳"看起来正常"但明显是均匀分布的假时间戳 | `use_half` 走 **bf16**，而 `torchaudio.forced_align` 只接受 fp16/fp32/fp64；官方此处是**裸 except**，静默吞掉 | 用 **fp16**：`torch.cuda.is_bf16_supported = lambda *a, **k: False` 后再 `use_half=True` |
| fp16 报 `value cannot be converted to type at::Half without overflow` | mask 用了 `-1e10`，超出 fp16 范围（max 65504） | mask 屏蔽值按 dtype 取：fp32 用 -1e10，fp16 用 -3e4 |
| fp16 报 `Input type (float) and bias type (Half)` | 只把模型转了半精度，输入特征没转 | 特征同步 `.half()`（官方 `asr.py` 在 `use_half` 时也这么做） |
| 每遇到一个新的输入长度就卡 0.5~1s | CUDA Graph 按精确长度缓存 → 长度各异时反复捕获/淘汰 | **长度分桶**（如 64 帧一档），桶内用 mask 屏蔽填充帧 |
| `ModuleNotFoundError: Could not import module 'PreTrainedModel' / 'GenerationMixin'`（import transformers 崩） | 精简环境无 `USERNAME` → `getpass.getuser()` 回退 `import pwd`（Windows 无此模块）→ `torch._inductor` 缓存目录初始化失败 | 导入前 `os.environ.setdefault("USERNAME","dsh")`（`USER` 一起补） |
| 转写文本中间冒出 `<eos><eos>`，且分数不累积 | CUDA Graph 内用了 Python 重绑定 `self.x = ...`（图里写的是**旧缓冲地址**，回放后新对象没被写） | 全部改**原地更新**：`copy_` / `add_` / `index_copy_` / `scatter_` |
| 解码每步比理论算力慢 200 倍 | 循环内写了 `if bool(tensor.any())` → **每步每层一次 GPU→CPU 同步**（16 层×86 步=1376 次） | 判断提到循环外做一次；循环内禁止 `bool()` / `.item()` / `.cpu()` |
| 24G 卡报 "44.5 GiB allocated" 假 OOM | 自己的脚本没包 `torch.no_grad()`，autograd 保留了全部中间激活 | `with torch.no_grad():`（推理脚本一律加） |
| GPU 上两个服务同时跑，测出来的 RTF 翻 2~5 倍 | GPU 争用 | **测速务必单跑**，且确认旧进程真的退出了（Windows 上 kill 父进程不一定会杀子进程） |
| CUDA Graph 捕获报错或静默变慢 | 捕获期做了同步操作 / 形状随步数变化 | 捕获前把 shape 全部固定；捕获期零同步 |
| 结果与官方差 1~2 字 | 静态填充改变了 softmax 归约顺序（beam 打平处） | 接受（交叉验证确认不更差），或改用逐字一致的 `fast` 方案 |

---

## 5. 最终结果（20 秒中文句子，单跑无争用）

| 方案 | 耗时 | RTF | 输出一致性 | 显存 |
|---|---|---|---|---|
| 官方 FireRedASR2-AED | 2.50 s | 0.125 | 基准 | 4.5 GB |
| + 增量解码（`fast`） | 1.67 s | 0.084 | **逐字一致** | 4.5 GB |
| + CUDA Graph（`graph`，fp32） | 0.62 s | 0.031 | 多数逐字一致 | 4.5 GB + 图缓存 |
| **+ fp16 + Ti 分桶（当前默认）** | **0.47 s** | **0.024** | **8/8 逐字一致** | **2.2 GB + 图缓存 ~1G** |
| faster-whisper large-v3-turbo int8（对照） | 0.45 s | 0.022 | 中文 30s 片段**错 5 处** | 1.0 GB |

线上服务实测（含 live 预览并发）：20s 句子 **0.44~0.86 s**（取决于 live 引擎选择），12s 句子 0.45 s。

**优化效果链条：2.50s → 1.67s → 0.62s → 0.47s（累计 5.3x），同时显存从 4.5GB 降到 2.2GB。**

**准确率对照**（同一段 30s 中文通话，逐字比对）：

| 实际内容 | whisper 输出 | FireRed 输出 |
|---|---|---|
| 办不了 | 断不了 ❌ | 办不了 ✅ |
| 七折优惠 | 汽车运会 ❌ | 七折优惠 ✅ |
| 最低档 | 最底层 ❌ | 最低档 ✅ |
| 改成折扣 | 改成这个套餐 ❌ | 改成折扣 ✅ |
| （结尾） | 幻觉"谢谢大家" ❌ | 无 ✅ |

**显存实测**（逐模型增量）：

| 组件 | fp32 | fp16（当前默认） |
|---|---|---|
| FireRedASR2-AED（20s 推理峰值） | +4.89 GB | **+2.18 GB** |
| FireRedPunc（BERT 标点） | +0.22 GB | +0.22 GB |
| campplus 声纹 | +0.04 GB | +0.04 GB |
| CUDA Graph 缓存（24 张，LRU） | ≈ +4.3 GB | ≈ +1 GB |
| **整机合计** | ≈ 10 GB | **≈ 3.5 GB** |

→ **现在不仅速度持平 whisper，显存也只比 whisper 栈多 2.5GB，准确率明显领先。**

---

## 6. 管线级冗余（容易被忽略的大头）

实时系统里 live 预览通常是"每 N 秒把**当前整句**重新转写一遍"。设句子时长 D，间隔 T：

```
重复处理音频量 ≈ D² / (2T)        # D=20s, T=1.2s → 167 秒音频的算力
```

即 **20 秒的句子要烧掉相当于 167 秒音频的 GPU 算力**，比 final 本身贵 8 倍以上。

**对策（按收益排序）**：
1. **自适应间隔**：`interval = max(1.2, D/4)`，句子越长越少刷（感知影响小，省算力多）
2. **只转写尾部窗口**（配合"已提交前缀"，只重转最后 5~8 秒）
3. **直接关闭 live**（一行开关），把 GPU 全留给 final
4. 解码本身快 4 倍后，这一项占比会自然下降，但仍建议做成可调开关

---

## 7. 适用边界（什么情况该用、什么情况别用）

**这些技术高效的前提**：
- 自回归/小批量解码，**每步算子多而小**（Transformer 解码器、逐帧模型）
- `GPU 实跑时间 ≪ 墙钟时间`（派发瓶颈）
- 形状可静态化（能预分配缓冲 + mask 填充）

**别用/没用**：
- 大矩阵、大 batch 的算力密集型（CUDA Graph 只省派发，省不出算力）
- 输入形状无法固定（宽动态 shape）→ 图会反复重捕获
- 需要严格逐位复现官方输出时（改用等价的"减少算子"手段）

**移植到其它项目前的检查清单**：
1. 先跑 Step 1/2/3 三步诊断，量出「GPU 时间 : 墙钟时间」比值
2. 比值 < 0.3 → 优先考虑 CUDA Graph / 算子融合 / 减少 Python 调用
3. 比值 > 0.7 → 纯算力瓶颈，考虑 fp16/量化/TensorRT，CUDA Graph 收益有限
4. 任何优化都要配**等价性验证脚本**（多样本逐 token 对比），不接受"看起来差不多"

---

## 8. 复现与验证脚本（模式可照搬）

| 脚本 | 用途 |
|---|---|
| `bench_op_overhead.py` | **首选**：量化每算子派发开销 + CUDA Graph 收益上限（决定要不要上图） |
| `profile_step.py` | kernel 级 profiling，算清每步算子数与 GPU/CPU 时间 |
| `breakdown_aed.py` | 组件级耗时拆解（含正确使用 `no_grad` 的写法） |
| `verify_fast_decode.py` | 增量解码 vs 官方：**逐字/时间戳/置信度**全字段对比 |
| `verify_graph.py` | CUDA Graph vs 官方：多样本等价性 + 首次/稳态耗时 |
| `measure_vram.py` | 逐模型显存增量实测（跨进程 GPU 查询不可靠时用进程内数据） |
| `e2e_firered.py` | WebSocket 端到端灌流测试（可指定 wav / 倍速 / 端口） |

**等价性验证的写法（关键）**：
同一批音频分别跑「官方」与「优化」，逐 token 对齐、打印**首个差异位置**与两侧片段，
再叠加"更长上下文解码"和"第三方模型"做交叉判断——只有这样才能区分
「优化引入的错误」与「beam 打平的合理差异」。
