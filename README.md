# FireRed 实时语音转写（实时字幕 + 声纹说话人分离）

> 基于小红书 FireRed 团队 [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S) 的**实时**语音转写系统：
> 浏览器麦克风 → WebSocket → 实时字幕 / 断句 / 标点 / **声纹说话人分离** / 离线定版精修。
>
> A real-time speech-to-text system built on FireRedASR2S, with **speaker diarization by voiceprint**.
> Two highlights: **(1) 5.3× decoding speedup on Windows/PyTorch (2.50s → 0.47s per 20s audio)** and
> **(2) a two-pass diarization design (online rough + offline finalized) that actually separates speakers reliably**.

---

## ✨ 亮点速览

### 1. 速度：20 秒音频 2.50s → 0.47s（RTF 0.125 → 0.024），显存 4.5G → 2.2G

| 阶段 | 20s 句子耗时 | RTF | 输出一致性 | 显存 |
|---|---|---|---|---|
| 官方 FireRedASR2-AED | 2.50 s | 0.125 | 基准 | 4.5 G |
| + 增量解码 | 1.67 s | 0.084 | **逐字一致** | 4.5 G |
| + CUDA Graph 整步回放 | 0.62 s | 0.031 | 多数逐字一致 | 4.5 G + 图缓存 |
| **+ fp16 + 输入长度分桶** | **0.47 s** | **0.024** | **8/8 用例逐字一致** | **2.2 G + 图缓存** |
| *faster-whisper large-v3-turbo int8（对照）* | *0.45 s* | *0.022* | *中文 30s 片段错 5 处* | *1.0 G* |

**结论：速度追平 whisper，中文准确率明显领先**（同一段 30s 中文通话，whisper 错 5 处，本系统全对）。

> 关键认知：Windows/PyTorch 上小算子密集的自回归解码，**瓶颈几乎从来不是算力，而是算子派发开销**
> （本机实测每算子约 42µs，解码一步约 240 个算子 → CPU 侧 10ms/步，GPU 实际只跑 0.8~5ms）。
> 用 CUDA Graph 干掉派发开销后，**瓶颈才轮到权重带宽 —— 这时 fp16 才真正开始有效**
> （在派发瓶颈阶段 fp16 实测 0% 收益，别提前下结论）。

### 2. 声纹说话人分离：在线粗略 + 离线定版（两遍式）

单靠在线阈值判定必然出错，所以做成两遍：

- **在线**（实时）：句内**声学变点分段**（库无关，能切出句内出现的陌生人）+ 段级复核
- **离线定版**（停止录音时 / 录制中每 ~60s）：
  段级全局聚类 → **成员音频拼接后重嵌入** → 挖出"未解释窗口"里漏掉的说话人 →
  k-means 迭代精炼 → 时序平滑 → 词级重新切句 → **整份回发改正结果**

实测（合成用例）：
- 4 人录音（通话两人 + 小说朗读 + 英文演讲）→ **4 人全对**，且某人再次出现被正确重认
- **"三种声音混在一句话里"**（女声 + 男声 + 英语，段间仅 0.15s 静音）→ 正确拆成 3 人
- 跨用例同一说话人正确匹配到同一 ID

---

## 🚀 快速开始

### 0. 环境要求
- Windows 10/11（Linux 亦可，路径/启动脚本需微调）
- Python 3.10+（实测 3.12）
- NVIDIA GPU（实测 RTX 3090 24G；fp16 下模型仅占 2.2G，8G 卡也够）
- 国内网络建议用 ModelScope 下模型（见下）

### 1. 取本仓库 + 官方源码与模型

```bash
git clone https://github.com/aihaili/firered-realtime-voiceprint.git
cd firered-realtime-voiceprint

# 官方 ASR 源码（本仓库与它同级放置，见步骤 3）
git clone https://github.com/FireRedTeam/FireRedASR2S.git
cd FireRedASR2S
```

再回到本节下方继续下模型。

### 1. 取官方源码与模型

```bash
git clone https://github.com/FireRedTeam/FireRedASR2S.git
cd FireRedASR2S

# 模型（ModelScope，国内直连快）
pip install -U modelscope
modelscope download --model xukaituo/FireRedASR2-AED  --local_dir ./pretrained_models/FireRedASR2-AED
modelscope download --model xukaituo/FireRedPunc      --local_dir ./pretrained_models/FireRedPunc
# 可选：FireRedVAD / FireRedLID
modelscope download --model xukaituo/FireRedVAD       --local_dir ./pretrained_models/FireRedVAD
```

### 2. 装依赖

```bash
pip install -r requirements.txt
# 官方 requirements 要求 torch==2.1.0+cu118 / numpy==1.26.1；
# 实测 torch 2.8.0+cu128 + numpy 2.x 也能跑通（本项目 CUDA Graph 方案基于该组合）
```

### 3. 放置本仓库文件

把本仓库的 `server_firered.py` / `firede_fast.py` / `firede_graph.py` / `index.html` /
`audio-worklet.js` 放到 **FireRedASR2S 的上一层目录**（即与本项目的目录结构一致）：

```
your-workspace/
├── server_firered.py      ← 本仓库
├── firede_fast.py
├── firede_graph.py
├── index.html
├── audio-worklet.js
├── FireRedASR2S/          ← 官方源码（git clone）
│   └── pretrained_models/ ← 上面下的模型
└── speakers_firered.json  ← 首次运行自动生成（声纹库）
```

### 4. 启动

```bash
python server_firered.py       # 或双击 start_firered.bat
# 浏览器打开 http://localhost:8766 ，点绿色按钮开始说话
```

就绪标志：日志出现 `ASR (FireRedASR2-AED) ready` / `punctuation (FireRedPunc) ready` / `speaker model ready`。

---

## 🏗 架构

```
浏览器 30ms PCM 帧 ──WebSocket──▶ FastAPI 服务端
                                   │
                    帧序状态机（30ms/帧，与到达速度无关）
                                   │
              句子切分：0.7s 静音 / 20s 上限 / 超 60 字挂起式分段
                                   │
        ┌──────────────────────────┴───────────────────────────┐
        │ final（句末）                                          │ live（每 1.2s）
        ▼                                                       ▼
  FireRedASR2-AED 转写(beam=3 + 词级时间戳)              AED 或 CTC 贪心
        │                                                （hybrid：长句用 CTC）
        ▼                                                       ▼
  句内声学变点分段（库无关）                             规则标点归一化
        │
        ▼
  段级声纹复核 → campplus 嵌入 vs 声纹库（归类/新建）
        │
        ▼
  FireRedPunc 逐段标点 ──▶ WebSocket final（气泡 + 说话人）
                                   │
                    ┌──────────────┴───────────────┐
                    │ 停止录音 / 录制中每 ~60s       │
                    ▼                              ▼
              离线定版：段级聚类 + 拼接重嵌入 + 未解释窗口挖掘
                        + 迭代精炼 + 中值滤波 + 词级重新切句
                                   │
                                   ▼
                      revision（整份改正后的气泡）→ 前端重建
```

---

## 🧠 两个核心技术

### 一、CUDA Graph 解码（`firede_graph.py`）

自回归 beam search 每步要跑 ~240 个小算子。本实现把**整步解码**（16 层 Transformer + beam 剪枝 + KV 重排）
全部塞进一张 CUDA Graph，之后每步只 `graph.replay()`：

- **所有跨步状态都是预分配静态缓冲、图内原地更新**（KV 缓存 / token / step / mask / scores）
- **变长输入用「静态填充 + 加性 mask」换静态形状**；位置索引用 device 上的 step 张量在图内自增，host 零交互
- **cross-attn K/V 每句只投影一次**（官方每步每层重算整段编码器输出，占了一半以上算力）
- **输入长度分桶**（默认 64 帧一档）：否则每遇到新句长就捕获一张图，实测出现 0.5~1s 卡顿
- 图按桶缓存（LRU），实测 43 次捕获后显存稳定不涨

三个必踩的坑（详见 `docs/PROJECT_NOTES.md`）：
1. **Python 重绑定会破坏图状态**：`self.x = ...` 在图里写的是旧缓冲地址 → 必须 `copy_` / `index_copy_` / `scatter_`
2. **捕获期禁止同步**：循环内 `if bool(tensor.any())` 会每步每层触发 GPU→CPU 同步（16 层 × 86 步 = 1376 次）
3. **beam 数必须从 config 读**（写死 1 会静默变成贪心解码）

### 二、两遍式声纹说话人分离（`server_firered.py`）

**在线**（够用即可）：
1. **句内声学变点分段**：逐词算 campplus 嵌入，与"当前段质心"相似度低于阈值就开新段，再合并相似段、吸收过短段。
   ⚠️ 反面教训：**用"声纹库质心给每个词归类"只能区分已知的人** —— 一句里出现陌生人时，每个词都会被塞给库里最像的人，
   整句被判成同一个已知说话人（实测：女声+男声+英语三种声音的一句话被全标成"说话人 3"）。
2. **段级复核**：每段用本段嵌入按阈值重判（认识→归类，不认识→新建），不采信切分阶段的结论。
3. 防碎片四件套：短段二次证据 / 灰区自愈 / 新建前先吸收 / 周期性合并。

**离线定版**（准确率的主要来源）：
| 步骤 | 做法 | 为什么 |
|---|---|---|
| A 定说话人集合 | final 段全局聚类 → 每簇**拼接音频重嵌入** → 簇间互并 → 与库比对 | 段音频长、声纹稳；纯窗口聚类会碎（85s 碎成 19 簇） |
| B 挖漏掉的人 | 窗口分配到段级质心，**相似度低于阈值的"未解释窗口"**单独聚类 → 保守门槛过滤 → 建新说话人 | 只出现在混合段里的人从未独立成簇，只有这步能挖出来（实测挖出 16 窗 ≈12s 的隐藏说话人） |
| C 迭代精炼 | k-means 式：用分配到的窗口音频重嵌入质心再重新分配（2 轮） | 段级质心可能是**混合质心**（整段混了两人） |
| D 时序平滑 | 窗口众数滤波 + **段级中值滤波**（A-B-A 且中间段短 → 三并为一） | 抑制抖动式过切分 |
| E 词级切句 | 词中点 → 最近窗口 → 说话人 → 连续合并 → **逐子段补标点** | 把一句话按说话人真正拆开 |
| F 清理 | 删掉本次会话新建但定版未用到的说话人 | 在线碎片自动清掉 |

回发 `revision` 时用**整份替换**（定版可能把一句拆成多句），前端按序重建气泡。

---

## ⚙️ 配置开关

| 变量 | 默认 | 含义 |
|---|---|---|
| `FIRERED_PORT` | 8766 | 端口 |
| `FIRERED_DECODE` | `graph` | 解码后端：`graph`(CUDA Graph) / `fast`(等价增量解码) / `official` |
| `FIRERED_HALF` | 1 | fp16（显存减半 + 解码 1.26×，输出逐字一致） |
| `FIRERED_MAX_GRAPHS` | 24 | CUDA Graph 缓存上限（每图约 180MB；分桶后 8~16 张足够） |
| `FIRERED_LIVE_ENGINE` | `hybrid` | live 预览后端：`aed` / `ctc`(快 50×，会漏字) / `hybrid` |
| `FIRERED_FINALIZE_ON_STOP` | 1 | 停止时离线定版 |
| `FIRERED_FINALIZE_PERIOD` | 60 | 录制中自动定版间隔秒（实际周期 = max(此值, 会话时长/10)） |
| `FIRERED_MATCH_THR` | 0.60 | 声纹归类阈值（越低越倾向合并） |
| `FIRERED_UNEXPLAINED_THR` | 0.35 | "未解释窗口"阈值（越低越保守，越不易多建说话人） |
| `FIRERED_CLUSTER_THR` | 0.55 | 离线段级聚类阈值 |
| `FIRERED_ASR_TIMING` | 0 | 打印每句转写耗时 + 显存 |
| `FIRERED_DB` | `speakers_firered.json` | 声纹库路径（换文件=从零开始） |

完整列表见 `server_firered.py` 顶部与 `docs/PROJECT_NOTES.md`。

---

## 🕳 踩坑清单（精华，完整的见 `docs/PROJECT_NOTES.md`）

| 症状 | 根因 | 修法 |
|---|---|---|
| 转写整段空文本，只有一个 `<sil>`；官方示例音频却正常 | 模型 fbank 期望 **int16 量级**采样值，传了 [-1,1] | 传前 `×32768`（注意声纹模型仍要 [-1,1]，**同进程两个模型约定相反**） |
| 词级时间戳是均匀分布的"假时间戳" | `use_half` 走 **bf16**，`torchaudio.forced_align` 只接受 fp16/fp32/fp64，官方此处是**裸 except** | `torch.cuda.is_bf16_supported = lambda *a, **k: False` 逼出 `.half()` |
| `value cannot be converted to type at::Half without overflow` | mask 用 `-1e10`，超出 fp16 范围（65504） | mask 值按 dtype 取：fp32 用 -1e10，fp16 用 -3e4 |
| `ModuleNotFoundError: Could not import module 'PreTrainedModel'` | 精简环境无 `USERNAME` → `getpass.getuser()` 回退 `import pwd`（Windows 无此模块） | 导入前 `os.environ.setdefault("USERNAME", "dsh")` |
| 24G 卡报 "44.5 GiB allocated" 假 OOM | profiling 脚本没包 `torch.no_grad()`，autograd 保留全部中间激活 | 推理脚本一律 `with torch.no_grad():` |
| 每遇新句长就卡 0.5~1s | CUDA Graph 按精确长度缓存 → 反复捕获/淘汰 | 输入长度分桶 + mask 屏蔽填充帧 |
| 转写文本中间冒出 `<eos><eos>`，beam 分数不累积 | CUDA Graph 内用了 Python 重绑定（写的是旧缓冲地址） | 全部改原地更新 |
| 解码每步比理论算力慢 200 倍 | 循环内 `bool(tensor)` 触发 GPU→CPU 同步 | 判断提到循环外；循环内禁止 `bool()`/`.item()`/`.cpu()` |
| 测速时快时慢差 2~5 倍 | 两个进程共用 GPU 互相抢 | 测速务必单跑；Windows 上 kill 父进程不一定杀子进程 |
| 陌生人被并进旧说话人 | 词级分类只选"最像的库内质心"，从不检查阈值 | 段级复核 + 整句门槛改为**库无关的声学分段** |

---

## 🧰 工具脚本（`tools/`）

| 脚本 | 用途 |
|---|---|
| `bench_op_overhead.py` | **首选**：量化每算子派发开销 + CUDA Graph 收益上限（决定要不要上图，5 分钟出结论） |
| `profile_step.py` | kernel 级 profiling：每步算子数、GPU/CPU 时间占比 |
| `breakdown_aed.py` | 组件级耗时拆解（含 `no_grad` 正确写法） |
| `bench_precision.py` | fp32 vs fp16 对比 |
| `bench_asr_ab.py` | FireRed vs faster-whisper 的准确率/速度对照 |
| `verify_fast_decode.py` / `verify_graph.py` / `verify_fp16.py` / `compare_fp16.py` | **等价性验证**：逐 token / 时间戳 / 置信度对比 |
| `measure_vram.py` | 逐模型显存增量实测 |
| `diag_speakers.py` | 声纹相似度矩阵（判断"到底几个人、换人点在哪"） |
| `diag_scale.py` | int16 标度问题的定位脚本（踩坑复现） |
| `make_4spk.py` / `make_3voice.py` | 合成多说话人测试音频（4 人 / 句内三种声音） |
| `e2e_firered.py` | WebSocket 端到端灌流测试（可指定 wav / 倍速 / 端口） |
| `ground_truth.py` / `show_db.py` / `show_args.py` / `check_mask.py` / `check_html_js.py` / `debug_graph_diff.py` | 各类排查小工具 |

---

## ⚠️ 已知局限

1. **重叠语音（两人同时说话）无法分离** —— 这是单通道声纹方案的固有边界，需要语音分离模型（如 MossFormer2）才能解
2. **跨录音的同一人**声纹相似度只有 0.4~0.5，与部分陌生人 0.5~0.6 有重叠区，单阈值不可能两侧都完美
3. 定版耗时约 5~7s / 85s 音频（主要为滑窗声纹；已做增量缓存）
4. 声纹模型仅用 campplus（192 维）；换更强的声纹模型（如 3D-Speaker ERes2NetV2）可进一步提升
5. 官方 FireRedASR2S 只在 Ubuntu 22.04 测过，本项目在 Windows 上跑通但属非官方支持路径

---

## 🔒 隐私说明

**本仓库不包含任何音频、声纹库或转录文本。** 声纹库（`speakers_firered.json`）在首次运行时自动生成，
属于**生物特征数据**，请勿提交到任何公开仓库（`.gitignore` 已默认排除）。
麦克风音频仅在本机内存中处理，不落盘、不外传（调试用的 `mic_debug.wav` 也请勿提交）。

---

## 📄 License

Apache-2.0（与上游 FireRedASR2S 一致）。

本仓库与之**集成但不内嵌**的上游作品（各自遵循其许可证）：
- [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S)（Apache-2.0）— ASR / 标点 / VAD / LID
- [FunASR / ModelScope](https://github.com/modelscope/FunASR) campplus 声纹模型（Apache-2.0）
- 测试脚本中可选使用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 做对照实验

## 🙏 致谢

- FireRed 团队（小红书）开源的 FireRedASR2S —— 本项目的 ASR/标点基础
- ModelScope 提供的 campplus 声纹模型
- 社区关于 CUDA Graph 加速自回归解码的诸多实践
