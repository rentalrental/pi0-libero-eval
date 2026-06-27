# π₀ VLA 在 LIBERO 上的微调与评估（个人学习项目）

> 对 **π₀ / π₀.₅ 视觉-语言-动作模型（VLA）** 在公开 **LIBERO** 操作 benchmark 上的动手复现与研究：
> LeRobot 数据转换 → JAX/Flax 训练 → LoRA 微调 → 归一化统计 → 动作分块 → WebSocket 策略服务推理 →
> 仿真 rollout 评估。

> ⚠️ **范围与 IP 说明：** 本仓库只包含基于**公开 LIBERO benchmark** 和开源
> [openpi](https://github.com/Physical-Intelligence/openpi) 代码库的工作。实习期间的**公司专有机器人数据
> 与内部代码不包含在内**，相关经验仅做概念层面的描述。

---

## 1. 这个项目展示了什么

| 能力 | 我做了什么 |
|---|---|
| **VLA 模型** | 上手 π₀ / π₀.₅（VLM 主干 + flow-matching 动作专家） |
| **数据管线** | 把演示数据统一转换为 **LeRobot** 训练格式，完成图像 / 状态 / 动作 / 语言指令字段映射 |
| **归一化** | 计算 **norm stats**（mean / std + q01/q99 分位数，增量 running statistics），用于输入/动作归一化 |
| **训练** | 跑通 **JAX/Flax** 后端：加载 `pi0_base` 权重，对 `gemma_2b_lora` 主干 + `gemma_300m_lora` 动作专家做 **LoRA 微调**（rank=16, alpha=16），冻结主干、关闭 EMA、30k 步，支持 checkpoint 保存与断点续训 |
| **动作表示** | **动作分块（action chunking）**——一次预测未来 H 步动作块（控制更平滑、隐藏推理延迟） |
| **推理服务** | 用 **WebSocket 策略服务（policy server）** 封装为远程推理（输入观测 → 输出动作序列） |
| **评估** | 在 **LIBERO** 仿真中执行模型 rollout，录制成功/失败视频，分析泛化效果 |

## 2. 背景——为什么是 π₀

π₀ 是一个 **VLA**：视觉-语言模型主干 + **flow-matching / 扩散式动作专家**，输出连续动作块。
在大规模跨本体操作数据上训练，再微调到下游任务。我重点研究的几个设计点：

- **为什么用 JAX**：`jit` 把函数 trace 成计算图，交给 **XLA 编译器**做**算子融合 + 整图优化**，
  生成贴硬件的机器码。快在"编译 + 融合"，不是"数组本身快"。
- **数据 transforms 管线**：`RepackTransform → Normalize → DeltaActions → TokenizePrompt →
  PadStatesAndActions → JAX 数组`。
- **归一化**：running stats 增量累计 mean + mean of squares（得 std），再用直方图估 q01/q99 分位数；
  归一化 = `(x − mean) / std` 或分位裁剪。
- **LoRA**：冻结主干，只训低秩 `A·B` 适配器（本配置 rank=16 / alpha=16）；从 `pi0_base` checkpoint
  接着在 LIBERO 数据上训，配合冻结主干 + 关闭 EMA 防灾难性遗忘。

## 3. 评估结果（LIBERO rollout）

在 LIBERO 的 [填: 子集，如 LIBERO-Long / LIBERO-90] 共 **78 个任务**上做了 rollout 评估，
成功与失败案例均有录制（见 [`video/`](video/)）。整体成功率约 **[填: __]%**。

典型任务表现：

| 任务 | 类型 | 结果 |
|---|---|---|
| 把黑碗放进柜子底层抽屉并关上 | 长程·带关闭子目标 | ✅ 成功 |
| 打开炉灶并把摩卡壶放上去 | 多步序列 | ⚠️ 部分成功 |
| 把字母汤罐和番茄酱都放进篮子 | 多物体 | ❌ 偏失败 |
| 白杯放盘子上、巧克力布丁放盘子右边 | 多物体·空间关系 | ❌ 偏失败 |

**失败分析**：长程、多物体、带"关抽屉/盖子"等复合子目标的任务，成功率明显低于单步抓放。
失败主要集中在两类——(1) **抓取阶段的位姿偏移**，导致后续动作链整体失败；
(2) **第二段子任务**（放完第一个物体后）出现策略漂移。

## 4. 技术栈

`Python` · `JAX` / `Flax` · `LoRA` · `LeRobot` · `LIBERO` · `WebSocket（策略服务）` · `Docker`

## 5. 如何复现

基于上游 [openpi](https://github.com/Physical-Intelligence/openpi)，配置名 `pi0_libero_low_mem_finetune`
（π₀ + LoRA，数据集 `physical-intelligence/libero`）：

```bash
# 1) 数据转 LeRobot 格式
uv run examples/libero/convert_libero_data_to_lerobot.py

# 2) 计算归一化统计
uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune

# 3) LoRA 微调（XLA 显存比例按显卡调）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune \
    --exp-name=pi0_libero_lora --overwrite

# 4) 起策略服务（远程推理）
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_libero_low_mem_finetune \
    --policy.dir=checkpoints/pi0_libero_low_mem_finetune/pi0_libero_lora/<step>

# 5) LIBERO rollout 评估（examples/libero，连上策略服务）
```

关键文件：`examples/libero/convert_libero_data_to_lerobot.py`（数据转换）、
`docs/norm_stats.md`（归一化）、`docs/remote_inference.md`（策略服务）、`examples/libero/main.py`（rollout）。

## 6. 下一步计划

- 量化各 LIBERO 子集的成功率，做成对比表
- 对比 π₀ / π₀.₅ / π₀-FAST 在长程任务上的表现差异
- 引入数据增强 + 增大 action chunk 长度复测，验证对长程/多物体任务的鲁棒性
- 探索 **sim2real**：仿真策略迁移到真机的差距与对策
