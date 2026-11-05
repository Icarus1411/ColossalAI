# Qwen3.5 纯文本训练接入

本接入覆盖 Qwen3.5 dense 的 `Qwen3_5ForCausalLM` 与
`Qwen3_5ForConditionalGeneration`，通过 ShardFormer policy 在 TP=1 下替换
Gated DeltaNet。以下记录纯文本训练的实现、实测精度及已覆盖的插件配置。

## 使用方式

环境沿用 Torch 2.13.0+cu130、Transformers 5.17.0、Triton 3.7.1，
可选内核依赖见 `requirements-gdn.lock.txt`。启动 Python 前设置
`TRITON_F32_DEFAULT=ieee`。

```python
import torch
from transformers import Qwen3_5ForConditionalGeneration
from colossalai.shardformer import ShardConfig, ShardFormer

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    snapshot_path, local_files_only=True, attn_implementation="eager"
)
model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
# 在 policy 冻结视觉塔之后创建优化器，再交给 DDP/ZeRO 等插件。
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)
```

使用 `HybridParallelPlugin(tp_size=1, pp_size=1)` 时由插件调用 policy，
无需手动再次优化。视觉塔保留在 checkpoint 中，但被冻结，纯文本前向不调用它。
policy 关闭 `use_cache`，拒绝传入视觉数据、KV cache、packed sequences，
不支持 TP/PP/SP/EP 或额外的 attention/norm/JIT 替换。

## 精度语义

HF 原生 Qwen3.5 GDN 在核心计算中将 Q/K/V/beta 转为 FP32。
整模型 policy 使用 `force_fp32_core=True` 匹配该配方，投影、卷积和输出仍采用
模型精度。BF16 提升到 FP32 后使用 16-token 分块，Triton 补偿扫描在对数域累加衰减，
FLA 的 KKT 内核构造因果三角系统。前向直接求解键和值的右端项，通过 FLA 更新跨块状态，
将块内输出与衰减后的查询对历史状态的读取分别计算再相加。反向沿用 FLA 的 WY 因子
重算和项目内稳定的门控梯度。PyTorch 的 GPU 矩阵乘和三角求解使用 FP32；
正式精度报告使用 IEEE 矩阵乘。FP64 仅用于独立验收参照。

## 验证入口

基础测试覆盖 FP32/BF16、CausalLM/ConditionalGeneration、梯度检查点、
lazy init、参数对象与 tied weight 保留、视觉塔冻结及实际 fused 前反向调用：

```bash
TRITON_F32_DEFAULT=ieee python -m pytest \
  tests/test_shardformer/test_model/test_shard_qwen3_5.py -v
```

该文件加入 CI GPU batch 5。镜像缺少可选 FLA 时会明确 skip；FLA 镜像同步由
镜像负责人另行处理，因此本地通过不表示公共 CI 镜像已具备依赖。

模型注册名为 `transformers_qwen3_5_{0_8b,2b}_tiny`，可追加 `_conditional`
和 `_fused`。这些是缩小配置，不能称为真实 0.8B/2B 权重结果。
原生条目用于空白对照和 HybridParallel policy；`_fused` 条目用于其他插件，
并通过独立 `reference_fn` 构建 HF 原生参照。

测试参考仅在每个模型实例上固定 HF PyTorch dispatch；不会修改 HF/FLA 全局函数。
FP64 真值使用独立递推公式和保留 FP64 的 norm、RoPE、attention、loss 计算，
低精度参照保持 HF 原生实现。统一工具检查输出、梯度、一步更新及多步 loss，
DP 等价、有效 token 加权和续训由公共 scenarios 执行。

```bash
TRITON_F32_DEFAULT=ieee python -m tests.test_shardformer.test_model.run_qwen3_5_acceptance \
  --entry transformers_qwen3_5_0_8b_tiny --config shardformer_fp32 --out report.json
```

真实权重验证脚本 `run_qwen3_5_real.py` 接收只读缓存的 `--snapshot`，进行
FP64 参照的前向比较与三步训练。它不代表真实权重逐层梯度、一步更新和续训
已完成全套 L2，也不提供收敛结论。

`run_qwen3_5_real_alignment.py` 进一步调用公共 `check_training`，检查真实
权重的输出、可训练参数梯度、首步更新和三步 loss。真实 2B 的完整验收可在 H20
使用 `PYTORCH_NO_CUDA_MEMORY_CACHING=1`，使 FP64 参照结束后释放的显存可供 NCCL 使用。

## 数值及 checkpoint 修复（2026-11-05）

本轮保留以下修复：

- GDN 四路投影按 HF 顺序建图，避免 BF16 反向累加顺序不同造成近零梯度变号。
- 稳定反向显式接住 `dk + dk2`、`dg + dg2` 的返回值；Gemini 的 ColoTensor
  会将原地加法改为非原地运算，旧写法会漏掉 WY 梯度贡献。
- ZeRO 与 Gemini 将 FP32 master weights 随 optimizer state 保存并恢复。
  新增字段 `_colossalai_master_param` 由专用 loader 消费，不传给底层优化器。
  旧 checkpoint 仍可加载，但无法找回此前未保存的低位；只加载模型不等于续训。
- Gemini 正确识别 tied parameter 别名；已复制到各卡的累积梯度不再额外乘卡数。
- FSDP 从底层 HF 模型保存配置，修复 `from_pretrained` 重载失败。
- 公共数值工具的 FSDP 参照在计算精度转换前保留原始 FP32 master weights，
  与 FSDP 实现一致。FP64 真值公式和判定阈值没有调整。修复前原生对原生也失败，
  修复后原生与 fused 两组各 115/115 通过。

已通过 GDN/模型回归 42/42、ZeRO checkpoint 回归 4/4、Gemini 两种 chunk
布局下的累积与 tied checkpoint 回归 2/2，以及参照 master 精度回归 1/1。
公共数值工具全套回归 64/64 通过；上述 pytest 合计 113 项，无 skip。
本次分块修复使用正常模型入口复验。真实 0.8B 和 2B 的 BF16 前向 logits 与 CE 均通过；
三步 BF16 训练分别通过 798/798 和 942/942，双卡 Hybrid BF16 参数梯度各通过 320/320。
0.8B 的独立前向 CE 绝对误差为 0.00041246（限值 0.00117971），2B tiny 的 Gemini
第三步目标误差为 0.00001280（限值 0.00001753）。算子专项以独立 FP64 递推对照输出
及 q、k、v、g、beta 的梯度，覆盖单 token 与跨块序列。

证据见 `logs/gpu-h20-5/20261105-140309-2B3-block-repair/rematerialized/`。微基准
使用单 H20、batch 2、8 heads、head dim 128，测 GDN 核心前向加反向。原路径与新路径在
65/512/2048 token 下的中位耗时分别为 2.81/7.16/24.08 和 3.28/7.65/24.69 ms。

## 验收范围

- FSDP 默认 flatten 配置仍不接受冻结视觉参数与可训练文本参数混合。
  已验收 FSDP 路径使用纯文本 `Qwen3_5ForCausalLM`。
- HybridParallel BF16 checkpoint 的 master-weight 恢复限制不在本轮修复范围；
  当前不能从 ZeRO/Gemini 的通过结果推断该路径也通过。
- 真实权重的分布式对照覆盖双卡 Hybrid BF16；其余插件矩阵使用 tiny 模型。整模型吞吐及收敛需另行评估。

此前数值及 checkpoint 修复的证据目录：`logs/gpu-h20-5/20261104-175044-2B3/`，包含修复前后报告、
独立诊断脚本、失败实验和最终回归。上一轮目录为
`logs/gpu-h20-5/20261104-160505-2B3/` 与
`logs/gpu-h20-4/20261104-162500-2B3/`。
CI 镜像同步继续由镜像负责人跟进；本次没有安装或升级依赖，没有提交或推送 Git。
