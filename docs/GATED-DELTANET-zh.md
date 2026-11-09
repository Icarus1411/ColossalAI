# Gated DeltaNet 共享训练层

FLA 0.6.0 + 项目内稳定 FP32 门控反向公式。FP32 的门控导数把块间项与块内项分开，
在求导公式中排除恒等抵消的对角项，减少两个含共同大项的求和结果相减带来的舍入误差。
Q/K/W 梯度仍使用完整公式；不是把小梯度清零，也不依赖 FP64 回退。

## 实现与依赖

- `gated_delta_net.py` 提供共享层及入口。保留 HF 参数名与对象，投影、卷积和归一化沿用原语义。
- `_gated_delta_net_fp32.py` 在项目内维护稳定 Triton kernel 和受限训练适配器，保留 FLA MIT 许可。
  前向、状态更新、WY 反向与归一化复用 FLA 0.6.0；FP32 门控反向使用本地稳定公式。
- BF16/FP16 继续使用原 FLA chunk64；FP32 使用本地稳定反向、chunk16，启动前必须设置
  `TRITON_F32_DEFAULT=ieee`。Torch 的 TF32 开关不足以约束 Triton。
- 没有修改 FLA 安装或进程全局函数。可选 FLA 依赖延迟导入，版本检查保留。
- API 的初始 recurrent state 固定为零，第一步衰减不影响结果，进入 FLA 前移除这一冗余依赖。
- `requirements-gdn.lock.txt` 固定两个可选 FLA 包及 wheel 哈希；这是可选包锁，不是完整环境锁。
  本轮没有安装或升级依赖。验证环境：H20、Torch 2.13.0+cu130、Transformers 5.17.0、Triton 3.7.1。

```python
from colossalai.shardformer.layer import FusedGatedDeltaNet

layer = FusedGatedDeltaNet.from_native_module(native_gdn)
output = layer(hidden_states, attention_mask=padding_mask)
```

支持 TP=1、完整序列训练、二维 padding mask、value heads 是 key heads 的整数倍。
不支持 cache decoding、packed sequences 或 TP>1；后续模型 policy 需确保 use_cache=False。
测试仅验证替换接口，没有实现整模型 policy，2B3 未开展。

## 官方统一入口验收

使用官方提交 `69c9ff52c160774cc40eebb83a77f4aa92ff46d3` 的
`tests.kit.numerics.scenarios.independent_impl.check_fwd_bwd`，不修改工具或默认阈值。
HF 原生 PyTorch 为对照；因其内部强制 FP32，FP64 真值使用独立、保持 dtype 的数学公式。
官方入口负责三路执行、缺失梯度检查和误差判定。Trace 观察器只保存原样结果。

`expect_calls` 同时检查 FLA 前向与实际使用的反向路径：FP32 为本地
`stable_chunk_bwd_dqkwg`，BF16/FP16 为原 FLA backward。
负例验证 FP32/BF16 的纯 PyTorch 回退不能冒充 fused 实现；测试还确认原 FLA kernel 未被全局替换。

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_F32_DEFAULT=ieee GDN_TEST_LOG_DIR=/path/to/logs/numerics \
/workspace-home/venv/bin/python -m pytest tests/test_shardformer/test_layer/test_gated_delta_net.py -v
```

本轮正式接入复验：**31 passed、0 failed、0 skipped**；27 个数值用例、207 个输出/梯度张量、
54 个实际前向/反向调用断言全部通过。覆盖 FP32/BF16/FP16、序列长度1/63/64/65/129、
head dim64/128、弱衰减；完整层 hidden_size64、长度65、key heads16、value heads16/32/48，含左右 padding。
原 FP32 48-head 门控投影梯度失败已在同一入口、同一默认标准下通过。

当前结果见 `logs/gpu-h20-5/20261104-142029-2B2-stable-gate/`，包括代码快照、补丁、命令、
原始日志、官方 AlignmentReport JSON、三路 safetensors 及一页验收记录。
旧版30通过1失败记录保留为历史证据，诊断过程见 `20261104-140858-2B2-fp32-diagnosis/`。

## 范围与待办

本次通过只适用于已测算子、缩小隐藏维度的完整单层与接口，不代表整模型 L2 或分布式训练。
尚未测吞吐、显存收益或收敛；稳定反向新增计算的性能成本需要单独评估。
本地适配器依赖 FLA 0.6.0 内部接口，升级前需复核并重跑矩阵。
2B1 CI 镜像和原 setuptools 冲突仍为独立待办；本轮不向 shared-results/numerics/ 发布整模型报告。

## Qwen3.5 策略的低精度路径

上文默认共享层的 FLA 调度与 Qwen3.5 策略启用的 `force_fp32_core` 路径不同。
Qwen 的 FP16 和 BF16 在核心内部提升到 FP32 计算，使用 64-token 分块，保留 HF 的归一化、
自然对数 gate 前缀和矩阵乘法舍入边界。`_gated_delta_net_fp16.py` 中的对齐核心同时服务这两种输入；
它在转置之后转换精度，保持 HF 的连续布局及后续浮点运算的舍入路径，并在核心出口恢复输入 dtype。不要在层入口提前转 FP32：同 dtype 的 Tensor.to() 可能直接返回原张量，从而保留不同的步长。FP32 的稳定 FLA 路径保持原样。

BF16 原有 16-token 前向和反向重建在修正左 padding 标签后的 2B 训练上暴露了舍入差异：
门控梯度、层归一化权重及后续 loss 超过参考误差预算。原生核心和 64-token 对齐核心均可消除
所定位的失败，单独恢复首 gate 的值不能消除。生产路径仍调用本地 Triton 前向/反向核，未使用 HF 回退。

`test_qwen_low_precision_core_alignment` 覆盖 FP16/BF16、显式精度和 autocast、
分块边界及弱衰减，并通过统一 `check_fwd_bwd` 断言 `_block_scores_kernel.run` 和
`_block_scores_bwd_kernel.run` 实际执行。负例拒绝原生回退。
`test_bf16_training_alignment` 覆盖六种 BF16 训练配置，保留统一工具的默认判分。

Qwen 评测数据的因果标签要求预测位置和目标位置都有效：左 padding 的最后一格不能被用来
预测第一个有效 token。真实权重数据准备复用同一规则；回归检查补齐后的 token loss 与未补齐序列等价。

`test_qwen_low_precision_layer_alignment` 额外覆盖两种低精度下 16/32/48 value heads、真实投影布局及左右 padding 的整层前后向，防止仅测独立核心遗漏入口精度转换问题。
