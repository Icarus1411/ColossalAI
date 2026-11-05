"""Shared, non-sharded Gated DeltaNet training layer (FLA 0.6.0).

The Qwen-style projection/conv/norm parameters retain their native names and
identities. Model policies can replace ``linear_attn`` through
``SubModuleReplacementDescription(..., FusedGatedDeltaNet)`` at TP size one.
Cache decoding and packed sequences are deliberately outside this training API.
FP32 uses a local stable gate-backward formula with the pinned FLA 0.6.0
forward and supporting kernels; see docs/GATED-DELTANET-zh.md.
"""

import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .parallel_module import ParallelModule

__all__ = ["FusedGatedDeltaNet", "fused_gated_delta_rule"]


@lru_cache(maxsize=1)
def _load_fla_kernel():
    try:
        versions = {name: version(name) for name in ("flash-linear-attention", "fla-core")}
    except PackageNotFoundError as exc:
        raise ImportError("FusedGatedDeltaNet requires flash-linear-attention==0.6.0 and fla-core==0.6.0") from exc
    if any(v != "0.6.0" for v in versions.values()):
        raise ImportError(f"FusedGatedDeltaNet was validated with FLA 0.6.0; found {versions}")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


def fused_gated_delta_rule(q, k, v, g, beta, *, align_bf16_blocks=False):
    """Return [batch, time, heads, value_dim] with differentiable FLA chunk kernels.

    Q/K/V use FP32, BF16 or FP16 on CUDA; gates are accumulated in FP32. Q/K are
    L2-normalized inside FLA. No native fallback or inference-only recurrent
    dispatch is allowed, including for a one-token training sequence. The Qwen
    policy enables align_bf16_blocks only for BF16 inputs promoted to FP32.
    """
    if q.device.type != "cuda" or q.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("FLA training requires CUDA FP32/BF16/FP16 Q/K/V")
    if q.dtype == torch.float32 and os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise ValueError(
            "FP32 GDN requires TRITON_F32_DEFAULT=ieee before starting Python; Torch TF32 flags do not suffice"
        )
    if any(x.device != q.device for x in (k, v, g, beta)) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("GDN inputs must share a device and Q/K/V must share a dtype")
    if q.ndim != 4 or v.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("Expected Q/K [B,T,H,K] and V [B,T,H,V]; expand grouped Q/K heads first")
    if q.shape[1] == 0 or g.shape != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("Expected a nonempty sequence and g/beta [B,T,H]")
    if align_bf16_blocks and q.dtype != torch.float32:
        raise ValueError("BF16-aligned blocks require Q/K/V promoted to FP32")
    # This training API always starts from a zero recurrent state. Its first
    # decay multiplies zero, so neither the output nor its gradient depends on
    # g[:, 0]. Remove that redundant dependency before the chunk cumsum to avoid
    # a spurious nonzero gradient from subtracting rounded prefix sums.
    g = torch.cat((torch.zeros_like(g[:, :1]), g[:, 1:]), dim=1)
    kernel = _load_fla_kernel()
    if q.dtype == torch.float32:
        from ._gated_delta_net_fp32 import stable_gated_delta_rule

        return stable_gated_delta_rule(q, k, v, g.float(), beta, align_bf16_blocks)
    output, _ = kernel(
        q=q, k=k, v=v, g=g.float(), beta=beta,
        use_qk_l2norm_in_kernel=True, output_final_state=False,
        chunk_size=64,
    )
    return output


class FusedGatedDeltaNet(ParallelModule):
    """Reusable Qwen-style GDN layer for data-parallel training at TP=1.

    ``from_native_module`` accepts the projection/conv/norm layout used by HF
    Qwen3.5. Reusing the native norm preserves checkpoint and precision semantics.
    Dense/MoE policy integration and full-model validation are separate tasks.
    """

    @staticmethod
    def from_native_module(module, process_group=None, force_fp32_core=False, **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported GDN replacement options: {sorted(kwargs)}")
        groups = process_group if isinstance(process_group, (tuple, list)) else [process_group]
        if any(group is not None and dist.get_world_size(group) != 1 for group in groups):
            raise NotImplementedError("FusedGatedDeltaNet supports TP size one only")
        if module.activation not in ("silu", "swish"):
            raise ValueError("Gated DeltaNet currently supports SiLU convolution activation only")
        if module.num_v_heads % module.num_k_heads:
            raise ValueError("Value heads must be a multiple of key heads")
        if module.conv1d.groups != module.conv_dim or module.conv1d.stride != (1,) or module.conv1d.dilation != (1,):
            raise ValueError("Expected a depthwise stride-one, dilation-one convolution")
        _load_fla_kernel()
        result = FusedGatedDeltaNet()
        result.force_fp32_core = force_fp32_core
        for name in (
            "hidden_size", "num_v_heads", "num_k_heads", "head_k_dim", "head_v_dim",
            "key_dim", "value_dim", "conv_kernel_size", "conv_dim", "layer_idx",
            "activation", "layer_norm_epsilon", "layer_type",
            "conv1d", "dt_bias", "A_log", "norm", "out_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
        ):
            setattr(result, name, getattr(module, name))
        result.training = module.training
        return result

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if cache_params is not None:
            raise NotImplementedError("GDN shared layer is training-only; set use_cache=False")
        if any(kwargs.get(name) is not None for name in ("cu_seq_lens_q", "cu_seq_lens_k", "cu_seqlens")):
            raise NotImplementedError("Packed sequences require boundary-aware convolution and are not supported")
        batch, length, _ = hidden_states.shape
        if attention_mask is not None:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError("GDN attention_mask must have shape [batch, sequence]")
            hidden_states = (hidden_states * attention_mask.unsqueeze(-1)).to(hidden_states.dtype)
        mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)
        # Preserve HF projection order: autograd accumulates their BF16 input
        # gradients in graph order, and BF16 addition is not associative.
        z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        mixed = F.silu(F.conv1d(
            mixed, self.conv1d.weight, self.conv1d.bias,
            padding=self.conv_kernel_size - 1, groups=self.conv_dim,
        )[..., :length]).transpose(1, 2)
        q, k, v = mixed.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
        q = q.reshape(batch, length, self.num_k_heads, self.head_k_dim)
        k = k.reshape(batch, length, self.num_k_heads, self.head_k_dim)
        v = v.reshape(batch, length, self.num_v_heads, self.head_v_dim)
        repeats = self.num_v_heads // self.num_k_heads
        if repeats > 1:
            q, k = q.repeat_interleave(repeats, dim=2), k.repeat_interleave(repeats, dim=2)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.force_fp32_core:
            # HF Qwen3.5's native core promotes Q/K/V and beta before L2
            # normalization and the recurrence. Match that precision recipe;
            # projections, convolution and the returned activations stay in
            # the model's dtype. The actual core is still fused Triton/FLA.
            output = fused_gated_delta_rule(
                q.float(), k.float(), v.float(), g, beta.float(), align_bf16_blocks=q.dtype == torch.bfloat16,
            ).to(q.dtype)
        else:
            output = fused_gated_delta_rule(q, k, v, g, beta)
        output = self.norm(output.reshape(-1, self.head_v_dim), z)
        return self.out_proj(output.reshape(batch, length, self.value_dim))
