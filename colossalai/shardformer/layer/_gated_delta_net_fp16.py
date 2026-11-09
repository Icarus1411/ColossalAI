"""Qwen low-precision GDN blocks with AMP rounding at matrix-product boundaries."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import exp as expf


@triton.jit
def _block_scores_kernel(
    Q, K, KB, PREFIX, ATTENTION, LOWER, WIDTH: tl.constexpr, N: tl.constexpr, BK: tl.constexpr,
    DOT_DTYPE: tl.constexpr,
):
    # One program shares a key tile between the attention and delta-rule products.
    block = tl.program_id(0)
    rows = tl.arange(0, N)
    cols = tl.arange(0, BK)
    qk = tl.full((N, N), 0, tl.float32)
    kk = tl.full((N, N), 0, tl.float32)
    for tile in range(tl.cdiv(WIDTH, BK)):
        features = cols + tile * BK
        offsets = (block * N + rows[:, None]) * WIDTH + features[None, :]
        mask = features[None, :] < WIDTH
        k = tl.load(K + offsets, mask, 0.).to(DOT_DTYPE)
        q = tl.load(Q + offsets, mask, 0.).to(DOT_DTYPE)
        kb = tl.load(KB + offsets, mask, 0.).to(DOT_DTYPE)
        qk = tl.dot(q, tl.trans(k), qk, input_precision="ieee")
        kk = tl.dot(kb, tl.trans(k), kk, input_precision="ieee")

    p = tl.load(PREFIX + block * N + rows)
    causal = rows[:, None] >= rows[None, :]
    decay = expf(tl.where(causal, p[:, None] - p[None, :], -float("inf")))
    offsets = (block * N + rows[:, None]) * N + rows[None, :]
    # AMP rounds the result of each matmul before multiplying by the FP32 decay.
    tl.store(ATTENTION + offsets, qk.to(DOT_DTYPE).to(tl.float32) * decay)
    tl.store(LOWER + offsets, kk.to(DOT_DTYPE).to(tl.float32) * decay)


@triton.jit
def _block_scores_bwd_kernel(
    PREFIX, ATTENTION, LOWER, DA, DL, DQK, DKK, DP, N: tl.constexpr,
):
    block = tl.program_id(0)
    rows = tl.arange(0, N)
    p = tl.load(PREFIX + block * N + rows)
    causal = rows[:, None] >= rows[None, :]
    decay = expf(tl.where(causal, p[:, None] - p[None, :], -float("inf")))
    offsets = (block * N + rows[:, None]) * N + rows[None, :]
    da = tl.load(DA + offsets)
    dl = tl.load(DL + offsets)
    tl.store(DQK + offsets, da * decay)
    tl.store(DKK + offsets, dl * decay)
    # A diagonal pair has exponent p_i - p_i and contributes no gate derivative.
    pairs = da * tl.load(ATTENTION + offsets) + dl * tl.load(LOWER + offsets)
    pairs = tl.where(rows[:, None] > rows[None, :], pairs, 0.)
    tl.store(DP + block * N + rows, tl.sum(pairs, 1) - tl.sum(pairs, 0))


class _BlockScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, k_beta, prefix):
        size = prefix.shape[-1]
        dot_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else q.dtype
        kernel_dtype = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}[dot_dtype]
        attention = prefix.new_empty((*prefix.shape, size))
        lower = torch.empty_like(attention)
        _block_scores_kernel[(prefix.numel() // size,)](
            q.contiguous(), k.contiguous(), k_beta.contiguous(), prefix.contiguous(), attention, lower,
            q.shape[-1], size, 32, kernel_dtype,
        )
        ctx.save_for_backward(q, k, k_beta, prefix, attention, lower)
        ctx.dot_dtype = dot_dtype
        return attention, lower

    @staticmethod
    def backward(ctx, d_attention, d_lower):
        q, k, k_beta, prefix, attention, lower = ctx.saved_tensors
        size = prefix.shape[-1]
        dqk = torch.empty_like(attention, dtype=ctx.dot_dtype)
        dkk = torch.empty_like(dqk)
        dp = torch.empty_like(prefix)
        _block_scores_bwd_kernel[(prefix.numel() // size,)](
            prefix.contiguous(), attention, lower, d_attention.contiguous(), d_lower.contiguous(),
            dqk, dkk, dp, size,
        )
        # Keep the two key gradients separate until they reach the FP32 input.
        with torch.autocast("cuda", enabled=False):
            q, k, k_beta = (x.to(ctx.dot_dtype) for x in (q, k, k_beta))
            dk = (dqk.transpose(-1, -2) @ q).float() + (dkk.transpose(-1, -2) @ k_beta).float()
            return dqk @ k, dk, dkk @ k, dp


def _normalize(x):
    # Preserve the two multiplication branches in HF's normalization backward.
    # square() reassociates their derivative and can round small FP16 updates differently.
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def qwen_fp16_gated_delta_rule(q, k, v, g, beta):
    """Compute 64-token WY blocks for FP16 or BF16 Qwen inputs.

    Keep the original private entry name for callers of the FP16 path. Convert
    to FP32 after the layout change, and return the original activation dtype.
    """
    output_dtype = q.dtype
    length = q.shape[1]
    size = 64
    # Keep the layout conversion and dtype conversion at the same boundary
    # as the model projections, including the final low-precision output conversion.
    q, k, v, g, beta = (
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format) for x in (q, k, v, g, beta)
    )
    q = _normalize(q) * q.shape[-1] ** -0.5
    k = _normalize(k)
    padding = (-length) % size
    q, k, v = (F.pad(x, (0, 0, 0, padding)) for x in (q, k, v))
    g, beta = (F.pad(x, (0, padding)).unflatten(2, (-1, size)) for x in (g, beta))
    q, k, v = (x.unflatten(2, (-1, size)) for x in (q, k, v))
    prefix = g.cumsum(-1)
    k_beta = k * beta.unsqueeze(-1)
    attention, lower = _BlockScores.apply(q, k, k_beta, prefix)

    decay = prefix.exp().unsqueeze(-1)
    # The unit-lower solves must stay FP32 even under autocast.
    values = torch.linalg.solve_triangular(lower, v * beta.unsqueeze(-1), upper=False, unitriangular=True)
    keys = torch.linalg.solve_triangular(lower, k_beta * decay, upper=False, unitriangular=True)
    reads = q * decay
    writes = k * (prefix[..., -1:] - prefix).exp().unsqueeze(-1)
    carry = prefix[..., -1].exp()[..., None, None]
    state = values.new_zeros((*q.shape[:2], q.shape[-1], v.shape[-1]))
    outputs = []
    # Unbind once so backward can assemble each block gradient in one pass.
    blocks = zip(*(tensor.unbind(2) for tensor in (values, keys, reads, attention, writes, carry)))
    for index, (value, key, read, scores, write, retention) in enumerate(blocks):
        residual = value - key @ state
        outputs.append(read @ state + scores @ residual)
        if index + 1 < q.shape[2]:
            state = state * retention + write.transpose(-1, -2) @ residual

    out = torch.stack(outputs, dim=2).flatten(2, 3)[:, :, :length]
    return out.transpose(1, 2).to(output_dtype, memory_format=torch.contiguous_format)
