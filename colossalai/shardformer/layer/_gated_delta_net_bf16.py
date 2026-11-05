"""Block operations for Qwen's promoted BF16 GDN. Inputs and accumulation are FP32."""

import struct

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.gated_delta_rule import chunk_fwd
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd_h
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.solve_tril import solve_tril


@triton.jit
def _log2_prefix_kernel(
    gate, prefix, T, H: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr,
    SCALE_HI: tl.constexpr, SCALE_LO: tl.constexpr,
):
    block = tl.program_id(0)
    heads = tl.program_id(1) * BH + tl.arange(0, BH)
    batch = tl.program_id(2)
    total = tl.full((BH,), 0, tl.float32)
    residual = tl.full((BH,), 0, tl.float32)
    # TwoSum retains the rounding error of each addition. Round only once after
    # scaling the accumulated natural-log prefix to FLA's log2 representation.
    for i in tl.static_range(BT):
        token = block * BT + i
        offset = (batch * T + token) * H + heads
        valid = (token < T) & (heads < H)
        value = tl.load(gate + offset, mask=valid, other=0.0)
        updated = total + value
        added = updated - total
        residual += (total - (updated - added)) + (value - added)
        total = updated
        product = total * SCALE_HI
        correction = tl.fma(total, SCALE_HI, -product) + total * SCALE_LO + residual * SCALE_HI
        tl.store(prefix + offset, product + correction, mask=valid)


def _gate_prefix(gate, block_size):
    batch, tokens, heads = gate.shape
    head_tile = min(32, triton.next_power_of_2(heads))
    scale_hi = struct.unpack("f", struct.pack("f", RCP_LN2))[0]
    gate = gate.contiguous()
    prefix = torch.empty_like(gate)
    _log2_prefix_kernel[(triton.cdiv(tokens, block_size), triton.cdiv(heads, head_tile), batch)](
        gate, prefix, tokens, heads, block_size, head_tile, scale_hi, RCP_LN2 - scale_hi,
        enable_fp_fusion=False,
    )
    return prefix


def _blocks(x, size):
    """[batch, time, heads, ...] to padded [batch, heads, chunks, size, ...]."""
    x = x.transpose(1, 2).contiguous()
    padding = (-x.shape[2]) % size
    return F.pad(x, (0, padding) if x.ndim == 3 else (0, 0, 0, padding)).unflatten(2, (-1, size))


def _tokens(x, length):
    return x.flatten(2, 3)[:, :, :length].transpose(1, 2).contiguous()


def aligned_block_forward(q, k, v, g, beta, block_size):
    """Use FLA's causal KKT and recurrent-state kernels with an FP32 block solve."""
    length = q.shape[1]
    prefix = _gate_prefix(g, block_size)
    lower = chunk_fwd.chunk_scaled_dot_kkt_fwd(k=k, g=prefix, beta=beta, chunk_size=block_size)
    log_decay = _blocks(prefix, block_size)
    decay = log_decay.exp2().unsqueeze(-1)
    keys = _blocks(k, block_size)
    rate = _blocks(beta, block_size).unsqueeze(-1)
    with torch.autocast("cuda", enabled=False):
        rhs = torch.cat(((keys * rate) * decay, _blocks(v, block_size) * rate), dim=-1)
        solution = torch.linalg.solve_triangular(_blocks(lower, block_size), rhs, upper=False, unitriangular=True)
        w, u = (_tokens(x, length) for x in solution.split((k.shape[-1], v.shape[-1]), dim=-1))
        state, residual, _ = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=prefix, initial_state=None, output_final_state=False, chunk_size=block_size,
        )
        query = _blocks(q, block_size)
        # Mask the upper triangle before exp2 so an unused future key cannot overflow.
        pair_decay = (log_decay.unsqueeze(-1) - log_decay.unsqueeze(-2)).tril().exp2().tril()
        local = ((query @ keys.transpose(-1, -2)) * pair_decay) @ _blocks(residual, block_size)
        history = (query * decay) @ state.transpose(1, 2)
        output = _tokens(local + history, length)
    # The analytical WY backward takes the inverse of the same unit-lower system.
    inverse = solve_tril(A=lower, output_dtype=k.dtype)
    return prefix, output, inverse
