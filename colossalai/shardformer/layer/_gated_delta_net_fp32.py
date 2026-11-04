# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Adapted from FLA 0.6.0 fla/ops/common/chunk_o.py and gated_delta_rule/chunk.py.
# Modifications: stable FP32 gate derivative and a local training-only adapter.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Stable FP32 backward for the pinned FLA 0.6.0 GDN training API.

Only the gate derivative formula differs from FLA. Native FLA forward, state
updates, WY backward and normalization kernels are reused without modifying
FLA globals. This private adapter supports zero initial state, expanded heads,
no cache/packed sequences and chunk size 16, as used by FusedGatedDeltaNet.
"""

import importlib

import torch
import triton
import triton.language as tl

from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2, unflatten_program_id
from fla.utils import (
    IS_NVIDIA_HOPPER,
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    check_shared_mem,
    input_guard,
)

_fla = importlib.import_module("fla.ops.gated_delta_rule.chunk")
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'USE_DW': lambda args: args['dw'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_DW', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dqkwg(
    q,
    k,
    v,
    g,
    g_gamma,
    h,
    do,
    dh,
    dq,
    dk,
    dw,
    dv,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_DW: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t = unflatten_program_id(tl.cdiv(K, BK))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV

    all = (B * T).to(tl.int64)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    v += (bos * HV + i_h) * V
    do += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K*V
    dh += (i_tg * HV + i_h).to(tl.int64) * K*V
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    dq += (bos * HV + i_h) * K
    dk += (bos * HV + i_h) * K

    # for delta rule only
    if USE_DW:
        dw += (bos * HV + i_h) * K
        dv += (bos * HV + i_h) * V

    if USE_G:
        dg += i_k * all * HV
        b_dg_last = tl.zeros([1], dtype=tl.float32) if USE_G else None
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_g_last = b_gamma * min(BT, T - i_t * BT)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_qk = m_t[:, None] & m_k[None, :]
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_hv = m_t[:, None] & m_v[None, :]
        p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
        p_do = do + o_t[:, None] * (HV*V) + o_v[None, :]
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            p_dh = dh + o_v[:, None] * K + o_k[None, :]
        else:
            p_h = h + o_v[:, None] + o_k[None, :] * V
            p_dh = dh + o_v[:, None] + o_k[None, :] * V
        m_h = m_v[:, None] & m_k[None, :]
        # [BT, BV]
        b_v = tl.load(p_v, mask=m_hv, other=0.0)
        b_do = tl.load(p_do, mask=m_hv, other=0.0)
        # [BV, BK]
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        b_dh = tl.load(p_dh, mask=m_h, other=0.0)
        if USE_G:
            b_dg_last += (tl.sum(b_h * b_dh))
        # [BT, BV] @ [BV, BT] -> [BT, BT]
        b_ds = tl.dot(b_do, tl.trans(b_v), b_ds)
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dq = tl.dot(b_do, b_h.to(b_do.dtype), b_dq)
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk = tl.dot(b_v, b_dh.to(b_v.dtype), b_dk)
        if USE_DW:
            p_dv = dv + o_t[:, None] * (HV*V) + o_v[None, :]
            b_dv = tl.load(p_dv, mask=m_hv, other=0.0)
            b_dw = tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype), b_dw)

    if USE_DW:
        p_dw = dw + o_t[:, None] * (HV*K) + o_k[None, :]
        tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), mask=m_qk)

    tl.debug_barrier()
    p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
    p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_k = tl.load(p_k, mask=m_qk, other=0.0)

    p_dq = dq + o_t[:, None] * (HV*K) + o_k[None, :]
    p_dk = dk + o_t[:, None] * (HV*K) + o_k[None, :]

    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    if USE_G:
        g += bos * HV + i_h
        dg += bos * HV + i_h
        p_g = g + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * HV)
        b_dg_last *= exp2(b_g_last)
        b_dq = b_dq * exp2(b_g)[:, None] * scale
        b_dk = b_dk * tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
        b_dg_last += tl.sum(b_dk * b_k)

        b_ds = tl.where(m_A, b_ds * exp2(b_g[:, None] - b_g[None, :]), 0) * scale
        b_ds = b_ds.to(b_k.dtype)
        # Gate derivative of the inter-chunk terms before adding local attention.
        gate_inter = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)
        # The local diagonal has g_i-g_i=0 and identically zero derivative.
        # Remove it symbolically instead of subtracting two separately rounded products.
        gate_pair = b_ds * tl.dot(b_q, tl.trans(b_k))
        gate_pair = tl.where(o_t[:, None] > o_t[None, :], gate_pair, 0.)
        gate_local = tl.sum(gate_pair, axis=1) - tl.sum(gate_pair, axis=0)
        # [BT, BK]
        b_dq = tl.dot(b_ds, b_k, b_dq)
        b_dk = tl.dot(tl.trans(b_ds), b_q, b_dk)

        b_dg = gate_inter + gate_local

        p_dg = dg + o_t * HV
        # reverse cumsum runs in a separate kernel to avoid a Triton compiler issue.
        b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_qk)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_qk)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_t)

    elif USE_G_GAMMA:
        b_dq = b_dq * exp2(b_g)[:, None] * scale
        b_dk = b_dk * tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
        b_ds = tl.where(m_A, b_ds * exp2(b_g[:, None] - b_g[None, :]), 0) * scale
        b_ds = b_ds.to(b_k.dtype)
        # [BT, BK]
        b_dq = tl.dot(b_ds, b_k, b_dq)
        b_dk = tl.dot(tl.trans(b_ds), b_q, b_dk)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_qk)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_qk)

    else:
        b_ds = tl.where(m_A, b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        b_dq = tl.dot(b_ds, b_k, b_dq)
        b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_qk)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_qk)


def stable_chunk_bwd_dqkwg(q, k, v, do, h, dh, w, g, dv, scale, chunk_size=16):
    """Launch the stable FP32 gate kernel; preserve native Q/K/W derivatives."""
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    # Match FLA's tiling for the same device and FP32 inputs.
    tiling = 128 if check_shared_mem("hopper", k.device.index) else (
        64 if check_shared_mem("ampere", k.device.index) or check_shared_mem("ada", k.device.index) else 32
    )
    bk = min(max(triton.next_power_of_2(key_dim), 16), tiling)
    bv = min(max(triton.next_power_of_2(value_dim), 16), tiling)
    nk = triton.cdiv(key_dim, bk)
    dq, dk, dw = torch.empty_like(q), torch.empty_like(k), torch.empty_like(w)
    dg = torch.empty(nk, *g.shape, device=g.device, dtype=torch.float32)
    chunk_bwd_kernel_dqkwg[(nk * triton.cdiv(length, chunk_size), batch * heads)](
        q=q, k=k, v=v, g=g, g_gamma=None, h=h, do=do, dh=dh,
        dq=dq, dk=dk, dw=dw, dv=dv, dg=dg,
        cu_seqlens=None, chunk_indices=None, scale=scale,
        B=batch, T=length, H=heads, HV=heads, K=key_dim, V=value_dim,
        BT=chunk_size, BK=bk, BV=bv, STATE_V_FIRST=False,
    )
    return dq, dk, dw, dg.sum(0)


class _StableGatedDeltaRule(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, beta):
        # Version and precision guards live in the shared layer's public entry.
        q, q_rstd = _fla.l2norm_fwd(q)
        k, k_rstd = _fla.l2norm_fwd(k)
        scale = q.shape[-1] ** -0.5
        g, out, a, _, _, _ = _fla.chunk_gated_delta_rule_fwd(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=None, output_final_state=False, chunk_size=16,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, a)
        ctx.scale = scale
        return out.to(q.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        q, q_rstd, k, k_rstd, v, g, beta, a = ctx.saved_tensors
        scale = ctx.scale
        # Native FLA 0.6.0 backward, restricted to this training API. Only the
        # dq/ dk/ dw/ dg launch below uses our stable algebraic gate derivative.
        w, u = _fla.recompute_w_u_fwd(k=k, v=v, beta=beta, A=a, g=g)
        h, v_new, _ = _fla.chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g, initial_state=None,
            output_final_state=False, chunk_size=16,
        )
        dv = _fla.chunk_bwd_dv_local(q=q, k=k, g=g, do=do, scale=scale, chunk_size=16)
        dh, _, dv = _fla.chunk_gated_delta_rule_bwd_dhu(
            q=q, k=k, w=w, g=g, h0=None, dht=None, do=do, dv=dv,
            scale=scale, chunk_size=16,
        )
        dq, dk, dw, dg = stable_chunk_bwd_dqkwg(
            q=q, k=k, v=v_new, w=w, g=g, h=h, dv=dv, do=do, dh=dh,
            scale=scale, chunk_size=16,
        )
        dk2, dv, db, dg2 = _fla.prepare_wy_repr_bwd(k=k, v=v, beta=beta, g=g, A=a, dw=dw, du=dv)
        dk.add_(dk2)
        dg.add_(dg2)
        dg = _fla.chunk_local_cumsum(dg, chunk_size=16, reverse=True)
        dq = _fla.l2norm_bwd(q, q_rstd, dq)
        dk = _fla.l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta)


def stable_gated_delta_rule(q, k, v, g, beta):
    """FP32 path; called lazily after the shared layer validates FLA 0.6.0."""
    return _StableGatedDeltaRule.apply(q, k, v, g, beta)
