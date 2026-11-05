"""Test-only HF native dispatch and dtype-preserving Qwen3.5 FP64 reference.

No installed library or global dispatch is modified. Low precision and FP32 use
HF's unwrapped PyTorch implementation. FP64 uses a direct recurrence and removes
HF's explicit FP32 casts from normalization, attention and rotary embeddings.
"""

import inspect
from types import FunctionType, MethodType

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf


def _with_globals(fn, **replacements):
    fn = inspect.unwrap(fn)
    result = FunctionType(fn.__code__, dict(fn.__globals__, **replacements), fn.__name__, fn.__defaults__, fn.__closure__)
    result.__kwdefaults__ = fn.__kwdefaults__
    return result


_native_gdn = _with_globals(
    hf.Qwen3_5GatedDeltaNet.forward,
    torch_chunk_gated_delta_rule=inspect.unwrap(hf.torch_chunk_gated_delta_rule),
    causal_conv1d_fn=inspect.unwrap(hf.causal_conv1d_fn),
)


def recurrence(q, k, v, g, beta):
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * q.shape[-1] ** -0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outputs = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp()[..., None, None]
        residual = (v[:, t] - (state * k[:, t, :, :, None]).sum(-2)) * beta[:, t, :, None]
        state = state + k[:, t, :, :, None] * residual.unsqueeze(-2)
        outputs.append((state * q[:, t, :, :, None]).sum(-2))
    return torch.stack(outputs, dim=1)


def gdn_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    if hidden_states.dtype != torch.float64:
        return _native_gdn(self, hidden_states, cache_params, attention_mask, **kwargs)
    if cache_params is not None:
        raise ValueError("FP64 training reference does not support caches")
    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(-1)
    batch, length, _ = hidden_states.shape
    mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)
    mixed = F.conv1d(mixed, self.conv1d.weight, self.conv1d.bias,
                     padding=self.conv_kernel_size - 1, groups=self.conv_dim)
    q, k, v = F.silu(mixed[..., :length]).transpose(1, 2).split((self.key_dim, self.key_dim, self.value_dim), -1)
    q = q.reshape(batch, length, self.num_k_heads, self.head_k_dim)
    k = k.reshape_as(q)
    repeats = self.num_v_heads // self.num_k_heads
    q, k = q.repeat_interleave(repeats, 2), k.repeat_interleave(repeats, 2)
    v = v.reshape(batch, length, self.num_v_heads, self.head_v_dim)
    g = -self.A_log.exp() * F.softplus(self.in_proj_a(hidden_states) + self.dt_bias)
    output = recurrence(q, k, v, g, self.in_proj_b(hidden_states).sigmoid())
    # Call the norm module as in HF so dtype auditing observes the same modules.
    output = self.norm(output.reshape(-1, self.head_v_dim), self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim))
    return self.out_proj(output.reshape(batch, length, self.value_dim))


def norm_forward(self, x):
    if x.dtype != torch.float64:
        return hf.Qwen3_5RMSNorm.forward(self, x)
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * (1 + self.weight)


def gated_norm_forward(self, x, gate):
    if x.dtype != torch.float64:
        return hf.Qwen3_5RMSNormGated.forward(self, x, gate)
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon) * self.weight * F.silu(gate)


def rotary_forward(self, x, position_ids):
    if x.dtype != torch.float64:
        return hf.Qwen3_5TextRotaryEmbedding.forward(self, x, position_ids)
    freq = self.inv_freq[None, None, :, None].double().expand(3, position_ids.shape[1], -1, 1)
    freq = (freq @ position_ids[:, :, None, :].double()).transpose(2, 3)
    return (self.recomposition_frequencies(freq.cos() * self.attention_scaling),
            self.recomposition_frequencies(freq.sin() * self.attention_scaling))


def eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    if query.dtype != torch.float64:
        return hf.eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    key = hf.repeat_kv(key, module.num_key_value_groups)
    value = hf.repeat_kv(value, module.num_key_value_groups)
    scores = query @ key.transpose(2, 3) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    weights = scores.softmax(-1)
    return (weights @ value).transpose(1, 2).contiguous(), weights


_attention_forward = _with_globals(hf.Qwen3_5Attention.forward, eager_attention_forward=eager_attention)


def causal_loss(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100, shift_labels=None, **kwargs):
    from transformers.loss.loss_utils import ForCausalLMLoss

    if logits.dtype != torch.float64:
        return ForCausalLMLoss(logits, labels, vocab_size, num_items_in_batch=num_items_in_batch,
                              ignore_index=ignore_index, shift_labels=shift_labels, **kwargs)
    if shift_labels is None:
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), shift_labels.reshape(-1),
                           ignore_index=ignore_index, reduction="sum" if num_items_in_batch is not None else "mean")
    return loss if num_items_in_batch is None else loss / num_items_in_batch


def native_reference(model):
    """Preserve HF classes/state_dict names while giving each instance native dispatch."""
    forwards = {
        hf.Qwen3_5GatedDeltaNet: gdn_forward,
        hf.Qwen3_5RMSNorm: norm_forward,
        hf.Qwen3_5RMSNormGated: gated_norm_forward,
        hf.Qwen3_5TextRotaryEmbedding: rotary_forward,
        hf.Qwen3_5Attention: _attention_forward,
    }
    for module in model.modules():
        if type(module) in forwards:
            module.forward = MethodType(forwards[type(module)], module)
    model.loss_function = causal_loss
    return model
