"""FLA training must align with native HF and independently computed FP64 truth."""

import copy
import importlib
import importlib.util
import inspect
import json
import math
import os
import subprocess
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from colossalai.shardformer.layer.gated_delta_net import FusedGatedDeltaNet, fused_gated_delta_rule
from tests.kit.numerics.scenarios import independent_impl

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("fla") is None,
    reason="GDN tests require CUDA and optional FLA 0.6.0 dependencies",
)


@pytest.fixture(autouse=True)
def strict_triton_precision(request):
    # Set this before the first FP32 FLA JIT in this isolated test module.
    # Torch's full_precision_matmul does not control Triton dot precision.
    spec = getattr(request.node, "callspec", None)
    if spec is not None and spec.params.get("dtype") in (torch.float32, torch.bfloat16):
        with patch.dict(os.environ, {"TRITON_F32_DEFAULT": "ieee"}):
            yield
    else:
        yield


def recurrence(q, k, v, g, beta):
    # Deliberately independent of both FLA and HF's chunk algorithm. Preserve
    # FP64 throughout: HF's native function explicitly casts to FP32 internally.
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


@contextmanager
def native_hf():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    # Installation of FLA changes HF's dispatch at import time. Unwrap all
    # relevant fallbacks, or this comparison could silently compare FLA to itself.
    chunk = inspect.unwrap(hf.torch_chunk_gated_delta_rule)
    with patch.object(hf, "torch_chunk_gated_delta_rule", chunk), patch.object(
        hf, "causal_conv1d_fn", inspect.unwrap(hf.causal_conv1d_fn)
    ):
        assert hf.torch_chunk_gated_delta_rule.__module__ == hf.__name__
        yield hf


@pytest.fixture
def kernel_calls(monkeypatch):
    kernel = importlib.import_module("fla.ops.gated_delta_rule.chunk")
    calls = {"forward": 0, "backward": 0}
    for key, name in (("forward", "chunk_gated_delta_rule_fwd"), ("backward", "chunk_gated_delta_rule_bwd")):
        original = getattr(kernel, name)

        def counted(*args, _key=key, _original=original, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(kernel, name, counted)
    return calls


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def expected_kernel_calls(dtype):
    kernel = importlib.import_module("fla.ops.gated_delta_rule.chunk")
    if dtype == torch.float32:
        stable = importlib.import_module("colossalai.shardformer.layer._gated_delta_net_fp32")
        backward = (stable, "stable_chunk_bwd_dqkwg")
    else:
        backward = (kernel, "chunk_gated_delta_rule_bwd")
    return [(kernel, "chunk_gated_delta_rule_fwd"), backward]


def check_alignment(test_fn, ref_fn, inputs, dtype, case, expected, kernel_targets=None):
    traces = {}
    original_capture = independent_impl.run_fwd_bwd

    def record_capture(*args, **kwargs):
        # Observe the scenario's exact runs for audit; never alter tensors,
        # the upstream gradient, missing-gradient checks or judging rules.
        trace = original_capture(*args, **kwargs)
        traces[kwargs["role"]] = trace
        return trace

    with patch.object(independent_impl, "run_fwd_bwd", record_capture):
        report = independent_impl.check_fwd_bwd(
            test_fn, ref_fn, inputs, dtype,
            expect_calls=expected_kernel_calls(dtype) if kernel_targets is None else kernel_targets,
        )  # No judge_kwargs: use the official defaults, including RMS-scaled tol.

    if os.environ.get("GDN_TEST_LOG_DIR"):
        directory = Path(os.environ["GDN_TEST_LOG_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "scope": "GDN operator/module acceptance; not a whole-model L2 report",
            "entrypoint": "tests.kit.numerics.scenarios.independent_impl.check_fwd_bwd",
            "case": case, "dtype": str(dtype), "grad_seed": 0,
            "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "passed": report.passed, "report": asdict(report),
            "traces": {role: trace.meta for role, trace in traces.items()},
        }
        (directory / f"{case}.json").write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False))
        for role, trace in traces.items():
            trace.save(directory / f"{case}-{role}.safetensors")
    assert set(traces) == {"test", "ref", "truth"}
    assert all(set(trace) == expected for trace in traces.values())
    assert all(torch.isfinite(t).all() for trace in traces.values() for t in trace.values())
    assert all(t.dtype == torch.float64 for t in traces["truth"].values())
    assert not any(trace.meta["missing_grads"] for trace in traces.values())
    report.assert_pass()
    return traces


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("length,heads,dim,decay", [
    (1, 2, 64, 1.0), (63, 2, 128, 1.0), (64, 2, 128, 1.0),
    (65, 2, 128, 1.0), (129, 2, 128, 1.0), (129, 2, 128, 0.01),
])
def test_kernel_alignment(dtype, length, heads, dim, decay):
    common = importlib.import_module("fla.ops.common.chunk_o")
    installed_backward = common.chunk_bwd_kernel_dqkwg
    torch.manual_seed(17)
    shape = (2, length, heads, dim)
    inputs = {name: torch.randn(shape, device="cuda") for name in ("q", "k", "v")}
    # Slow decay exercises long-lived states across multiple chunks.
    inputs["g"] = -decay * F.softplus(torch.randn(shape[:3], device="cuda"))
    inputs["beta"] = torch.rand(shape[:3], device="cuda")
    with native_hf() as hf:
        def reference(q, k, v, g, beta):
            if q.dtype == torch.float64:
                return recurrence(q, k, v, g, beta)
            return hf.torch_chunk_gated_delta_rule(q, k, v, g.float(), beta, use_qk_l2norm_in_kernel=True)[0]

        traces = check_alignment(
            fused_gated_delta_rule, reference, inputs, dtype, f"kernel-{dtype}-{length}-{dim}-{decay}",
            {"out", *(f"grad/{name}" for name in inputs)},
        )
    test = traces["test"]
    # The local FP32 adapter must not monkey-patch FLA for other users/models.
    assert common.chunk_bwd_kernel_dqkwg is installed_backward
    assert all(test[f"grad/{name}"].abs().sum() > 0 for name in inputs if length > 1 or name != "g")


def make_native(value_heads=16):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    config = Qwen3_5TextConfig(
        hidden_size=64, num_hidden_layers=1, layer_types=["linear_attention"],
        linear_num_key_heads=16, linear_num_value_heads=value_heads,
        linear_key_head_dim=128, linear_value_head_dim=128,
    )
    return Qwen3_5GatedDeltaNet(config, 0).cuda()


def truth_layer_forward(self, hidden_states, attention_mask=None):
    # Independent, dtype-preserving layer math. Never call the implementation
    # under test or HF norm/gate functions that silently reduce FP64 to FP32.
    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(-1)
    batch, length, _ = hidden_states.shape
    x = self.in_proj_qkv(hidden_states).transpose(1, 2)
    x = F.conv1d(x, self.conv1d.weight, self.conv1d.bias, padding=self.conv_kernel_size - 1, groups=self.conv_dim)
    q, k, v = F.silu(x[..., :length]).transpose(1, 2).split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = q.reshape(batch, length, self.num_k_heads, self.head_k_dim)
    k = k.reshape_as(q)
    repeat = self.num_v_heads // self.num_k_heads
    q, k = q.repeat_interleave(repeat, 2), k.repeat_interleave(repeat, 2)
    v = v.reshape(batch, length, self.num_v_heads, self.head_v_dim)
    g = -self.A_log.exp() * F.softplus(self.in_proj_a(hidden_states) + self.dt_bias)
    out = recurrence(q, k, v, g, self.in_proj_b(hidden_states).sigmoid())
    out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + self.layer_norm_epsilon)
    z = self.in_proj_z(hidden_states).reshape_as(out)
    out = out * self.norm.weight * F.silu(z)
    return self.out_proj(out.reshape(batch, length, self.value_dim))


def reference_layer_forward(self, hidden_states, attention_mask=None):
    # HF's native code forces FP32 internally, even for double inputs. Keep the
    # same mathematical reference in FP64 via the independent formula above.
    if hidden_states.dtype == torch.float64:
        return truth_layer_forward(self, hidden_states, attention_mask)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    return Qwen3_5GatedDeltaNet.forward(self, hidden_states, attention_mask=attention_mask)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("value_heads", [16, 32, 48])
def test_layer_alignment(dtype, value_heads):
    torch.manual_seed(29)
    native = make_native(value_heads)
    fused = FusedGatedDeltaNet.from_native_module(copy.deepcopy(native))
    reference = copy.deepcopy(native)
    reference.forward = MethodType(reference_layer_forward, reference)
    inputs = {"hidden_states": torch.randn(2, 65, 64, device="cuda")}
    # Both left/right padding, with no gradient through padding tokens.
    mask = torch.ones(2, 65, device="cuda", dtype=torch.bool)
    mask[0, :7], mask[1, -9:] = False, False
    inputs["attention_mask"] = mask
    expected = {"out", "grad/hidden_states", *(f"grad/param/{name}" for name, _ in native.named_parameters())}
    with native_hf():
        traces = check_alignment(fused, reference, inputs, dtype, f"layer-{dtype}-heads{value_heads}", expected)
    test = traces["test"]
    assert torch.count_nonzero(test["grad/hidden_states"][~mask.cpu()]) == 0


def test_conversion_and_guards():
    native = make_native().eval()
    fused = FusedGatedDeltaNet.from_native_module(native)
    assert not fused.training
    assert list(native.state_dict()) == list(fused.state_dict())
    for name, param in native.named_parameters():
        assert dict(fused.named_parameters())[name] is param
    fused.load_state_dict(native.state_dict(), strict=True)
    native.load_state_dict(fused.state_dict(), strict=True)
    x = torch.randn(1, 2, 64, device="cuda")
    with pytest.raises(NotImplementedError, match="use_cache=False"):
        fused(x, cache_params=object())
    with pytest.raises(NotImplementedError, match="Packed sequences"):
        fused(x, cu_seq_lens_q=torch.tensor([0, 2], device="cuda"))
    with pytest.raises(ValueError, match="attention_mask"):
        fused(x, attention_mask=torch.ones(1, 1, 2, 2, device="cuda"))
    with patch.dict(os.environ, {"TRITON_F32_DEFAULT": "tf32"}):
        with pytest.raises(ValueError, match="TRITON_F32_DEFAULT=ieee"):
            fused(x)
    with patch.dict(os.environ, {"TRITON_F32_DEFAULT": "ieee"}):
        q = torch.randn(1, 2, 2, 64, device="cuda")
        gate = torch.randn(1, 2, 2, device="cuda")
        with pytest.raises(ValueError, match="promotion belongs inside the core"):
            fused_gated_delta_rule(q, q, q, gate, gate, align_bf16_blocks=True)
    with pytest.raises(ValueError, match="CUDA FP32/BF16/FP16"):
        fused.cpu()(x.cpu())
    with patch("colossalai.shardformer.layer.gated_delta_net._load_fla_kernel", side_effect=ImportError("missing FLA")):
        with pytest.raises(ImportError, match="missing FLA"):
            FusedGatedDeltaNet.from_native_module(native)


def test_unified_entry_rejects_unfused_path():
    torch.manual_seed(17)
    inputs = {name: torch.randn(1, 17, 2, 64, device="cuda") for name in ("q", "k", "v")}
    inputs["g"] = -F.softplus(torch.randn(1, 17, 2, device="cuda"))
    inputs["beta"] = torch.rand(1, 17, 2, device="cuda")
    # A numerically correct PyTorch fallback must still fail the fused-path
    # contract. Exercise the official assertion rather than our own counter.
    for dtype in (torch.bfloat16, torch.float32):
        report = independent_impl.check_fwd_bwd(
            recurrence, recurrence, inputs, dtype, expect_calls=expected_kernel_calls(dtype),
        )
        called = [entry for entry in report.entries if entry.name.startswith("<called>/")]
        assert len(called) == 2 and all(not entry.passed for entry in called)
        assert all(entry.passed for entry in report.entries if not entry.name.startswith("<called>/"))
        assert not report.passed


def test_fp32_weights_autocast(kernel_calls):
    layer = FusedGatedDeltaNet.from_native_module(make_native())
    x = torch.randn(1, 17, 64, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        layer(x).float().square().mean().backward()
    assert kernel_calls == {"forward": 1, "backward": 1}
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        p.dtype == torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()
    )


def test_shardformer_replacement(kernel_calls):
    from colossalai.shardformer import ShardConfig, ShardFormer
    from colossalai.shardformer.policies.base_policy import (
        ModulePolicyDescription,
        Policy,
        SubModuleReplacementDescription,
    )

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = make_native()

        def forward(self, x):
            return self.linear_attn(x)

    class LayerPolicy(Policy):
        def config_sanity_check(self):
            assert self.shard_config.tensor_parallel_size == 1

        def preprocess(self):
            return self.model

        def module_policy(self):
            return {Block: ModulePolicyDescription(sub_module_replacement=[
                SubModuleReplacementDescription("linear_attn", FusedGatedDeltaNet)
            ])}

        def postprocess(self):
            return self.model

    model = Block()
    parameters = dict(model.named_parameters())
    model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model, policy=LayerPolicy())
    assert isinstance(model.linear_attn, FusedGatedDeltaNet)
    assert all(param is parameters[name] for name, param in model.named_parameters())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(torch.randn(1, 17, 64, device="cuda")).float().square().mean().backward()
    assert kernel_calls == {"forward": 1, "backward": 1}



@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("length,key_dim,value_dim,decay", [
    (1, 32, 64, 1.0),
    (63, 64, 32, 1.0),
    (64, 64, 64, 1.0),
    (65, 128, 128, 1.0),
    (129, 73, 40, 1.0),
    (129, 128, 128, 0.01),
    (512, 128, 128, 0.1),
])
def test_qwen_low_precision_core_alignment(dtype, autocast, length, key_dim, value_dim, decay):
    from colossalai.shardformer.layer import _gated_delta_net_fp16 as fp16

    torch.manual_seed(17)
    shape = (2, length, 2)
    inputs = {
        "q": torch.randn(*shape, key_dim, device="cuda"),
        "k": torch.randn(*shape, key_dim, device="cuda"),
        "v": torch.randn(*shape, value_dim, device="cuda"),
        "g": -decay * F.softplus(torch.randn(shape, device="cuda")),
        "beta": torch.rand(shape, device="cuda"),
    }

    def candidate(q, k, v, g, beta):
        with torch.autocast("cuda", dtype=dtype, enabled=autocast):
            if dtype == torch.bfloat16:
                return fused_gated_delta_rule(q, k, v, g.float(), beta, align_bf16_blocks=True)
            return fused_gated_delta_rule(q, k, v, g.float(), beta, align_fp16_blocks=True)

    with native_hf() as hf:
        def reference(q, k, v, g, beta):
            if q.dtype == torch.float64:
                return recurrence(q, k, v, g, beta)
            with torch.autocast("cuda", dtype=dtype, enabled=autocast):
                return hf.torch_chunk_gated_delta_rule(
                    q, k, v, g.float(), beta, use_qk_l2norm_in_kernel=True,
                )[0]

        traces = check_alignment(
            candidate, reference, inputs, dtype,
            f"qwen-{str(dtype).split('.')[-1]}-{autocast}-{length}-{key_dim}-{value_dim}-{decay}",
            {"out", *(f"grad/{name}" for name in inputs)},
            kernel_targets=[
                (fp16._block_scores_kernel, "run"),
                (fp16._block_scores_bwd_kernel, "run"),
            ],
        )
    assert torch.count_nonzero(traces["test"]["grad/g"][:, 0]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qwen_low_precision_alignment_rejects_native_fallback(dtype):
    from colossalai.shardformer.layer import _gated_delta_net_fp16 as fp16

    torch.manual_seed(17)
    shape = (1, 17, 2, 64)
    inputs = {name: torch.randn(shape, device="cuda") for name in ("q", "k", "v")}
    inputs.update(g=-torch.rand(shape[:3], device="cuda"), beta=torch.rand(shape[:3], device="cuda"))
    with native_hf() as hf:
        def reference(q, k, v, g, beta):
            if q.dtype == torch.float64:
                return recurrence(q, k, v, g, beta)
            return hf.torch_chunk_gated_delta_rule(
                q, k, v, g.float(), beta, use_qk_l2norm_in_kernel=True,
            )[0]

        report = independent_impl.check_fwd_bwd(
            reference, reference, inputs, dtype,
            expect_calls=[(fp16._block_scores_kernel, "run"), (fp16._block_scores_bwd_kernel, "run")],
        )
    calls = [entry for entry in report.entries if entry.name.startswith("<called>/")]
    assert len(calls) == 2
    assert all(not entry.passed and entry.reason == "never called" for entry in calls)
    assert not report.passed


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value_heads", [16, 32, 48])
def test_qwen_low_precision_layer_alignment(dtype, value_heads):
    from colossalai.shardformer.layer import _gated_delta_net_fp16 as aligned

    torch.manual_seed(29)
    native = make_native(value_heads)
    fused = FusedGatedDeltaNet.from_native_module(copy.deepcopy(native), force_fp32_core=True)
    reference = copy.deepcopy(native)
    reference.forward = MethodType(reference_layer_forward, reference)
    mask = torch.ones(2, 65, device="cuda", dtype=torch.bool)
    mask[0, :7], mask[1, -9:] = False, False
    inputs = {"hidden_states": torch.randn(2, 65, 64, device="cuda"), "attention_mask": mask}
    expected = {"out", "grad/hidden_states", *(f"grad/param/{name}" for name, _ in native.named_parameters())}
    with native_hf():
        traces = check_alignment(
            fused, reference, inputs, dtype, f"qwen-layer-{dtype}-heads{value_heads}", expected,
            kernel_targets=[(aligned._block_scores_kernel, "run"), (aligned._block_scores_bwd_kernel, "run")],
        )
    assert torch.count_nonzero(traces["test"]["grad/hidden_states"][~mask.cpu()]) == 0
