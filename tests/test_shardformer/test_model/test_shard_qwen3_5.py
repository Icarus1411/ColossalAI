"""Qwen3.5 TP=1 policy, text-only training and actual fused-kernel coverage."""

import importlib
import importlib.util
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import transformers

from colossalai.lazy import LazyInitContext
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.layer import FusedGatedDeltaNet
from tests.kit.model_zoo.transformers.qwen3_5 import build_model, causal_labels, data_gen, token_count, token_loss

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("fla") is None
    or not hasattr(transformers, "Qwen3_5ForCausalLM"),
    reason="Qwen3.5 GDN training requires CUDA and optional FLA 0.6.0",
)


@pytest.fixture(autouse=True)
def ieee_triton(monkeypatch):
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")


def test_causal_loss_ignores_padding_prediction_positions():
    ids = torch.tensor([[0, 0, 7, 8, 9], [7, 8, 9, 0, 0]])
    mask = ids.ne(0)
    labels = causal_labels(ids, mask)
    batch = {"labels": labels}
    logits = torch.randn(2, 5, 16, requires_grad=True)
    loss = token_loss(SimpleNamespace(logits=logits), batch)
    # Each padded row must have exactly the two targets of its unpadded [7, 8, 9].
    prediction_logits = torch.stack([logits[0, 2], logits[0, 3], logits[1, 0], logits[1, 1]])
    expected = torch.nn.functional.cross_entropy(
        prediction_logits, torch.tensor([8, 9, 8, 9]), reduction="sum",
    )
    torch.testing.assert_close(loss, expected)
    assert token_count(batch) == 4
    loss.backward()
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    assert torch.count_nonzero(logits.grad[0, 2:4]) > 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="DP alignment requires two GPUs")
@pytest.mark.parametrize("backend", ["ddp", "gemini"])
def test_fp16_token_weighted_padding(backend):
    from colossalai.booster.mixed_precision import FP16TorchMixedPrecision
    from colossalai.booster.plugin import GeminiPlugin, TorchDDPPlugin
    from tests.kit.numerics.scenarios import data_parallel

    if backend == "ddp":
        plugin_fn, mp = TorchDDPPlugin, FP16TorchMixedPrecision(init_scale=1024)
    else:
        plugin_fn = partial(GeminiPlugin, precision="fp16", initial_scale=1024, enable_gradient_accumulation=True)
        mp = None
    data_parallel.check(
        "transformers_qwen3_5_2b_tiny_fused", plugin_fn, mp,
        world_size=2, micro_batches=4, token_weighted=True,
    ).assert_pass()


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_text_policy(conditional, checkpoint, dtype):
    torch.manual_seed(42)
    model = build_model(conditional=conditional).cuda().to(dtype).train()
    original = {name: id(p) for name, p in model.named_parameters()}
    model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
    assert original == {name: id(p) for name, p in model.named_parameters()}
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert sum(isinstance(m, FusedGatedDeltaNet) for m in model.modules()) == 3
    if checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vision_before = {}
    if conditional:
        vision_before = {n: p.detach().clone() for n, p in model.model.visual.named_parameters()}
        assert not any(p.requires_grad for p in model.model.visual.parameters())
        model.model.visual.register_forward_pre_hook(lambda *args: pytest.fail("text path called vision tower"))
    batch = {k: v.cuda() for k, v in data_gen().items()}
    intra = importlib.import_module("fla.ops.gated_delta_rule.chunk_fwd")
    backward = (importlib.import_module("colossalai.shardformer.layer._gated_delta_net_fp32"),
                "stable_chunk_bwd_dqkwg")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    before = model.lm_head.weight.detach().clone()
    with patch.object(intra, "chunk_scaled_dot_kkt_fwd", wraps=intra.chunk_scaled_dot_kkt_fwd) as fwd, \
            patch.object(backward[0], backward[1], wraps=getattr(*backward)) as bwd:
        result = model(**batch)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert fwd.call_count >= 3 and bwd.call_count >= 3
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    optimizer.step()
    assert not torch.equal(before, model.lm_head.weight)
    if conditional:
        assert all(torch.equal(vision_before[n], p) and p.grad is None for n, p in model.model.visual.named_parameters())


def test_lazy_init():
    with LazyInitContext(default_device=torch.device("cuda")):
        model = build_model(conditional=True)
    model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
    output = model(**{k: v.cuda() for k, v in data_gen().items()})
    output.loss.backward()
    assert torch.isfinite(output.loss)


def test_reject_unsupported_inputs_and_configuration():
    model = build_model(conditional=True, fused=True).cuda()
    batch = {k: v.cuda() for k, v in data_gen().items()}
    for extra in ({"use_cache": True}, {"pixel_values": torch.zeros(1, device="cuda")},
                  {"cu_seq_lens_q": torch.tensor([0, 65], device="cuda")}):
        with pytest.raises(NotImplementedError):
            model(**batch, **extra)
    with pytest.raises(NotImplementedError, match="enable_flash_attention"):
        ShardFormer(ShardConfig(enable_tensor_parallelism=False, enable_flash_attention=True)).optimize(build_model())


def test_reapply_text_policy_preserves_fused_modules():
    model = build_model(conditional=True, fused=True).cuda()
    layers = [m for m in model.modules() if isinstance(m, FusedGatedDeltaNet)]
    assert len(layers) == 3
    sharder = ShardFormer(ShardConfig(enable_tensor_parallelism=False))
    model, _ = sharder.optimize(model)
    assert layers == [m for m in model.modules() if isinstance(m, FusedGatedDeltaNet)]
    layers[0].force_fp32_core = False
    with pytest.raises(AssertionError, match="has been replaced"):
        sharder.optimize(model)


def test_stable_backward_with_colo_tensor():
    from colossalai.shardformer.layer import fused_gated_delta_rule
    from colossalai.tensor.colo_tensor import ColoTensor

    torch.manual_seed(123)
    shape = (1, 17, 2, 64)
    values = [torch.randn(shape, device="cuda") for _ in range(3)]
    values += [-torch.rand(shape[:3], device="cuda"), torch.rand(shape[:3], device="cuda")]
    upstream = torch.randn(shape, device="cuda")
    gradients = []
    for subclass in (False, True):
        leaves = [value.detach().clone().requires_grad_() for value in values]
        inputs = [x.as_subclass(ColoTensor) for x in leaves] if subclass else leaves
        output = fused_gated_delta_rule(*inputs)
        gradients.append(torch.autograd.grad((output * upstream).sum(), leaves))
    for plain, colo in zip(*gradients):
        torch.testing.assert_close(colo, plain, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("length,dim", [(1, 32), (17, 64), (65, 128)])
def test_promoted_bf16_blocks_against_fp64(length, dim):
    from colossalai.shardformer.layer import fused_gated_delta_rule
    from tests.kit.model_zoo.transformers.qwen3_5_reference import recurrence
    from tests.kit.numerics.capture.utils import full_precision_matmul

    torch.manual_seed(101 + length)
    shape = (2, length, 2, dim)
    values = [torch.randn(shape, device="cuda").bfloat16().float() for _ in range(3)]
    values.extend([-torch.rand(shape[:-1], device="cuda"), torch.rand(shape[:-1], device="cuda").bfloat16().float()])
    upstream = torch.randn_like(values[0])

    def evaluate(reference):
        inputs = [(x.double() if reference else x).detach().requires_grad_() for x in values]
        out = recurrence(*inputs) if reference else fused_gated_delta_rule(*inputs, align_bf16_blocks=True)
        return (out, *torch.autograd.grad(out, inputs, upstream.to(out)))

    with full_precision_matmul():
        actual, expected = evaluate(False), evaluate(True)
    for x, target in zip(actual, expected):
        torch.testing.assert_close(x.double(), target, rtol=3e-5, atol=1e-5)
    assert torch.count_nonzero(actual[4][:, 0]) == 0


def _check_fp16_training(rank, world_size, port, backend, entry):
    import colossalai
    from colossalai.booster.mixed_precision import FP16TorchMixedPrecision
    from colossalai.booster.plugin import GeminiPlugin, TorchDDPPlugin
    from tests.kit.numerics.scenarios.independent_impl import check_training, training_traces

    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    if backend == "ddp":
        plugin, mp = TorchDDPPlugin(), FP16TorchMixedPrecision(init_scale=1024)
    else:
        plugin, mp = GeminiPlugin(precision="fp16", initial_scale=1024, enable_gradient_accumulation=True), None
    if backend == "gemini":
        from tests.kit.numerics.capture.plugins import FirstMomentIsGradient
        from tests.kit.numerics.capture.training import default_optimizer, prepare, run_boosted
        from tests.kit.numerics.judge import judge, require_gradients

        # Isolate Qwen fusion using HF native kernels under the same optimizer.
        # The standard acceptance reference unscales in FP16 before its FP32
        # update, unlike Gemini. Keep that official result in the acceptance
        # report; it is not an independent test of the fused Qwen arithmetic.
        test, _, truth = training_traces(entry, plugin, mp, steps=3)
        native = prepare(entry.removesuffix("_fused"), steps=3, seed=0)
        reference = run_boosted(
            native, GeminiPlugin(precision="fp16", initial_scale=1024, enable_gradient_accumulation=True),
            None, FirstMomentIsGradient(default_optimizer), role="ref",
        )
        require_gradients(judge(test, reference, truth), test=test, ref=reference, truth=truth).assert_pass()
    else:
        check_training(entry, plugin, mp, steps=3).assert_pass()


@pytest.mark.parametrize("entry,backend", [
    ("transformers_qwen3_5_0_8b_tiny_conditional_fused", "ddp"),
    ("transformers_qwen3_5_0_8b_tiny_conditional_fused", "gemini"),
    ("transformers_qwen3_5_2b_tiny_fused", "ddp"),
    ("transformers_qwen3_5_2b_tiny_fused", "gemini"),
])
def test_fp16_training_alignment(backend, entry):
    from colossalai.testing import spawn

    spawn(_check_fp16_training, 1, backend=backend, entry=entry)
