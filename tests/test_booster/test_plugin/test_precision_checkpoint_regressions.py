"""Precision and distributed-state regressions without optional Apex kernels."""

import copy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.testing import spawn


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)
        # Checkpointing must also handle parameters without optimizer moments.
        self.unused = nn.Parameter(torch.randn(3))

    def forward(self, x):
        return self.linear(x)


def _zero_roundtrip(rank, world_size, port, stage, offload, sharded, directory):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)

    def build():
        torch.manual_seed(123)
        model = SmallModel().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00123)
        booster = Booster(plugin=LowLevelZeroPlugin(stage=stage, precision="bf16", cpu_offload=offload))
        model, optimizer, *_ = booster.boost(model, optimizer)
        return booster, model, optimizer

    torch.manual_seed(300 + rank)
    batches = [torch.randn(2, 8, device="cuda") for _ in range(3)]

    def step(booster, model, optimizer, i):
        loss = (model(batches[i]).float() - 0.37).square().mean()
        booster.backward(loss, optimizer)
        optimizer.step()
        optimizer.zero_grad()

    booster, model, optimizer = build()
    # No optimizer state exists yet: serialization still preserves all masters.
    optimizer.load_state_dict(optimizer.state_dict())
    step(booster, model, optimizer, 0)
    masters = [p.detach().clone() for p in optimizer.get_working_to_master_map().values()]
    weights = Path(directory) / "model"
    optim_path = Path(directory) / ("optimizer" if sharded else "optimizer.pt")
    booster.save_model(model, str(weights), shard=True, use_safetensors=True)
    booster.save_optimizer(optimizer, str(optim_path), shard=sharded, size_per_shard=0.0001)
    dist.barrier()
    for i in (1, 2):
        step(booster, model, optimizer, i)
    expected = {n: p.detach().clone() for n, p in model.unwrap().named_parameters()}

    other_booster, other_model, other_optimizer = build()
    other_booster.load_model(other_model, str(weights))
    other_booster.load_optimizer(other_optimizer, str(optim_path))
    for saved, loaded in zip(masters, other_optimizer.get_working_to_master_map().values()):
        assert torch.equal(saved, loaded), "FP32 master weights lost bits during checkpointing"
    assert all("_colossalai_master_param" not in state for state in other_optimizer.optim.state.values())
    for i in (1, 2):
        step(other_booster, other_model, other_optimizer, i)
    for name, param in other_model.unwrap().named_parameters():
        assert torch.equal(expected[name], param), name

    # Legacy optimizer checkpoints without the new optional field still load.
    legacy = copy.deepcopy(other_optimizer.state_dict())
    for state in legacy["state"].values():
        state.pop("_colossalai_master_param", None)
    other_optimizer.load_state_dict(legacy)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("stage,offload,sharded", [(1, False, True), (2, False, True),
                                                (2, False, False), (2, True, True)])
def test_zero_master_weight_resume(tmp_path, stage, offload, sharded):
    spawn(_zero_roundtrip, 2, stage=stage, offload=offload, sharded=sharded, directory=str(tmp_path))


class TiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(8, 8, bias=False)
        self.right = nn.Linear(8, 8, bias=False)
        self.right.weight = self.left.weight

    def forward(self, x):
        return self.right(self.left(x).tanh())


def _data():
    return {"x": torch.randn(4, 8)}


def _outputs(x):
    return {"output": x}


def _gemini(keep_gathered):
    from colossalai.booster.plugin import GeminiPlugin

    return GeminiPlugin(
        precision="bf16", enable_gradient_accumulation=True,
        chunk_config_dict={size: {"chunk_size": 512, "keep_gathered": keep_gathered} for size in (1, 2)},
        chunk_init_device=torch.device("cuda"),
    )


def _entry():
    from tests.kit.model_zoo.registry import ModelAttribute
    from tests.kit.numerics import NumericsSpec

    return TiedModel, _data, _outputs, None, ModelAttribute(numerics=NumericsSpec())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("keep_gathered", [False, True])
def test_gemini_accumulation_and_tied_resume(keep_gathered):
    from functools import partial
    from tests.kit.numerics.scenarios import data_parallel, resume

    factory = partial(_gemini, keep_gathered)
    # Two ranks with two micro-batches each must match one rank with four.
    data_parallel.check(_entry(), factory, world_size=2, micro_batches=4).assert_pass()
    # Exercise both gathered and sharded optimizer state, including master weights.
    resume.check(_entry(), factory, world_size=2, shard_optimizer=True).assert_pass()


def test_reference_preserves_original_fp32_master():
    from functools import partial
    from colossalai.accelerator import get_accelerator
    from tests.kit.numerics.capture.recipes import Recipe
    from tests.kit.numerics.capture.training import Prepared, run_reference

    class ScalarModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0037))

        def forward(self):
            return self.weight

    model = ScalarModel().to(get_accelerator().get_current_device())
    prepared = Prepared("scalar-master", model, model, [{}], _outputs, None, seed=0)
    optimizer_fn = partial(torch.optim.SGD, lr=0.002)
    recipe = Recipe(torch.bfloat16, master_weights=True, master_from_fp32=True)
    trace = run_reference(prepared, recipe, optimizer_fn)
    expected = (model.weight.detach().cpu() - 0.002 * trace["grad/param/weight"].float()).bfloat16()
    assert torch.equal(trace["param/weight"], expected)
    cast_master = run_reference(prepared, Recipe(torch.bfloat16, master_weights=True), optimizer_fn)
    assert not torch.equal(cast_master["param/weight"], expected), "fixture must detect rounded master initialization"
