"""Offline real-checkpoint forward alignment and full text-parameter SFT smoke.

The FP64 forward reference is test-only. Training uses the same policy as tiny
tests, with a frozen vision tower and LowLevelZeRO-2 BF16 FP32-master AdamW.
"""

import argparse
import copy
import gc
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import spawn
from tests.kit.model_zoo.transformers.qwen3_5 import prepare_text
from tests.kit.model_zoo.transformers.qwen3_5_reference import native_reference
from tests.kit.numerics import Trace, judge
from tests.kit.numerics.capture.utils import full_precision_matmul


def worker(rank, world_size, port, snapshot, out, steps, precision, forward_only):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    torch.manual_seed(20261104)
    torch.cuda.reset_peak_memory_stats()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, dtype=getattr(torch, {"bf16": "bfloat16", "fp32": "float32"}[precision]),
        attn_implementation="eager"
    )
    model = prepare_text(native_reference(model)).cuda().train()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = ["Explain why the sky appears blue. " * 10, "Write a short Python function that adds two numbers. " * 4]
    batch = tokenizer(texts, padding="max_length", truncation=True, max_length=65, return_tensors="pt")
    batch = {k: v.cuda() for k, v in batch.items() if k in ("input_ids", "attention_mask")}
    batch["labels"] = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)
    directory = Path(out).with_suffix("")
    directory.mkdir(exist_ok=True)
    with full_precision_matmul(), torch.no_grad():
        truth_model = copy.deepcopy(model).double()
        result = truth_model(**batch)
        truth = Trace.capture({"logits": result.logits.cpu(), "loss": result.loss.cpu()}, role="truth")
        del truth_model, result
        gc.collect()
        torch.cuda.empty_cache()
        result = model(**batch)
        reference = Trace.capture({"logits": result.logits.cpu(), "loss": result.loss.cpu()}, role="ref")
        del result
    model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
    intra = importlib.import_module("fla.ops.gated_delta_rule.chunk_fwd")
    stable = importlib.import_module("colossalai.shardformer.layer._gated_delta_net_fp32")
    with patch.object(intra, "chunk_scaled_dot_kkt_fwd", wraps=intra.chunk_scaled_dot_kkt_fwd) as fwd:
        with full_precision_matmul(), torch.no_grad():
            result = model(**batch)
            test = Trace.capture({"logits": result.logits.cpu(), "loss": result.loss.cpu()}, role="test")
        assert fwd.call_count == model.config.text_config.layer_types.count("linear_attention")
    alignment = judge(test, reference, truth)
    for role, trace in (("test", test), ("ref", reference), ("truth", truth)):
        trace.save(directory / f"forward-{role}.safetensors")
    payload = {"snapshot": snapshot, "world_size": world_size, "precision": precision,
               "forward_alignment": asdict(alignment), "forward_passed": alignment.passed,
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
               "losses": []}
    Path(out).write_text(json.dumps(payload, indent=2))
    print(alignment.format(), flush=True)
    if forward_only:
        alignment.assert_pass()
        torch.distributed.destroy_process_group()
        return
    del result, test, reference, truth
    gc.collect()
    torch.cuda.empty_cache()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    target = model.model.language_model.layers[0].linear_attn.in_proj_qkv.weight
    before = target.detach().clone()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    booster = Booster(plugin=LowLevelZeroPlugin(stage=2, precision=precision))
    model, optimizer, *_ = booster.boost(model, optimizer)
    with patch.object(stable, "stable_chunk_bwd_dqkwg", wraps=stable.stable_chunk_bwd_dqkwg) as backward:
        for step in range(steps):
            result = model(**batch)
            booster.backward(result.loss, optimizer)
            missing, nonfinite = [], []
            for name, parameter in model.unwrap().named_parameters():
                gradient = optimizer.get_param_grad(parameter)
                if parameter.requires_grad:
                    if gradient is None:
                        missing.append(name)
                    elif not torch.isfinite(gradient).all():
                        nonfinite.append(name)
                else:
                    assert gradient is None
            assert not missing and not nonfinite, (missing, nonfinite)
            optimizer.step()
            optimizer.zero_grad()
            payload["losses"].append(result.loss.item())
            print(json.dumps({"step": step, "loss": result.loss.item()}), flush=True)
        payload["fused_backward_calls"] = backward.call_count
    payload["weight_updated"] = not torch.equal(before, target)
    payload["training_passed"] = payload["weight_updated"] and backward.call_count >= 18 * steps
    payload["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    Path(out).write_text(json.dumps(payload, indent=2))
    assert payload["training_passed"]
    alignment.assert_pass()
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()
    spawn(worker, args.world_size, snapshot=args.snapshot, out=args.out, steps=args.steps,
          precision=args.precision, forward_only=args.forward_only)


if __name__ == "__main__":
    main()
