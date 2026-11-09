"""Full real-checkpoint training alignment through the unmodified shared tool."""
import argparse
import importlib
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

import colossalai
from colossalai.testing import spawn
from tests.kit.model_zoo.registry import ModelAttribute
from tests.kit.model_zoo.transformers.qwen3_5 import outputs, loss_fn, prepare_text, causal_labels
from tests.kit.model_zoo.transformers.qwen3_5_reference import native_reference
from tests.kit.numerics import NumericsSpec
from tests.kit.numerics.acceptance import CONFIGS
from tests.kit.numerics.scenarios.independent_impl import check_training


def model_fn(snapshot):
    return prepare_text(native_reference(Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot, dtype=torch.float32, attn_implementation="eager", local_files_only=True)))


def data_fn(snapshot):
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(["Explain why the sky appears blue. " * 10,
                       "Write a short Python function that adds two numbers. " * 4],
                      padding="max_length", truncation=True, max_length=65, return_tensors="pt")
    batch = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")}
    batch["labels"] = causal_labels(batch["input_ids"], batch["attention_mask"])
    return batch


def worker(rank, world_size, port, snapshot, config, out, steps):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    entry = (partial(model_fn, snapshot), partial(data_fn, snapshot), outputs, loss_fn,
             ModelAttribute(has_control_flow=True, numerics=NumericsSpec(prepare_fn=prepare_text)))
    recipe = CONFIGS[config]
    intra = importlib.import_module("fla.ops.gated_delta_rule.chunk_fwd")
    stable = importlib.import_module("colossalai.shardformer.layer._gated_delta_net_fp32")
    forward, backward = (intra, "chunk_scaled_dot_kkt_fwd"), (stable, "stable_chunk_bwd_dqkwg")
    if config == "shardformer_bf16":
        aligned = importlib.import_module("colossalai.shardformer.layer._gated_delta_net_fp16")
        forward, backward = (aligned._block_scores_kernel, "run"), (aligned._block_scores_bwd_kernel, "run")
    with patch.object(forward[0], forward[1], wraps=getattr(*forward)) as fwd, \
            patch.object(backward[0], backward[1], wraps=getattr(*backward)) as bwd:
        report = check_training(entry, recipe.plugin_fn(), recipe.mixed_precision, steps=steps)
    payload = {"snapshot": snapshot, "config": config, "steps": steps, "passed": report.passed,
               "fused_forward_calls": fwd.call_count, "fused_backward_calls": bwd.call_count,
               "report": asdict(report), "peak_memory_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
    Path(out).write_text(json.dumps(payload, indent=2))
    print(report.format(), flush=True)
    assert fwd.call_count >= 18 * steps and bwd.call_count >= 18 * steps
    report.assert_pass()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--config", default="shardformer_fp32")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spawn(worker, 1, snapshot=args.snapshot, config=args.config, out=args.out, steps=args.steps)
