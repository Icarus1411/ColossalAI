"""Use the shared numerical judges and record every Qwen3.5 result."""

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

import colossalai
from colossalai.testing import spawn
from tests.kit.numerics.acceptance import CONFIGS, run_acceptance
from tests.kit.numerics.scenarios import independent_impl


def numeric_worker(rank, world_size, port, entry, config, out, blank, steps):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    if blank:
        report = independent_impl.blank_control(entry, steps=steps)
        calls = {}
    else:
        recipe = CONFIGS[config]
        kernel = importlib.import_module("fla.ops.gated_delta_rule.chunk")
        intra = importlib.import_module("fla.ops.gated_delta_rule.chunk_fwd")
        stable = importlib.import_module("colossalai.shardformer.layer._gated_delta_net_fp32")
        original_traces = independent_impl.training_traces

        def record_traces(*args, **kwargs):
            traces = original_traces(*args, **kwargs)
            for role, trace in zip(("test", "ref", "truth"), traces):
                trace.save(str(Path(out).with_suffix("")) + f"-{role}.safetensors")
            return traces

        with patch.object(independent_impl, "training_traces", record_traces), \
                patch.object(intra, "chunk_scaled_dot_kkt_fwd", wraps=intra.chunk_scaled_dot_kkt_fwd) as fwd, \
                patch.object(kernel, "chunk_gated_delta_rule_bwd", wraps=kernel.chunk_gated_delta_rule_bwd) as bwd, \
                patch.object(stable, "stable_chunk_bwd_dqkwg", wraps=stable.stable_chunk_bwd_dqkwg) as stable_bwd:
            report = independent_impl.check_training(entry, recipe.plugin_fn(), recipe.mixed_precision, steps=steps)
        calls = {"forward": fwd.call_count, "native_backward": bwd.call_count, "stable_backward": stable_bwd.call_count}
        if "fused" in entry or config.startswith("shardformer"):
            assert calls["forward"] >= 3 * steps
            assert calls["stable_backward"] >= 3 * steps
    payload = {"entry": entry, "config": config, "blank": blank, "steps": steps,
               "calls": calls, "passed": report.passed, "report": asdict(report)}
    Path(out).write_text(json.dumps(payload, indent=2))
    print(report.format(), flush=True)
    report.assert_pass()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", default="transformers_qwen3_5_0_8b_tiny")
    parser.add_argument("--config", default="shardformer_fp32")
    parser.add_argument("--out", required=True)
    parser.add_argument("--blank", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--checks")
    args = parser.parse_args()
    if args.checks:
        result = run_acceptance(args.entry, args.config.split(","), args.checks.split(","), steps=args.steps)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(json.dumps(result.get("summary", {})), flush=True)
        if result.get("summary", {}).get("failed", 0):
            raise SystemExit(1)
    else:
        spawn(numeric_worker, 1, entry=args.entry, config=args.config, out=args.out, blank=args.blank, steps=args.steps)


if __name__ == "__main__":
    main()
