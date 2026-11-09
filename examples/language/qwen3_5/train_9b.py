"""Offline Qwen3.5-9B text training, CPU AdamW offload, and DP measurements."""
import argparse
import hashlib
import subprocess
import sys
import json
import math
import os
import re
import resource
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.shardformer import ShardConfig, ShardFormer


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--gsm8k-sft", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval-every", type=int, default=16)
    parser.add_argument("--sft-validation-count", type=int, default=128)
    parser.add_argument("--sft-test-count", type=int, default=128)
    parser.add_argument("--kernel-audit", action="store_true", help="Count actual Triton launches; omit for timing")
    return parser.parse_args()


def evaluate(model, tokenizer, data, downstream):
    model.eval()
    with torch.no_grad():
        losses = []
        for ids in data["valid"][:2048].long().view(-1, 256).cuda():
            batch = ids[None, :]
            losses.append(model(input_ids=batch, labels=batch).loss.item())
        metrics = {"validation_nll": sum(losses) / len(losses)}
        metrics["validation_ppl"] = math.exp(metrics["validation_nll"])
        if downstream:
            predictions = []
            for item in data["gsm8k"]:
                prompt = "Question: " + item["question"] + "\nGive only the numerical answer.\nAnswer:"
                ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
                tokens = []
                for _ in range(12):
                    token = model(input_ids=ids, logits_to_keep=1).logits[:, -1].argmax(-1, keepdim=True)
                    tokens.append(token.item())
                    answer = tokenizer.decode(tokens, skip_special_tokens=True)
                    if tokens[-1] == tokenizer.eos_token_id or "\n" in answer.strip(" "):
                        break
                    ids = torch.cat([ids, token], dim=1)
                answer = answer.strip().splitlines()
                number = re.fullmatch(r"[+-]?\d+(?:\.\d+)?", answer[0].replace(",", "").strip()) if answer else None
                hit = bool(number and float(number.group()) == float(item["gold"].replace(",", "")))
                predictions.append({"question": item["question"], "gold": item["gold"], "output": answer, "correct": hit})
            metrics["gsm8k"] = predictions
    model.train()
    return metrics


def main():
    args = arguments()
    colossalai.launch_from_torch()
    rank, world = dist.get_rank(), dist.get_world_size()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"rank-{rank}.jsonl").exists():
        raise FileExistsError(f"Choose a fresh output directory: {out}")
    torch.manual_seed(20261105)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = torch.load(args.data, map_location="cpu", weights_only=True)
    manifest = json.loads(args.data.with_suffix(".json").read_text())
    snapshot = manifest["model"]
    if min(args.steps, args.length, args.batch, args.lr, args.eval_every, args.sft_validation_count, args.sft_test_count) <= 0:
        raise ValueError("steps, length, batch and lr must be positive")
    if args.gsm8k_sft and world != 1:
        raise ValueError("GSM8K SFT currently supports a single rank")
    required = args.steps * args.length * args.batch * world
    if not args.gsm8k_sft and required > data["train"].numel():
        raise ValueError(f"Need {required} tokens")
    kernel_counts = {"forward": 0, "backward": 0}
    if args.kernel_audit:
        from colossalai.shardformer.layer import _gated_delta_net_fp16 as core

        def instrument(kernel, name):
            original = kernel.run
            def counted(*a, **kw):
                kernel_counts[name] += 1
                return original(*a, **kw)
            kernel.run = counted
        instrument(core._block_scores_kernel, "forward")
        instrument(core._block_scores_bwd_kernel, "backward")
    raw = Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    )
    if args.reference:
        from tests.kit.model_zoo.transformers.qwen3_5 import prepare_text
        from tests.kit.model_zoo.transformers.qwen3_5_reference import native_reference

        raw = native_reference(prepare_text(raw))
    else:
        raw, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(raw)
    raw.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in raw.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    assert all(not p.requires_grad for p in raw.model.visual.parameters())
    assert all(p.requires_grad for p in raw.model.language_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, fused=True)
    plugin = LowLevelZeroPlugin(stage=2, precision="bf16", cpu_offload=True)
    booster = Booster(plugin=plugin)
    model, optim, *_ = booster.boost(raw, optimizer)
    model.train()
    target = raw.model.language_model.layers[0].linear_attn.in_proj_qkv.weight
    probe = target.detach().flatten()[:65536].clone()
    def record(event):
        with (out / f"rank-{rank}.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
    repo = Path(__file__).resolve().parents[3]
    def git(*items):
        return subprocess.check_output(["git", "-C", str(repo), *items], text=True).strip()
    sources = [Path(__file__), Path(__file__).with_name("gsm8k_sft.py")]
    record({"provenance": {"git_sha": git("rev-parse", "HEAD"),
                            "tracked_diff": git("status", "--porcelain", "--untracked-files=no"),
                            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                            "data_manifest": manifest, "command": sys.argv,
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "threads_per_rank": torch.get_num_threads(), "seed": 20261105,
                            "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT")}})
    record({"args": {k: str(v) for k, v in vars(args).items()}, "rank": rank, "world": world,
            "trainable": n, "frozen": sum(p.numel() for p in raw.parameters() if not p.requires_grad),
            "device": torch.cuda.get_device_name(), "torch": str(torch.__version__), "snapshot": snapshot})
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if args.gsm8k_sft:
        import gsm8k_sft

        sft_train = gsm8k_sft.examples(tokenizer, manifest["gsm8k"], "train", 0, args.steps * args.batch)
        sft_valid = gsm8k_sft.examples(tokenizer, manifest["gsm8k"], "train", len(sft_train), args.sft_validation_count)
        sft_test = gsm8k_sft.examples(tokenizer, manifest["gsm8k"], "test", 16, args.sft_test_count)
        splits = {"train": sft_train, "validation": sft_valid, "test": sft_test}
        question_sets = {name: {r[3] for r in rows} for name, rows in splits.items()}
        assert not (question_sets["train"] & question_sets["validation"])
        assert not ((question_sets["train"] | question_sets["validation"]) & question_sets["test"])
        record({"sft_splits": {name: {"count": len(rows), "sha256": hashlib.sha256(
            json.dumps(rows, ensure_ascii=False).encode()).hexdigest()} for name, rows in splits.items()},
                "sft_protocol": {"train_skip": 0, "validation_skip": len(sft_train), "test_skip": 16,
                                 "order": "dataset order, one epoch", "generation_max_tokens": 12,
                                 "loss": "numeric answer and EOS only", "thinking": False}})
    else:
        train = data["train"][:required].long().reshape(args.steps, world, args.batch, args.length)

    def evaluation(at):
        if args.gsm8k_sft:
            measurements = {"validation": gsm8k_sft.evaluate(model, tokenizer, sft_valid, False)}
            if at in (0, args.steps):
                measurements["test"] = gsm8k_sft.evaluate(model, tokenizer, sft_test, True)
        else:
            measurements = evaluate(model, tokenizer, data, at == args.steps)
        record({"eval_at": at, **measurements})

    if args.eval:
        evaluation(0)
    for step in range(args.steps):
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if args.gsm8k_sft:
            rows = sft_train[step * args.batch:(step + 1) * args.batch]
            feed = gsm8k_sft.batch(rows, tokenizer.pad_token_id)
        else:
            ids = train[step, rank].cuda()
            feed = {"input_ids": ids, "labels": ids}
        loss = model(**feed).loss
        if not torch.isfinite(loss.detach()).item():
            raise FloatingPointError("Non-finite loss before optimizer update")
        booster.backward(loss, optim)
        optim.step()
        optim.zero_grad()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        value = loss.detach().float().clone()
        dist.all_reduce(value)
        value = value.item() / world
        assert math.isfinite(value)
        record({"step": step + 1, "loss": value, "seconds": elapsed,
                "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20})
        if step == 0:
            masters = optimizer.param_groups[0]["params"]
            assert len(masters) == len(params) and all(p.device.type == "cpu" and p.dtype == torch.float32 for p in masters)
            moments = [optimizer.state[p][k] for p in masters for k in ("exp_avg", "exp_avg_sq")]
            assert all(t.device.type == "cpu" and t.dtype == torch.float32 for t in moments)
            changed = (target.detach().flatten()[:probe.numel()] != probe).sum().item()
            assert changed > 0
            record({"changed_probe_elements": changed, "master_tensors": len(masters), "cpu_fp32_moment_gib":
                    sum(t.numel() * t.element_size() for t in moments) / 2**30})
        if args.eval and ((step + 1) % args.eval_every == 0 or step + 1 == args.steps):
            evaluation(step + 1)
    if args.kernel_audit:
        record({"actual_kernel_launches": kernel_counts})
        if args.reference:
            assert kernel_counts == {"forward": 0, "backward": 0}
        else:
            assert min(kernel_counts.values()) > 0
    record({"completed_steps": args.steps})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
