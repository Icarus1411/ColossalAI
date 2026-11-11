# Qwen3.5-9B CPU offload training and evaluation

Offline text-only full-parameter BF16 training with frozen vision tower, activation checkpointing and ZeRO-2 CPU FP32 AdamW master weights/states. Both the policy and native HF reference use the same ColossalAI optimizer/offload implementation; this compares model implementations, not two independent training frameworks.

Run from the repository root using the existing environment (no extra installs):

```bash
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 TRITON_F32_DEFAULT=ieee
RUN=/workspace/ColossalAI/logs/my-machine/my-fresh-run
mkdir -p "$RUN"
python examples/language/qwen3_5/prepare_data.py --hub /mnt/beegfs/ColossalAI/shared-cache/hub --out "$RUN/data.pt"
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --standalone --nproc-per-node=1 examples/language/qwen3_5/train_9b.py \
  --data "$RUN/data.pt" --out "$RUN/smoke" --steps 2 --length 128 --kernel-audit
```

GSM8K small-task convergence: use `--gsm8k-sft --steps 64 --batch 8 --lr 5e-6 --eval --eval-every 16`. Train on the first 512 training rows, validate on the next 128, and test on 128 test rows starting at zero-based index 16. Hashes include tokenized inputs and answers; question overlap is rejected. Dataset order is fixed and each training row is used once. Only numeric answer/EOS tokens contribute to loss. Thinking is disabled. Test exact numeric accuracy uses greedy decoding capped at 12 tokens, only before and after training. This is a small answer-only fine-tuning experiment, not a full GSM8K chain-of-thought leaderboard evaluation. Add `--reference` for the paired native HF model, with a new output directory.

Performance: use `--steps 10 --length 512 --batch 1`, with 1, 2 or 4 visible GPUs and matching `--nproc-per-node`. Discard the first two timed steps. Report per-step maximum elapsed time across ranks. GPU allocated/reserved peaks reset before each training step. CPU RSS is the process lifetime maximum. Do not enable kernel instrumentation for timing or overlap timing runs with other CPU offload experiments. With a fixed batch per rank this is weak scaling; CPU threads and global batch increase with world size. For a fixed global batch of 4 use batch 4/2/1 on world size 1/2/4.

Capacity: each fresh-process trial must finish two optimizer steps; record the largest passing value and the adjacent CUDA OOM at the chosen grid resolution. An arbitrary exception is not an OOM boundary.

`--kernel-audit` counts real forward/backward Triton launches without retaining tensor arguments. The policy must launch both kernels; the native reference must launch neither. Every run checks frozen vision/all-trainable text parameters, CPU FP32 optimizer tensors, a changed weight probe, finite loss before updates, and logs source SHA, tracked working-tree status, input hashes and arguments. Updated model weights are not saved; this entry point produces experiment evidence and metrics.
