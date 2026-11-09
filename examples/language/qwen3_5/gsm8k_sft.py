"""Answer-only GSM8K SFT data and an independent held-out evaluation."""
import re
from pathlib import Path

import pyarrow.parquet as pq
import torch


def examples(tokenizer, root, split, skip, count):
    path = next((Path(root) / "main").glob(f"{split}-*.parquet"))
    from itertools import islice

    batches = pq.ParquetFile(path).iter_batches(batch_size=128)
    rows = (row for chunk in batches for row in chunk.to_pylist())
    selected = []
    for row in islice(rows, skip, skip + count):
        question = row["question"]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question + "\nGive only the final numeric answer."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        answer = tokenizer.encode(gold, add_special_tokens=False) + [tokenizer.eos_token_id]
        selected.append((prefix, answer, gold, question))
    assert len(selected) == count
    return selected


def batch(rows, pad):
    length = max(len(ids) + len(answer) for ids, answer, _, _ in rows)
    tokens = torch.full((len(rows), length), pad, dtype=torch.long)
    labels = torch.full_like(tokens, -100)
    mask = torch.zeros_like(tokens)
    for i, (ids, answer, _, _) in enumerate(rows):
        end = len(ids) + len(answer)
        tokens[i, :end] = torch.tensor(ids + answer)
        labels[i, len(ids):end] = torch.tensor(answer)
        mask[i, :end] = 1
    return {"input_ids": tokens.cuda(), "labels": labels.cuda(), "attention_mask": mask.cuda()}


@torch.no_grad()
def evaluate(model, tokenizer, rows, generate):
    model.eval()
    nll, count = 0.0, 0
    for i in range(0, len(rows), 4):
        feed = batch(rows[i:i+4], tokenizer.pad_token_id)
        size = (feed["labels"] != -100).sum().item()
        nll += model(**feed).loss.item() * size
        count += size
    metrics = {"answer_nll": nll / count, "target_tokens": count}
    if generate:
        predictions = []
        for prompt, _, gold, question in rows:
            ids = torch.tensor([prompt], device="cuda")
            generated = []
            for _ in range(12):
                token = model(input_ids=ids, logits_to_keep=1).logits[:, -1].argmax(-1, keepdim=True)
                generated.append(token.item())
                text = tokenizer.decode(generated, skip_special_tokens=True).strip(" ")
                if generated[-1] == tokenizer.eos_token_id or "\n" in text:
                    break
                ids = torch.cat([ids, token], 1)
            line = text.strip().splitlines()
            number = re.fullmatch(r"[+-]?\d+(?:\.\d+)?", line[0].replace(",", "").strip()) if line else None
            predictions.append({"question": question, "gold": gold, "output": text,
                                "correct": bool(number and float(number.group()) == float(gold))})
        metrics["predictions"] = predictions
    model.train()
    return metrics
