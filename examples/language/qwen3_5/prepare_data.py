"""Prepare deterministic, document-disjoint FineWeb-Edu streams and GSM8K test questions."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer


def snapshot(hub, repo):
    root = hub / repo
    return root / "snapshots" / (root / "refs/main").read_text().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = snapshot(args.hub, "models--Qwen--Qwen3.5-9B")
    corpus = snapshot(args.hub, "datasets--HuggingFaceFW--fineweb-edu")
    gsm = snapshot(args.hub, "datasets--openai--gsm8k")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    shards = sorted((corpus / "sample/10BT").glob("*.parquet"))
    manifest = {"model": str(model), "gsm8k": str(gsm)}
    data = {}
    for name, shard, count in (("train", shards[0], 524288), ("valid", shards[1], 8192)):
        ids = []
        docs = 0
        for rows in pq.ParquetFile(shard).iter_batches(columns=["text"], batch_size=16):
            for row in rows.to_pylist():
                ids.extend(tokenizer.encode(row["text"], add_special_tokens=False))
                ids.append(tokenizer.eos_token_id)
                docs += 1
                if len(ids) >= count:
                    break
            if len(ids) >= count:
                break
        assert len(ids) >= count
        data[name] = torch.tensor(ids[:count], dtype=torch.int32)
        manifest[name] = {"shard": str(shard), "docs": docs, "tokens": count,
                          "sha256": hashlib.sha256(data[name].numpy().tobytes()).hexdigest()}
    rows = next(pq.ParquetFile(next((gsm / "main").glob("test-*.parquet"))).iter_batches(batch_size=16))
    data["gsm8k"] = [{"question": row["question"], "gold": row["answer"].split("####")[-1].strip()} for row in rows.to_pylist()]
    torch.save(data, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
