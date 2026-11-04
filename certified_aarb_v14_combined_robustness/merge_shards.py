"""Merge completed shards and compute gate counts from their records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from study import BANKS, summarize_bank


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="development only; never use for publication")
    args = parser.parse_args()

    records = []
    for shard in range(args.shards):
        path = OUT / f"records_shard_{shard}_of_{args.shards}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing shard: {path}")
        records.extend(
            json.loads(line) for line in path.read_text().splitlines()
            if line.strip()
        )
    records.sort(key=lambda record: record["id"])
    ids = [record["id"] for record in records]
    expected = sorted(
        f"{bank['name']}-{index:04d}"
        for bank in BANKS for index in range(2 ** bank["power"])
    )
    if not args.allow_incomplete and ids != expected:
        missing = sorted(set(expected) - set(ids))
        duplicates = len(ids) - len(set(ids))
        raise SystemExit(
            f"incomplete archive: missing={missing[:10]}, duplicates={duplicates}"
        )

    summaries = {
        bank["name"]: summarize_bank([
            record for record in records
            if record["id"].startswith(bank["name"] + "-")
        ])
        for bank in BANKS
    }
    status = "PASS" if (
        ids == expected and all(
            summary["finite_count"] == summary["sample_count"]
            and summary["error_count"] == 0
            and summary["support_success_count"] == summary["proposed_stable_count"]
            for summary in summaries.values()
        )
    ) else "REVIEW"
    save_json(OUT / "results.json", {
        "records": records,
        "summaries": summaries,
        "status": status,
    })
    save_json(OUT / "results_summary.json", {
        "status": status,
        "summaries": summaries,
    })
    print(json.dumps({"status": status, "summaries": summaries}, indent=2))
    if status != "PASS" and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
