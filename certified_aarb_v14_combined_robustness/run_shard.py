"""Run one independent and resumable shard of the v14 audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from scipy.stats import qmc

from study import BANKS, STRESS_DIMENSION, evaluate_record


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="development-only records per bank for this shard")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise SystemExit("invalid shard index")

    path = OUT / f"records_shard_{args.shard}_of_{args.shards}.jsonl"
    if args.force and path.exists():
        path.unlink()
    completed = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                completed.add(json.loads(line)["id"])

    started = time.time()
    completed_now = 0
    with path.open("a") as stream:
        for bank in BANKS:
            unit = qmc.Sobol(
                d=STRESS_DIMENSION, scramble=True, seed=bank["seed"]
            ).random_base2(m=bank["power"])
            indices = list(range(args.shard, len(unit), args.shards))
            if args.limit is not None:
                indices = indices[:args.limit]
            total = len(indices)
            for position, index in enumerate(indices, 1):
                record_id = f"{bank['name']}-{index:04d}"
                if record_id in completed:
                    continue
                record = evaluate_record(
                    record_id, unit[index], bank["radius"], 21, 41
                )
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                completed.add(record_id)
                completed_now += 1
                if position % 8 == 0 or position == total:
                    print(
                        f"shard {args.shard}: {bank['name']} {position}/{total}; "
                        f"new={completed_now}; elapsed={time.time()-started:.1f}s",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
