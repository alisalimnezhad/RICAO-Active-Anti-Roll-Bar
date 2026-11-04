"""One-click reproduction driver for the complete v14 combined audit.

Open this file in VS Code/PyCharm and press Run.  Existing shard files are
resumed automatically.  Use ``python run_all.py --force`` for a clean rerun.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def command(arguments):
    return [sys.executable, *map(str, arguments)]


def run(arguments, env):
    cmd = command(arguments)
    print("\nRUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="discard shard checkpoints and recompute all records")
    parser.add_argument("--sequential", action="store_true",
                        help="run the two shards sequentially instead of concurrently")
    args = parser.parse_args()
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    shard_commands = []
    for shard in (0, 1):
        arguments = [HERE / "run_shard.py", "--shard", shard, "--shards", 2]
        if args.force:
            arguments.append("--force")
        shard_commands.append(command(arguments))

    if args.sequential:
        for cmd in shard_commands:
            print("\nRUN:", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    else:
        print("\nStarting two independent solver shards...", flush=True)
        processes = [
            subprocess.Popen(cmd, cwd=ROOT, env=env) for cmd in shard_commands
        ]
        return_codes = [process.wait() for process in processes]
        if any(code != 0 for code in return_codes):
            raise SystemExit(f"shard failure: return codes {return_codes}")

    run([HERE / "merge_shards.py", "--shards", 2], env)
    run([HERE / "refine_critical.py"], env)
    run([HERE / "independent_verify.py"], env)
    run([HERE / "make_artifacts.py"], env)
    run([HERE / "finalize_archive.py"], env)
    run([HERE / "test_robustness.py"], env)
    release_audit = ROOT / "certified_aarb_v14_release" / "audit_release.py"
    if release_audit.exists():
        run([release_audit], env)
    print("\nV14 COMBINED AUDIT: COMPLETE", flush=True)


if __name__ == "__main__":
    main()