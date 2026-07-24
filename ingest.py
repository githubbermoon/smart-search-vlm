#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

STACK_ROOT = Path(os.getenv("SMART_STACK_ROOT", Path(__file__).resolve().parent))
PYTHON_BIN = STACK_ROOT / ".venv" / "bin" / "python"
MM_CLI = STACK_ROOT / "mm_cli.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Compatibility wrapper: routes ingest.py to multimodal stack")
    parser.add_argument("--vlm-model", default="")
    parser.add_argument("--embed-model", default="")
    parser.add_argument("--safe-reprocess", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-print-fields", action="store_true")

    # Legacy memory-gate args are accepted for compatibility only.
    parser.add_argument("--memory-threshold-mb", type=int, default=0)
    parser.add_argument("--memory-gate-mode", choices=["wait", "skip", "fail"], default="wait")
    parser.add_argument("--memory-timeout-sec", type=int, default=180)
    parser.add_argument("--memory-poll-sec", type=float, default=5.0)
    parser.add_argument("--memory-relief-cmd", default="")

    return parser.parse_known_args()


def main() -> int:
    args, unknown = parse_args()
    if unknown:
        print(f"[WARN] Ignoring unsupported args: {' '.join(unknown)}", file=sys.stderr)

    if args.memory_threshold_mb > 0:
        print(
            "[WARN] Legacy per-image memory gate options are ignored in ingest.py wrapper. "
            "Use ./run_guarded_ingest.sh for one-time memory-guarded ingest.",
            file=sys.stderr,
        )

    env = dict(os.environ)
    if args.vlm_model:
        env["SMART_STACK_VLM_MODEL"] = args.vlm_model
    if args.embed_model:
        env["SMART_STACK_TEXT_MODEL"] = args.embed_model

    cmd = [str(PYTHON_BIN), str(MM_CLI), "ingest-inbox"]
    if args.safe_reprocess:
        cmd.append("--safe-reprocess")
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])

    proc = subprocess.run(cmd, cwd=str(STACK_ROOT), env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
