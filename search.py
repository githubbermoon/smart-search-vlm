#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

STACK_ROOT = Path(os.getenv("SMART_STACK_ROOT", Path(__file__).resolve().parent))
PYTHON_BIN = STACK_ROOT / ".venv" / "bin" / "python"
MM_CLI = STACK_ROOT / "mm_cli.py"
MARKER = "@@SMARTSTACK_JSON@@"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility wrapper: routes search.py to multimodal stack")
    parser.add_argument("query", nargs="?", default="", help="Search query")
    parser.add_argument("-n", "--top-k", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--embed-model", default="")
    parser.add_argument("--no-notes", action="store_true")
    parser.add_argument("--with-notes", action="store_true")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--open-app", default="")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[WARN] Ignoring unsupported args: {' '.join(unknown)}", file=sys.stderr)
    return args


def _normalize_results(mm_payload: dict, min_score: float) -> list[dict]:
    rows: list[dict] = []
    for item in mm_payload.get("results", []):
        try:
            score = float(item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score < min_score:
            continue

        path = str(item.get("file_path", ""))
        rows.append(
            {
                "source": "image",
                "filename": Path(path).name or "unknown",
                "caption": str(item.get("caption", "") or ""),
                "tags": item.get("tags", []) if isinstance(item.get("tags", []), list) else [],
                "score": f"{score:.4f}",
                "obsidian_path": path,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    query = args.query.strip()
    if not query:
        print("Query is required.", file=sys.stderr)
        return 2

    cmd = [
        str(PYTHON_BIN),
        str(MM_CLI),
        "search",
        query,
        "-n",
        str(max(1, args.top_k)),
        "--json",
    ]

    proc = subprocess.run(cmd, cwd=str(STACK_ROOT), text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        return proc.returncode

    try:
        mm_payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"Failed to decode multimodal JSON output: {exc}", file=sys.stderr)
        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
        return 3

    payload = {
        "query": query,
        "embed_model": args.embed_model or "multimodal-default",
        "top_k": max(1, args.top_k),
        "min_score": float(args.min_score),
        "results": _normalize_results(mm_payload, float(args.min_score)),
    }

    if args.open:
        print("[WARN] --open is not supported by multimodal wrapper; use SmartStackUI or mm_cli directly.", file=sys.stderr)

    if args.json:
        print(f"{MARKER}{json.dumps(payload, ensure_ascii=False)}")
        return 0

    print(f"Query: {query}")
    print(f"Results: {len(payload['results'])}")
    for i, row in enumerate(payload["results"], start=1):
        print(f"{i}. {row['filename']} | score={row['score']}")
        if row["caption"]:
            print(f"   caption: {row['caption']}")
        if row["obsidian_path"]:
            print(f"   path: {row['obsidian_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
