#!/Users/pranjal/garage/smart_stack/.venv/bin/python3
"""OpenClaw-friendly wrapper for Smart Stack image search."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

STACK_ROOT = Path("/Users/pranjal/garage/smart_stack")
SEARCH = STACK_ROOT / "search.py"
PYTHON = STACK_ROOT / ".venv" / "bin" / "python"
MARKER = "@@SMARTSTACK_JSON@@"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenClaw wrapper for Smart Stack search")
    p.add_argument("query", nargs="+", help="Search query")
    p.add_argument("-n", "--top-k", type=int, default=5, help="Top results")
    p.add_argument("--min-score", type=float, default=0.0, help="Minimum similarity score")
    p.add_argument("--embed-model", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--with-notes", action="store_true", help="Include note vectors")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    query = " ".join(args.query).strip()

    cmd = [
        str(PYTHON),
        str(SEARCH),
        query,
        "--embed-model",
        args.embed_model,
        "-n",
        str(max(1, args.top_k)),
        "--min-score",
        str(args.min_score),
        "--json",
    ]
    if not args.with_notes:
        cmd.append("--no-notes")

    proc = subprocess.run(cmd, cwd=str(STACK_ROOT), text=True, capture_output=True)
    output = "\n".join([proc.stdout or "", proc.stderr or ""]).strip()

    if proc.returncode != 0:
        print("Search failed.")
        if output:
            print(output)
        raise SystemExit(proc.returncode)

    line = next((ln for ln in output.splitlines() if MARKER in ln), "")
    if not line:
        print("No JSON payload returned by search.")
        if output:
            print(output)
        raise SystemExit(2)

    payload = line.split(MARKER, 1)[1]
    data = json.loads(payload)
    rows = data.get("results", [])

    if not rows:
        print("No matches found.")
        return

    print(f"Query: {query}")
    print(f"Model: {data.get('embed_model', '')}")
    print(f"Results: {len(rows)}")
    for idx, row in enumerate(rows, start=1):
        source = str(row.get("source", "?"))
        filename = str(row.get("filename", "unknown"))
        score = str(row.get("score", "-"))
        caption = str(row.get("caption", "")).strip()
        path = str(row.get("obsidian_path", ""))
        print(f"{idx}. [{source}] {filename} | score={score}")
        if caption:
            print(f"   caption: {caption}")
        if path:
            print(f"   path: {path}")


if __name__ == "__main__":
    main()
