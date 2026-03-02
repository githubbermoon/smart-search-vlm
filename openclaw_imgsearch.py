#!/Users/pranjal/garage/smart_stack/.venv/bin/python3
"""OpenClaw-friendly wrapper for Smart Stack image search."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

STACK_ROOT = Path("/Users/pranjal/garage/smart_stack")
MM_CLI = STACK_ROOT / "mm_cli.py"
PYTHON = STACK_ROOT / ".venv" / "bin" / "python"


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
        str(MM_CLI),
        "search",
        query,
        "-n",
        str(max(1, args.top_k)),
        "--json",
    ]
    if args.embed_model:
        print(f"[WARN] --embed-model is ignored by multimodal CLI at query time ({args.embed_model}).")
    if args.with_notes:
        print("[WARN] --with-notes is ignored. Multimodal index is image-first.")

    proc = subprocess.run(cmd, cwd=str(STACK_ROOT), text=True, capture_output=True)
    output = "\n".join([proc.stdout or "", proc.stderr or ""]).strip()

    if proc.returncode != 0:
        print("Search failed.")
        if output:
            print(output)
        raise SystemExit(proc.returncode)

    data = json.loads(proc.stdout)
    rows = []
    for row in data.get("results", []):
        score = float(row.get("score", 0.0) or 0.0)
        if score < args.min_score:
            continue
        file_path = str(row.get("file_path", ""))
        rows.append(
            {
                "source": str(row.get("source", "image")),
                "filename": Path(file_path).name or "unknown",
                "score": f"{score:.4f}",
                "caption": str(row.get("caption", "")),
                "path": file_path,
            }
        )

    if not rows:
        print("No matches found.")
        return

    print(f"Query: {query}")
    print(f"Routing: {data.get('routing_mode', '')}")
    print(f"Results: {len(rows)}")
    for idx, row in enumerate(rows, start=1):
        source = str(row.get("source", "?"))
        filename = str(row.get("filename", "unknown"))
        score = str(row.get("score", "-"))
        caption = str(row.get("caption", "")).strip()
        path = str(row.get("path", ""))
        print(f"{idx}. [{source}] {filename} | score={score}")
        if caption:
            print(f"   caption: {caption}")
        if path:
            print(f"   path: {path}")


if __name__ == "__main__":
    main()
