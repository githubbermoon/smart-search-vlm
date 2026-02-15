#!/Users/pranjal/garage/smart_stack/.venv/bin/python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

# Suppress warnings and configure logging to stderr
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from mm_stack.api import (
    evaluate, ingest_image, reembed_all, search, chat, explain, compare,
    ingest_path as api_ingest_path, rescan as api_rescan, rescan_watched as api_rescan_watched,
    watch_add, watch_remove, watch_toggle, watch_list,
    exclude_add, exclude_remove, exclude_list,
)
from mm_stack.config import StackConfig
from mm_stack.evaluation import ensure_eval_fixture
from mm_stack.ingestion import MultimodalIngestor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal Smart Stack CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest_one = sub.add_parser("ingest-image", help="Ingest one image")
    ingest_one.add_argument("path", help="Image file path")
    ingest_one.add_argument("--safe-reprocess", action="store_true")

    ingest_inbox = sub.add_parser("ingest-inbox", help="Ingest inbox directory")
    ingest_inbox.add_argument("--limit", type=int, default=0)
    ingest_inbox.add_argument("--safe-reprocess", action="store_true")

    search_cmd = sub.add_parser("search", help="Search multimodal indexes")
    search_cmd.add_argument("query", nargs="?", default="")
    search_cmd.add_argument("--image-path", default="")
    search_cmd.add_argument("-n", "--top-k", type=int, default=10)
    search_cmd.add_argument("--json", action="store_true")

    chat_cmd = sub.add_parser("chat", help="Chat with your images")
    chat_cmd.add_argument("query", help="Question about your images")
    chat_cmd.add_argument("-n", "--top-k", type=int, default=3)
    chat_cmd.add_argument("--json", action="store_true")

    explain_cmd = sub.add_parser("explain", help="Explain a query's intent and related concepts")
    explain_cmd.add_argument("query", help="Query to explain")

    compare_cmd = sub.add_parser("compare", help="Compare a query against indexed content")
    compare_cmd.add_argument("query", help="Query driving comparison")

    sub.add_parser("reembed-all", help="Re-embed stale entries")

    eval_cmd = sub.add_parser("evaluate", help="Run evaluation harness")
    eval_cmd.add_argument("--fixture", default="")
    eval_cmd.add_argument("--init-fixture", action="store_true")

    # ── Index-in-Place commands ──
    ingest_path_cmd = sub.add_parser("ingest-path", help="Ingest file or folder in-place (no copy)")
    ingest_path_cmd.add_argument("path", help="File or directory to ingest")
    ingest_path_cmd.add_argument("--safe-reprocess", action="store_true")

    sub.add_parser("rescan", help="Rescan indexed files for changes (inode/size/mtime)")
    sub.add_parser("rescan-all", help="Rescan all watched folders")

    # ── Watch commands ──
    watch_add_cmd = sub.add_parser("watch-add", help="Add a watched folder")
    watch_add_cmd.add_argument("path", help="Folder path")

    watch_rm_cmd = sub.add_parser("watch-remove", help="Remove a watched folder")
    watch_rm_cmd.add_argument("path", help="Folder path")

    watch_toggle_cmd = sub.add_parser("watch-toggle", help="Toggle a watched folder on/off")
    watch_toggle_cmd.add_argument("path", help="Folder path")

    sub.add_parser("watch-list", help="List watched folders")

    # ── Exclude commands ──
    excl_add_cmd = sub.add_parser("exclude-add", help="Add an exclusion pattern")
    excl_add_cmd.add_argument("pattern", help="Glob pattern or path")

    excl_rm_cmd = sub.add_parser("exclude-remove", help="Remove an exclusion pattern")
    excl_rm_cmd.add_argument("pattern", help="Glob pattern or path")

    sub.add_parser("exclude-list", help="List exclusion patterns")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StackConfig()

    if args.cmd == "ingest-image":
        out = ingest_image(args.path, safe_reprocess=args.safe_reprocess, cfg=cfg)
    elif args.cmd == "ingest-inbox":
        engine = MultimodalIngestor(cfg)
        out = engine.ingest_inbox(limit=max(0, args.limit), safe_reprocess=args.safe_reprocess)
    elif args.cmd == "search":
        query = args.query.strip()
        image_path = args.image_path.strip() or None
        if not query and not image_path:
            raise SystemExit("Provide query text or --image-path")
        out = search(query=query, image_path=image_path, top_k=max(1, args.top_k), cfg=cfg)
    elif args.cmd == "chat":
        # Use streaming for better UX in CLI, but buffer for JSON output if requested
        if args.json:
            out = chat(query=args.query, top_k=max(1, args.top_k), cfg=cfg)
        else:
            # Interactive Stream
            print(f"Thinking...", end="", flush=True)
            from mm_stack.api import stream_chat
            accumulated = ""
            for event in stream_chat(query=args.query, top_k=max(1, args.top_k), cfg=cfg):
                if event["type"] == "token":
                    # Clear "Thinking..." on first token logic could be added, but simple append is fine
                    if not accumulated:
                        print("\r", end="") # Clear line
                    print(event["content"], end="", flush=True)
                    accumulated += event["content"]
                elif event["type"] == "complete":
                    print("\n\n-- Sources --")
                    for s in event["sources"]:
                        print(f"[{s['score']:.2f}] {Path(s['file_path']).name}")
                    print(f"\nConfidence: {event['confidence']} (Grounded: {event['grounded_score']:.2f})")
            return
    elif args.cmd == "explain":
        from mm_stack.api import explain
        out = explain(args.query, cfg)
    elif args.cmd == "compare":
        from mm_stack.api import compare
        out = compare(args.query, cfg)
        import dataclasses
        if dataclasses.is_dataclass(out):
             out = dataclasses.asdict(out)
    elif args.cmd == "reembed-all":
        out = reembed_all(cfg)
    elif args.cmd == "evaluate":
        if args.init_fixture:
            path = ensure_eval_fixture(cfg)
            out = {"fixture_initialized": str(path)}
        else:
            out = evaluate(cfg, fixture_path=(args.fixture or None))
    elif args.cmd == "ingest-path":
        out = api_ingest_path(args.path, safe_reprocess=args.safe_reprocess, cfg=cfg)
    elif args.cmd == "rescan":
        out = api_rescan(cfg)
    elif args.cmd == "rescan-all":
        out = api_rescan_watched(cfg)
    elif args.cmd == "watch-add":
        out = watch_add(args.path, cfg)
    elif args.cmd == "watch-remove":
        out = watch_remove(args.path, cfg)
    elif args.cmd == "watch-toggle":
        out = watch_toggle(args.path, cfg)
    elif args.cmd == "watch-list":
        out = watch_list(cfg)
    elif args.cmd == "exclude-add":
        out = exclude_add(args.pattern, cfg)
    elif args.cmd == "exclude-remove":
        out = exclude_remove(args.pattern, cfg)
    elif args.cmd == "exclude-list":
        out = exclude_list(cfg)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
