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

from mm_stack.api import evaluate, ingest_image, reembed_all, search
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

    sub.add_parser("reembed-all", help="Re-embed stale entries")

    eval_cmd = sub.add_parser("evaluate", help="Run evaluation harness")
    eval_cmd.add_argument("--fixture", default="")
    eval_cmd.add_argument("--init-fixture", action="store_true")

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
    elif args.cmd == "reembed-all":
        out = reembed_all(cfg)
    elif args.cmd == "evaluate":
        if args.init_fixture:
            path = ensure_eval_fixture(cfg)
            out = {"fixture_initialized": str(path)}
        else:
            out = evaluate(cfg, fixture_path=(args.fixture or None))
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
