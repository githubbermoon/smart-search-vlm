#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
import warnings
from pathlib import Path

# Suppress warnings and configure logging to stderr
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from mm_stack.api import (
    evaluate, reembed_all, search, chat, explain, compare,
    context_lens,
    timeline,
    photos_list,
    cluster_recalc, cluster_label, cluster_list, cluster_items,
    watch_live as api_watch_live,
    watch_add, watch_remove, watch_toggle, watch_list,
    exclude_add, exclude_remove, exclude_list,
)
from mm_stack.config import StackConfig
from mm_stack.evaluation import ensure_eval_fixture
from mm_stack.ingest_telemetry import IngestTelemetry, TelemetryOptions
from mm_stack.ingestion import MultimodalIngestor


def _add_ingest_telemetry_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Emit detailed ingest stage counters to stderr as JSON lines",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Emit progress event every N processed items (default: 10)",
    )
    parser.add_argument(
        "--webhook-url",
        default="",
        help="Optional webhook URL for ingest telemetry events (POST JSON)",
    )
    parser.add_argument(
        "--webhook-timeout-sec",
        type=float,
        default=2.0,
        help="Webhook timeout in seconds (default: 2.0)",
    )


def _new_ingestor(
    *,
    cfg: StackConfig,
    args: argparse.Namespace,
    command_name: str,
    image_batch_size: int | None,
) -> tuple[MultimodalIngestor, IngestTelemetry]:
    telemetry = IngestTelemetry(
        run_id=str(uuid.uuid4()),
        options=TelemetryOptions(
            command=command_name,
            emit_to_stderr=bool(getattr(args, "progress", False)),
            webhook_url=str(getattr(args, "webhook_url", "") or "").strip(),
            webhook_timeout_sec=float(getattr(args, "webhook_timeout_sec", 2.0)),
        ),
    )
    ingestor = MultimodalIngestor(
        cfg,
        image_batch_size=image_batch_size,
        telemetry=telemetry,
        progress_every=max(1, int(getattr(args, "progress_every", 10))),
    )
    telemetry.emit(
        "cli_command_started",
        image_batch_size=(image_batch_size or int(cfg.ingest_image_batch_size)),
    )
    return ingestor, telemetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal Smart Stack CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest_one = sub.add_parser("ingest-image", help="Ingest one image")
    ingest_one.add_argument("path", help="Image file path")
    ingest_one.add_argument("--safe-reprocess", action="store_true")
    ingest_one.add_argument(
        "--image-batch-size",
        type=int,
        default=0,
        help="Max image candidates per ingest batch (0=default from config/env)",
    )
    _add_ingest_telemetry_flags(ingest_one)

    ingest_inbox = sub.add_parser("ingest-inbox", help="Ingest inbox directory")
    ingest_inbox.add_argument("--limit", type=int, default=0)
    ingest_inbox.add_argument("--safe-reprocess", action="store_true")
    ingest_inbox.add_argument(
        "--image-batch-size",
        type=int,
        default=0,
        help="Max image candidates per ingest batch (0=default from config/env)",
    )
    _add_ingest_telemetry_flags(ingest_inbox)

    search_cmd = sub.add_parser("search", help="Search multimodal indexes")
    search_cmd.add_argument("query", nargs="?", default="")
    search_cmd.add_argument("--image-path", default="")
    search_cmd.add_argument("-n", "--top-k", type=int, default=10)
    search_cmd.add_argument("--mode", choices=["auto", "keyword", "semantic"], default="auto")
    search_cmd.add_argument(
        "--auto-strategy",
        choices=["legacy", "hybrid"],
        default="hybrid",
        help="Auto-mode retrieval strategy",
    )
    search_cmd.add_argument("--verify", action="store_true", help="Enable VLM verification (off by default)")
    search_cmd.add_argument(
        "--semantic-fallback-threshold",
        type=int,
        default=0,
        help="In auto mode, run semantic fallback when keyword hits <= threshold",
    )
    search_cmd.add_argument("--json", action="store_true")
    search_cmd.add_argument("--intent-debug", action="store_true", help="Include parsed query intent in output")

    chat_cmd = sub.add_parser("chat", help="Chat with your images")
    chat_cmd.add_argument("query", help="Question about your images")
    chat_cmd.add_argument("-n", "--top-k", type=int, default=3)
    chat_cmd.add_argument("--image-id", default="", help="Optional pinned image id for focused chat")
    chat_cmd.add_argument("--file-path", default="", help="Optional pinned image file path for focused chat")
    chat_cmd.add_argument(
        "--history-json",
        default="",
        help="Optional JSON array chat history: [{\"role\":\"user|assistant\",\"content\":\"...\"}]",
    )
    chat_cmd.add_argument(
        "--history-file",
        default="",
        help="Optional path to JSON file containing chat history array",
    )
    chat_cmd.add_argument("--json", action="store_true")

    explain_cmd = sub.add_parser("explain", help="Explain a query's intent and related concepts")
    explain_cmd.add_argument("query", help="Query to explain")

    compare_cmd = sub.add_parser("compare", help="Compare a query against indexed content")
    compare_cmd.add_argument("query", help="Query driving comparison")

    context_lens_cmd = sub.add_parser("context-lens", help="Show contextual neighbors for one indexed image")
    context_lens_cmd.add_argument("--image-id", default="", help="Indexed image id")
    context_lens_cmd.add_argument("--file-path", default="", help="Indexed file path")
    context_lens_cmd.add_argument("-n", "--top-k", type=int, default=8)

    timeline_cmd = sub.add_parser("timeline", help="Aggregate indexed items by time buckets")
    timeline_cmd.add_argument("--granularity", default="month", choices=["year", "month", "day"])
    timeline_cmd.add_argument("--query", default="", help="Optional text filter over metadata")
    timeline_cmd.add_argument("--limit", type=int, default=240, help="Max buckets returned")

    photos_list_cmd = sub.add_parser("photos-list", help="List indexed photos")
    photos_list_cmd.add_argument("--limit", type=int, default=500, help="Max photos returned")
    photos_list_cmd.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    photos_list_cmd.add_argument(
        "--exclude-missing",
        action="store_true",
        help="Exclude files that are missing on disk",
    )
    photos_list_cmd.add_argument(
        "--check-exists",
        action="store_true",
        help="Stat each path on disk to set exists_on_disk (slower on large/remote folders)",
    )

    sub.add_parser("reembed-all", help="Re-embed stale entries")

    eval_cmd = sub.add_parser("evaluate", help="Run evaluation harness")
    eval_cmd.add_argument("--fixture", default="")
    eval_cmd.add_argument("--init-fixture", action="store_true")

    # ── Index-in-Place commands ──
    ingest_path_cmd = sub.add_parser("ingest-path", help="Ingest file or folder in-place (no copy)")
    ingest_path_cmd.add_argument("path", help="File or directory to ingest")
    ingest_path_cmd.add_argument("--safe-reprocess", action="store_true")
    ingest_path_cmd.add_argument(
        "--image-batch-size",
        type=int,
        default=0,
        help="Max image candidates per ingest batch (0=default from config/env)",
    )
    _add_ingest_telemetry_flags(ingest_path_cmd)

    rescan_cmd = sub.add_parser("rescan", help="Rescan indexed files for changes (inode/size/mtime)")
    rescan_cmd.add_argument(
        "--image-batch-size",
        type=int,
        default=0,
        help="Max image candidates per ingest batch (0=default from config/env)",
    )
    _add_ingest_telemetry_flags(rescan_cmd)
    rescan_all_cmd = sub.add_parser("rescan-all", help="Rescan all watched folders")
    rescan_all_cmd.add_argument(
        "--image-batch-size",
        type=int,
        default=0,
        help="Max image candidates per ingest batch (0=default from config/env)",
    )
    _add_ingest_telemetry_flags(rescan_all_cmd)
    watch_live_cmd = sub.add_parser("watch-live", help="Run low-RAM realtime path watcher")
    watch_live_cmd.add_argument(
        "--hourly-refresh-min",
        type=int,
        default=60,
        help="Minutes between lightweight watcher refresh passes (default: 60)",
    )
    watch_live_cmd.add_argument(
        "--debounce-ms",
        type=int,
        default=1200,
        help="Debounce window for create events in milliseconds (default: 1200)",
    )
    watch_live_cmd.add_argument(
        "--move-grace-sec",
        type=float,
        default=5.0,
        help="Grace period before marking deletes as stale (default: 5.0)",
    )
    watch_live_cmd.add_argument(
        "--initial-refresh",
        action="store_true",
        help="Run one lightweight refresh at startup",
    )

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
    
    # ── Memory Cluster commands ──
    cluster_parser = sub.add_parser("cluster", help="Manage memory clusters")
    cluster_parser.add_argument("--auto", action="store_true", help="Recalculate and auto-label clusters")
    cluster_parser.add_argument("--recalc", action="store_true", help="Recalculate clusters (K-Means)")
    cluster_parser.add_argument("--label", action="store_true", help="Auto-label pending clusters (VLM)")
    cluster_parser.add_argument("--list", action="store_true", help="List clusters")
    cluster_parser.add_argument("--items", default="", help="List items for one cluster_id")
    cluster_parser.add_argument("--n-clusters", type=int, default=20, help="Number of clusters (default: 20)")
    cluster_parser.add_argument("--limit", type=int, default=120, help="Limit for list/items")
    cluster_parser.add_argument("--min-items", type=int, default=1, help="Min items per cluster for list")

    # ── Collection commands ──
    coll_create = sub.add_parser("collection-create", help="Create a smart collection")
    coll_create.add_argument("name", help="Name of collection")
    coll_create.add_argument("query", help="Query string (e.g. 'tag:invoice')")

    sub.add_parser("collection-list", help="List all collections")

    coll_eval = sub.add_parser("collection-eval", help="Evaluate a collection")
    coll_eval.add_argument("name", help="Name of collection")

    coll_del = sub.add_parser("collection-delete", help="Delete a collection")
    coll_del.add_argument("name", help="Name of collection")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StackConfig()
    batch_size = max(1, int(args.image_batch_size)) if hasattr(args, "image_batch_size") and int(args.image_batch_size) > 0 else None

    if args.cmd == "ingest-image":
        ingestor, telemetry = _new_ingestor(
            cfg=cfg,
            args=args,
            command_name=args.cmd,
            image_batch_size=batch_size,
        )
        try:
            out = ingestor.ingest_image(args.path, safe_reprocess=args.safe_reprocess)
            telemetry.emit("cli_command_completed", **out)
        finally:
            telemetry.close()
    elif args.cmd == "ingest-inbox":
        ingestor, telemetry = _new_ingestor(
            cfg=cfg,
            args=args,
            command_name=args.cmd,
            image_batch_size=batch_size,
        )
        try:
            out = ingestor.ingest_inbox(limit=max(0, args.limit), safe_reprocess=args.safe_reprocess)
            telemetry.emit("cli_command_completed", **out)
        finally:
            telemetry.close()
    elif args.cmd == "search":
        query = args.query.strip()
        image_path = args.image_path.strip() or None
        if not query and not image_path:
            raise SystemExit("Provide query text or --image-path")
        out = search(
            query=query,
            image_path=image_path,
            top_k=max(1, args.top_k),
            mode=args.mode,
            auto_strategy=args.auto_strategy,
            verify=bool(args.verify),
            semantic_fallback_threshold=max(0, int(args.semantic_fallback_threshold)),
            cfg=cfg,
        )
        if args.intent_debug and query:
            from mm_stack.query_planner import parse_query
            out["query_intent_debug"] = parse_query(query).to_dict()
    elif args.cmd == "chat":
        attached_image_id = args.image_id.strip() or None
        attached_file_path = args.file_path.strip() or None
        history: list[dict[str, str]] | None = None
        history_blob = ""
        if args.history_file.strip():
            try:
                history_blob = Path(args.history_file).read_text(encoding="utf-8")
            except Exception as exc:
                raise SystemExit(f"Invalid --history-file: {exc}")
        elif args.history_json.strip():
            history_blob = args.history_json

        if history_blob.strip():
            try:
                parsed = json.loads(history_blob)
                if isinstance(parsed, list):
                    history = parsed
                else:
                    raise ValueError("history_json must decode to a JSON list")
            except Exception as exc:
                raise SystemExit(f"Invalid --history-json: {exc}")

        # Use streaming for better UX in CLI, but buffer for JSON output if requested
        if args.json:
            out = chat(
                query=args.query,
                top_k=max(1, args.top_k),
                cfg=cfg,
                attached_image_id=attached_image_id,
                attached_file_path=attached_file_path,
                history=history,
            )
        else:
            # Interactive Stream
            print(f"Thinking...", end="", flush=True)
            from mm_stack.api import stream_chat
            accumulated = ""
            for event in stream_chat(
                query=args.query,
                top_k=max(1, args.top_k),
                cfg=cfg,
                attached_image_id=attached_image_id,
                attached_file_path=attached_file_path,
                history=history,
            ):
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
                    if event.get("timings"):
                        print(f"Timings: {event['timings']}")
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
    elif args.cmd == "context-lens":
        image_id = args.image_id.strip() or None
        file_path = args.file_path.strip() or None
        if not image_id and not file_path:
            raise SystemExit("Provide --image-id or --file-path")
        out = context_lens(image_id=image_id, file_path=file_path, top_k=max(1, args.top_k), cfg=cfg)
    elif args.cmd == "timeline":
        out = timeline(
            granularity=args.granularity,
            query=(args.query.strip() or None),
            limit=max(1, args.limit),
            cfg=cfg,
        )
    elif args.cmd == "photos-list":
        out = photos_list(
            limit=max(1, int(args.limit)),
            offset=max(0, int(args.offset)),
            include_missing=not bool(args.exclude_missing),
            check_paths=bool(args.check_exists),
            cfg=cfg,
        )
    elif args.cmd == "reembed-all":
        out = reembed_all(cfg)
    elif args.cmd == "evaluate":
        if args.init_fixture:
            path = ensure_eval_fixture(cfg)
            out = {"fixture_initialized": str(path)}
        else:
            out = evaluate(cfg, fixture_path=(args.fixture or None))
    elif args.cmd == "ingest-path":
        ingestor, telemetry = _new_ingestor(
            cfg=cfg,
            args=args,
            command_name=args.cmd,
            image_batch_size=batch_size,
        )
        try:
            out = ingestor.ingest_path(args.path, safe_reprocess=args.safe_reprocess)
            telemetry.emit("cli_command_completed", **out)
        finally:
            telemetry.close()
    elif args.cmd == "rescan":
        ingestor, telemetry = _new_ingestor(
            cfg=cfg,
            args=args,
            command_name=args.cmd,
            image_batch_size=batch_size,
        )
        try:
            out = ingestor.rescan_stale()
            telemetry.emit("cli_command_completed", **out)
        finally:
            telemetry.close()
    elif args.cmd == "rescan-all":
        ingestor, telemetry = _new_ingestor(
            cfg=cfg,
            args=args,
            command_name=args.cmd,
            image_batch_size=batch_size,
        )
        try:
            out = ingestor.rescan_watched()
            telemetry.emit("cli_command_completed", **out)
        finally:
            telemetry.close()
    elif args.cmd == "watch-live":
        out = api_watch_live(
            hourly_refresh_min=max(1, int(args.hourly_refresh_min)),
            debounce_ms=max(50, int(args.debounce_ms)),
            move_grace_sec=max(0.0, float(args.move_grace_sec)),
            initial_refresh=bool(args.initial_refresh),
            cfg=cfg,
        )
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
    elif args.cmd == "cluster":
        if args.auto:
            out = cluster_recalc(
                n_clusters=max(1, int(args.n_clusters)),
                auto_label=True,
                cfg=cfg,
            )
        elif args.list:
            clusters = cluster_list(
                limit=max(1, int(args.limit)),
                min_items=max(0, int(args.min_items)),
                cfg=cfg,
            )
            out = {"count": len(clusters), "clusters": clusters}
        elif args.items.strip():
            items = cluster_items(
                args.items.strip(),
                limit=max(1, int(args.limit)),
                cfg=cfg,
            )
            out = {
                "cluster_id": args.items.strip(),
                "count": len(items),
                "items": items,
            }
        else:
            out = {}
            if args.recalc:
                out.update(
                    cluster_recalc(
                        n_clusters=max(1, int(args.n_clusters)),
                        auto_label=False,
                        cfg=cfg,
                    )
                )
            if args.label:
                out.update(cluster_label(cfg))
            if not args.recalc and not args.label:
                out = {"msg": "Specify --auto, --list, --items, --recalc, or --label"}
    elif args.cmd == "collection-create":
        from mm_stack.collections import CollectionManager
        cm = CollectionManager(cfg)
        cid = cm.create_collection(args.name, args.query)
        out = {"created": cid, "name": args.name}
    elif args.cmd == "collection-list":
        from mm_stack.collections import CollectionManager
        cm = CollectionManager(cfg)
        out = cm.list_collections()
    elif args.cmd == "collection-eval":
        from mm_stack.collections import CollectionManager
        cm = CollectionManager(cfg)
        out = cm.evaluate_collection(args.name)
    elif args.cmd == "collection-delete":
        from mm_stack.collections import CollectionManager
        cm = CollectionManager(cfg)
        deleted = cm.delete_collection(args.name)
        out = {"deleted": deleted, "name": args.name}
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
