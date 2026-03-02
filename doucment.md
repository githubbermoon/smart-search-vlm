# Smart Stack Emergency Kill Switch

## Purpose
The SmartStackUI now includes an emergency red kill switch to free RAM by terminating Smart Stack background workers (especially lingering `python3.11` model/daemon processes).

## Where It Is
- Main header: red circular power button.
- Expanded controls: `Emergency Kill` button.
- Menus: `Emergency Kill Switch` action.

## What It Kills
The kill switch targets Smart Stack processes, including:
- `.../smart_stack/.venv/bin/python`
- `mm_stack.text_embed_daemon`
- `mm_cli.py`
- `search.py`
- `ingest.py`
- `openclaw_imgsearch.py`
- `notes_index.py`
- `run_guarded_ingest.sh`
- daemon child processes (for example `multiprocessing.resource_tracker`)

## How It Works
1. Stops active search process from UI state.
2. Resolves daemon PID from:
   - `/tmp/smart_stack_text_embed_<uid>.sock.pid`
3. Kills daemon process tree first (children + parent).
4. Collects remaining Smart Stack worker PIDs and kills them (TERM then KILL).
5. Cleans daemon artifacts:
   - `/tmp/smart_stack_text_embed_<uid>.sock`
   - `/tmp/smart_stack_text_embed_<uid>.sock.pid`
6. Logs `killed=` and `remaining=` in UI logs.

## Notes
- This action is scoped to Smart Stack workers, not all Python processes on the machine.
- SmartStackUI app remains open; only worker processes are terminated.
- If a worker respawns, it is usually due to a new query/action triggering model usage again.

## Quick Verify Command
Run this in terminal to verify no Smart Stack worker remains:

```bash
ps -axo pid,ppid,rss,command | rg "python3\.11|smart_stack|mm_stack\.text_embed_daemon|resource_tracker"
```

