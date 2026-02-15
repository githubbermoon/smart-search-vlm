#!/usr/bin/env bash
set -euo pipefail

STACK_ROOT="/Users/pranjal/garage/smart_stack"
PYTHON_BIN="${STACK_ROOT}/.venv/bin/python"
INGEST_SCRIPT="${STACK_ROOT}/ingest.py"

MEM_THRESHOLD_MB="${SMART_STACK_MEMORY_THRESHOLD_MB:-8704}"
MEM_GATE_MODE="${SMART_STACK_MEMORY_GATE_MODE:-wait}"
MEM_TIMEOUT_SEC="${SMART_STACK_MEMORY_TIMEOUT_SEC:-180}"
MEM_POLL_SEC="${SMART_STACK_MEMORY_POLL_SEC:-5}"
MEM_RELIEF_CMD="${SMART_STACK_MEMORY_RELIEF_CMD:-bash /Users/pranjal/clawdGIT/scripts/purge_and_run.sh --threshold-mb 8704 --relief-only}"

exec "${PYTHON_BIN}" "${INGEST_SCRIPT}" \
  --memory-threshold-mb "${MEM_THRESHOLD_MB}" \
  --memory-gate-mode "${MEM_GATE_MODE}" \
  --memory-timeout-sec "${MEM_TIMEOUT_SEC}" \
  --memory-poll-sec "${MEM_POLL_SEC}" \
  --memory-relief-cmd "${MEM_RELIEF_CMD}" \
  "$@"
