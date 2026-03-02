#!/usr/bin/env bash
set -euo pipefail

STACK_ROOT="/Users/pranjal/garage/smart_stack"
PYTHON_BIN="${STACK_ROOT}/.venv/bin/python"
MM_CLI="${STACK_ROOT}/mm_cli.py"

MEM_THRESHOLD_MB="${SMART_STACK_MEMORY_THRESHOLD_MB:-8704}"
MEM_GATE_MODE="${SMART_STACK_MEMORY_GATE_MODE:-wait}"
MEM_TIMEOUT_SEC="${SMART_STACK_MEMORY_TIMEOUT_SEC:-180}"
MEM_POLL_SEC="${SMART_STACK_MEMORY_POLL_SEC:-5}"
MEM_RELIEF_CMD="${SMART_STACK_MEMORY_RELIEF_CMD:-bash /Users/pranjal/clawdGIT/scripts/purge_and_run.sh --threshold-mb 8704 --relief-only}"

get_used_mb() {
  vm_stat | awk '
    /page size of/ { gsub("\\.", "", $8); page_size=$8 }
    /Pages active:/ { gsub("\\.", "", $3); active=$3 }
    /Pages wired down:/ { gsub("\\.", "", $4); wired=$4 }
    END {
      if (page_size == 0) page_size = 4096
      used = (active + wired) * page_size / 1048576
      printf("%.0f\n", used)
    }
  '
}

memory_gate_once() {
  local threshold="$1"
  local mode="$2"
  local timeout_sec="$3"
  local poll_sec="$4"
  local relief_cmd="$5"
  local start_ts
  local used
  local relief_ran=0

  if [[ "${threshold}" -le 0 ]]; then
    return 0
  fi

  start_ts="$(date +%s)"

  while true; do
    used="$(get_used_mb)"
    if [[ "${used}" -le "${threshold}" ]]; then
      echo "[INFO] Memory guard: used=${used}MB threshold=${threshold}MB mode=${mode}"
      return 0
    fi

    if [[ "${relief_ran}" -eq 0 ]] && [[ -n "${relief_cmd}" ]]; then
      echo "[INFO] Memory guard high (${used}MB > ${threshold}MB). Running relief command once..."
      bash -lc "${relief_cmd}" || true
      relief_ran=1
      continue
    fi

    case "${mode}" in
      skip)
        echo "[WARN] Memory high (${used}MB). Skipping ingest run."
        return 10
        ;;
      fail)
        echo "[ERROR] Memory high (${used}MB > ${threshold}MB). Failing ingest run."
        return 2
        ;;
      wait)
        if (( "$(date +%s)" - start_ts >= timeout_sec )); then
          echo "[ERROR] Memory did not recover below ${threshold}MB within ${timeout_sec}s."
          return 3
        fi
        echo "[INFO] Memory high (${used}MB > ${threshold}MB). Waiting ${poll_sec}s..."
        sleep "${poll_sec}"
        ;;
      *)
        echo "[ERROR] Invalid SMART_STACK_MEMORY_GATE_MODE=${mode}. Use wait|skip|fail."
        return 4
        ;;
    esac
  done
}

SAFE_REPROCESS=0
LIMIT=0
IMAGE_BATCH_SIZE="${SMART_STACK_INGEST_IMAGE_BATCH_SIZE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --safe-reprocess)
      SAFE_REPROCESS=1
      shift
      ;;
    --limit)
      LIMIT="${2:-0}"
      shift 2
      ;;
    --image-batch-size)
      IMAGE_BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "[WARN] Ignoring unsupported legacy arg: $1"
      shift
      ;;
  esac
done

gate_rc=0
memory_gate_once "${MEM_THRESHOLD_MB}" "${MEM_GATE_MODE}" "${MEM_TIMEOUT_SEC}" "${MEM_POLL_SEC}" "${MEM_RELIEF_CMD}" || gate_rc=$?
if [[ "${gate_rc}" -eq 10 ]]; then
  exit 0
fi
if [[ "${gate_rc}" -ne 0 ]]; then
  exit "${gate_rc}"
fi

args=("${PYTHON_BIN}" "${MM_CLI}" "ingest-inbox")
if [[ "${SAFE_REPROCESS}" -eq 1 ]]; then
  args+=("--safe-reprocess")
fi
if [[ "${LIMIT}" != "0" ]]; then
  args+=("--limit" "${LIMIT}")
fi
if [[ -n "${IMAGE_BATCH_SIZE}" ]] && [[ "${IMAGE_BATCH_SIZE}" != "0" ]]; then
  args+=("--image-batch-size" "${IMAGE_BATCH_SIZE}")
fi

exec "${args[@]}"
