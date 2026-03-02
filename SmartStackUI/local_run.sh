#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTFIX_DIR="${SCRIPT_DIR}/.swiftfix"
BUILD_DIR="${SCRIPT_DIR}/.build-local"
SOURCE_DIR="${SCRIPT_DIR}/Sources/SmartStackUI"
BINARY_PATH="${BUILD_DIR}/SmartStackUI"
SWIFTLANG_STAMP="${SWIFTFIX_DIR}/swiftlang_token.txt"

# Bash 3.2 compat: avoid mapfile. Use standard array building.
SOURCE_FILES=()
while IFS= read -r file; do
    SOURCE_FILES+=("$file")
done < <(find "${SOURCE_DIR}" -name "*.swift")

# Use xcrun to find the active SDK path (more robust than hardcoding)
CLT_SDK="$(xcrun --show-sdk-path)"
LOCAL_SDK="${SWIFTFIX_DIR}/MacOSX.sdk"
EMPTY_MODULEMAP="${SWIFTFIX_DIR}/empty.modulemap"

build_only=0
clean_sdk=0
force_build=0
new_instance=0
kill_switch=1
kill_threshold_mb="${SMART_STACK_KILL_THRESHOLD_MB:-15872}"
kill_free_pct_threshold="${SMART_STACK_KILL_FREE_PCT:-8}"
kill_breach_count="${SMART_STACK_KILL_BREACH_COUNT:-3}"
kill_poll_sec="${SMART_STACK_KILL_POLL_SEC:-3}"
kill_python_count_threshold="${SMART_STACK_KILL_PY_COUNT_THRESHOLD:-6}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-only)
            build_only=1
            shift
            ;;
        --clean-sdk)
            clean_sdk=1
            shift
            ;;
        --force-build)
            force_build=1
            shift
            ;;
        --new-instance)
            new_instance=1
            shift
            ;;
        --no-killswitch)
            kill_switch=0
            shift
            ;;
        --kill-threshold-mb)
            [[ $# -lt 2 ]] && { echo "[ERROR] --kill-threshold-mb requires a value"; exit 2; }
            kill_threshold_mb="$2"
            shift 2
            ;;
        --kill-free-pct)
            [[ $# -lt 2 ]] && { echo "[ERROR] --kill-free-pct requires a value"; exit 2; }
            kill_free_pct_threshold="$2"
            shift 2
            ;;
        --kill-breach-count)
            [[ $# -lt 2 ]] && { echo "[ERROR] --kill-breach-count requires a value"; exit 2; }
            kill_breach_count="$2"
            shift 2
            ;;
        --kill-poll-sec)
            [[ $# -lt 2 ]] && { echo "[ERROR] --kill-poll-sec requires a value"; exit 2; }
            kill_poll_sec="$2"
            shift 2
            ;;
        --kill-python-count)
            [[ $# -lt 2 ]] && { echo "[ERROR] --kill-python-count requires a value"; exit 2; }
            kill_python_count_threshold="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: ./local_run.sh [--build-only] [--clean-sdk] [--force-build] [--new-instance] [--no-killswitch]"
            echo "                    [--kill-threshold-mb N] [--kill-free-pct N] [--kill-breach-count N] [--kill-poll-sec N] [--kill-python-count N]"
            exit 2
            ;;
    esac
done

is_int() {
    [[ "${1:-}" =~ ^[0-9]+$ ]]
}

for n in "$kill_threshold_mb" "$kill_free_pct_threshold" "$kill_breach_count" "$kill_poll_sec" "$kill_python_count_threshold"; do
    if ! is_int "$n"; then
        echo "[ERROR] Numeric kill-switch argument expected, got: '$n'"
        exit 2
    fi
done

if [[ ! -d "${CLT_SDK}" ]]; then
    echo "[ERROR] SDK not found at ${CLT_SDK}"
    exit 1
fi

mkdir -p "${SWIFTFIX_DIR}" "${BUILD_DIR}"

if [[ "${clean_sdk}" -eq 1 ]]; then
    rm -rf "${LOCAL_SDK}"
fi

if [[ ! -d "${LOCAL_SDK}" ]]; then
    echo "[INFO] Copying SDK into local writable cache..."
    cp -R "${CLT_SDK}" "${LOCAL_SDK}"
fi

swift_version_line="$(swiftc -version 2>&1 | tr '\n' ' ')"
# Extract swiftlang version token
swiftlang_token="$(printf '%s\n' "${swift_version_line}" | sed -n 's/.*\(swiftlang-[^ )]*\).*/\1/p')"

if [[ -z "${swiftlang_token}" ]]; then
    echo "[ERROR] Could not parse swiftlang token from: ${swift_version_line}"
    exit 1
fi

previous_swiftlang_token=""
if [[ -f "${SWIFTLANG_STAMP}" ]]; then
    previous_swiftlang_token="$(cat "${SWIFTLANG_STAMP}")"
fi

# Patch SDK if needed (SDK mismatch workaround)
if [[ "${clean_sdk}" -eq 1 || "${swiftlang_token}" != "${previous_swiftlang_token}" ]]; then
    echo "[INFO] Patching swiftinterface compiler stamp to ${swiftlang_token}..."
    swiftinterface_list="${SWIFTFIX_DIR}/swiftinterface_files.list"
    if rg -l 'swiftlang-[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' "${LOCAL_SDK}" --glob '*.swiftinterface' > "${swiftinterface_list}"; then
        while IFS= read -r swiftinterface_file; do
            sed -E -i '' "s/swiftlang-[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/${swiftlang_token}/g" "${swiftinterface_file}"
        done < "${swiftinterface_list}"
    fi
    rm -f "${swiftinterface_list}"
    printf '%s\n' "${swiftlang_token}" > "${SWIFTLANG_STAMP}"
fi

cat > "${EMPTY_MODULEMAP}" <<'EOF'
// intentionally empty
EOF

# VFS overlay to mask strict SDK checks if needed
VFS_OVERLAY="${SWIFTFIX_DIR}/vfs_overlay.yaml"
cat > "${VFS_OVERLAY}" <<EOF
{
  "version": 0,
  "roots": [
    {
      "name": "/Library/Developer/CommandLineTools/usr/include/swift/bridging.modulemap",
      "type": "file",
      "external-contents": "${EMPTY_MODULEMAP}"
    }
  ]
}
EOF

need_build=0
if [[ "${force_build}" -eq 1 ]]; then
    need_build=1
elif [[ ! -x "${BINARY_PATH}" ]]; then
    need_build=1
else
    # Check if any source file is newer than binary
    for src in "${SOURCE_FILES[@]}"; do
        if [[ "$src" -nt "${BINARY_PATH}" ]]; then
            need_build=1
            break
        fi
    done
fi

if [[ "${need_build}" -eq 1 ]]; then
    echo "[INFO] Building SmartStackUI via swiftc..."
    swiftc \
        -parse-as-library \
        -sdk "${LOCAL_SDK}" \
        -Xcc -ivfsoverlay -Xcc "${VFS_OVERLAY}" \
        "${SOURCE_FILES[@]}" \
        -o "${BINARY_PATH}"
    echo "[INFO] Built binary: ${BINARY_PATH}"
else
    echo "[INFO] Build skipped (up-to-date): ${BINARY_PATH}"
fi

if [[ "${build_only}" -eq 1 ]]; then
    exit 0
fi

# Prevent accidental multi-instance launches unless explicitly requested.
if [[ "${new_instance}" -eq 0 ]]; then
    existing_pid="$(pgrep -x SmartStackUI | head -n 1 || true)"
    if [[ -n "${existing_pid}" ]]; then
        echo "[INFO] SmartStackUI already running (pid ${existing_pid}). Reusing existing instance."
        osascript -e 'tell application "SmartStackUI" to activate' >/dev/null 2>&1 || true
        exit 0
    fi
fi

echo "[INFO] Launching SmartStackUI..."
"${BINARY_PATH}" &
app_pid=$!

smart_stack_root="$(cd "${SCRIPT_DIR}/.." && pwd)"
watchdog_pid=""

kill_descendants() {
    local parent_pid="$1"
    local children
    children="$(pgrep -P "${parent_pid}" || true)"
    if [[ -z "${children}" ]]; then
        return
    fi
    while IFS= read -r child_pid; do
        [[ -z "${child_pid}" ]] && continue
        kill_descendants "${child_pid}"
        kill -TERM "${child_pid}" 2>/dev/null || true
    done <<< "${children}"
}

kill_stack_processes() {
    local target_pid="$1"
    echo "[KILLSWITCH] Triggered. Stopping SmartStackUI (pid ${target_pid}) and Smart Stack subprocesses..."
    report_top_memory_hogs
    kill_descendants "${target_pid}"
    # Kill any duplicate UI instances too.
    pkill -x SmartStackUI 2>/dev/null || true
    kill -TERM "${target_pid}" 2>/dev/null || true
    sleep 1
    kill -KILL "${target_pid}" 2>/dev/null || true
    pkill -f "${smart_stack_root}/\\.venv/bin/python" 2>/dev/null || true
    pkill -f "${smart_stack_root}/(mm_cli\\.py|search\\.py|ingest\\.py|notes_index\\.py|run_guarded_ingest\\.sh)" 2>/dev/null || true
}

active_wired_mb() {
    local page_size active wired
    page_size="$(vm_stat | awk '/page size of/ {gsub("[^0-9]","",$8); print $8; exit}')"
    active="$(vm_stat | awk '/Pages active/ {gsub("\\.","",$3); print $3; exit}')"
    wired="$(vm_stat | awk '/Pages wired down/ {gsub("\\.","",$4); print $4; exit}')"
    awk -v ps="${page_size}" -v a="${active}" -v w="${wired}" 'BEGIN {printf "%.0f", ((a+w)*ps)/(1024*1024)}'
}

system_free_pct() {
    memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print int($2); exit}'
}

smartstack_instance_count() {
    pgrep -x SmartStackUI | wc -l | awk '{print $1}'
}

smartstack_python_count() {
    pgrep -f "${smart_stack_root}/\\.venv/bin/python" | wc -l | awk '{print $1}'
}

report_top_memory_hogs() {
    echo "[WATCHDOG] Top RAM hogs now:"
    ps -axo pid,%mem,rss,command | sort -k3 -nr | head -n 8 | awk '{
        pid=$1; pmem=$2; rss_mb=int($3/1024); cmd="";
        for(i=4;i<=NF;i++){cmd=cmd $i " "}
        printf "  pid=%s rss=%sMB mem=%s%% cmd=%s\n", pid, rss_mb, pmem, cmd
    }'
}

report_hogs_over_mb() {
    local limit_mb="$1"
    ps -axo pid,%mem,rss,command | awk -v lim="${limit_mb}" '
        NR==1 { next }
        {
            rss_mb=int($3/1024)
            if (rss_mb >= lim) {
                cmd=""
                for(i=4;i<=NF;i++){cmd=cmd $i " "}
                printf "  pid=%s rss=%sMB mem=%s%% cmd=%s\n", $1, rss_mb, $2, cmd
            }
        }
    ' | head -n 10
}

start_killswitch_watchdog() {
    local target_pid="$1"
    (
        local breaches=0
        while kill -0 "${target_pid}" 2>/dev/null; do
            local used_mb free_pct ss_count py_count
            used_mb="$(active_wired_mb)"
            free_pct="$(system_free_pct)"
            ss_count="$(smartstack_instance_count)"
            py_count="$(smartstack_python_count)"
            [[ -z "${free_pct}" ]] && free_pct=100

            if (( used_mb > kill_threshold_mb || free_pct <= kill_free_pct_threshold || ss_count > 1 || py_count > kill_python_count_threshold )); then
                breaches=$((breaches + 1))
                echo "[WATCHDOG] breach ${breaches}/${kill_breach_count} (used=${used_mb}MB, free=${free_pct}%, smartstack_instances=${ss_count}, smartstack_python=${py_count})"
            else
                breaches=0
            fi

            if (( breaches >= kill_breach_count )); then
                kill_stack_processes "${target_pid}"
                exit 137
            fi
            sleep "${kill_poll_sec}"
        done
    ) &
    watchdog_pid=$!
}

cleanup_watchdog() {
    if [[ -n "${watchdog_pid}" ]] && kill -0 "${watchdog_pid}" 2>/dev/null; then
        kill "${watchdog_pid}" 2>/dev/null || true
    fi
}

trap cleanup_watchdog EXIT INT TERM

if [[ "${kill_switch}" -eq 1 ]]; then
    echo "[INFO] Kill-switch watchdog ON (threshold=${kill_threshold_mb}MB, free<=${kill_free_pct_threshold}%, python_count>${kill_python_count_threshold}, breaches=${kill_breach_count}, poll=${kill_poll_sec}s)"
    echo "[INFO] Preflight RAM scan (processes >= 500MB):"
    report_hogs_over_mb 500
    start_killswitch_watchdog "${app_pid}"
else
    echo "[INFO] Kill-switch watchdog OFF"
fi

app_status=0
watchdog_status=0

# Avoid hanging shell sessions: actively observe the launched app PID.
while kill -0 "${app_pid}" 2>/dev/null; do
    sleep 1
done

wait "${app_pid}" || app_status=$?

if [[ -n "${watchdog_pid}" ]]; then
    if kill -0 "${watchdog_pid}" 2>/dev/null; then
        kill "${watchdog_pid}" 2>/dev/null || true
    else
        wait "${watchdog_pid}" || watchdog_status=$?
    fi
fi

if [[ "${watchdog_status}" -eq 137 ]]; then
    exit 137
fi

exit "${app_status}"
