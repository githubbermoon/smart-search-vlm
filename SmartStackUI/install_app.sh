#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY_PATH="${SCRIPT_DIR}/.build-local/SmartStackUI"
APP_NAME="SmartStackUI.app"
DEFAULT_DEST="${HOME}/Applications/${APP_NAME}"
DEST_PATH="${DEFAULT_DEST}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)
            [[ $# -lt 2 ]] && { echo "[ERROR] --dest requires a value"; exit 2; }
            DEST_PATH="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: ./install_app.sh [--dest /absolute/path/SmartStackUI.app]"
            exit 2
            ;;
    esac
done

if [[ "${DEST_PATH}" != *.app ]]; then
    DEST_PATH="${DEST_PATH}/${APP_NAME}"
fi

mkdir -p "$(dirname "${DEST_PATH}")"

echo "[INFO] Ensuring latest binary exists..."
"${SCRIPT_DIR}/local_run.sh" --build-only

if [[ ! -x "${BINARY_PATH}" ]]; then
    echo "[ERROR] Binary missing at ${BINARY_PATH}"
    exit 1
fi

echo "[INFO] Installing app bundle at ${DEST_PATH}"
rm -rf "${DEST_PATH}"
mkdir -p "${DEST_PATH}/Contents/MacOS" "${DEST_PATH}/Contents/Resources"

cat > "${DEST_PATH}/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>SmartStackUI</string>
    <key>CFBundleIdentifier</key>
    <string>local.pranjal.smartstackui</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>SmartStackUI</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundleVersion</key>
    <string>3</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

cp "${BINARY_PATH}" "${DEST_PATH}/Contents/Resources/SmartStackUI.bin"
chmod +x "${DEST_PATH}/Contents/Resources/SmartStackUI.bin"

cat > "${DEST_PATH}/Contents/MacOS/SmartStackUI" <<EOF
#!/usr/bin/env bash
set -euo pipefail

APP_EXEC_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="\$(cd "\${APP_EXEC_DIR}/.." && pwd)"
BUNDLED_BIN="\${APP_ROOT}/Resources/SmartStackUI.bin"
DEV_BIN="${BINARY_PATH}"
STACK_ROOT="${STACK_ROOT}"

KILL_SWITCH="\${SMART_STACK_KILL_SWITCH:-1}"
KILL_THRESHOLD_MB="\${SMART_STACK_KILL_THRESHOLD_MB:-15872}"
KILL_FREE_PCT="\${SMART_STACK_KILL_FREE_PCT:-8}"
KILL_BREACH_COUNT="\${SMART_STACK_KILL_BREACH_COUNT:-3}"
KILL_POLL_SEC="\${SMART_STACK_KILL_POLL_SEC:-3}"
KILL_PY_COUNT_THRESHOLD="\${SMART_STACK_KILL_PY_COUNT_THRESHOLD:-6}"

if [[ -x "\${DEV_BIN}" ]] && ([[ ! -x "\${BUNDLED_BIN}" ]] || [[ "\${DEV_BIN}" -nt "\${BUNDLED_BIN}" ]]); then
    cp "\${DEV_BIN}" "\${BUNDLED_BIN}"
    chmod +x "\${BUNDLED_BIN}"
fi

if [[ ! -x "\${BUNDLED_BIN}" ]]; then
    echo "[LAUNCHER] Missing bundled binary at \${BUNDLED_BIN}" >&2
    exit 1
fi

existing_pid="\$(pgrep -f "\${BUNDLED_BIN}" | head -n 1 || true)"
if [[ -n "\${existing_pid}" ]]; then
    osascript -e 'tell application "SmartStackUI" to activate' >/dev/null 2>&1 || true
    exit 0
fi

kill_descendants() {
    local parent_pid="\$1"
    local children
    children="\$(pgrep -P "\${parent_pid}" || true)"
    if [[ -z "\${children}" ]]; then
        return
    fi
    while IFS= read -r child_pid; do
        [[ -z "\${child_pid}" ]] && continue
        kill_descendants "\${child_pid}"
        kill -TERM "\${child_pid}" 2>/dev/null || true
    done <<< "\${children}"
}

active_wired_mb() {
    local page_size active wired
    page_size="\$(vm_stat | awk '/page size of/ {gsub("[^0-9]","",\$8); print \$8; exit}')"
    active="\$(vm_stat | awk '/Pages active/ {gsub("\\\\.","",\$3); print \$3; exit}')"
    wired="\$(vm_stat | awk '/Pages wired down/ {gsub("\\\\.","",\$4); print \$4; exit}')"
    awk -v ps="\${page_size}" -v a="\${active}" -v w="\${wired}" 'BEGIN {printf "%.0f", ((a+w)*ps)/(1024*1024)}'
}

system_free_pct() {
    memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",\$2); print int(\$2); exit}'
}

smartstack_python_count() {
    pgrep -f "\${STACK_ROOT}/\\.venv/bin/python" | wc -l | awk '{print \$1}'
}

report_top_memory_hogs() {
    ps -axo pid,%mem,rss,command | sort -k3 -nr | head -n 8 | awk '{
        pid=\$1; pmem=\$2; rss_mb=int(\$3/1024); cmd="";
        for(i=4;i<=NF;i++){cmd=cmd \$i " "}
        printf "[WATCHDOG] pid=%s rss=%sMB mem=%s%% cmd=%s\\n", pid, rss_mb, pmem, cmd
    }'
}

kill_stack_processes() {
    local target_pid="\$1"
    echo "[KILLSWITCH] Triggered. Stopping SmartStackUI and Smart Stack subprocesses..." >&2
    report_top_memory_hogs >&2
    kill_descendants "\${target_pid}"
    pkill -f "\${BUNDLED_BIN}" 2>/dev/null || true
    kill -TERM "\${target_pid}" 2>/dev/null || true
    sleep 1
    kill -KILL "\${target_pid}" 2>/dev/null || true
    pkill -f "\${STACK_ROOT}/\\.venv/bin/python" 2>/dev/null || true
    pkill -f "\${STACK_ROOT}/(mm_cli\\.py|search\\.py|ingest\\.py|notes_index\\.py|run_guarded_ingest\\.sh)" 2>/dev/null || true
}

"\${BUNDLED_BIN}" &
app_pid=\$!
watchdog_pid=""

if [[ "\${KILL_SWITCH}" == "1" ]]; then
    (
        breaches=0
        while kill -0 "\${app_pid}" 2>/dev/null; do
            used_mb="\$(active_wired_mb)"
            free_pct="\$(system_free_pct)"
            py_count="\$(smartstack_python_count)"
            [[ -z "\${free_pct}" ]] && free_pct=100
            if (( used_mb > KILL_THRESHOLD_MB || free_pct <= KILL_FREE_PCT || py_count > KILL_PY_COUNT_THRESHOLD )); then
                breaches=\$((breaches + 1))
            else
                breaches=0
            fi
            if (( breaches >= KILL_BREACH_COUNT )); then
                kill_stack_processes "\${app_pid}"
                exit 137
            fi
            sleep "\${KILL_POLL_SEC}"
        done
    ) &
    watchdog_pid=\$!
fi

while kill -0 "\${app_pid}" 2>/dev/null; do
    sleep 1
done

wait "\${app_pid}" || true
if [[ -n "\${watchdog_pid}" ]]; then
    kill "\${watchdog_pid}" 2>/dev/null || true
fi
exit 0
EOF
chmod +x "${DEST_PATH}/Contents/MacOS/SmartStackUI"

# Remove quarantine if present (safe if absent).
xattr -dr com.apple.quarantine "${DEST_PATH}" 2>/dev/null || true

echo "[INFO] App installed."
echo "[INFO] Mode: bundled-binary launcher + kill-switch watchdog (no compile on click)"
echo "[INFO] Open with: open \"${DEST_PATH}\""
