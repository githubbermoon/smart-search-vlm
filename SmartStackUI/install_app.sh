#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY_PATH="${SCRIPT_DIR}/.build-local/SmartStackUI"
APP_NAME="SmartStackUI.app"
DEFAULT_DEST="${HOME}/Applications/${APP_NAME}"
DEST_PATH="${DEFAULT_DEST}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)
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
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

cp "${BINARY_PATH}" "${DEST_PATH}/Contents/MacOS/SmartStackUI"
chmod +x "${DEST_PATH}/Contents/MacOS/SmartStackUI"

# Remove quarantine if present (safe if absent).
xattr -dr com.apple.quarantine "${DEST_PATH}" 2>/dev/null || true

echo "[INFO] App installed."
echo "[INFO] Open with: open \"${DEST_PATH}\""
