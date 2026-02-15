#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFTFIX_DIR="${SCRIPT_DIR}/.swiftfix"
BUILD_DIR="${SCRIPT_DIR}/.build-local"
SOURCE_FILE="${SCRIPT_DIR}/Sources/SmartStackUI/main.swift"
BINARY_PATH="${BUILD_DIR}/SmartStackUI"
SWIFTLANG_STAMP="${SWIFTFIX_DIR}/swiftlang_token.txt"

CLT_SDK="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
LOCAL_SDK="${SWIFTFIX_DIR}/MacOSX.sdk"
EMPTY_MODULEMAP="${SWIFTFIX_DIR}/empty.modulemap"
VFS_OVERLAY="${SWIFTFIX_DIR}/vfs_overlay.yaml"

build_only=0
clean_sdk=0
force_build=0

for arg in "$@"; do
    case "$arg" in
        --build-only)
            build_only=1
            ;;
        --clean-sdk)
            clean_sdk=1
            ;;
        --force-build)
            force_build=1
            ;;
        *)
            echo "[ERROR] Unknown argument: $arg"
            echo "Usage: ./local_run.sh [--build-only] [--clean-sdk] [--force-build]"
            exit 2
            ;;
    esac
done

if [[ ! -d "${CLT_SDK}" ]]; then
    echo "[ERROR] CLT SDK not found at ${CLT_SDK}"
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
swiftlang_token="$(printf '%s\n' "${swift_version_line}" | sed -n 's/.*\(swiftlang-[^ )]*\).*/\1/p')"
if [[ -z "${swiftlang_token}" ]]; then
    echo "[ERROR] Could not parse swiftlang token from: ${swift_version_line}"
    exit 1
fi

previous_swiftlang_token=""
if [[ -f "${SWIFTLANG_STAMP}" ]]; then
    previous_swiftlang_token="$(cat "${SWIFTLANG_STAMP}")"
fi

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
elif [[ "${SOURCE_FILE}" -nt "${BINARY_PATH}" ]]; then
    need_build=1
elif [[ "${SWIFTLANG_STAMP}" -nt "${BINARY_PATH}" ]]; then
    need_build=1
fi

if [[ "${need_build}" -eq 1 ]]; then
    echo "[INFO] Building SmartStackUI..."
    swiftc \
        -parse-as-library \
        -sdk "${LOCAL_SDK}" \
        -vfsoverlay "${VFS_OVERLAY}" \
        "${SOURCE_FILE}" \
        -o "${BINARY_PATH}"
    echo "[INFO] Built binary: ${BINARY_PATH}"
else
    echo "[INFO] Build skipped (up-to-date): ${BINARY_PATH}"
fi

if [[ "${build_only}" -eq 1 ]]; then
    exit 0
fi

echo "[INFO] Launching SmartStackUI..."
"${BINARY_PATH}"
