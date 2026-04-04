#!/bin/bash
set -euo pipefail

APP_NAME="aigo"
CMD_PATH="./cmd/server"
OUTPUT_DIR="./dist"


VERSION="${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo "dev")}"
LDFLAGS="-s -w -X main.version=${VERSION}"

build() {
    local os="$1"
    local arch="$2"
    local output="${OUTPUT_DIR}/${APP_NAME}-${os}-${arch}"
    echo "Building ${output} ..."

    if [ "$arch" == "amd64" ]; then
        CC="x86_64-linux-gnu-gcc"
    elif [ "$arch" == "arm64" ]; then
        CC="aarch64-linux-gnu-gcc"
    fi
    CGO_ENABLED=1 GOOS="${os}" GOARCH="${arch}" go build \
        -ldflags="${LDFLAGS}" \
        -o "${output}" \
        "${CMD_PATH}"

    chmod +x "${output}"
    echo "  -> $(file "${output}")"
    echo "  -> $(du -h "${output}" | cut -f1)"
}

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

build linux arm64
build linux amd64

echo ""
echo "Done. Output:"
ls -lh "${OUTPUT_DIR}/"
