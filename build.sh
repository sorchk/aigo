#!/bin/bash
set -euo pipefail

APP_NAME="aigo"
CMD_PATH="./cmd/server"
OUTPUT_DIR="./dist"


VERSION="${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo "dev")}"
LDFLAGS="-s -w -X main.version=${VERSION}"

# 交叉编译策略:
#   CROSS=auto (默认)  缺少交叉工具链时跳过该目标, 只构建能构建的
#   CROSS=1             强制构建全部目标, 缺工具链直接报错
#   CROSS=0             只构建当前主机架构
CROSS="${CROSS:-auto}"

# 旧 glibc 目标兼容构建:
#   COMPAT_BASE=ubuntu:18.04 ./build.sh   在 Ubuntu 18.04 (glibc 2.27) 容器内编译本机架构
#   COMPAT_BASE=ubuntu:22.04 ./build.sh   在 Ubuntu 22.04 (glibc 2.35) 容器内编译本机架构
# 适用于把二进制放到 glibc 更老的机器/容器上运行 (如 JetPack 5 = Ubuntu 18.04)。
COMPAT_BASE="${COMPAT_BASE:-}"
GO_VERSION="${GO_VERSION:-1.24.13}"

host_arch() {
    case "$(uname -m)" in
        x86_64)  echo "amd64" ;;
        aarch64) echo "arm64" ;;
        *)       echo "$(uname -m)" ;;
    esac
}

cc_for_arch() {
    case "$1" in
        amd64) echo "x86_64-linux-gnu-gcc" ;;
        arm64) echo "aarch64-linux-gnu-gcc" ;;
        *)     echo "gcc" ;;
    esac
}

build() {
    local os="$1"
    local arch="$2"
    local output="${OUTPUT_DIR}/${APP_NAME}-${os}-${arch}"
    local host
    host="$(host_arch)"
    local cc
    cc="$(cc_for_arch "${arch}")"

    if [ "${arch}" != "${host}" ] && [ "${CROSS}" = "0" ]; then
        echo "skip ${output}: CROSS=0 (只构建 ${host})"
        return 0
    fi

    if ! command -v "${cc}" >/dev/null 2>&1; then
        if [ "${arch}" = "${host}" ]; then
            cc="gcc"   # 本机原生编译器
        elif [ "${CROSS}" = "1" ]; then
            echo "error: 缺少交叉编译器 ${cc}，请先安装后重试" >&2
            echo "  Debian/Ubuntu: sudo apt-get install -y gcc-${cc%%-linux-gnu-gcc}-linux-gnu" >&2
            echo "  或改用: docker buildx build --platform linux/amd64 --target cpu ." >&2
            exit 1
        else
            echo "skip ${output}: 缺少交叉编译器 ${cc} (CROSS=1 可强制, 或用 Docker 构建)"
            return 0
        fi
    fi

    echo "Building ${output} ... (CC=${cc})"

    # 注意: CC 必须以「环境变量前缀」方式传给 go 子进程。
    # 之前写成单独赋值 `CC=...` 不会导出, go 会退回默认的 `gcc`,
    # 在 ARM 主机上构建 amd64 时 cgo 会带上 -m64 → gcc: unrecognized option '-m64'。
    CGO_ENABLED=1 GOOS="${os}" GOARCH="${arch}" CC="${cc}" go build \
        -ldflags="${LDFLAGS}" \
        -o "${output}" \
        "${CMD_PATH}"

    chmod +x "${output}"
    echo "  -> $(file "${output}")"
    echo "  -> $(du -h "${output}" | cut -f1)"
}

build_compat() {
    local os="$1"
    local arch="$2"
    local output="${OUTPUT_DIR}/${APP_NAME}-${os}-${arch}"

    echo "Building ${output} ... (compat base ${COMPAT_BASE})"

    # 复用宿主机 Go module 缓存，避免每次重新下载依赖
    local cache_args=()
    if [ -d "${HOME}/go/pkg/mod" ]; then
        cache_args=(-v "${HOME}/go/pkg/mod":/root/go/pkg/mod -e GOMODCACHE=/root/go/pkg/mod)
    fi

    docker run --rm -v "${PWD}":/src -w /src ${cache_args[@]+"${cache_args[@]}"} "${COMPAT_BASE}" bash -c "
        set -e
        apt-get update >/dev/null
        apt-get install -y --no-install-recommends ca-certificates curl gcc libc6-dev >/dev/null
        GOARCH=\$(case \"\$(uname -m)\" in x86_64) echo amd64 ;; aarch64) echo arm64 ;; *) echo \$(uname -m) ;; esac)
        curl -fsSL \"https://go.dev/dl/go${GO_VERSION}.linux-\${GOARCH}.tar.gz\" -o /tmp/go.tgz
        tar -C /usr/local -xzf /tmp/go.tgz
        PATH=/usr/local/go/bin:\$PATH CGO_ENABLED=1 go build -ldflags='-s -w -X main.version=${VERSION}' -o '${output}' ./cmd/server
    "
    sudo chown "$(id -u):$(id -g)" "${output}" 2>/dev/null || true
    chmod +x "${output}"
    echo "  -> $(file "${output}")"
    echo "  -> $(du -h "${output}" | cut -f1)"
}

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

if [ -n "${COMPAT_BASE}" ]; then
    # 容器内编译只能产出容器(=本机)架构
    case "$(uname -m)" in
        x86_64)  build_compat linux amd64 ;;
        aarch64) build_compat linux arm64 ;;
    esac
else
    build linux arm64
    build linux amd64
fi

echo ""
echo "Done. Output:"
ls -lh "${OUTPUT_DIR}/"
