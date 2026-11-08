#!/bin/bash
set -e

name=aigo
ver=${1:-0.0.3}
build_date=$(date +"%Y%m%d")
tag="${ver}_${build_date}"

# 只构建指定变体: ./docker.sh 0.0.5 jetpack5 ; 不带第二个参数则全量构建
only_variant="${2:-}"

export DOCKER_CLI_EXPERIMENTAL=enabled
export DOCKER_BUILDKIT=1

variant_platform() {
  case "$1" in
    cpu)                                   echo "linux/amd64,linux/arm64" ;;
    cuda13|cuda12|cuda11|rocm)             echo "linux/amd64" ;;
    vulkan)                                echo "linux/amd64,linux/arm64" ;;
    jetpack5|jetpack6)                     echo "linux/arm64" ;;
    *)                                     echo "linux/amd64,linux/arm64" ;;
  esac
}

build_variant() {
  local variant=$1
  local platform=$2

  echo "=== Building ${variant} (${platform}) ==="

  docker buildx build \
    --platform "${platform}" \
    --target "${variant}" \
    --push \
    --tag "sorc/${name}:${tag}-${variant}" \
    --tag "sorc/${name}:${ver}-${variant}" \
    --tag "sorc/${name}:${variant}" \
    .
}

if [ -n "${only_variant}" ]; then
  # 单变体模式（JetPack 5 / 6 等）
  build_variant "${only_variant}" "$(variant_platform "${only_variant}")"
  echo ""
  echo "=== Single-variant build complete ==="
  echo "Image: sorc/${name}:${only_variant}"
  exit 0
fi

# CPU (amd64 + arm64)
build_variant cpu "linux/amd64,linux/arm64"

# CUDA 13 (amd64 only, ORT GPU x64)
build_variant cuda13 "linux/amd64"

# CUDA 12 (amd64 only, ORT GPU x64)
build_variant cuda12 "linux/amd64"

# CUDA 11 (amd64 only)
build_variant cuda11 "linux/amd64"

# ROCm 7.x (amd64 only)
build_variant rocm "linux/amd64"

# Vulkan (amd64 + arm64)
build_variant vulkan "linux/amd64,linux/arm64"

# JetPack 5 (arm64 only)
build_variant jetpack5 "linux/arm64"

# JetPack 6 (arm64 only)
build_variant jetpack6 "linux/arm64"

# Tag CPU as latest
echo "=== Tagging latest ==="
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --target cpu \
  --push \
  --tag "sorc/${name}:${tag}" \
  --tag "sorc/${name}:${ver}" \
  --tag "sorc/${name}:latest" \
  .

echo "=== All builds complete ==="
echo "Images:"
echo "  sorc/${name}:latest              (CPU, multi-arch)"
echo "  sorc/${name}:cpu                 (CPU, multi-arch)"
echo "  sorc/${name}:cuda13              (CUDA 13.0, amd64)"
echo "  sorc/${name}:cuda12              (CUDA 12.6, amd64)"
echo "  sorc/${name}:cuda11              (CUDA 11.8, amd64)"
echo "  sorc/${name}:rocm                (ROCm 7.x, amd64)"
echo "  sorc/${name}:vulkan              (Vulkan, multi-arch)"
echo "  sorc/${name}:jetpack5            (JetPack 5, arm64)"
echo "  sorc/${name}:jetpack6            (JetPack 6, arm64)"
