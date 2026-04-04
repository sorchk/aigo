#!/bin/bash
set -e

name=aigo
ver=${1:-0.0.2}
build_date=$(date +"%Y%m%d")
tag="${ver}_${build_date}"

export DOCKER_CLI_EXPERIMENTAL=enabled
export DOCKER_BUILDKIT=1

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

# CPU (amd64 + arm64)
build_variant cpu "linux/amd64,linux/arm64"

# CUDA 12 (amd64 only, ORT GPU x64)
build_variant cuda12 "linux/amd64"

# CUDA 11 (amd64 only)
build_variant cuda11 "linux/amd64"

# ROCm (amd64 only)
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
echo "  sorc/${name}:cuda12              (CUDA 12.6, amd64)"
echo "  sorc/${name}:cuda11              (CUDA 11.8, amd64)"
echo "  sorc/${name}:rocm                (ROCm 6.3, amd64)"
echo "  sorc/${name}:vulkan              (Vulkan, multi-arch)"
echo "  sorc/${name}:jetpack5            (JetPack 5, arm64)"
echo "  sorc/${name}:jetpack6            (JetPack 6, arm64)"
