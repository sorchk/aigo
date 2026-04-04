# =============================================================================
# Go AI Model Hub - Multi-platform Dockerfile
# =============================================================================
#
# Build targets (--target):
#   cpu        - CPU only (default)               [amd64, arm64]
#   cuda11     - NVIDIA CUDA 11.8                 [amd64]
#   cuda12     - NVIDIA CUDA 12.6                 [amd64]
#   rocm       - AMD ROCm                         [amd64]
#   vulkan     - Vulkan (AMD/Intel/ARM GPUs)      [amd64, arm64]
#   jetpack5   - NVIDIA JetPack 5 (L4T r35.x)     [arm64]
#   jetpack6   - NVIDIA JetPack 6 (L4T r36.x)     [arm64]
#
# Build examples:
#   docker build -t aigo:cpu .
#   docker build --target cuda12 -t aigo:cuda12 .
#   docker buildx build --platform linux/amd64 --target cuda12 -t aigo:cuda12 .
#   docker buildx build --platform linux/arm64 --target jetpack6 -t aigo:jetpack6 .
#
# Run examples:
#   docker run -p 8000:8000 aigo:cpu
#   docker run --gpus all -p 8000:8000 aigo:cuda12
#   docker run --device /dev/kfd --device /dev/dri -p 8000:8000 aigo:rocm
#   docker run --device /dev/dri -p 8000:8000 aigo:vulkan
#
# llama-server:
#   Mount at runtime:  -v /path/to/llama-server:/usr/bin/llama-server
#   Or build custom:   FROM aigo:cuda12
#                       COPY llama-server /usr/bin/llama-server
#
# Environment variables:
#   ONNXRUNTIME_LIB    - Path to ORT shared library
#   LLAMA_SERVER_PATH  - Path to llama-server binary
#   API_TOKEN          - Bearer token auth
#   HOST / PORT        - Listen address (default 0.0.0.0:8000)
# =============================================================================

# ======================== Go Builder ========================
FROM golang:1.23-bookworm AS builder

ARG TARGETARCH

WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-aarch64-linux-gnu \
        gcc-x86-64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$TARGETARCH" = "arm64" ]; then \
      CC=aarch64-linux-gnu-gcc; \
    else \
      CC=x86_64-linux-gnu-gcc; \
    fi && \
    CGO_ENABLED=1 GOARCH=${TARGETARCH} CC=${CC} \
    go build -ldflags="-s -w" -o /server ./cmd/server

# ======================== ORT CPU Downloader ========================
FROM debian:bookworm-slim AS ort-cpu

ARG TARGETARCH
ARG ORT_VERSION=1.19.2

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*

RUN if [ "$TARGETARCH" = "arm64" ]; then ORT_ARCH=aarch64; else ORT_ARCH=x64; fi && \
    wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz" \
        -O /tmp/ort.tgz && \
    tar xzf /tmp/ort.tgz -C /opt && \
    mv "/opt/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}" /opt/onnxruntime && \
    cd /opt/onnxruntime && tar cf /tmp/ort-libs.tar lib/ && \
    rm -rf /tmp/ort.tgz /opt/onnxruntime/include

# ======================== ORT GPU Downloader ========================
FROM debian:bookworm-slim AS ort-gpu

ARG ORT_VERSION=1.19.2

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" \
        -O /tmp/ort.tgz && \
    tar xzf /tmp/ort.tgz -C /opt && \
    mv "/opt/onnxruntime-linux-x64-gpu-${ORT_VERSION}" /opt/onnxruntime && \
    cd /opt/onnxruntime && tar cf /tmp/ort-libs.tar lib/ && \
    rm -rf /tmp/ort.tgz /opt/onnxruntime/include

# ======================== CPU ========================
FROM debian:bookworm-slim AS cpu

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ort-cpu /tmp/ort-libs.tar /tmp/
RUN tar xf /tmp/ort-libs.tar -C /usr/ && ldconfig && rm /tmp/ort-libs.tar

COPY --from=builder /server /app/server

WORKDIR /app
ENV ONNXRUNTIME_LIB=/usr/lib/libonnxruntime.so
ENV LLAMA_SERVER_PATH=/usr/bin/llama-server
EXPOSE 8000
CMD ["./server"]

# ======================== CUDA 12 ========================
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS cuda12

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ort-gpu /tmp/ort-libs.tar /tmp/
RUN tar xf /tmp/ort-libs.tar -C /usr/ && ldconfig && rm /tmp/ort-libs.tar

COPY --from=builder /server /app/server

WORKDIR /app
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV ONNXRUNTIME_LIB=/usr/lib/libonnxruntime.so
ENV LLAMA_SERVER_PATH=/usr/bin/llama-server
EXPOSE 8000
CMD ["./server"]

# ======================== CUDA 11 ========================
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS cuda11

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ort-cpu /tmp/ort-libs.tar /tmp/
RUN tar xf /tmp/ort-libs.tar -C /usr/ && ldconfig && rm /tmp/ort-libs.tar

COPY --from=builder /server /app/server

WORKDIR /app
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV ONNXRUNTIME_LIB=/usr/lib/libonnxruntime.so
ENV LLAMA_SERVER_PATH=/usr/bin/llama-server
EXPOSE 8000
CMD ["./server"]

# ======================== ROCm ========================
FROM rocm/dev-ubuntu-22.04:6.3 AS rocm

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ort-cpu /tmp/ort-libs.tar /tmp/
RUN tar xf /tmp/ort-libs.tar -C /usr/ && ldconfig && rm /tmp/ort-libs.tar

COPY --from=builder /server /app/server

WORKDIR /app
ENV ONNXRUNTIME_LIB=/usr/lib/libonnxruntime.so
ENV LLAMA_SERVER_PATH=/usr/bin/llama-server
ENV ROCM_PATH=/opt/rocm
EXPOSE 8000
CMD ["./server"]

# ======================== Vulkan ========================
FROM debian:bookworm-slim AS vulkan

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libvulkan1 \
        mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ort-cpu /tmp/ort-libs.tar /tmp/
RUN tar xf /tmp/ort-libs.tar -C /usr/ && ldconfig && rm /tmp/ort-libs.tar

COPY --from=builder /server /app/server

WORKDIR /app
ENV ONNXRUNTIME_LIB=/usr/lib/libonnxruntime.so
ENV LLAMA_SERVER_PATH=/usr/bin/llama-server
EXPOSE 8000
CMD ["./server"]

# ======================== JetPack 5 ========================
FROM nvcr.io/nvidia/l4t-jetpack:r35.4.1 AS jetpack5

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

ARG ORT_VERSION=1.19.2
RUN wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz" \
        -O /tmp/ort.tgz && \
    tar xzf /tmp/ort.tgz -C /opt && \
    cp "/opt/onnxruntime-linux-aarch64-${ORT_VERSION}/lib/"* /usr/lib/ && \
    ldconfig && \
    rm -rf /tmp/ort.tgz "/opt/onnxruntime-linux-aarch64-${ORT_VERSION}"

COPY --from=builder /server /app/server

WORKDIR /app
ENV ONNXRUNTIME_LIB=/usr/lib/libonnxruntime.so
ENV LLAMA_SERVER_PATH=/usr/bin/llama-server
EXPOSE 8000
CMD ["./server"]

# ======================== JetPack 6 ========================
FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0 AS jetpack6

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

ARG ORT_VERSION=1.19.2
RUN wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz" \
        -O /tmp/ort.tgz && \
    tar xzf /tmp/ort.tgz -C /opt && \
    cp "/opt/onnxruntime-linux-aarch64-${ORT_VERSION}/lib/"* /usr/lib/ && \
    ldconfig && \
    rm -rf /tmp/ort.tgz "/opt/onnxruntime-linux-aarch64-${ORT_VERSION}"

COPY --from=builder /server /app/server

WORKDIR /app
ENV ONNXRUNTIME_LIB=/usr/lib/libonnxruntime.so
ENV LLAMA_SERVER_PATH=/usr/bin/llama-server
EXPOSE 8000
CMD ["./server"]
