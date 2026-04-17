FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 基础开发工具
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    wget \
    gnupg \
    lsb-release \
    software-properties-common \
    build-essential \
    ninja-build \
    git \
    gdb \
    ccache \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    bash-completion \
 && rm -rf /var/lib/apt/lists/*

# 安装npm和codex
ARG NODE_VERSION=22.22.2

RUN ARCH="$(dpkg --print-architecture)" && \
    case "${ARCH}" in \
      amd64) NODE_ARCH="x64" ;; \
      arm64) NODE_ARCH="arm64" ;; \
      *) echo "Unsupported arch: ${ARCH}" && exit 1 ;; \
    esac && \
    curl -fsSLO "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz" && \
    tar -xJf "node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz" -C /usr/local --strip-components=1 && \
    rm "node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz" && \
    npm i -g @openai/codex && \
    node -v && npm -v && codex --version

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install cmake

ARG TORCH_VERSION=2.4.1
ARG TORCHVISION_VERSION=0.19.1
ARG TORCHAUDIO_VERSION=2.4.1

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
    numpy \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-nsight-systems-12-4 \
    cuda-nsight-compute-12-4 \
    && rm -rf /var/lib/apt/lists/*

# 装更新一点的 clangd / clang（Jammy 官方仓库默认偏旧）
RUN mkdir -p /etc/apt/keyrings && \
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /etc/apt/keyrings/llvm.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" > /etc/apt/sources.list.d/llvm.list && \
    apt-get update && apt-get install -y \
      clang-20 \
      clangd-20 \
      clang-format-20 \
      clang-tidy-20 \
      lldb-20 \
 && rm -rf /var/lib/apt/lists/*

# 给 VS Code / shell 一个稳定路径
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 200 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-20 200 && \
    update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-20 200

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:/usr/lib/ccache:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /workspace