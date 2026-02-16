FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install NeurInSpectre + dev tooling. The extra index enables CUDA wheels for torch
# when available (falls back to CPU wheels if not).
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu121

# Verify installation (no network; best-effort inventory).
RUN neurinspectre doctor

ENTRYPOINT ["neurinspectre"]

