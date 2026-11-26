# CUDA 12.6 + Python 3.10 environment for this repo.
#
# Build from the repository root (context excludes datasets, keys, and encoded weights):
#   docker build -f docker/Dockerfile -t wsm .
#
# Run with the repo mounted. Liberate is not pip-installed: HE/path_setup.py
# adds HE/liberate/src, and the CPython 3.10 extensions already in that tree
# need this image's CUDA runtime plus the torch libraries installed below.
#   docker run --gpus all --rm -it -v "$PWD":/workspace -w /workspace wsm
#   python3 Plain/poly_model_inference.py
#   python3 HE/HE_model_inference.py --dataset mrpc --sample 0

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        build-essential \
        git \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
        --index-url https://download.pytorch.org/whl/cu126 \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

CMD ["python3"]
