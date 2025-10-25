FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential \
    bash tini \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ARG CONDA_DIR=/opt/conda
ENV CONDA_DIR=${CONDA_DIR}
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
 && rm -f /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}

WORKDIR /workspace

# Copy repository
COPY . /workspace/associative-recurrent-memory-transformer
WORKDIR /workspace/associative-recurrent-memory-transformer

# Create Python environment and install dependencies
RUN bash create_env.sh

# Default environment and caches
ENV CONDA_DEFAULT_ENV=pretrain \
    PATH=${CONDA_DIR}/envs/pretrain/bin:${PATH} \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    WANDB_DIR=/workspace/wandb

RUN mkdir -p ${HF_HOME} ${WANDB_DIR}

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]


