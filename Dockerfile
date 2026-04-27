ARG CUDA_VERSION=13.2.1-cudnn-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        python3-dev \
        curl \
        git \
        build-essential \
        && ln -sf /usr/bin/python3 /usr/bin/python \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed pip wheel

RUN python3 -m pip install --no-cache-dir --break-system-packages \
        'vllm==0.19.1' \
        auto-round \
        hf_transfer \
        huggingface_hub

RUN mkdir -p /data/models /data/logs

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

COPY scripts/ /usr/local/bin/scripts/
ENV PATH="/usr/local/bin/scripts:${PATH}"

ENV MODEL_DIR=/data/models
ENV LOG_DIR=/data/logs
ENV MODEL_REPO=Lorbus/Qwen3.6-27B-int4-AutoRound
ENV PORT=1234
ENV SERVED_MODEL_NAME=qwen3.6-27b
ENV MAX_MODEL_LEN=200000
ENV MAX_NUM_SEQS=3
ENV GPU_MEMORY_UTIL=0.92
ENV TEMPERATURE=0.6
ENV TOP_P=0.95
ENV TOP_K=20
ENV MIN_P=0.0
ENV PRESENCE_PENALTY=0
ENV REPETITION_PENALTY=1.0
ENV REASONING_PARSER=qwen3
ENV MODEL_DOWNLOAD=0

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["serve"]
