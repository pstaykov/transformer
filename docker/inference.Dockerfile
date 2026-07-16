# Inference container: plain Ubuntu, no GPU needed - serve.py runs the
# NumPy model on CPU. Serves the showcase page + chat API (see serve.py).
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        python3-dev \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Build the bbpe tokenizer's Python extension so checkpoints trained with
# --tokenizer bbpe work here too (without it, only --tokenizer byte works).
COPY tokenizer/src ./tokenizer/src
COPY tokenizer/include ./tokenizer/include
COPY tokenizer/tools ./tokenizer/tools
COPY tokenizer/setup.py ./tokenizer/setup.py
RUN pip install --no-cache-dir pybind11 setuptools \
    && pip install --no-cache-dir ./tokenizer --no-build-isolation

COPY model.py generate.py serve.py ./
COPY utils ./utils
COPY web ./web

EXPOSE 8000
ENTRYPOINT ["python3", "serve.py"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
