FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg \
    FACESLIM_OUTPUT=/data/output

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libdbus-1-3 \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libice6 \
    libsm6 \
    libxext6 \
    libxkbcommon0 \
    libxrender1 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-docker.txt

COPY FaceSlim.py FaceSlim_v1.py runtime_hook_mp.py ./
COPY icon.png ./

VOLUME ["/data"]
ENTRYPOINT ["python", "FaceSlim_v1.py"]
CMD ["--help"]
