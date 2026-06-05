# Mall Heatmap - CPU-only image. Bundles torch + ultralytics so the client needs only Docker.
FROM python:3.13-slim

# System libs: OpenCV needs libGL/libglib at import; build-essential lets pip build any
# sdist-only dependency (e.g. lap) when no wheel is available for this Python version.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU build of torch first, so ultralytics doesn't pull the multi-GB CUDA wheels
# (useless without a GPU and far too large).
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# Remaining Python deps (torch already satisfied above). Copy requirements before the app code
# so editing source doesn't invalidate the cached dependency layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + bundled model weights.
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
