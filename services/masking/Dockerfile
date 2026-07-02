FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps.
# Separated from source copy so this layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Model pre-download ────────────────────────────────────────────────────────
# Models are baked into this Docker layer so they are available instantly on
# every container start — no network download needed at runtime.
#
# Each ARG maps to the corresponding env var in main.py. Change a model ID here
# (or pass --build-arg at docker build time) to swap models; Docker will only
# rebuild from this layer down, re-downloading just the changed model.
#
# Cache layout inside the image:
#   /app/model_cache/          ← HF_HOME (SAM 2/3, Depth, CLIPSeg, …)
#   /app/model_cache/u2net/    ← U2NET_HOME (rembg ONNX checkpoints)
ARG SEGMENT_MODEL=isnet-general-use
ARG SAM2_MODEL_ID=facebook/sam2-hiera-small
ARG DEPTH_MODEL_ID=depth-anything/Depth-Anything-V2-Small-hf
ARG GROUND_MODEL_ID=CIDAS/clipseg-rd64-refined

COPY preload_models.py .
RUN SEGMENT_MODEL=$SEGMENT_MODEL \
    SAM2_MODEL_ID=$SAM2_MODEL_ID \
    DEPTH_MODEL_ID=$DEPTH_MODEL_ID \
    GROUND_MODEL_ID=$GROUND_MODEL_ID \
    python preload_models.py

# Copy the rest of the service code AFTER pre-downloading so source edits
# don't bust the model cache layer above.
COPY . .

# Tell main.py to use the pre-downloaded cache at runtime.
ENV MODEL_CACHE_DIR=/app/model_cache

# Hugging Face Spaces expects the app to run on port 7860
ENV PORT=7860
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "main.py"]
