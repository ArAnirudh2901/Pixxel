# Pixxel Masking Service

A focused FastAPI microservice for AI **masking** — SAM 3.1 (subject / instance /
text grounding), Depth Anything V2, SAM 2 click/box, rembg saliency, CLIPSeg
fallback. Split out from `services/segment` so masking deploys as its **own
Hugging Face Space**, independent of the erase/inpaint service. The Next.js app
reaches it via `MASKING_SERVICE_URL` (its API routes proxy here).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# SAM 3.1 (gated — optional but recommended; falls back to rembg/CLIPSeg if absent)
pip install "git+https://github.com/facebookresearch/sam3.git"
hf auth login                      # account must be approved at hf.co/facebook/sam3.1
cp .env.example .env               # tweak as needed
python main.py                     # serves on PORT (default 8002 local, 7860 in Docker/HF)
```

## Endpoints

- `GET  /health`            – service + model availability
- `POST /warmup`            – eager-load SAM 3 / SAM 2 / Depth / CLIPSeg
- `POST /segment`           – subject saliency matte (RGBA PNG)
- `POST /segment/instances` – SAM 3.1 multi-instance concept masks (JSON)
- `POST /sam2/click`        – point/box promptable mask (greyscale PNG)
- `POST /depth`             – Depth Anything V2 depth map (greyscale PNG)
- `POST /ground/text`       – text-grounded mask (SAM 3 → CLIPSeg fallback, JSON)

## Deploy (Hugging Face Space)

Docker Space using the bundled `Dockerfile` (pre-downloads rembg/SAM2/Depth/CLIPSeg
at build; SAM 3.1 loads at runtime). Set `CORS_ORIGINS` to your app origin and
enable Persistent Storage → `MODEL_CACHE_DIR=/data/models`. Point the app's
`MASKING_SERVICE_URL` at the Space URL.
