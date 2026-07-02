"""Pre-download all model weights into /app/model_cache at Docker build time.

Invoked by the Dockerfile RUN step so weights are baked into the image layer.
Model IDs are read from ARG-injected env vars (defaulting to the same values
main.py uses), so changing a model ID in the Dockerfile ARGs triggers a
targeted layer rebuild rather than a full image rebuild.
"""

from __future__ import annotations

import os
import sys

CACHE = "/app/model_cache"
os.makedirs(CACHE, exist_ok=True)
os.makedirs(os.path.join(CACHE, "u2net"), exist_ok=True)

# Pin all HuggingFace and rembg downloads to the persistent cache dir.
# These must be set before any model library is imported.
os.environ["HF_HOME"] = CACHE
os.environ["U2NET_HOME"] = os.path.join(CACHE, "u2net")

SEGMENT_MODEL  = os.getenv("SEGMENT_MODEL",  "isnet-general-use")
SAM2_MODEL_ID  = os.getenv("SAM2_MODEL_ID",  "facebook/sam2-hiera-small")
DEPTH_MODEL_ID = os.getenv("DEPTH_MODEL_ID", "depth-anything/Depth-Anything-V2-Small-hf")
GROUND_MODEL_ID = os.getenv("GROUND_MODEL_ID", "CIDAS/clipseg-rd64-refined")

errors: list[str] = []


def _step(label: str) -> None:
    print(f"\n── {label}", flush=True)


# ── rembg ONNX checkpoint ─────────────────────────────────────────────────────
_step(f"rembg / {SEGMENT_MODEL}")
try:
    from rembg import new_session  # type: ignore
    new_session(SEGMENT_MODEL)
    print("  done", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    errors.append(f"rembg/{SEGMENT_MODEL}: {e}")


# ── SAM 2 ─────────────────────────────────────────────────────────────────────
_step(f"SAM 2 / {SAM2_MODEL_ID}")
try:
    from transformers import Sam2Model, Sam2Processor  # type: ignore
    Sam2Processor.from_pretrained(SAM2_MODEL_ID)
    Sam2Model.from_pretrained(SAM2_MODEL_ID)
    print("  done", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    errors.append(f"sam2/{SAM2_MODEL_ID}: {e}")


# ── Depth Anything V2 ─────────────────────────────────────────────────────────
_step(f"Depth Anything V2 / {DEPTH_MODEL_ID}")
try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # type: ignore
    AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
    print("  done", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    errors.append(f"depth/{DEPTH_MODEL_ID}: {e}")


# ── CLIPSeg ───────────────────────────────────────────────────────────────────
_step(f"CLIPSeg / {GROUND_MODEL_ID}")
try:
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor  # type: ignore
    CLIPSegProcessor.from_pretrained(GROUND_MODEL_ID)
    CLIPSegForImageSegmentation.from_pretrained(GROUND_MODEL_ID)
    print("  done", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    errors.append(f"clipseg/{GROUND_MODEL_ID}: {e}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "─" * 60, flush=True)
if errors:
    print(f"Pre-download finished with {len(errors)} error(s):", flush=True)
    for err in errors:
        print(f"  • {err}", flush=True)
    sys.exit(1)
else:
    print("All models pre-downloaded successfully.", flush=True)
