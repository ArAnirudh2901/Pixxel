"""Phosmith mask service.

Wraps `rembg` and Meta SAM 3.1 in a tiny FastAPI HTTP API so the Next.js AI
routes can call background-removal and segmentation models locally.

Free-tier friendly: no GPU required, but auto-uses CUDA (NVIDIA) or
MPS (Apple Silicon) when available.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np

# OpenCV + SciPy power the saliency-matte cleanup (hole-fill, speck removal,
# morphological close, distance transform). Both are hard deps of the listed
# requirements, but degrade gracefully: if either is missing, clean_matte()
# returns the matte untouched rather than crashing the request.
try:
    import cv2  # type: ignore
    from scipy import ndimage  # type: ignore
    _MATTE_CLEANUP = True
except Exception:  # pragma: no cover - optional accel
    cv2 = None  # type: ignore
    ndimage = None  # type: ignore
    _MATTE_CLEANUP = False

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw, ImageFilter, ImageOps, UnidentifiedImageError
from rembg import new_session, remove
from starlette.concurrency import run_in_threadpool

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("mask-service")

# Load a local services/segment/.env (if present) BEFORE reading any config, so
# `bun run mask:dev` can be configured without exporting shell vars. Best-effort:
# python-dotenv is an indirect dep; absence just means env comes from the shell.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except Exception:  # pragma: no cover - optional
    pass

# ─── Config ──────────────────────────────────────────────────────────────────

# Persistent model cache directory.
# Set MODEL_CACHE_DIR to a directory that survives process restarts, e.g.:
#   - HuggingFace Spaces with persistent storage: /data/models
#   - Docker volume: /models
# When set, both HF_HOME (transformers/diffusers) and U2NET_HOME (rembg ONNX)
# are redirected here so models are downloaded once and reused across restarts.
# Leave unset to use the platform defaults (~/.cache/huggingface, ~/.u2net).
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "").strip() or None

if MODEL_CACHE_DIR:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    # HF_HOME covers transformers, huggingface_hub, diffusers, tokenizers
    os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)
    # U2NET_HOME covers all rembg ONNX checkpoints
    _u2net_dir = os.path.join(MODEL_CACHE_DIR, "u2net")
    os.makedirs(_u2net_dir, exist_ok=True)
    os.environ.setdefault("U2NET_HOME", _u2net_dir)
    log.info("model cache pinned to %r (HF_HOME + U2NET_HOME)", MODEL_CACHE_DIR)

MODEL_NAME = os.getenv("SEGMENT_MODEL", "isnet-general-use").strip()
DEPTH_MODEL_ID = os.getenv(
    "DEPTH_MODEL_ID", "depth-anything/Depth-Anything-V2-Small-hf"
).strip()
DEPTH_CACHE_MAX = int(os.getenv("DEPTH_CACHE_MAX", "20").strip())
# Cap the per-entry depth-map size we'll cache. A 2048×2048 uint8 map
# is 4 MB; larger maps get recomputed every time (cheap relative to
# network I/O, and bounds peak memory at ~80 MB even with the full
# 20-entry budget). Defends against pathological inputs that bypass
# the Node route's dimension cap.
DEPTH_CACHE_MAX_PIXELS = int(os.getenv("DEPTH_CACHE_MAX_PIXELS", str(2048 * 2048)).strip())
# Cap the input image's longest side. The model runs internally at
# ~518×518 anyway, and resizing the depth map back to a very large
# output is O(n²) Lanczos work that can OOM the process. The Node
# route (`/api/ai/depth`) applies the same cap; this defends the
# service against direct curls that bypass the route.
DEPTH_MAX_SIDE = int(os.getenv("DEPTH_MAX_SIDE", "2048").strip())
PORT = int(os.getenv("PORT", "8001").strip())
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "24").strip())
# Reject /segment inputs whose longest side exceeds this (defense-in-depth vs
# direct curls that bypass the Node route's MAX_MODEL_SIDE cap). Mirrors the
# /depth handler's DEPTH_MAX_SIDE.
SEGMENT_MAX_SIDE = int(os.getenv("SEGMENT_MAX_SIDE", "2048").strip())
SHAPE_MASK_MAX_SIDE = int(os.getenv("SHAPE_MASK_MAX_SIDE", "2048").strip())
SHAPE_MASK_MAX_POINTS = int(os.getenv("SHAPE_MASK_MAX_POINTS", "10000").strip())

# Heavy torch models load LAZILY on first use to keep resident footprint small.
# Set SEGMENT_EAGER_MODELS=1 to preload at startup (lowest first-request latency).
SEGMENT_EAGER_MODELS = os.getenv("SEGMENT_EAGER_MODELS", "0").strip() not in ("0", "false", "False", "")

# ─── SAM 3 concept segmentation ──────────────────────────────────────────────
# SAM 3 is preferred anywhere this service needs open-vocabulary subject or
# concept masks. It is optional because Meta's official package/checkpoints are
# gated and have newer runtime requirements than the lightweight local service.
SAM3_ENABLE = os.getenv("SAM3_ENABLE", "1").strip() not in ("0", "false", "False", "")
SAM3_MODEL_ID = os.getenv("SAM3_MODEL_ID", "facebook/sam3.1").strip()
SAM3_CHECKPOINT_PATH = os.getenv("SAM3_CHECKPOINT_PATH", "").strip() or None
# 0.25 (not 0.5): at 0.5 SAM 3.1 only commits the highest-confidence core of a
# concept — e.g. "sky" bound just the bright centre (~30%). 0.25 lets it segment
# the full concept (whole sky ~80%, complete subject) while box prompts, which
# are high-confidence, are unaffected.
SAM3_CONFIDENCE = float(os.getenv("SAM3_CONFIDENCE", "0.25").strip())
SAM3_SUBJECT_PROMPT = os.getenv("SAM3_SUBJECT_PROMPT", "main subject").strip() or "main subject"
SAM3_INSTANCES_MAX = int(os.getenv("SAM3_INSTANCES_MAX", "24").strip())
SAM3_EAGER = os.getenv("SAM3_EAGER", "0").strip() not in ("0", "false", "False", "")

# High-precision matting refinement of SAM 3 masks: turn the coarse binary SAM
# mask into a hair-accurate soft alpha via a trimap + alpha-matting solve guided
# by the RGB image (PyMatting closed-form, with a pure-cv2 guided-filter
# fallback). Off by default (adds CPU latency); enable with SAM3_REFINE_MATTING=1.
SAM3_REFINE_MATTING = os.getenv("SAM3_REFINE_MATTING", "0").strip() not in ("0", "false", "False", "")
SAM3_REFINE_METHOD = os.getenv("SAM3_REFINE_METHOD", "pymatting_cf").strip()  # pymatting_cf | pymatting_knn | guided
SAM3_REFINE_MAX_SIDE = int(os.getenv("SAM3_REFINE_MAX_SIDE", "1024").strip())  # cap matting resolution (CPU perf/RAM)
SAM3_REFINE_ERODE = int(os.getenv("SAM3_REFINE_ERODE", "6").strip())          # sure-fg shrink (px @ full res)
SAM3_REFINE_DILATE = int(os.getenv("SAM3_REFINE_DILATE", "18").strip())       # unknown-band width — room for hair

# ─── Matte-cleanup tuning ────────────────────────────────────────────────────
# Only fill interior holes SMALLER than this fraction of the subject's area —
# those are model drop-outs (a leaf's dropped veins/center). Larger holes are
# GENUINE see-through gaps (a donut/ring, eyeglass lens, the gap between an arm
# and the torso) and must stay transparent.
MATTE_HOLE_FILL_MAX_FRAC = float(os.getenv("MATTE_HOLE_FILL_MAX_FRAC", "0.02").strip())
# Recover (solidify) only DEEP-interior pixels whose alpha is at or below this —
# i.e. regions the model essentially dropped. Genuinely semi-transparent pixels
# above it (frosted glass, smoke, a soft hair gradient) are LEFT soft.
MATTE_FAINT_RECOVER_MAX = int(os.getenv("MATTE_FAINT_RECOVER_MAX", "96").strip())
# ...and only when the subject has a confidently-solid CORE at least this big a
# fraction of its area. A subject that is mostly faint IS translucent — recover
# nothing, keep its alpha as-is.
MATTE_FAINT_MIN_SOLID_FRAC = float(os.getenv("MATTE_FAINT_MIN_SOLID_FRAC", "0.50").strip())
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]

# ─── Auto-crop tuning ────────────────────────────────────────────────────────
# Default padding around the union of detected subjects when computing the
# DEPTH-aware crop, expressed as a fraction of the SHORTER box side.
# 0.08 keeps a comfortable breathing room without losing too much frame.
CROP_SUBJECT_PADDING = float(os.getenv("CROP_SUBJECT_PADDING", "0.08").strip())
# Subject-aware crop is a COMPOSITION, not a subject extraction: the crop is
# sized so the subject's longer side spans ~this fraction of the crop, leaving
# deliberate breathing room / context around it. 0.62 ≈ subject fills ~⅔ of the
# frame with ~⅓ context — a natural hero composition. Lower = more context.
CROP_SUBJECT_TARGET_FRAC = float(os.getenv("CROP_SUBJECT_TARGET_FRAC", "0.62").strip())
# When there is no distinct compact saliency region and the salient bbox
# sprawls across at least this fraction of the frame, the image is a scene, not
# a subject. Subject-aware then composes the whole scene (max-area aspect fit,
# gently biased toward the salient region) instead of zooming into a blob.
CROP_SCENE_SPREAD = float(os.getenv("CROP_SCENE_SPREAD", "0.45").strip())
# How far a scene crop is biased toward the salient centroid (0 = image centre,
# 1 = fully on the salient region). A gentle pull keeps the composition balanced.
CROP_SCENE_BIAS = float(os.getenv("CROP_SCENE_BIAS", "0.35").strip())
# Content-fill (border trim) considers a pixel "content" when its absolute
# difference from the dominant border colour exceeds this 0..255 threshold.
CROP_CONTENT_TRIM_THRESH = int(os.getenv("CROP_CONTENT_TRIM_THRESH", "16").strip())
# Minimum fraction of the image the content-fill crop must keep — guards
# against accidentally cropping to nothing on a near-solid image.
CROP_CONTENT_MIN_FRAC = float(os.getenv("CROP_CONTENT_MIN_FRAC", "0.20").strip())
# Depth-aware foreground = the N percentile of nearest pixels (largest depth).
# 0.45 typically isolates the foreground plane.
CROP_DEPTH_FOREGROUND_PCT = float(os.getenv("CROP_DEPTH_FOREGROUND_PCT", "0.45").strip())
# Snap-to-rule-of-thirds strength when subject centroid is close to a power
# point (0 = no snap, 1 = always snap).
CROP_THIRDS_SNAP = float(os.getenv("CROP_THIRDS_SNAP", "0.65").strip())
# Cap the longest side accepted by /crop/auto. The endpoint composes results
# from segment + depth, both of which already cap at 2048.
CROP_MAX_SIDE = int(os.getenv("CROP_MAX_SIDE", "2048").strip())

# ─── Text-grounded masking (/ground/text — Step 9: NL mask pipeline) ────────
# Free-text phrase ("the red jacket", "the waterfall") → mask, via SAM 3's
# open-vocabulary concept grounding.
GROUND_MAX_SIDE = int(os.getenv("GROUND_MAX_SIDE", "2048").strip())
GROUND_MAX_PHRASES = int(os.getenv("GROUND_MAX_PHRASES", "4").strip())

# rembg >=2.0.59 model registry.
# Licenses: isnet-*, u2net*, silueta = MIT; bria-rmbg = CC BY-NC
# (non-commercial).
# Sizes and recommended use-cases are documented in README.md.
ALLOWED_MODELS = {
    "isnet-general-use",
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "bria-rmbg",
    # birefnet-* need rembg >= 2.0.57; birefnet-general is the quality pick
    # for backlit/complex mattes (~930 MB first download).
    "birefnet-general",
    "birefnet-general-lite",
    "birefnet-portrait",
    "birefnet-dis",
    "birefnet-hrsod",
    "birefnet-massive",
}

if MODEL_NAME not in ALLOWED_MODELS:
    log.warning("unknown SEGMENT_MODEL=%r; falling back to isnet-general-use", MODEL_NAME)
    MODEL_NAME = "isnet-general-use"


# ─── Execution-provider auto-detect (ONNX / rembg) ──────────────────────────

def detect_providers() -> List[str]:
    """Pick the ONNX Runtime execution providers for the rembg session.

    Order of preference: CUDA (NVIDIA) > CPU. CoreML is deliberately NOT
    preferred on Apple Silicon: this process also loads the torch/MPS models
    (SAM 3, Depth Anything), and the rembg CoreML session intermittently
    HARD-FAILS at inference under that Neural-Engine contention (ONNXRuntime
    error -1, "Unable to compute the prediction"), which 500s /crop/auto. CoreML
    is also ~6× slower than CPU for isnet-general-use here. Opt back in with
    SEGMENT_PROVIDERS=CoreMLExecutionProvider,CPUExecutionProvider.
    """
    override = os.getenv("SEGMENT_PROVIDERS", "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]

    providers: List[str] = ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort  # type: ignore
        if "CUDAExecutionProvider" in set(ort.get_available_providers()):
            providers.insert(0, "CUDAExecutionProvider")
            log.info("ONNX CUDA execution provider detected (NVIDIA GPU)")
    except Exception as e:  # pragma: no cover - best effort
        log.debug("onnxruntime provider probe failed: %s", e)
    return providers


def _rembg_matte(img: "Image.Image") -> "Image.Image":
    """rembg saliency matte ('L'), self-healing to a CPU session on provider
    failure (e.g. a CoreML session set via SEGMENT_PROVIDERS dying under MPS
    contention). CPU is the default, so on a healthy box this is a passthrough."""
    try:
        return remove(img, session=app.state.session, only_mask=True)
    except Exception as e:
        if app.state.providers == ["CPUExecutionProvider"]:
            raise
        log.warning(
            "rembg inference failed on %s (%s); rebuilding session on CPU",
            app.state.providers, e,
        )
        app.state.session = new_session(app.state.model_name, providers=["CPUExecutionProvider"])
        app.state.providers = ["CPUExecutionProvider"]
        return remove(img, session=app.state.session, only_mask=True)


# ─── Torch device ────────────────────────────────────────────────────────────

def detect_torch_device():
    """Pick the best torch device on this machine.

    Order of preference: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
    Returns the torch.device and a short label.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return None, "torch-missing"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


DEPTH_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _image_hash(img: Image.Image) -> str:
    """Stable hash of a PIL image's pixel data for cache keys (depth maps).
    Collisions on 16 hex chars (64 bits) are astronomically unlikely for any
    image a user will upload."""
    return hashlib.sha256(img.tobytes()).hexdigest()[:16]


def _depth_predict(app, img: Image.Image) -> np.ndarray:
    """Run Depth Anything V2 on `img`, caching the depth map by image hash.

    Returns a `np.ndarray` of shape `(H, W)` with dtype `uint8` (0..255).
    White = near, black = far. The map is at the input image's
    resolution — no resize needed.
    """
    key = _image_hash(img)
    cached = DEPTH_CACHE.get(key)
    if cached is not None:
        DEPTH_CACHE.move_to_end(key)
        log.info("Depth cache hit (%d entries)", len(DEPTH_CACHE))
        return cached

    import torch  # type: ignore
    processor = app.state.depth_processor
    model = app.state.depth_model
    device = app.state.depth_device

    t0 = time.perf_counter()
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(pixel_values=inputs.pixel_values)
    # `predicted_depth` is (1, H, W). Normalise per-image to 0..255 so
    # the user can pick a meaningful near/far range. Per-image
    # normalisation is the right default — Depth Anything V2 returns
    # relative depth, not metric, so absolute thresholds would be
    # image-dependent anyway.
    depth = outputs.predicted_depth.squeeze(0).cpu().numpy()
    d_min = float(depth.min())
    d_max = float(depth.max())
    if d_max - d_min < 1e-6:
        # Flat depth (extremely rare — uniform-colour images). Avoid
        # division-by-zero by writing 0 (the mask will be empty).
        normalised = np.zeros_like(depth, dtype=np.uint8)
    else:
        normalised = ((depth - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)

    # Skip caching huge depth maps — see DEPTH_CACHE_MAX_PIXELS rationale.
    if normalised.size <= DEPTH_CACHE_MAX_PIXELS:
        DEPTH_CACHE[key] = normalised
        while len(DEPTH_CACHE) > DEPTH_CACHE_MAX:
            DEPTH_CACHE.popitem(last=False)
    else:
        log.info(
            "Depth %s %dx%d — skipping cache (size %d > %d pixels)",
            app.state.depth_model_id, img.width, img.height,
            normalised.size, DEPTH_CACHE_MAX_PIXELS,
        )

    log.info(
        "Depth %s %dx%d in %.2fs (cache: %d entries)",
        app.state.depth_model_id,
        img.width, img.height,
        time.perf_counter() - t0,
        len(DEPTH_CACHE),
    )
    return normalised


# ─── High-precision matting refinement (hair-level edges) ────────────────────
# SAM 3 returns a coarse object mask; to reach pixel precision on fine structure
# (hair, fur, thin/wispy edges) we build a trimap from the binary mask (erode →
# sure-foreground, dilate → unknown band, outside → sure-background) and solve a
# matting problem guided by the RGB image, so the soft alpha follows real image
# gradients inside the unknown band. PyMatting closed-form is the quality path; a
# pure-cv2 guided filter is the no-extra-dependency fallback.

def _disk(px: int) -> np.ndarray:
    px = max(1, int(px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))


def _make_trimap(binary_u8: np.ndarray, erode_px: int, dilate_px: int) -> np.ndarray:
    """Binary mask (0/255) → trimap float32: 1.0 sure-fg, 0.0 sure-bg, 0.5 unknown.
    A wide dilate band gives the matting solver room to recover hair / fine strands
    that extend past the coarse SAM mask boundary."""
    m = (binary_u8 > 127).astype(np.uint8) * 255
    fg = cv2.erode(m, _disk(erode_px))
    near = cv2.dilate(m, _disk(dilate_px))
    tri = np.full(m.shape, 0.5, np.float32)
    tri[near == 0] = 0.0
    tri[fg == 255] = 1.0
    return tri


def _guided_filter_np(I: np.ndarray, p: np.ndarray, r: int = 8, eps: float = 1e-4) -> np.ndarray:
    """Edge-aware guided filter (He et al.) — pure numpy/cv2, the fast matting
    fallback. I = grayscale guide [0,1], p = trimap [0,1]."""
    k = (2 * r + 1, 2 * r + 1)
    blur = lambda x: cv2.blur(x, k)
    mean_I, mean_p = blur(I), blur(p)
    var_I = blur(I * I) - mean_I * mean_I
    cov_Ip = blur(I * p) - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return blur(a) * I + blur(b)


def _fine_matte(rgb: np.ndarray, tri: np.ndarray, method: str) -> np.ndarray:
    """Trimap + RGB → refined alpha float32 [0,1]."""
    if method.startswith("pymatting"):
        from pymatting import estimate_alpha_cf, estimate_alpha_knn  # type: ignore
        img = rgb.astype(np.float64) / 255.0
        fn = estimate_alpha_knn if "knn" in method else estimate_alpha_cf
        return np.clip(fn(img, tri.astype(np.float64)), 0.0, 1.0).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    q = _guided_filter_np(gray, tri.astype(np.float32))
    return np.where(tri == 1.0, 1.0, np.where(tri == 0.0, 0.0, np.clip(q, 0.0, 1.0))).astype(np.float32)


def _refine_alpha_with_matting(alpha_u8: np.ndarray, rgb: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """Refine a coarse SAM 3 binary mask into a hair-accurate soft alpha via a
    trimap + matting solve guided by the RGB image. Runs at SAM3_REFINE_MAX_SIDE
    for CPU perf / RAM, then upsamples the alpha. Returns HxW uint8; falls back to
    the input alpha on any error or if cv2 is missing."""
    if not SAM3_REFINE_MATTING or cv2 is None:
        return alpha_u8
    try:
        if rgb is None or getattr(rgb, "ndim", 0) != 3:
            return alpha_u8
        H, W = binary_mask.shape[:2]
        bin255 = (binary_mask > 0).astype(np.uint8) * 255
        ms = SAM3_REFINE_MAX_SIDE
        scale = min(1.0, ms / float(max(H, W))) if ms > 0 else 1.0
        if scale < 1.0:
            w2, h2 = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
            rgb_s = cv2.resize(rgb, (w2, h2), interpolation=cv2.INTER_AREA)
            bin_s = cv2.resize(bin255, (w2, h2), interpolation=cv2.INTER_NEAREST)
            er = max(2, int(round(SAM3_REFINE_ERODE * scale)))
            di = max(3, int(round(SAM3_REFINE_DILATE * scale)))
        else:
            rgb_s, bin_s, er, di = rgb, bin255, SAM3_REFINE_ERODE, SAM3_REFINE_DILATE
        tri = _make_trimap(bin_s, er, di)
        alpha_f = _fine_matte(np.ascontiguousarray(rgb_s), tri, SAM3_REFINE_METHOD)
        out = (np.clip(alpha_f, 0.0, 1.0) * 255.0).astype(np.uint8)
        if scale < 1.0:
            out = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
        return out
    except Exception as e:
        log.warning("SAM 3 matting refine failed (%s); using coarse alpha", e)
        return alpha_u8


# ─── Saliency-matte cleanup ──────────────────────────────────────────────────

def clean_matte(matte_u8: np.ndarray, rgb: "np.ndarray | None" = None) -> np.ndarray:
    """Clean a saliency matte into a complete, solid subject mask while
    preserving soft edges.

    This is the fix for under-segmented subjects (e.g. a backlit fig leaf whose
    translucent lobes/veins the model drops, leaving Swiss-cheese holes and
    floating specks). Strategy:

      1. LOW threshold (alpha>24) -> binary capturing faint/semi-opaque regions.
      2. binary_fill_holes -> fill interior translucent holes (veins/center).
      3. small morphological CLOSE -> bridge tiny gaps without swallowing real
         concavities (leaf lobes must stay separated).
      4. connected-component speck removal by ABSOLUTE+RELATIVE area floor —
         keeps legitimately-detached real fragments, drops detection noise.
         (NOT largest-only: a leaf can have separated tip pieces.)
      5. gate the ORIGINAL soft matte by the dilated cleaned binary (WHERE),
         then solidify ONLY genuine interior holes (geometry-exact, from
         fill_holes) and faint DEEP-interior pixels (distance-transform gated)
         to 255 — so anti-aliased edges survive verbatim; the binary decides
         WHERE, the soft matte decides the edge profile.

    HxW uint8 in -> HxW uint8 out. ~240 ms at 2048² on a single CPU core.
    Returns the input untouched if OpenCV/SciPy are unavailable.
    """
    if not _MATTE_CLEANUP:
        return matte_u8
    if matte_u8.ndim != 2:
        matte_u8 = matte_u8[..., 0]
    h, w = matte_u8.shape
    matte = matte_u8  # keep original soft matte untouched

    # Kernel size scales with the image so behaviour is resolution-independent.
    # ~0.35% of the smaller side, clamped to [3, 9] and forced odd. At 1024 ->
    # 5px: bridges 1-2px sampling gaps but is far smaller than a leaf-lobe gap
    # (tens of px), so genuine concavities between lobes survive.
    k = int(round(min(h, w) * 0.0035))
    k = max(3, min(9, k))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # 1. LOW threshold so faint/semi-opaque halo regions are captured.
    binary = (matte > 24).astype(np.uint8)
    if not binary.any():
        return matte  # nothing salient — leave as-is (empty stays empty)

    # 2. Bridge tiny outline gaps (CLOSE = dilate then erode -> no net growth,
    #    so concave gaps wider than the kernel are NOT bridged). Holes stay open.
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3. Connected-component speck removal -> the subject CORE (holes still
    #    open). Floor = max(0.05% of image, 64px): the relative term scales with
    #    resolution, the absolute floor keeps the threshold sane on tiny images.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_area = max(int(0.0005 * h * w), 64)
    keep = np.zeros(num, dtype=bool)
    keep[0] = False  # background label 0 is never kept
    areas = stats[:, cv2.CC_STAT_AREA]
    keep[1:] = areas[1:] >= min_area
    core = keep[labels].astype(np.uint8)
    if not core.any():
        # Every component fell below the floor (tiny subject). Fall back to the
        # threshold binary so we never return an empty mask for a real subject.
        core = binary

    # 4. The subject's solid extent = core with interior holes filled. This is
    #    the WHERE region (and the basis for distance/depth). The holes it adds
    #    over `core` are the interior holes — but we only FILL the SMALL ones
    #    (model drop-outs); large holes are GENUINE see-through gaps (donut/ring,
    #    eyeglass lens, the gap between an arm and the torso) and stay transparent.
    clean_bin = ndimage.binary_fill_holes(core).astype(np.uint8)
    subject_area = int(clean_bin.sum()) or 1
    holes = ((clean_bin > 0) & (core == 0)).astype(np.uint8)
    small_holes = np.zeros((h, w), dtype=bool)
    if holes.any():
        hn, hlabels, hstats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
        max_hole_area = MATTE_HOLE_FILL_MAX_FRAC * subject_area
        small_ids = [i for i in range(1, hn) if hstats[i, cv2.CC_STAT_AREA] <= max_hole_area]
        if small_ids:
            small_holes = np.isin(hlabels, small_ids)

    # 5. Composite. WHERE from clean_bin; EDGE PROFILE from the soft matte. A
    #    genuine large hole keeps matte~0 (gated below) and is never solidified.
    region = cv2.dilate(clean_bin, kernel, iterations=1)
    gated = np.where(region > 0, matte, 0).astype(np.uint8)
    out = gated.copy()
    out[small_holes] = 255  # fill small model drop-outs only

    # Faint recovery: only when the subject is mostly solid (so a translucent
    # subject is left alone), and only for DEEP near-dropped pixels (alpha in
    # (24, FAINT_MAX]) — so genuine mid/low-alpha translucency is preserved and
    # a fully-zero genuine hole (alpha 0, fails matte>24) is never touched.
    solid_frac = float(((matte >= 128) & (clean_bin > 0)).sum()) / subject_area
    if solid_frac >= MATTE_FAINT_MIN_SOLID_FRAC:
        dist = cv2.distanceTransform(clean_bin, cv2.DIST_L2, 3)
        deep = dist > (3.0 * k)  # covers a Gaussian rim up to sigma~k
        faint_core = deep & (matte > 24) & (matte <= MATTE_FAINT_RECOVER_MAX)
        out[faint_core] = 255
    return out


# ─── SAM 3 helpers ───────────────────────────────────────────────────────────

_SAM3_LOCK = threading.Lock()
_SAM3_INFER_LOCK = threading.Lock()


def _sam3_loadable() -> bool:
    """Cheap probe: is the official SAM 3 package importable?"""
    try:
        return bool(importlib.util.find_spec("sam3") and importlib.util.find_spec("torch"))
    except Exception:
        return False


def _sam3_device_label() -> str:
    """SAM 3's builder reliably handles CUDA and CPU. Avoid MPS model/tensor
    mismatches in the upstream helper by using CPU on Apple Silicon."""
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sam3_checkpoint_kwargs():
    if SAM3_CHECKPOINT_PATH:
        return {"checkpoint_path": SAM3_CHECKPOINT_PATH, "load_from_HF": False}
    # The current preferred checkpoint is SAM 3.1. Older upstream builders only
    # download SAM 3 by default, so resolve 3.1 explicitly when requested.
    if SAM3_MODEL_ID.lower().replace("_", ".") in {"sam3.1", "facebook/sam3.1"}:
        from sam3.model_builder import download_ckpt_from_hf  # type: ignore

        return {
            "checkpoint_path": download_ckpt_from_hf(version="sam3.1"),
            "load_from_HF": False,
        }
    return {"checkpoint_path": None, "load_from_HF": True}


_SAM3_CPU_PATCHED = False


def _patch_sam3_fused_mlp_for_cpu() -> None:
    """Two upstream sam3/torch issues only reproduce on CPU on Apple Silicon;
    patch both once before the first CPU load.

    1. sam3.model.vitdet's MLP block always routes fc1+activation through
       sam3.perflib.fused.addmm_act, which unconditionally casts to bfloat16
       for GPU tensor-core throughput. On CPU this leaves fc1's output in
       bf16 while fc2's float32 weights are untouched, crashing every
       forward pass with "mat1 and mat2 must have the same dtype". Replace
       the symbol vitdet.py actually calls with a plain float32 path.

    2. sam3.model.geometry_encoders calls tensor.pin_memory() before moving
       box-scale tensors onto the inference device. On this machine (torch
       2.12.0 + Apple Silicon, MPS backend available), pin_memory() alone -
       with no MPS tensor involved anywhere - crashes with "Attempted to set
       the storage of a tensor on device cpu to a storage on a different
       device mps:0" (reproduces from a bare `torch.tensor([1.]).pin_memory()`
       call). Pinning only speeds up host->CUDA transfers and is a no-op
       benefit-wise on CPU, so disable it process-wide while running SAM 3
       on CPU."""
    global _SAM3_CPU_PATCHED
    if _SAM3_CPU_PATCHED:
        return
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    import sam3.model.vitdet as vitdet_mod  # type: ignore

    def _addmm_act_cpu(activation, linear, mat1):
        x = linear(mat1)
        if activation in (F.relu, nn.ReLU):
            return F.relu(x)
        if activation in (F.gelu, nn.GELU):
            return F.gelu(x)
        raise ValueError(f"Unexpected activation {activation}")

    vitdet_mod.addmm_act = _addmm_act_cpu
    torch.Tensor.pin_memory = lambda self, *a, **kw: self
    _SAM3_CPU_PATCHED = True


def _ensure_triton_stub() -> None:
    """SAM 3's kernels (edt / nms / connected-components) do a bare ``import
    triton`` at module load, but Triton ships no wheel for macOS / Apple
    Silicon, so that import raises ``ModuleNotFoundError`` before SAM 3 can load
    and the service silently falls back to saliency. The actual CPU compute
    paths already exist -- ``sam3.perflib`` dispatches to numpy/skimage whenever
    tensors are not CUDA, and the fused MLP is patched above -- so the only
    missing piece is the ``triton`` symbol itself. Install a permissive stub
    that satisfies the import-time references; its callables are never executed
    because every dispatcher routes to its CPU branch off-GPU. No-op on a real
    CUDA host where the genuine Triton package is importable."""
    try:
        import triton  # noqa: F401  (real Triton present, e.g. a CUDA box)
        return
    except Exception:
        pass
    # Load torch's compile stack (and torchvision, which pulls it in) WHILE
    # Triton is genuinely absent, so torch._inductor records "no Triton" and
    # won't later try to import real-Triton submodules (triton.backends.*)
    # through our stub. SAM 3's import drags in torchvision ->
    # torch._dynamo -> torch._inductor, which probes Triton; doing it here
    # first makes that probe resolve to "absent" and cache it.
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import torch._inductor.runtime.hints  # noqa: F401
    except Exception:
        pass
    import sys
    import types

    class _Dummy:
        # Usable as @triton.jit (returns the fn), @triton.autotune(...) (returns
        # a decorator), triton.Config(...) / triton.cdiv(...) (returns self), and
        # is subscriptable / iterable so nothing raises at decoration time.
        def __call__(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return self

        def __getattr__(self, name):
            return self

        def __getitem__(self, key):
            return self

        def __iter__(self):
            return iter(())

    _dummy = _Dummy()

    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            return _dummy

    triton_mod = _Stub("triton")
    triton_mod.__path__ = []  # mark as a package so `import triton.language` resolves
    triton_mod.__version__ = "3.0.0"  # satisfy any version probes
    triton_mod.__file__ = "triton_stub/__init__.py"
    triton_mod.jit = _dummy
    triton_mod.autotune = _dummy
    triton_mod.heuristics = _dummy
    triton_mod.Config = _dummy
    triton_mod.cdiv = lambda a, b: (int(a) + int(b) - 1) // int(b)
    language_mod = _Stub("triton.language")
    language_mod.__file__ = "triton_stub/language.py"
    language_mod.constexpr = _dummy
    triton_mod.language = language_mod
    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = language_mod
    log.warning(
        "Triton is unavailable (expected on macOS/Apple Silicon); installed a CPU "
        "stub so SAM 3 can load. Inference runs on CPU and will be slow."
    )


_SAM3_CUDA_REDIRECTED = False


def _redirect_cuda_to_cpu() -> None:
    """sam3 hardcodes ``device="cuda"`` / ``.cuda()`` in several spots that run on
    the IMAGE path regardless of the requested device — e.g.
    ``position_encoding.PositionEmbeddingSine`` precompute, ``vl_combiner``,
    ``decoder`` — which crashes on a CUDA-less torch (macOS / Apple Silicon) with
    "Torch not compiled with CUDA enabled". On a host without CUDA, coerce every
    cuda tensor placement to cpu. This only changes behaviour for cuda requests,
    which would otherwise hard-crash here, so it cannot affect the working
    CPU / CoreML paths. No-op on a real CUDA box."""
    global _SAM3_CUDA_REDIRECTED
    if _SAM3_CUDA_REDIRECTED:
        return
    import torch  # type: ignore

    if torch.cuda.is_available():
        return

    def _coerce(device):
        if device is None:
            return device
        try:
            text = str(device)
        except Exception:
            return device
        return "cpu" if text.startswith("cuda") else device

    for _name in (
        "zeros", "ones", "empty", "full", "tensor", "as_tensor", "arange",
        "randn", "rand", "randint", "zeros_like", "ones_like", "linspace",
    ):
        _orig = getattr(torch, _name, None)
        if _orig is None:
            continue

        def _make(orig):
            def _wrapped(*args, **kwargs):
                if "device" in kwargs:
                    kwargs["device"] = _coerce(kwargs["device"])
                return orig(*args, **kwargs)

            return _wrapped

        setattr(torch, _name, _make(_orig))

    # TorchScript's frontend rejects the *args/**kwargs wrappers above, and
    # sam3's interactive predictor scripts its transform pipeline at build time
    # (SAM2Transforms: torch.jit.script(...) → torchvision Normalize →
    # torch.as_tensor, now a wrapper). Off-GPU, scripting buys nothing over
    # eager — and eager keeps the device coercions applying — so hand back the
    # object unscripted. No-op on a real CUDA host (guarded above).
    torch.jit.script = lambda obj, *a, **k: obj

    _orig_to = torch.Tensor.to

    def _tensor_to(self, *args, **kwargs):
        args = tuple(
            _coerce(a) if isinstance(a, (str, torch.device)) else a for a in args
        )
        if "device" in kwargs:
            kwargs["device"] = _coerce(kwargs["device"])
        return _orig_to(self, *args, **kwargs)

    torch.Tensor.to = _tensor_to
    torch.Tensor.cuda = lambda self, *a, **k: self  # already cpu -> stay cpu
    # pin_memory() pins to the MPS "accelerator" on Apple Silicon, yielding an
    # mps:0 storage that then collides with cpu tensors (e.g. sam3
    # geometry_encoders._encode_boxes: `scale.pin_memory().to(cpu)`). Pinning
    # only helps async host->GPU copies we never do on cpu, so make it a no-op.
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    try:
        torch.cuda.current_device = lambda: 0  # avoid raise in unguarded callers
    except Exception:
        pass

    _SAM3_CUDA_REDIRECTED = True
    log.warning(
        "CUDA unavailable; redirecting SAM 3's hardcoded cuda tensor placement to "
        "cpu. Inference runs on CPU (slow). No-op on a real CUDA host."
    )


def _load_sam3(app: FastAPI) -> bool:
    """Load SAM 3 into app.state (blocking). Caller holds _SAM3_LOCK."""
    if not SAM3_ENABLE:
        app.state.sam3_load_failed = True
        return False
    try:
        _ensure_triton_stub()
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
        from sam3.model_builder import build_sam3_image_model  # type: ignore

        device = _sam3_device_label()
        if device == "cpu":
            _patch_sam3_fused_mlp_for_cpu()
            _redirect_cuda_to_cpu()
        log.info("loading SAM 3 model %r onto %s ...", SAM3_MODEL_ID, device)
        t0 = time.perf_counter()
        kwargs = _sam3_checkpoint_kwargs()
        model = build_sam3_image_model(
            device=device,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            **kwargs,
        )
        processor = Sam3Processor(
            model,
            device=device,
            confidence_threshold=SAM3_CONFIDENCE,
        )
        app.state.sam3_model = model
        app.state.sam3_processor = processor
        app.state.sam3_device = device
        app.state.sam3_model_id = SAM3_MODEL_ID
        app.state.sam3_available = True
        log.info("SAM 3 ready in %.1fs on %s", time.perf_counter() - t0, device)
        return True
    except ImportError:
        log.warning(
            "SAM 3 package not installed; SAM 3-backed routes will fall back. "
            "Install facebookresearch/sam3 and authenticate with Hugging Face "
            "for the gated checkpoints."
        )
        app.state.sam3_load_failed = True
        return False
    except Exception as e:  # pragma: no cover - defensive
        log.exception("failed to load SAM 3: %s", e)
        app.state.sam3_load_failed = True
        return False


def _ensure_sam3(app: FastAPI):
    """Lazily load SAM 3; return True/'cold' if ready, False if unavailable."""
    if getattr(app.state, "sam3_available", False):
        return True
    if getattr(app.state, "sam3_load_failed", False):
        return False
    with _SAM3_LOCK:
        if getattr(app.state, "sam3_available", False):
            return True
        if getattr(app.state, "sam3_load_failed", False):
            return False
        ok = _load_sam3(app)
        return "cold" if ok else False


def _bbox_of(mask: np.ndarray) -> "list[int]":
    """Tight [x, y, w, h] bounding box of a boolean mask (mask is non-empty)."""
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _mask_png_b64(alpha: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(alpha, "L").save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _soft_alpha_from_mask(mask: np.ndarray) -> np.ndarray:
    alpha = (mask.astype(np.uint8)) * 255
    return np.asarray(
        Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(1.0)),
        dtype=np.uint8,
    )


def _instances_from_sam3_output(output: dict, label: str) -> "list[dict]":
    masks_t = output.get("masks")
    if masks_t is None:
        return []
    masks = masks_t.detach().cpu().numpy() if hasattr(masks_t, "detach") else np.asarray(masks_t)
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, :, :]

    scores_t = output.get("scores")
    scores = (
        scores_t.detach().cpu().numpy().reshape(-1)
        if hasattr(scores_t, "detach")
        else np.asarray(scores_t if scores_t is not None else [], dtype=np.float32).reshape(-1)
    )

    out: "list[dict]" = []
    for i, raw in enumerate(masks):
        mb = raw > 0
        area = int(mb.sum())
        if area <= 0:
            continue
        out.append({
            "class_id": -1,
            "label": label,
            "confidence": float(scores[i]) if i < len(scores) else 1.0,
            "mask": mb,
            "area": area,
            "source": "sam3",
        })
    out.sort(key=lambda inst: -inst["area"])
    return out[:SAM3_INSTANCES_MAX]


def _sam3_instances_for_prompt(app, img: Image.Image, prompt: str) -> "list[dict]":
    processor = app.state.sam3_processor
    with _SAM3_INFER_LOCK:
        state = processor.set_image(img)
        output = processor.set_text_prompt(state=state, prompt=prompt)
    return _instances_from_sam3_output(output, prompt)


def _sam3_box_mask(app, img: Image.Image, box_xyxy: "tuple[float, float, float, float]"):
    """SAM 3.1 box-prompted selection of the single object inside a rectangle.

    `box_xyxy` is (x0, y0, x1, y1) in image pixels. Returns
    (mask uint8 HxW white=selected, score float). The mask is all-zero when
    SAM finds no confident object in the box. Uses the detector's geometric
    prompt (Sam3Processor.add_geometric_prompt), which expects the box as
    [cx, cy, w, h] normalised to 0..1 — the strongest single-object prompt
    SAM 3 supports on a still image.
    """
    import torch  # type: ignore

    W, H = img.width, img.height
    x0, y0, x1, y1 = box_xyxy
    box_norm = [
        ((x0 + x1) / 2.0) / W,
        ((y0 + y1) / 2.0) / H,
        (x1 - x0) / W,
        (y1 - y0) / H,
    ]
    processor = app.state.sam3_processor
    with _SAM3_INFER_LOCK:
        state = processor.set_image(img)
        out = processor.add_geometric_prompt(box=box_norm, label=True, state=state)

    masks = out.get("masks")
    scores = out.get("scores")
    if masks is None or len(masks) == 0:
        return np.zeros((H, W), dtype=np.uint8), 0.0

    s = scores.detach().cpu().numpy().reshape(-1) if scores is not None else np.array([0.0])
    best = int(s.argmax())
    m = masks[best].detach().cpu().numpy()
    m = np.squeeze(m)
    # `masks` may be probabilities (0..1) or raw logits; threshold accordingly.
    binary = m > (0.0 if float(m.min()) < 0.0 else 0.5)
    return (binary.astype(np.uint8) * 255), float(s[best])


def _sam3_ground_results(app, img: Image.Image, phrases: "list[str]", rgb: np.ndarray) -> "list[dict]":
    processor = app.state.sam3_processor
    frame_area = float(img.width * img.height) or 1.0
    results = []

    with _SAM3_INFER_LOCK:
        state = processor.set_image(img)
        for phrase in phrases:
            output = processor.set_text_prompt(state=state, prompt=phrase)
            instances = _instances_from_sam3_output(output, phrase)
            if not instances:
                results.append({
                    "phrase": phrase,
                    "found": False,
                    "score": 0.0,
                    "coverage": 0.0,
                    "bbox": None,
                    "components": 0,
                    "refined": True,
                    "source": "sam3",
                    "maskPng": None,
                })
                continue

            union = _union_from_instances(instances, img.width, img.height) > 0
            alpha = _soft_alpha_from_mask(union)
            if SAM3_REFINE_MATTING:
                alpha = _refine_alpha_with_matting(alpha, rgb, union)
            if _MATTE_CLEANUP:
                try:
                    alpha = clean_matte(alpha, rgb)
                except Exception:
                    log.exception("clean_matte failed after SAM 3 grounding; using raw SAM 3 mask")
            bbox = _bbox_from_mask(alpha > 127)
            results.append({
                "phrase": phrase,
                "found": True,
                "score": round(max(float(i["confidence"]) for i in instances), 4),
                "coverage": round(float((alpha > 127).sum()) / frame_area, 4),
                "bbox": list(bbox) if bbox else None,
                "components": len(instances),
                "refined": True,
                "source": "sam3",
                "maskPng": _mask_png_b64(alpha),
            })

    return results


def _union_from_instances(instances: "list[dict]", width: int, height: int) -> np.ndarray:
    union = np.zeros((height, width), dtype=np.uint8)
    for inst in instances:
        union[inst["mask"]] = 255
    return union


def _saliency_instance(label: str, matte: np.ndarray) -> "dict | None":
    mb = matte > 127
    if not mb.any():
        return None
    return {
        "class_id": -1,
        "label": label,
        "confidence": 1.0,
        "mask": mb,
        "area": int(mb.sum()),
        "source": "saliency",
    }


# ─── Lazy model loaders (Depth) ──────────────────────────────────────────────
# These hold the heavy torch models. By default they load on first use so the
# resident footprint stays small. Loads are serialised per-model with a lock so
# two concurrent first-requests don't both load. A permanent failure (e.g. torch
# not installed) is remembered so we don't retry the heavy load on every request.

_DEPTH_LOCK = threading.Lock()


def _torch_stack_loadable() -> bool:
    """Cheap capability probe: are torch + transformers importable WITHOUT
    actually importing them (the import is the heavy ~200 MB+ cost)?"""
    try:
        return bool(
            importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")
        )
    except Exception:  # pragma: no cover - find_spec on a broken install
        return False


def _load_depth(app: FastAPI) -> bool:
    """Load Depth Anything V2 into app.state (blocking). Caller holds _DEPTH_LOCK."""
    try:
        import torch  # type: ignore  # noqa: F401
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # type: ignore

        device, device_label = detect_torch_device()
        if device is None:
            raise ImportError("torch not available")

        log.info("loading Depth model %r onto %s ...", DEPTH_MODEL_ID, device_label)
        t2 = time.perf_counter()
        app.state.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        app.state.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(device)
        app.state.depth_model.eval()
        app.state.depth_device = device
        app.state.depth_model_id = DEPTH_MODEL_ID
        app.state.depth_available = True
        log.info("Depth ready in %.1fs on %s", time.perf_counter() - t2, device_label)
        return True
    except ImportError:
        log.warning("torch / transformers not installed; /depth disabled.")
        app.state.depth_load_failed = True
        return False
    except Exception as e:  # pragma: no cover - defensive
        log.exception("failed to load Depth model: %s", e)
        app.state.depth_load_failed = True
        return False


def _ensure_depth(app: FastAPI):
    """Lazily load Depth Anything V2; return True/'cold' if ready, False if failed.
    Returns 'cold' on first load (model just downloaded/loaded)."""
    if app.state.depth_available:
        return True
    if getattr(app.state, "depth_load_failed", False):
        return False
    with _DEPTH_LOCK:
        if app.state.depth_available:
            return True
        if getattr(app.state, "depth_load_failed", False):
            return False
        ok = _load_depth(app)
        return "cold" if ok else False


# ─── LaMa inpainting ─────────────────────────────────────────────────────────

_LAMA_LOCK = threading.Lock()


def _lama_loadable() -> bool:
    """Cheap probe: is simple-lama-inpainting importable?"""
    try:
        return bool(importlib.util.find_spec("simple_lama_inpainting"))
    except Exception:
        return False


def _load_lama(app: FastAPI) -> bool:
    """Load LaMa into app.state (blocking). Caller holds _LAMA_LOCK."""
    try:
        import torch  # type: ignore
        from simple_lama_inpainting import SimpleLama  # type: ignore

        log.info("loading LaMa inpainting model ...")
        t0 = time.perf_counter()

        # The LaMa checkpoint was saved with CUDA tensors. On machines without
        # CUDA (e.g. Mac / CPU-only Linux) torch.jit.load fails because it
        # tries to deserialise onto the CUDA backend which doesn't exist.
        # Monkey-patch torch.jit.load to force map_location='cpu' so the
        # checkpoint is remapped transparently.
        _orig_jit_load = torch.jit.load
        def _cpu_jit_load(f, *args, **kw):
            # Absorb ANY caller-supplied map_location — positional (args[0],
            # swallowed by *args and not forwarded) or keyword (popped from kw)
            # — then force "cpu". The previous signature only caught a
            # positional value; simple-lama calls `torch.jit.load(path,
            # map_location=device)` with the KEYWORD form, which slipped into
            # **kw and collided with the hardcoded map_location="cpu"
            # ("got multiple values for keyword argument 'map_location'").
            kw.pop("map_location", None)
            return _orig_jit_load(f, map_location="cpu", **kw)
        torch.jit.load = _cpu_jit_load
        try:
            app.state.lama_model = SimpleLama()
        finally:
            torch.jit.load = _orig_jit_load

        app.state.lama_available = True
        log.info("LaMa ready in %.1fs", time.perf_counter() - t0)
        return True
    except ImportError:
        log.warning(
            "simple-lama-inpainting not installed; /inpaint disabled. "
            "Install with: pip install simple-lama-inpainting"
        )
        app.state.lama_load_failed = True
        return False
    except Exception as e:
        log.exception("failed to load LaMa: %s", e)
        app.state.lama_load_failed = True
        return False


def _ensure_lama(app: FastAPI):
    """Lazily load LaMa; return True/'cold' if ready, False if failed.
    Returns 'cold' on first load (model just downloaded/loaded)."""
    if getattr(app.state, "lama_available", False):
        return True
    if getattr(app.state, "lama_load_failed", False):
        return False
    with _LAMA_LOCK:
        if getattr(app.state, "lama_available", False):
            return True
        if getattr(app.state, "lama_load_failed", False):
            return False
        ok = _load_lama(app)
        return "cold" if ok else False


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading rembg model %r ...", MODEL_NAME)
    t0 = time.perf_counter()
    providers = detect_providers()
    log.info("execution providers: %s", providers)

    try:
        app.state.session = new_session(MODEL_NAME, providers=providers)
        app.state.model_name = MODEL_NAME
        app.state.providers = providers
        log.info("model %r ready in %.1fs", MODEL_NAME, time.perf_counter() - t0)
    except Exception as e:  # pragma: no cover - defensive
        log.exception("failed to load %r: %s", MODEL_NAME, e)
        log.info("falling back to u2net (CPU only)")
        app.state.session = new_session("u2net", providers=["CPUExecutionProvider"])
        app.state.model_name = "u2net"
        app.state.providers = ["CPUExecutionProvider"]

    # SAM 3.1 (subject / text-grounding / box-select) and Depth load lazily
    # unless warmed; record model ids up front for /health and headers.
    app.state.sam3_available = False
    app.state.sam3_load_failed = not SAM3_ENABLE
    app.state.sam3_model_id = SAM3_MODEL_ID
    app.state.depth_available = False
    app.state.depth_load_failed = False
    app.state.depth_model_id = DEPTH_MODEL_ID
    app.state.lama_available = False
    app.state.lama_load_failed = False

    if SAM3_EAGER:
        await run_in_threadpool(_ensure_sam3, app)

    if SEGMENT_EAGER_MODELS:
        await run_in_threadpool(_ensure_depth, app)
    else:
        log.info(
            "SAM 3 and Depth will load lazily on first use "
            "(set SAM3_EAGER=1 or SEGMENT_EAGER_MODELS=1 to preload)."
        )

    yield

    DEPTH_CACHE.clear()
    log.info("shutting down mask service")


app = FastAPI(
    title="Phosmith Mask Service",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


# ─── Hardening middleware ────────────────────────────────────────────────────

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Reject oversize POSTs before the body is read (defends against
    memory/disk exhaustion: a 10 GB upload would otherwise be spooled
    to disk by Starlette before any size check runs).
    """
    if request.method == "POST":
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
            return Response(
                content=(
                    f"file too large (Content-Length: {int(cl) // (1024*1024)}MB"
                    f" > {MAX_UPLOAD_MB}MB)"
                ),
                status_code=413,
            )
    return await call_next(request)


async def _read_limited(image: UploadFile) -> bytes:
    """Stream-read an UploadFile into memory, aborting if it exceeds the
    upload limit (defends against chunked uploads that bypass the
    Content-Length middleware above). Reads the underlying SpooledTemporaryFile
    in 1 MB chunks via the threadpool so we never block the event loop
    and never buffer the whole body before checking size.
    """
    contents = bytearray()
    while True:
        chunk = await run_in_threadpool(image.file.read, 1024 * 1024)
        if not chunk:
            break
        contents.extend(chunk)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413,
                f"file too large (> {MAX_UPLOAD_MB}MB)",
            )
    return bytes(contents)


def _parse_shape_points(raw: str, width: int, height: int) -> list[tuple[float, float]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"invalid points JSON: {e}")

    if not isinstance(data, list) or len(data) < 3:
        raise HTTPException(400, "points must be an array with at least 3 [x, y] pairs")
    if len(data) > SHAPE_MASK_MAX_POINTS:
        raise HTTPException(
            400,
            f"too many points: {len(data)} > {SHAPE_MASK_MAX_POINTS}",
        )

    points: list[tuple[float, float]] = []
    for i, point in enumerate(data):
        if not (isinstance(point, list) and len(point) == 2):
            raise HTTPException(400, f"point #{i} must be [x, y]; got {point!r}")
        x, y = point
        if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
            raise HTTPException(400, f"point #{i} x must be a finite number; got {x!r}")
        if not isinstance(y, (int, float)) or isinstance(y, bool) or not math.isfinite(y):
            raise HTTPException(400, f"point #{i} y must be a finite number; got {y!r}")
        if not (-1 <= x <= width + 1 and -1 <= y <= height + 1):
            raise HTTPException(
                400,
                f"point #{i} ({x}, {y}) is outside mask bounds ({width}x{height})",
            )
        points.append((float(x), float(y)))
    return points


def _rasterize_shape_mask(width: int, height: int, points: list[tuple[float, float]]) -> tuple[bytes, str]:
    """Fill a closed path to an RGBA PNG mask.

    White+alpha inside the polygon and transparent outside matches the editor's
    brush texture contract: plain brush layers sample alpha, smart-brush layers
    sample the red channel.
    """
    if cv2 is not None:
        mask = np.zeros((height, width, 4), dtype=np.uint8)
        poly = np.array(
            [[[int(round(x)), int(round(y))] for x, y in points]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, poly, color=(255, 255, 255, 255), lineType=cv2.LINE_AA)
        out = Image.fromarray(mask, "RGBA")
        engine = "opencv-fillpoly"
    else:
        out = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(out)
        draw.polygon(points, fill=(255, 255, 255, 255))
        engine = "pillow-polygon"

    buf = io.BytesIO()
    out.save(buf, format="PNG", compress_level=1)
    return buf.getvalue(), engine


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> dict:
    return {
        "status": "ok",
        "message": "Phosmith Mask Service is running"
    }

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model": app.state.model_name,
        "providers": app.state.providers,
        "max_upload_mb": MAX_UPLOAD_MB,
        "subject_engine": "sam3" if getattr(app.state, "sam3_available", False) else "saliency",
        "subject_prompt": SAM3_SUBJECT_PROMPT,
        "matte_cleanup": _MATTE_CLEANUP,
        "lazy_models": not SEGMENT_EAGER_MODELS,
        # `*_available` = CAPABLE of serving the endpoint (already loaded, or
        # loadable on first use). `*_loaded` = the heavy model is resident now.
        "sam3_available": getattr(app.state, "sam3_available", False)
        or (SAM3_ENABLE and _sam3_loadable() and not getattr(app.state, "sam3_load_failed", False)),
        "sam3_loaded": getattr(app.state, "sam3_available", False),
        "sam3_model": app.state.sam3_model_id,
        "sam3_confidence": SAM3_CONFIDENCE,
        "depth_available": app.state.depth_available
        or (_torch_stack_loadable() and not app.state.depth_load_failed),
        "depth_loaded": app.state.depth_available,
        "depth_model": app.state.depth_model_id,
        "shape_mask_engine": "opencv-fillpoly" if cv2 is not None else "pillow-polygon",
        "lama_available": getattr(app.state, "lama_available", False)
        or _lama_loadable(),
        "lama_loaded": getattr(app.state, "lama_available", False),
    }


@app.post("/warmup")
async def warmup() -> dict:
    """Pre-load all lazy models in background so the first real request is fast.

    Call this once after the Space starts to avoid cold-load latency on the
    first user interaction. Returns which models were loaded (vs already warm).
    """
    results = {}

    # SAM 3
    sam3_was_loaded = getattr(app.state, "sam3_available", False)
    sam3_status = await run_in_threadpool(_ensure_sam3, app)
    results["sam3"] = (
        "already_loaded" if sam3_was_loaded
        else "loaded" if sam3_status else "failed"
    )

    # Depth Anything V2
    depth_was_loaded = app.state.depth_available
    depth_status = await run_in_threadpool(_ensure_depth, app)
    results["depth"] = (
        "already_loaded" if depth_was_loaded
        else "loaded" if depth_status else "failed"
    )

    # LaMa inpainting
    lama_was_loaded = getattr(app.state, "lama_available", False)
    lama_status = _ensure_lama(app)
    results["lama"] = (
        "already_loaded" if lama_was_loaded
        else "loaded" if lama_status else "failed"
    )

    log.info("warmup results: %s", results)
    return {"status": "ok", "models": results}


@app.post("/inpaint")
async def inpaint(
    image: UploadFile = File(..., alias="image"),
    mask: UploadFile = File(..., alias="mask"),
) -> Response:
    """Inpaint masked regions using LaMa.

    Accepts:
      - image: the source image (JPEG/PNG)
      - mask: a white-on-black mask where white = region to fill

    Returns the inpainted image as PNG.
    """
    _lama_status = _ensure_lama(app)
    if not _lama_status:
        raise HTTPException(
            501,
            "LaMa inpainting is not available. "
            "Install with: pip install simple-lama-inpainting",
        )
    _lama_cold = _lama_status == "cold"

    image_bytes = await _read_limited(image)
    mask_bytes = await _read_limited(mask)
    if not image_bytes or not mask_bytes:
        raise HTTPException(400, "empty upload")
    # Decode defensively (same as /crop/auto) — garbage bytes must 400, not 500.
    try:
        # exif_transpose BEFORE convert: camera JPEGs store rotation in EXIF,
        # and inpainting the raw pixels would return a result rotated relative
        # to what the caller (and their mask) sees. No-op without EXIF.
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        msk = Image.open(io.BytesIO(mask_bytes)).convert("L")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise HTTPException(400, f"could not decode image or mask: {e}")

    def _run(img=img, msk=msk):
        # Resize mask to match image if needed
        if msk.size != img.size:
            msk = msk.resize(img.size, Image.NEAREST)

        # LaMa expects mask > 0 = inpaint region. Threshold to binary.
        msk_np = np.array(msk)
        msk_np = (msk_np > 16).astype(np.uint8) * 255
        msk = Image.fromarray(msk_np, mode="L")

        result = app.state.lama_model(img, msk)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return buf.getvalue()

    from starlette.concurrency import run_in_threadpool  # type: ignore

    png_bytes = await run_in_threadpool(_run)
    resp_headers = {"Cache-Control": "no-store"}
    if _lama_cold:
        resp_headers["X-Cold-Load"] = "true"
        log.info("LaMa cold load — first request after model download")
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers=resp_headers,
    )


@app.post("/shape/fill")
async def shape_fill(
    width: int = Form(...),
    height: int = Form(...),
    points: str = Form(...),
) -> Response:
    """Rasterize a closed editor path into a filled RGBA mask.

    Form fields:
        width/height: target mask texture dimensions
        points: JSON array of `[x, y]` pairs in target-mask pixel coords

    Returns an RGBA PNG: white+alpha inside the path, transparent outside.
    """
    if width < 1 or height < 1:
        raise HTTPException(400, "width and height must be positive")
    if max(width, height) > SHAPE_MASK_MAX_SIDE:
        raise HTTPException(
            413,
            f"mask too large ({width}x{height}); max longest side is {SHAPE_MASK_MAX_SIDE}px",
        )

    parsed_points = _parse_shape_points(points, width, height)
    t0 = time.perf_counter()
    try:
        png, engine = await run_in_threadpool(_rasterize_shape_mask, width, height, parsed_points)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("shape fill failed")
        raise HTTPException(500, f"shape fill failed: {e}")

    elapsed = time.perf_counter() - t0
    log.info(
        "shape fill %dx%d (%d points) via %s in %.3fs -> %dB",
        width,
        height,
        len(parsed_points),
        engine,
        elapsed,
        len(png),
    )
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Model": engine,
            "X-Width": str(width),
            "X-Height": str(height),
            "X-Elapsed-Ms": str(int(elapsed * 1000)),
        },
    )



# ─── Auto-crop helpers ──────────────────────────────────────────────────────



def _parse_aspect(raw: "str | None") -> "float | None":
    """Parse an aspect ratio from `"W:H"`, `"W/H"`, or a bare float `"1.5"`.

    Returns `None` for falsy / unparseable input (callers treat that as
    "freeform — no aspect constraint"). Tolerates whitespace and the `x`
    separator (e.g. `"16x9"`).
    """
    if not raw:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    for sep in (":", "/", "x"):
        if sep in s:
            a, b = s.split(sep, 1)
            try:
                num = float(a.strip())
                den = float(b.strip())
                if den <= 0 or num <= 0:
                    return None
                return num / den
            except ValueError:
                return None
    try:
        v = float(s)
        return v if v > 0 else None
    except ValueError:
        return None


def _clip_box_to_image(
    box: "tuple[float, float, float, float]", W: int, H: int
) -> "tuple[int, int, int, int]":
    """Clip `(x, y, w, h)` to the image bounds and round to ints.

    Always returns a box of at least 1×1 inside the image — callers downstream
    rely on the box being non-degenerate.
    """
    x, y, w, h = box
    x = max(0.0, min(float(W) - 1, x))
    y = max(0.0, min(float(H) - 1, y))
    w = max(1.0, min(float(W) - x, w))
    h = max(1.0, min(float(H) - y, h))
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def _bbox_from_mask(mask: np.ndarray) -> "tuple[int, int, int, int] | None":
    """Tight bbox `(x, y, w, h)` of the True pixels in a boolean mask, or
    `None` if the mask is entirely empty."""
    if mask is None or mask.size == 0 or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _expand_box(
    box: "tuple[int, int, int, int]",
    W: int,
    H: int,
    pad_frac: float,
) -> "tuple[int, int, int, int]":
    """Expand a box by `pad_frac` of its SHORTER side, then clip to image."""
    x, y, w, h = box
    pad = max(0.0, pad_frac) * float(min(w, h))
    return _clip_box_to_image((x - pad, y - pad, w + 2 * pad, h + 2 * pad), W, H)


def _fit_aspect(
    box: "tuple[int, int, int, int]",
    aspect: float,
    W: int,
    H: int,
    *,
    anchor: "tuple[float, float] | None" = None,
) -> "tuple[int, int, int, int]":
    """Fit a box of the requested aspect ratio around `box`, keeping the box
    fully inside the image. Grows the shorter side (never crops the subject)
    unless we hit an edge — then shifts the box toward `anchor` to maximise
    subject inclusion.

    `anchor` is the (cx, cy) the result should try to centre on. Defaults to
    the input box centre.
    """
    x, y, w, h = box
    cur_aspect = w / float(h)
    if aspect >= cur_aspect:
        # Need wider — grow horizontally.
        new_w = h * aspect
        new_h = h
    else:
        # Need taller — grow vertically.
        new_w = w
        new_h = w / aspect

    # If growth exceeds image dims, shrink uniformly to fit (we ALWAYS prefer
    # the requested aspect over keeping the subject 1:1).
    scale = min(1.0, W / new_w, H / new_h)
    new_w *= scale
    new_h *= scale

    ax = anchor[0] if anchor else x + w / 2.0
    ay = anchor[1] if anchor else y + h / 2.0
    new_x = ax - new_w / 2.0
    new_y = ay - new_h / 2.0

    # Clip to image bounds (this shifts the rect rather than shrinking it).
    if new_x < 0:
        new_x = 0
    if new_y < 0:
        new_y = 0
    if new_x + new_w > W:
        new_x = W - new_w
    if new_y + new_h > H:
        new_y = H - new_h
    return _clip_box_to_image((new_x, new_y, new_w, new_h), W, H)


def _compose_anchor(
    centroid: "tuple[float, float]",
    bbox: "tuple[float, float, float, float]",
    crop_w: float,
    crop_h: float,
    W: int,
    H: int,
    snap_strength: float = CROP_THIRDS_SNAP,
) -> "tuple[float, float]":
    """Where a crop of size `(crop_w, crop_h)` should be CENTRED to compose the
    subject professionally.

    Two ideas a senior photo tool gets right and a naive "nudge to a fixed
    power point" gets wrong:

    1. **Lead room (a.k.a. nose room).** A subject sitting in the left half of
       the frame is almost always facing/moving toward the open space on its
       right, so the crop should leave room on THAT side — i.e. place the
       subject on the left third, not shove it against the right edge. We infer
       facing from the subject's position in the frame (the photographer's
       original placement) and pull it toward the corresponding third.

    2. **Never cut the subject, never trim lopsidedly.** The pull is clamped so
       the whole subject stays inside the crop with a small margin. When the
       crop is simply too small to contain the subject on an axis (a wide group
       forced to 1:1), we centre on the subject bbox on that axis so the
       unavoidable trim is symmetric instead of dropping everyone on one side.
    """
    cx, cy = centroid
    bx, by, bw, bh = bbox
    s = max(0.0, min(1.0, snap_strength))

    def axis(c, b0, bdim, cdim, dim, strength):
        # Lead-room target third, chosen from the subject's side of the frame.
        third = (1.0 / 3.0) if (b0 + bdim / 2.0) < dim / 2.0 else (2.0 / 3.0)
        a = c + (0.5 - third) * cdim * strength
        margin = 0.04 * cdim
        if cdim >= bdim + 2 * margin:
            # Clamp so the subject (plus margin) stays fully inside the crop.
            lo = b0 + bdim + margin - cdim / 2.0
            hi = b0 - margin + cdim / 2.0
            a = min(max(a, lo), hi)
        else:
            # Crop can't hold the subject on this axis → balanced centre cut.
            a = b0 + bdim / 2.0
        # Keep the crop itself inside the image.
        if cdim <= dim:
            a = min(max(a, cdim / 2.0), dim - cdim / 2.0)
        else:
            a = dim / 2.0
        return a

    # Horizontal lead room is the strong cue; vertical gets a gentler pull so we
    # don't shove a standing subject's feet against the bottom edge.
    ax = axis(cx, bx, bw, crop_w, W, s)
    ay = axis(cy, by, bh, crop_h, H, s * 0.5)
    return ax, ay


def _compute_subject_crop(
    subject_mask: "np.ndarray | None",
    W: int,
    H: int,
    *,
    aspect: "float | None" = None,
    target_frac: float = CROP_SUBJECT_TARGET_FRAC,
    centroid: "tuple[float, float] | None" = None,
    has_subject: bool = True,
) -> "dict | None":
    """Subject-aware crop = a COMPOSITION around the subject, not a subject
    extraction.

    Two regimes:

    1. A distinct subject (SAM 3 instance, or a saliency blob that doesn't sprawl
       across the frame): size the crop so the subject's longer side spans
       ~`target_frac` of it — leaving deliberate breathing room/context — then
       nudge to rule-of-thirds and fit `aspect`. This is the fix for "subject-
       aware shouldn't crop ONLY the subject": the surrounding scene is kept.

    2. No distinct subject (`has_subject=False`) and the salient region sprawls
       across the frame (a landscape/scene): DON'T zoom into the saliency blob.
       Compose the whole scene — a max-area `aspect` fit gently biased toward
       the salient centroid, or the full frame when freeform. This is the fix
       for scenes like a canyon-and-sky where there is no single subject.

    Returns `None` when no usable subject mask is supplied — the caller should
    fall back to content-fill or to the whole frame.
    """
    if subject_mask is None:
        return None
    mask_bool = subject_mask > 127 if subject_mask.dtype == np.uint8 else subject_mask
    bbox = _bbox_from_mask(mask_bool)
    if bbox is None:
        return None

    # Centroid drives both thirds-snap and the scene bias.
    if centroid is None:
        ys, xs = np.nonzero(mask_bool)
        if xs.size == 0:
            return None
        centroid = (float(xs.mean()), float(ys.mean()))
    cx, cy = centroid

    bx, by, bw, bh = bbox
    frame_area = float(W * H) or 1.0
    subj_area = float(mask_bool.sum())
    spread = (bw * bh) / frame_area  # how much of the frame the subject bbox spans

    # ── Regime 2: scene composition (no distinct subject) ───────────────────
    is_scene = (not has_subject) and spread >= CROP_SCENE_SPREAD
    if is_scene:
        if aspect is not None:
            bias = max(0.0, min(1.0, CROP_SCENE_BIAS))
            ax = cx * bias + (W / 2.0) * (1.0 - bias)
            ay = cy * bias + (H / 2.0) * (1.0 - bias)
            x, y, w, h = _compute_aspect_crop(W, H, aspect, centroid=(ax, ay))["box"]
        else:
            x, y, w, h = 0, 0, W, H
        crop_area = float(w * h)
        return {
            "box": [x, y, w, h],
            "score": round(min(1.0, 0.45 + 0.2 * (1.0 - abs(spread - 0.6))), 4),
            "aspect_ratio": round(w / float(h), 4) if h else None,
            "rationale": (
                "scene composition (no distinct subject)"
                + (f", fitted to {aspect:.3f}:1" if aspect else "")
            ),
            "centroid": [round(float(cx), 1), round(float(cy), 1)],
            "subject_coverage": round(subj_area / frame_area, 4),
            "already_tight": (crop_area / frame_area) >= 0.92,
        }

    # ── Regime 1: compose AROUND the subject with breathing room ────────────
    # Size the crop so the subject occupies ~target_frac of it (context kept,
    # not extracted); fit the aspect; then PLACE it for lead room without ever
    # cutting the subject or trimming a group lopsidedly.
    f = max(0.2, min(0.95, target_frac))
    crop_w = min(float(W), bw / f)
    crop_h = min(float(H), bh / f)
    if aspect is not None:
        # Reuse the tested aspect fitter for SIZE only; we re-place below.
        fitted = _fit_aspect(
            _clip_box_to_image((bx, by, crop_w, crop_h), W, H), aspect, W, H
        )
        crop_w, crop_h = float(fitted[2]), float(fitted[3])

    ax, ay = _compose_anchor((cx, cy), (bx, by, bw, bh), crop_w, crop_h, W, H)
    x, y, w, h = _clip_box_to_image(
        (ax - crop_w / 2.0, ay - crop_h / 2.0, crop_w, crop_h), W, H
    )
    crop_area = float(w * h)
    # Score: higher when the subject is well-included AND well-placed (not the
    # whole frame, not a tight zoom). Peak around the target share.
    crop_frame_ratio = crop_area / frame_area if frame_area > 0 else 0.0
    subj_share = subj_area / crop_area if crop_area > 0 else 0.0
    score = round(min(1.0, 0.55 + 0.25 * (1.0 - abs(subj_share - f)) + 0.2 * (1.0 - crop_frame_ratio)), 4)
    # Flag when the crop box essentially IS the full frame — the subject already
    # fills the frame, so there's little to gain. The client still PLACES this
    # box (adjustable) rather than skipping it.
    already_tight = crop_frame_ratio >= 0.92
    return {
        "box": [x, y, w, h],
        "score": score,
        "aspect_ratio": round(w / float(h), 4) if h else None,
        "rationale": (
            f"composed around subject (~{int(f * 100)}% frame share, lead-room placed)"
            + (f", fitted to {aspect:.3f}:1" if aspect else "")
        ),
        "centroid": [round(float(cx), 1), round(float(cy), 1)],
        "subject_coverage": round(subj_area / frame_area, 4),
        "already_tight": already_tight,
    }


def _compute_content_fill_crop(
    rgb: np.ndarray,
    *,
    aspect: "float | None" = None,
    thresh: int = CROP_CONTENT_TRIM_THRESH,
    min_frac: float = CROP_CONTENT_MIN_FRAC,
) -> "dict | None":
    """Trim near-solid borders (white/black mats, sky padding, letterboxes).

    Heuristic: sample the dominant colour from a thin border ring, then find
    the tight bbox of pixels whose max-channel distance from that colour
    exceeds `thresh`. Falls back to None when the trim would remove more than
    `1 - min_frac` of the frame (guard against accidental nuke on near-solid
    images like flatlays).
    """
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    H, W = rgb.shape[:2]
    if W < 16 or H < 16:
        return None

    # Sample a ~2% border ring for the dominant colour.
    ring = max(2, int(0.02 * min(W, H)))
    border = np.concatenate([
        rgb[:ring].reshape(-1, 3),
        rgb[-ring:].reshape(-1, 3),
        rgb[:, :ring].reshape(-1, 3),
        rgb[:, -ring:].reshape(-1, 3),
    ], axis=0)
    bg = np.median(border, axis=0).astype(np.int32)

    # Channel-wise distance from background.
    diff = np.abs(rgb.astype(np.int32) - bg[None, None, :]).max(axis=2)
    content = diff > int(thresh)

    if not content.any():
        return None

    # Cheap denoise: project onto rows & cols and threshold projection to
    # ignore single-pixel noise speckles in the margins.
    col_any = content.sum(axis=0) > max(2, H // 200)
    row_any = content.sum(axis=1) > max(2, W // 200)
    if not col_any.any() or not row_any.any():
        return None

    x0 = int(np.argmax(col_any))
    x1 = int(W - 1 - np.argmax(col_any[::-1]))
    y0 = int(np.argmax(row_any))
    y1 = int(H - 1 - np.argmax(row_any[::-1]))
    if x1 <= x0 or y1 <= y0:
        return None

    box = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    if (box[2] * box[3]) / float(W * H) < float(min_frac):
        return None

    if aspect is not None:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        box = _fit_aspect(box, aspect, W, H, anchor=(cx, cy))

    x, y, w, h = _clip_box_to_image(box, W, H)
    trimmed = 1.0 - (w * h) / float(W * H)
    return {
        "box": [x, y, w, h],
        "score": round(min(1.0, 0.4 + 0.6 * trimmed), 4),
        "aspect_ratio": round(w / float(h), 4) if h else None,
        "rationale": f"trimmed {trimmed * 100:.0f}% near-solid border (Δ>{thresh})",
        "already_tight": trimmed < 0.08,
    }


def _compute_depth_crop(
    depth: "np.ndarray | None",
    W: int,
    H: int,
    *,
    aspect: "float | None" = None,
    pct: float = CROP_DEPTH_FOREGROUND_PCT,
    padding: float = CROP_SUBJECT_PADDING,
) -> "dict | None":
    """Depth-aware foreground crop. Selects the nearest `pct` fraction of
    pixels by depth (white = near in Depth Anything V2), bboxes them and pads.
    """
    if depth is None or depth.size == 0:
        return None
    if depth.shape[0] != H or depth.shape[1] != W:
        # Resize via PIL — depth maps are smooth so bilinear is fine.
        d_img = Image.fromarray(depth, mode="L").resize((W, H), Image.BILINEAR)
        depth = np.asarray(d_img, dtype=np.uint8)

    # Threshold at the top `pct` of depth values.
    flat = depth.reshape(-1)
    if flat.size == 0:
        return None
    cutoff = int(np.quantile(flat, 1.0 - float(pct)))
    fg = depth >= max(1, cutoff)
    bbox = _bbox_from_mask(fg)
    if bbox is None:
        return None

    padded = _expand_box(bbox, W, H, padding)
    if aspect is not None:
        ys, xs = np.nonzero(fg)
        cx = float(xs.mean()) if xs.size else (padded[0] + padded[2] / 2.0)
        cy = float(ys.mean()) if ys.size else (padded[1] + padded[3] / 2.0)
        final = _fit_aspect(padded, aspect, W, H, anchor=(cx, cy))
    else:
        final = padded
    x, y, w, h = final
    frame_area = float(W * H) or 1.0
    crop_area = float(w * h)
    return {
        "box": [x, y, w, h],
        "score": round(min(1.0, 0.45 + 0.55 * (fg.sum() / frame_area)), 4),
        "aspect_ratio": round(w / float(h), 4) if h else None,
        "rationale": f"top {int(pct * 100)}% depth percentile padded {int(padding * 100)}%",
        "already_tight": (crop_area / frame_area) >= 0.92,
    }


def _compute_aspect_crop(
    W: int,
    H: int,
    aspect: float,
    *,
    centroid: "tuple[float, float] | None" = None,
) -> dict:
    """Maximum-area crop at `aspect`, centred on `centroid` (defaults to image
    centre). The cheap "preset" mode — no model required."""
    if W <= 0 or H <= 0 or aspect <= 0:
        raise ValueError("invalid aspect arguments")
    if aspect >= W / float(H):
        new_w = float(W)
        new_h = W / aspect
    else:
        new_h = float(H)
        new_w = H * aspect

    if centroid is None:
        cx, cy = W / 2.0, H / 2.0
    else:
        cx, cy = centroid
    box = (cx - new_w / 2.0, cy - new_h / 2.0, new_w, new_h)
    # Use _fit_aspect's clipping behaviour for free.
    x, y, w, h = _fit_aspect(
        _clip_box_to_image(box, W, H), aspect, W, H, anchor=(cx, cy)
    )
    return {
        "box": [x, y, w, h],
        "score": 0.5,
        "aspect_ratio": round(w / float(h), 4) if h else None,
        "rationale": f"max-area fit to {aspect:.3f}:1 around {'subject' if centroid else 'centre'}",
    }


def _subjects_from_instances(
    instances: "list[dict]", W: int, H: int
) -> "tuple[list[dict], np.ndarray, tuple[float, float] | None]":
    """Reduce concept instance dicts to the three things /crop/auto's subject
    strategy needs: the JSON `subjects` payload, a uint8 union mask over the
    top-N instances, and an area-weighted centroid (more stable than a global
    matte centroid for multi-subject photos)."""
    top = sorted(instances, key=lambda i: -i["area"])[:SAM3_INSTANCES_MAX]
    payload: "list[dict]" = []
    union = np.zeros((H, W), dtype=np.uint8)
    total = sx = sy = 0.0
    for idx, inst in enumerate(top):
        ys, xs = np.nonzero(inst["mask"])
        if xs.size == 0:
            continue
        payload.append({
            "index": idx,
            "label": inst["label"],
            "class_id": inst["class_id"],
            "confidence": round(float(inst["confidence"]), 4),
            "source": inst["source"],
            "bbox": _bbox_of(inst["mask"]),
            "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
        })
        union = np.maximum(union, inst["mask"].astype(np.uint8) * 255)
        a = float(inst["area"])
        sx += xs.mean() * a
        sy += ys.mean() * a
        total += a
    centroid = (sx / total, sy / total) if total > 0 else None
    return payload, union, centroid


@app.post("/crop/auto")
async def crop_auto(
    image: UploadFile = File(..., alias="image"),
    aspect: "str | None" = Form(None),
    mode: str = Form("all"),
    padding: "float | None" = Form(None),
) -> JSONResponse:
    """Production auto-crop. Returns one or more crop boxes computed from the
    input image, all expressed in the input image's ORIGINAL pixel coordinates.

    Form fields:
        image:   JPEG/PNG/WebP, max MAX_UPLOAD_MB.
        aspect:  optional `"W:H"` / `"W/H"` / float. When supplied, every
                 strategy's box is fitted to this ratio.
        mode:    one of `"subject" | "aspect" | "content" | "depth" | "all"`.
                 Default `"all"` runs every strategy that's available
                 (depth-aware needs Depth Anything; subject prefers SAM 3).
        padding: optional float (0..1) overriding `CROP_SUBJECT_PADDING`.

    Response shape:

        {
          "width": int, "height": int,
          "aspect": float | null,
          "ran": ["subject", "aspect", "content", "depth"],
          "crops": {
            "subject": { "box": [x,y,w,h], "score": 0..1,
                         "aspect_ratio": float, "rationale": str,
                         "centroid": [cx,cy] } | null,
            "aspect":  { ... } | null,
            "content": { ... } | null,
            "depth":   { ... } | null
          },
          "subjects": [ { index, label, confidence, bbox } ],
          "recommended": "subject" | "depth" | "content" | "aspect",
          "elapsed_ms": int
        }

    The endpoint NEVER raises on a per-strategy failure — it just emits `null`
    for that strategy and continues. The caller picks whichever box best fits
    its UX; `"recommended"` is the highest-scoring one.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(415, f"unsupported content-type: {image.content_type}")

    mode = (mode or "all").strip().lower()
    if mode not in {"subject", "aspect", "content", "depth", "all"}:
        raise HTTPException(400, f"invalid mode: {mode}")

    aspect_value = _parse_aspect(aspect)
    if aspect_value is None and mode == "aspect":
        raise HTTPException(400, "mode=aspect requires an `aspect` field (e.g. 16:9)")

    pad = CROP_SUBJECT_PADDING if padding is None else max(0.0, min(1.0, float(padding)))

    contents = await _read_limited(image)
    if not contents:
        raise HTTPException(400, "empty upload")

    t0 = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(contents))
        img.load()
        # Camera JPEGs store rotation in EXIF; crop boxes are computed on raw
        # pixels, so without this they come back rotated vs what the caller sees.
        img = ImageOps.exif_transpose(img)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise HTTPException(400, f"could not decode image: {e}")

    if img.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N"):
        # 16/32-bit greyscale: convert("RGB") truncates instead of rescaling,
        # blacking the frame out — rescale to 8-bit first.
        img = Image.fromarray(np.clip(np.asarray(img, dtype=np.float32) / 257.0, 0, 255).astype(np.uint8), "L")
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    if max(img.width, img.height) > CROP_MAX_SIDE:
        raise HTTPException(
            413,
            f"image too large ({img.width}x{img.height}); max longest side {CROP_MAX_SIDE}px",
        )

    rgb_img = img.convert("RGB")
    W, H = rgb_img.size
    rgb = np.asarray(rgb_img, dtype=np.uint8)

    crops: "dict[str, dict | None]" = {}
    ran: list[str] = []
    subjects_payload: list[dict] = []

    # Cheap modes first.
    if mode in ("aspect", "all") and aspect_value is not None:
        try:
            crops["aspect"] = _compute_aspect_crop(W, H, aspect_value)
            ran.append("aspect")
        except Exception:
            log.exception("aspect crop failed")
            crops["aspect"] = None

    if mode in ("content", "all"):
        try:
            crops["content"] = await run_in_threadpool(
                _compute_content_fill_crop, rgb, aspect=aspect_value
            )
            ran.append("content")
        except Exception:
            log.exception("content-fill crop failed")
            crops["content"] = None

    # Subject-aware: SAM 3 first. The crop box only needs a subject region, not
    # a pixel-perfect alpha, so SAM 3 concept instances provide the strongest
    # signal. If SAM 3 is unavailable or finds nothing, fall back to the generic
    # saliency matte so the tool remains usable in lightweight local setups.
    if mode in ("subject", "all"):
        try:
            union_mask: "np.ndarray | None" = None
            subject_centroid: "tuple[float, float] | None" = None

            if await run_in_threadpool(_ensure_sam3, app):
                try:
                    instances = await run_in_threadpool(
                        _sam3_instances_for_prompt, app, rgb_img, SAM3_SUBJECT_PROMPT
                    )
                except Exception:
                    log.exception("SAM 3 subject detection failed in /crop/auto")
                    instances = []
                if instances:
                    payload, union, subject_centroid = _subjects_from_instances(
                        instances, W, H
                    )
                    subjects_payload.extend(payload)
                    union_mask = union

            if union_mask is None:
                matte_img = await run_in_threadpool(_rembg_matte, img)
                raw_matte = np.asarray(matte_img.convert("L"), dtype=np.uint8)
                matte = await run_in_threadpool(clean_matte, raw_matte, rgb)
                union_mask = matte.copy()

            # `has_subject` is True only when SAM 3 found instances. A fallback
            # saliency matte alone can be diffuse, so the crop treats it as a
            # scene when it sprawls across the frame.
            crops["subject"] = _compute_subject_crop(
                union_mask, W, H,
                aspect=aspect_value, centroid=subject_centroid,
                has_subject=bool(subjects_payload),
            )
            ran.append("subject")
        except Exception:
            log.exception("subject crop failed")
            crops["subject"] = None

    # Depth-aware: only run when explicitly asked or in "all" AND the model
    # is loaded — don't trigger a 500MB lazy load just for an opportunistic
    # alternative.
    if mode == "depth" or (mode == "all" and getattr(app.state, "depth_available", False)):
        try:
            if mode == "depth":
                # User explicitly asked → ensure depth is loaded.
                if not await run_in_threadpool(_ensure_depth, app):
                    raise HTTPException(
                        501,
                        "Depth Anything V2 not available on this server.",
                    )
            if getattr(app.state, "depth_available", False):
                depth_arr = await run_in_threadpool(_depth_predict, app, rgb_img)
                crops["depth"] = await run_in_threadpool(
                    _compute_depth_crop, depth_arr, W, H, aspect=aspect_value, padding=pad
                )
                ran.append("depth")
        except HTTPException:
            raise
        except Exception:
            log.exception("depth crop failed")
            crops["depth"] = None

    # Pick a recommendation: prefer subject > depth > content > aspect, but
    # break ties on score and skip nulls.
    PREFERENCE = ["subject", "depth", "content", "aspect"]
    candidates = [(k, crops[k]) for k in PREFERENCE if crops.get(k)]
    recommended = None
    if candidates:
        recommended = max(
            candidates, key=lambda kv: (kv[1].get("score") or 0.0, -PREFERENCE.index(kv[0]))
        )[0]

    elapsed = time.perf_counter() - t0
    log.info(
        "crop/auto %s (%dx%d) mode=%s ran=%s rec=%s aspect=%s in %.2fs",
        image.filename or "<unnamed>", W, H, mode, ran, recommended,
        f"{aspect_value:.3f}" if aspect_value else "—", elapsed,
    )

    return JSONResponse({
        "width": W,
        "height": H,
        "aspect": round(aspect_value, 6) if aspect_value else None,
        "mode": mode,
        "ran": ran,
        "crops": crops,
        "subjects": subjects_payload,
        "recommended": recommended,
        "elapsed_ms": int(elapsed * 1000),
    }, headers={"Cache-Control": "no-store"})


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )
