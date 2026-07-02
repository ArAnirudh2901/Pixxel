"""Pixxel mask service.

Wraps `rembg` and Hugging Face `transformers` SAM 2 (Hiera-Small) in a
tiny FastAPI HTTP API so the Next.js AI routes can call background-removal
and click-to-select models locally without Docker.

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
from PIL import Image, ImageFilter, UnidentifiedImageError
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
SAM2_MODEL_ID = os.getenv("SAM2_MODEL_ID", "facebook/sam2-hiera-small").strip()
SAM2_CACHE_MAX = int(os.getenv("SAM2_CACHE_MAX", "20").strip())
SAM2_MAX_CLICKS = int(os.getenv("SAM2_MAX_CLICKS", "50").strip())
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
PORT = int(os.getenv("PORT", "8002").strip())
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "24").strip())
# Reject /segment inputs whose longest side exceeds this (defense-in-depth vs
# direct curls that bypass the Node route's MAX_MODEL_SIDE cap). Mirrors the
# /depth handler's DEPTH_MAX_SIDE.
SEGMENT_MAX_SIDE = int(os.getenv("SEGMENT_MAX_SIDE", "2048").strip())

# By default, SAM 2 and Depth Anything (the heavy torch models behind the
# /sam2 and /depth endpoints) are loaded LAZILY — on first use of their
# endpoint — instead of at startup. This keeps the resident footprint small
# so the service fits a small free-tier host; the core "Select Subject"
# (/segment) path never needs them.
# Set SEGMENT_EAGER_MODELS=1 to preload everything at startup (lowest
# first-request latency, ~1.5-2.5 GB resident).
SEGMENT_EAGER_MODELS = os.getenv("SEGMENT_EAGER_MODELS", "0").strip() not in ("0", "false", "False", "")

# ─── SAM 3 concept segmentation ──────────────────────────────────────────────
# SAM 3 is preferred anywhere this service needs open-vocabulary subject or
# concept masks. It is optional because Meta's official package/checkpoints are
# gated and have newer runtime requirements than the lightweight local service.
SAM3_ENABLE = os.getenv("SAM3_ENABLE", "1").strip() not in ("0", "false", "False", "")
SAM3_MODEL_ID = os.getenv("SAM3_MODEL_ID", "facebook/sam3.1").strip()
SAM3_CHECKPOINT_PATH = os.getenv("SAM3_CHECKPOINT_PATH", "").strip() or None
SAM3_CONFIDENCE = float(os.getenv("SAM3_CONFIDENCE", "0.5").strip())
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


# ─── Text-grounded masking (/ground/text — Step 9: NL mask pipeline) ────────
# CLIPSeg turns a free-text phrase ("the red jacket", "the waterfall") into a
# coarse relevance heatmap; connected components above threshold are then
# refined with SAM 2 box+point prompts for crisp instance-quality edges.
# rd64-refined is ~600 MB and lazy-loads exactly like SAM 2 / Depth.
GROUND_MODEL_ID = os.getenv("GROUND_MODEL_ID", "CIDAS/clipseg-rd64-refined").strip()
GROUND_MAX_SIDE = int(os.getenv("GROUND_MAX_SIDE", "2048").strip())
# CLIPSeg's sigmoid map lives in a compressed range (true positives often
# peak at only 0.35..0.6), so the binarisation threshold is RELATIVE to the
# map's own peak: pixel >= max(GROUND_FLOOR, GROUND_THRESHOLD * peak). An
# absolute cut (the naive 0.4) silently drops weak-but-real targets.
GROUND_THRESHOLD = float(os.getenv("GROUND_THRESHOLD", "0.55").strip())
# Absolute floor under the relative cut — keeps near-zero noise out of the
# mask even when the peak itself is tiny.
GROUND_FLOOR = float(os.getenv("GROUND_FLOOR", "0.10").strip())
# Below this peak probability the phrase is reported as not found at all.
GROUND_MIN_PEAK = float(os.getenv("GROUND_MIN_PEAK", "0.25").strip())
# Ignore components smaller than this fraction of the frame (heatmap noise).
GROUND_MIN_AREA_FRAC = float(os.getenv("GROUND_MIN_AREA_FRAC", "0.001").strip())
GROUND_MAX_PHRASES = int(os.getenv("GROUND_MAX_PHRASES", "4").strip())
GROUND_MAX_COMPONENTS = int(os.getenv("GROUND_MAX_COMPONENTS", "4").strip())
# SAM 2 refinement is the latency hot spot on CPU — cap how many components
# get the treatment; the rest fall back to the (cleaned) CLIPSeg mask.
GROUND_REFINE_TOP = int(os.getenv("GROUND_REFINE_TOP", "2").strip())

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
}

if MODEL_NAME not in ALLOWED_MODELS:
    log.warning("unknown SEGMENT_MODEL=%r; falling back to isnet-general-use", MODEL_NAME)
    MODEL_NAME = "isnet-general-use"


# ─── Execution-provider auto-detect (ONNX / rembg) ──────────────────────────

def detect_providers() -> List[str]:
    """Pick the best ONNX Runtime execution providers for this machine.

    Order of preference: CUDA (NVIDIA) > CoreML (Apple Silicon) > CPU.
    """
    override = os.getenv("SEGMENT_PROVIDERS", "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]

    providers: List[str] = ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort  # type: ignore
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
            log.info("ONNX CUDA execution provider detected (NVIDIA GPU)")
        elif "CoreMLExecutionProvider" in available:
            providers.insert(0, "CoreMLExecutionProvider")
            log.info("ONNX CoreML execution provider detected (Apple Silicon GPU)")
    except Exception as e:  # pragma: no cover - best effort
        log.debug("onnxruntime provider probe failed: %s", e)
    return providers


# ─── Optional torch / SAM 2 loader ──────────────────────────────────────────

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


SAM2_EMBEDDING_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
DEPTH_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _image_hash(img: Image.Image) -> str:
    """Stable hash of a PIL image's pixel data for cache keys (SAM 2
    embeddings, depth maps). Collisions on 16 hex chars (64 bits) are
    astronomically unlikely for any image a user will upload."""
    return hashlib.sha256(img.tobytes()).hexdigest()[:16]


def _sam2_encode(app, img: Image.Image):
    """Run the SAM 2 image encoder, caching by image-content hash.

    Returns (image_embeddings, original_sizes) on the model's device.
    """
    key = _image_hash(img)
    cached = SAM2_EMBEDDING_CACHE.get(key)
    if cached is not None:
        SAM2_EMBEDDING_CACHE.move_to_end(key)
        log.info("SAM 2 embedding cache hit (%d entries)", len(SAM2_EMBEDDING_CACHE))
        return cached

    import torch  # type: ignore
    processor = app.state.sam2_processor
    model = app.state.sam2_model
    device = app.state.sam2_device

    t0 = time.perf_counter()
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.inference_mode():
        embeddings = model.get_image_embeddings(pixel_values=inputs.pixel_values)

    value = (embeddings, inputs["original_sizes"])
    SAM2_EMBEDDING_CACHE[key] = value
    while len(SAM2_EMBEDDING_CACHE) > SAM2_CACHE_MAX:
        SAM2_EMBEDDING_CACHE.popitem(last=False)

    log.info(
        "SAM 2 embedded %dx%d in %.2fs (cache: %d entries)",
        img.width, img.height,
        time.perf_counter() - t0,
        len(SAM2_EMBEDDING_CACHE),
    )
    return value


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
    """sam3.model.vitdet's MLP block always routes fc1+activation through
    sam3.perflib.fused.addmm_act, which unconditionally casts to bfloat16 for
    GPU tensor-core throughput. On CPU this leaves fc1's output in bf16 while
    fc2's float32 weights are untouched, so every forward pass crashes with
    "mat1 and mat2 must have the same dtype". Replace the symbol vitdet.py
    actually calls with a plain float32 fc1->activation, only on CPU."""
    global _SAM3_CPU_PATCHED
    if _SAM3_CPU_PATCHED:
        return
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


# ─── Lazy model loaders (SAM 2 / Depth) ──────────────────────────────────────
# These hold the heavy torch models. By default they load on first use so the
# resident footprint stays small. Loads are serialised per-model with a lock so
# two concurrent first-requests don't both load. A permanent failure (e.g. torch
# not installed) is remembered so we don't retry the heavy load on every request.

_SAM2_LOCK = threading.Lock()
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


def _load_sam2(app: FastAPI) -> bool:
    """Load SAM 2 into app.state (blocking). Caller holds _SAM2_LOCK."""
    try:
        import torch  # type: ignore  # noqa: F401
        from transformers import Sam2Model, Sam2Processor  # type: ignore

        device, device_label = detect_torch_device()
        if device is None:
            raise ImportError("torch not available")

        log.info("loading SAM 2 model %r onto %s ...", SAM2_MODEL_ID, device_label)
        t1 = time.perf_counter()
        app.state.sam2_processor = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
        app.state.sam2_model = Sam2Model.from_pretrained(SAM2_MODEL_ID).to(device)
        app.state.sam2_model.eval()
        app.state.sam2_device = device
        app.state.sam2_model_id = SAM2_MODEL_ID
        app.state.sam2_available = True
        log.info("SAM 2 ready in %.1fs on %s", time.perf_counter() - t1, device_label)
        return True
    except ImportError:
        log.warning("torch / transformers not installed; /sam2/click disabled.")
        app.state.sam2_load_failed = True
        return False
    except Exception as e:  # pragma: no cover - defensive
        log.exception("failed to load SAM 2: %s", e)
        app.state.sam2_load_failed = True
        return False


def _ensure_sam2(app: FastAPI):
    """Lazily load SAM 2; return True/'cold' if ready, False if failed.
    Returns 'cold' on first load (model just downloaded/loaded)."""
    if app.state.sam2_available:
        return True
    if getattr(app.state, "sam2_load_failed", False):
        return False
    with _SAM2_LOCK:
        if app.state.sam2_available:
            return True
        if getattr(app.state, "sam2_load_failed", False):
            return False
        ok = _load_sam2(app)
        return "cold" if ok else False


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


_GROUND_LOCK = threading.Lock()


def _load_ground(app: FastAPI) -> bool:
    """Load CLIPSeg into app.state (blocking). Caller holds _GROUND_LOCK."""
    try:
        import torch  # type: ignore  # noqa: F401
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor  # type: ignore

        device, device_label = detect_torch_device()
        if device is None:
            raise ImportError("torch not available")

        log.info("loading CLIPSeg model %r onto %s ...", GROUND_MODEL_ID, device_label)
        t = time.perf_counter()
        app.state.ground_processor = CLIPSegProcessor.from_pretrained(GROUND_MODEL_ID)
        app.state.ground_model = CLIPSegForImageSegmentation.from_pretrained(GROUND_MODEL_ID).to(device)
        app.state.ground_model.eval()
        app.state.ground_device = device
        app.state.ground_model_id = GROUND_MODEL_ID
        app.state.ground_available = True
        log.info("CLIPSeg ready in %.1fs on %s", time.perf_counter() - t, device_label)
        return True
    except ImportError:
        log.warning("torch / transformers not installed; /ground/text disabled.")
        app.state.ground_load_failed = True
        return False
    except Exception as e:  # pragma: no cover - defensive
        log.exception("failed to load CLIPSeg: %s", e)
        app.state.ground_load_failed = True
        return False


def _ensure_ground(app: FastAPI):
    """Lazily load CLIPSeg; return True/'cold' if ready, False if failed.
    Returns 'cold' on first load (model just downloaded/loaded)."""
    if getattr(app.state, "ground_available", False):
        return True
    if getattr(app.state, "ground_load_failed", False):
        return False
    with _GROUND_LOCK:
        if getattr(app.state, "ground_available", False):
            return True
        if getattr(app.state, "ground_load_failed", False):
            return False
        ok = _load_ground(app)
        return "cold" if ok else False


def _ground_heatmaps(app, img: Image.Image, phrases: "list[str]") -> np.ndarray:
    """CLIPSeg relevance maps for `phrases` against `img`.

    Returns float32 array of shape (len(phrases), H, W) in 0..1 at the
    IMAGE's resolution (the model's 352² logits are bilinearly upsampled).
    """
    import torch  # type: ignore

    processor = app.state.ground_processor
    model = app.state.ground_model
    device = app.state.ground_device

    inputs = processor(
        text=phrases,
        images=[img] * len(phrases),
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    logits = outputs.logits  # (N, 352, 352) — or (352, 352) when N == 1
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    out = np.empty((len(phrases), img.height, img.width), dtype=np.float32)
    for i in range(probs.shape[0]):
        m = Image.fromarray((probs[i] * 255.0).astype(np.uint8), mode="L")
        m = m.resize((img.width, img.height), Image.BILINEAR)
        out[i] = np.asarray(m, dtype=np.float32) / 255.0
    return out


def _sam2_refine_box(app, img: Image.Image, box: "tuple[int, int, int, int]",
                     point: "tuple[float, float] | None" = None) -> "np.ndarray | None":
    """Refine a coarse region into a crisp SAM 2 mask using a box prompt
    (plus an optional positive point at the region's confidence peak).

    `box` is (x, y, w, h) in image pixels. Returns a bool (H, W) mask or
    None when SAM 2 is unavailable or inference fails — callers fall back
    to the coarse mask.
    """
    if not _ensure_sam2(app):
        return None
    try:
        import torch  # type: ignore

        x, y, w, h = box
        # SAM 2 input_boxes: [image, object, [x0, y0, x1, y1]] = 3 levels.
        boxes = [[[float(x), float(y), float(x + w), float(y + h)]]]
        kwargs = dict(images=img, input_boxes=boxes, return_tensors="pt")
        if point is not None:
            # input_points: [image, object, point, [x, y]] = 4 levels.
            kwargs["input_points"] = [[[[float(point[0]), float(point[1])]]]]
            kwargs["input_labels"] = [[[1]]]

        embeddings, original_sizes = _sam2_encode(app, img)
        processor = app.state.sam2_processor
        model = app.state.sam2_model
        device = app.state.sam2_device

        inputs = processor(**kwargs).to(device)
        inputs["image_embeddings"] = embeddings
        inputs.pop("pixel_values", None)
        with torch.inference_mode():
            outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), original_sizes.cpu(),
        )
        scores = outputs.iou_scores.cpu().numpy()[0].reshape(-1)
        best = int(scores.argmax())
        return masks[0][0][best].cpu().numpy().astype(bool)
    except Exception:
        log.exception("SAM 2 box refinement failed; using coarse mask")
        return None


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

    # SAM 3, SAM 2 (click-to-select), and Depth Anything V2 are loaded lazily
    # unless explicitly warmed. Their configured model ids are recorded up
    # front so /health and response headers can report them.
    app.state.sam3_available = False
    app.state.sam3_load_failed = not SAM3_ENABLE
    app.state.sam3_model_id = SAM3_MODEL_ID
    app.state.sam2_available = False
    app.state.sam2_load_failed = False
    app.state.sam2_model_id = SAM2_MODEL_ID
    app.state.depth_available = False
    app.state.depth_load_failed = False
    app.state.depth_model_id = DEPTH_MODEL_ID
    app.state.ground_available = False
    app.state.ground_load_failed = False
    app.state.ground_model_id = GROUND_MODEL_ID

    if SAM3_EAGER:
        await run_in_threadpool(_ensure_sam3, app)

    if SEGMENT_EAGER_MODELS:
        await run_in_threadpool(_ensure_sam2, app)
        await run_in_threadpool(_ensure_depth, app)
    else:
        log.info(
            "SAM 3, SAM 2, and Depth will load lazily on first use "
            "(set SAM3_EAGER=1 or SEGMENT_EAGER_MODELS=1 to preload)."
        )

    yield

    SAM2_EMBEDDING_CACHE.clear()
    DEPTH_CACHE.clear()
    log.info("shutting down mask service")


app = FastAPI(
    title="Pixxel Masking Service",
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


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> dict:
    return {
        "status": "ok",
        "message": "Pixxel Mask Service is running"
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
        "sam2_available": app.state.sam2_available
        or (_torch_stack_loadable() and not app.state.sam2_load_failed),
        "sam2_loaded": app.state.sam2_available,
        "sam2_model": app.state.sam2_model_id,
        "depth_available": app.state.depth_available
        or (_torch_stack_loadable() and not app.state.depth_load_failed),
        "depth_loaded": app.state.depth_available,
        "depth_model": app.state.depth_model_id,
        "ground_available": app.state.ground_available
        or (_torch_stack_loadable() and not app.state.ground_load_failed),
        "ground_loaded": app.state.ground_available,
        "ground_model": app.state.ground_model_id,
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

    # SAM 2
    sam2_was_loaded = app.state.sam2_available
    sam2_status = await run_in_threadpool(_ensure_sam2, app)
    results["sam2"] = (
        "already_loaded" if sam2_was_loaded
        else "loaded" if sam2_status else "failed"
    )

    # Depth Anything V2
    depth_was_loaded = app.state.depth_available
    depth_status = await run_in_threadpool(_ensure_depth, app)
    results["depth"] = (
        "already_loaded" if depth_was_loaded
        else "loaded" if depth_status else "failed"
    )

    # CLIPSeg grounding
    ground_was_loaded = getattr(app.state, "ground_available", False)
    ground_status = await run_in_threadpool(_ensure_ground, app)
    results["ground"] = (
        "already_loaded" if ground_was_loaded
        else "loaded" if ground_status else "failed"
    )

    log.info("warmup results: %s", results)
    return {"status": "ok", "models": results}


@app.post("/segment")
async def segment(image: UploadFile = File(..., alias="image")) -> Response:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(415, f"unsupported content-type: {image.content_type}")

    contents = await _read_limited(image)
    if not contents:
        raise HTTPException(400, "empty upload")

    t0 = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(contents))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        # PIL throws UnidentifiedImageError for "not an image at all" and
        # OSError (Truncated File Read, decode error) for "image header
        # was valid but body is corrupt". SyntaxError covers malformed
        # PNG chunks. All four are user-input problems, not server
        # bugs — return 400, not 500.
        raise HTTPException(400, f"could not decode image: {e}")

    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    # Reject pathologically large inputs (defense-in-depth vs direct curls that
    # bypass the Node route's MAX_MODEL_SIDE).
    if max(img.width, img.height) > SEGMENT_MAX_SIDE:
        raise HTTPException(
            413,
            f"image too large ({img.width}x{img.height}); "
            f"max longest side is {SEGMENT_MAX_SIDE}px",
        )

    img_rgb = img.convert("RGB")
    np_rgb = np.asarray(img_rgb)

    subject_mode = "sam3"
    subjects = 0
    final_alpha = None

    if await run_in_threadpool(_ensure_sam3, app):
        try:
            instances = await run_in_threadpool(
                _sam3_instances_for_prompt, app, img_rgb, SAM3_SUBJECT_PROMPT
            )
        except Exception:
            log.exception("SAM 3 subject segmentation failed; using saliency fallback")
            instances = []
        if instances:
            subjects = len(instances)
            union_bin = _union_from_instances(instances, img.width, img.height) > 0
            final_alpha = _soft_alpha_from_mask(union_bin)
            if SAM3_REFINE_MATTING:
                final_alpha = await run_in_threadpool(
                    _refine_alpha_with_matting, final_alpha, np_rgb, union_bin
                )

    if final_alpha is None:
        subject_mode = "saliency"
        try:
            matte_img = await run_in_threadpool(
                remove, img, session=app.state.session, only_mask=True
            )
        except Exception as e:
            log.exception("rembg.remove failed")
            raise HTTPException(500, f"segmentation failed: {e}")
        raw_matte = np.asarray(matte_img.convert("L"), dtype=np.uint8)
        final_alpha = await run_in_threadpool(clean_matte, raw_matte)
        subjects = 1 if final_alpha.any() else 0

    # Compose RGBA: original colours with the computed subject alpha.
    rgba = np.dstack([np_rgb, final_alpha]).astype(np.uint8)
    out = Image.fromarray(rgba, "RGBA")

    buf = io.BytesIO()
    out.save(buf, format="PNG", compress_level=1)
    elapsed = time.perf_counter() - t0
    log.info(
        "segmented %s (%dx%d, %dKB) mode=%s subjects=%d in %.2fs -> %dKB",
        image.filename or "<unnamed>",
        img.width,
        img.height,
        len(contents) // 1024,
        subject_mode,
        subjects,
        elapsed,
        buf.tell() // 1024,
    )

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Model": app.state.sam3_model_id if subject_mode == "sam3" else app.state.model_name,
            "X-Subject-Mode": subject_mode,
            "X-Subjects": str(subjects),
            "X-Elapsed-Ms": str(int(elapsed * 1000)),
        },
    )


@app.post("/segment/instances")
async def segment_instances(
    image: UploadFile = File(..., alias="image"),
    prompt: str = Form(SAM3_SUBJECT_PROMPT),
) -> JSONResponse:
    """SAM 3-first concept instance detection.

    Where /segment unions matching instances into one alpha, this returns one
    greyscale mask PER SAM 3 concept instance, with label, confidence, bounding
    box and area. The default prompt is `SAM3_SUBJECT_PROMPT`, and callers can
    supply a narrower noun phrase such as "person", "red jacket", or "dog".

    Response JSON:
        {
          "width": int, "height": int,
          "model": str, "prompt": str,
          "count": int,
          "instances": [
            { "index": 0, "label": "person",
              "confidence": 0.94, "source": "sam3" | "saliency",
              "bbox": [x, y, w, h], "area": int, "area_frac": float,
              "centroid": [cx, cy],
              "mask_png": "<base64 greyscale PNG, white=match>" },
            ...
          ],
          "union_png": "<base64 greyscale PNG of all matches>"
        }
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(415, f"unsupported content-type: {image.content_type}")

    contents = await _read_limited(image)
    if not contents:
        raise HTTPException(400, "empty upload")

    t0 = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(contents))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise HTTPException(400, f"could not decode image: {e}")

    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    if max(img.width, img.height) > SEGMENT_MAX_SIDE:
        raise HTTPException(
            413,
            f"image too large ({img.width}x{img.height}); "
            f"max longest side is {SEGMENT_MAX_SIDE}px",
        )

    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    concept = str(prompt or SAM3_SUBJECT_PROMPT).strip() or SAM3_SUBJECT_PROMPT

    mode = "sam3"
    source_model = app.state.sam3_model_id
    instances = []
    if await run_in_threadpool(_ensure_sam3, app):
        try:
            instances = await run_in_threadpool(
                _sam3_instances_for_prompt, app, img_rgb, concept
            )
        except Exception:
            log.exception("SAM 3 instance detection failed; falling back to saliency")
            instances = []

    if not instances:
        mode = "saliency"
        source_model = app.state.model_name
        try:
            matte_img = await run_in_threadpool(
                remove, img, session=app.state.session, only_mask=True
            )
        except Exception as e:
            log.exception("rembg.remove failed")
            raise HTTPException(500, f"segmentation failed: {e}")
        raw_matte = np.asarray(matte_img.convert("L"), dtype=np.uint8)
        matte = await run_in_threadpool(clean_matte, raw_matte)
        inst = _saliency_instance(concept, matte)
        instances = [inst] if inst else []

    instances.sort(key=lambda i: -i["area"])
    truncated = len(instances) > SAM3_INSTANCES_MAX
    instances = instances[:SAM3_INSTANCES_MAX]

    def _build_payload():
        frame_area = float(w * h) or 1.0
        union = np.zeros((h, w), dtype=np.uint8)
        union_bin = np.zeros((h, w), dtype=bool)
        items = []
        for idx, inst in enumerate(instances):
            mb = inst["mask"]
            alpha = _soft_alpha_from_mask(mb)
            union = np.maximum(union, alpha)
            union_bin |= mb
            ys, xs = np.nonzero(mb)
            items.append({
                "index": idx,
                "label": inst["label"],
                "class_id": inst["class_id"],
                "confidence": round(float(inst["confidence"]), 4),
                "source": inst["source"],
                "bbox": _bbox_of(mb),
                "area": int(inst["area"]),
                "area_frac": round(inst["area"] / frame_area, 5),
                "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
                "mask_png": _mask_png_b64(alpha),
            })
        union_out = union
        if items and SAM3_REFINE_MATTING:
            union_out = _refine_alpha_with_matting(union, np.asarray(img_rgb), union_bin)
        return items, (_mask_png_b64(union_out) if items else None)

    items, union_b64 = await run_in_threadpool(_build_payload)

    elapsed = time.perf_counter() - t0
    log.info(
        "segment/instances %s (%dx%d) mode=%s prompt=%r count=%d%s in %.2fs",
        image.filename or "<unnamed>", w, h, mode, concept, len(items),
        " (truncated)" if truncated else "", elapsed,
    )

    return JSONResponse(
        {
            "width": w,
            "height": h,
            "model": source_model,
            "prompt": concept,
            "mode": mode,
            "count": len(items),
            "truncated": truncated,
            "instances": items,
            "union_png": union_b64,
            "elapsed_ms": int(elapsed * 1000),
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/sam2/click")
async def sam2_click(
    image: UploadFile = File(..., alias="image"),
    clicks: "str | None" = Form(None),
    box: "str | None" = Form(None),
) -> Response:
    """Click- and/or box-prompted semantic masking with SAM 2.

    Form fields:
        image:  image file
        clicks: optional JSON array of `[x, y, label]` tuples, where `label`
                is 1 (positive / include this point) or 0 (negative / exclude).
        box:    optional JSON `[x0, y0, x1, y1]` box prompt — "select the
                object inside this rectangle". Boxes are SAM 2's strongest
                single prompt for whole-object selection; clicks can be
                combined with a box to refine it.

    At least one of `clicks` / `box` is required.
    Returns a greyscale PNG mask: white = include, black = exclude.
    """
    # Lazily load SAM 2 on first use (no-op once loaded).
    _sam2_status = await run_in_threadpool(_ensure_sam2, app)
    if not _sam2_status:
        raise HTTPException(
            501,
            "SAM 2 not available on this server. "
            "Install torch + transformers and restart, "
            "or set SAM2_MODEL_ID to a model in your HF cache.",
        )
    _sam2_cold = _sam2_status == "cold"

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(415, f"unsupported content-type: {image.content_type}")

    contents = await _read_limited(image)
    if not contents:
        raise HTTPException(400, "empty upload")

    try:
        img = Image.open(io.BytesIO(contents))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise HTTPException(400, f"could not decode image: {e}")

    if img.mode != "RGB":
        img = img.convert("RGB")

    clicks_data = []
    if clicks is not None and str(clicks).strip():
        try:
            clicks_data = json.loads(clicks)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"invalid clicks JSON: {e}")
        if not isinstance(clicks_data, list):
            raise HTTPException(400, "clicks must be an array of [x, y, label] tuples")

    box_data = None
    if box is not None and str(box).strip():
        try:
            box_data = json.loads(box)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"invalid box JSON: {e}")
        if not (isinstance(box_data, list) and len(box_data) == 4):
            raise HTTPException(400, f"box must be [x0, y0, x1, y1]; got {box_data!r}")
        for i, v in enumerate(box_data):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise HTTPException(400, f"box[{i}] must be a finite number; got {v!r}")
        x0, y0, x1, y1 = box_data
        if not (0 <= x0 < x1 <= img.width and 0 <= y0 < y1 <= img.height):
            raise HTTPException(
                400,
                f"box {box_data} is degenerate or outside image bounds ({img.width}x{img.height})",
            )

    if not clicks_data and box_data is None:
        raise HTTPException(400, "provide clicks and/or a box prompt")

    if len(clicks_data) > SAM2_MAX_CLICKS:
        raise HTTPException(
            400,
            f"too many clicks: {len(clicks_data)} > {SAM2_MAX_CLICKS}",
        )

    for i, c in enumerate(clicks_data):
        if not (isinstance(c, list) and len(c) == 3):
            raise HTTPException(400, f"click #{i} must be [x, y, label]; got {c!r}")
        x, y, label = c
        # `bool` is a subclass of `int` in Python, so it must be excluded
        # explicitly (otherwise `True in (0, 1)` is `True`).
        if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
            raise HTTPException(400, f"click #{i} x must be a finite number; got {x!r}")
        if not isinstance(y, (int, float)) or isinstance(y, bool) or not math.isfinite(y):
            raise HTTPException(400, f"click #{i} y must be a finite number; got {y!r}")
        if not (isinstance(label, int) and not isinstance(label, bool) and label in (0, 1)):
            raise HTTPException(400, f"click #{i} label must be 0 or 1; got {label!r}")
        if not (0 <= x < img.width and 0 <= y < img.height):
            raise HTTPException(
                400,
                f"click #{i} ({x}, {y}) is outside image bounds ({img.width}x{img.height})",
            )

    import torch  # type: ignore
    # SAM 2 input_points format: [image, object, point, [x, y]] = 4 levels.
    points = [[[[c[0], c[1]] for c in clicks_data]]] if clicks_data else None
    # SAM 2 input_labels format: [image, object, point_label] = 3 levels.
    labels = [[[c[2] for c in clicks_data]]] if clicks_data else None
    # SAM 2 input_boxes format: [image, object, [x0, y0, x1, y1]] = 3 levels.
    boxes = [[list(map(float, box_data))]] if box_data is not None else None

    t0 = time.perf_counter()
    try:
        embeddings, original_sizes = _sam2_encode(app, img)
        processor = app.state.sam2_processor
        model = app.state.sam2_model
        device = app.state.sam2_device

        prompt_kwargs = {}
        if points is not None:
            prompt_kwargs["input_points"] = points
            prompt_kwargs["input_labels"] = labels
        if boxes is not None:
            prompt_kwargs["input_boxes"] = boxes
        inputs = processor(
            images=img,
            return_tensors="pt",
            **prompt_kwargs,
        ).to(device)
        inputs["image_embeddings"] = embeddings
        inputs.pop("pixel_values", None)

        with torch.inference_mode():
            outputs = model(**inputs)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            original_sizes.cpu(),
        )
        scores = outputs.iou_scores.cpu().numpy()[0].reshape(-1)
        best_idx = int(scores.argmax())
        best_mask = masks[0][0][best_idx].cpu().numpy().astype(np.uint8) * 255
    except Exception as e:
        log.exception("sam2.click failed")
        raise HTTPException(500, f"sam2 inference failed: {e}")

    elapsed = time.perf_counter() - t0
    log.info(
        "SAM 2 click (%d points) on %dx%d in %.2fs (score=%.3f)",
        len(clicks_data), img.width, img.height, elapsed, float(scores[best_idx]),
    )

    mask_img = Image.fromarray(best_mask, mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG", compress_level=1)

    resp_headers = {
            "Cache-Control": "no-store",
            "X-Model": app.state.sam2_model_id or "sam2",
            "X-Score": f"{float(scores[best_idx]):.4f}",
            "X-Elapsed-Ms": str(int(elapsed * 1000)),
        }
    if _sam2_cold:
        resp_headers["X-Cold-Load"] = "true"
        log.info("SAM 2 cold load — first request after model download")

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers=resp_headers,
    )


@app.post("/depth")
async def depth(image: UploadFile = File(..., alias="image")) -> Response:
    """Monocular depth estimation with Depth Anything V2.

    Form fields:
        image:  image file (JPEG/PNG/WebP)

    Returns a greyscale PNG depth map at the input image's resolution.
    White (255) = nearest to the camera, black (0) = farthest. Per-image
    min-max normalisation is applied so the user can pick a meaningful
    near/far range on the resulting 0..1 slider.

    The map is cached by image-content hash (LRU, max `DEPTH_CACHE_MAX`
    entries); repeats against the same image return the cached result
    in milliseconds.
    """
    # Lazily load Depth Anything V2 on first use (no-op once loaded).
    _depth_status = await run_in_threadpool(_ensure_depth, app)
    if not _depth_status:
        raise HTTPException(
            501,
            "Depth Anything V2 not available on this server. "
            "Install torch + transformers and restart, "
            "or set DEPTH_MODEL_ID to a model in your HF cache.",
        )
    _depth_cold = _depth_status == "cold"

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(415, f"unsupported content-type: {image.content_type}")

    contents = await _read_limited(image)
    if not contents:
        raise HTTPException(400, "empty upload")

    try:
        img = Image.open(io.BytesIO(contents))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise HTTPException(400, f"could not decode image: {e}")

    if img.mode != "RGB":
        img = img.convert("RGB")

    # Reject inputs whose longest side exceeds DEPTH_MAX_SIDE. The model
    # runs internally at ~518×518, and the route (or this handler)
    # resizes the depth map back to the input's dimensions — a 12K
    # image would trigger a 144M-op Lanczos resize and a 144 MB array.
    # The Node route applies the same cap; this is defense-in-depth.
    if max(img.width, img.height) > DEPTH_MAX_SIDE:
        raise HTTPException(
            413,
            f"image too large ({img.width}x{img.height}); "
            f"max longest side is {DEPTH_MAX_SIDE}px",
        )

    t0 = time.perf_counter()
    try:
        depth_arr = _depth_predict(app, img)
    except Exception as e:
        log.exception("depth predict failed")
        raise HTTPException(500, f"depth inference failed: {e}")
    elapsed = time.perf_counter() - t0

    # Depth Anything V2 runs at a fixed internal resolution (~518×518) and
    # returns the depth map at that size. We resize to the input image's
    # resolution so the user can drop it straight onto the original
    # canvas at 1:1. Lanczos is the right kernel for greyscale maps:
    # smooth in flat areas, preserves sharp depth edges better than
    # nearest-neighbour and avoids the blocky artefacts from bilinear.
    if depth_arr.shape != (img.height, img.width):
        depth_native = Image.fromarray(depth_arr, mode="L")
        depth_resized = depth_native.resize(
            (img.width, img.height), resample=Image.LANCZOS
        )
        depth_arr = np.array(depth_resized, dtype=np.uint8)

    log.info(
        "Depth %s on %dx%d in %.2fs",
        app.state.depth_model_id, img.width, img.height, elapsed,
    )

    depth_img = Image.fromarray(depth_arr, mode="L")
    buf = io.BytesIO()
    depth_img.save(buf, format="PNG", compress_level=1)

    resp_headers = {
            "Cache-Control": "no-store",
            "X-Model": app.state.depth_model_id or "depth",
            "X-Width": str(img.width),
            "X-Height": str(img.height),
            "X-Elapsed-Ms": str(int(elapsed * 1000)),
        }
    if _depth_cold:
        resp_headers["X-Cold-Load"] = "true"
        log.info("Depth cold load — first request after model download")

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers=resp_headers,
    )


@app.post("/ground/text")
async def ground_text(
    image: UploadFile = File(..., alias="image"),
    phrases: str = Form(...),
    threshold: "float | None" = Form(None),
    refine: str = Form("1"),
) -> JSONResponse:
    """Text-grounded masking: free-text phrase(s) → soft mask(s).

    Pipeline per phrase: SAM 3 concept segmentation when available; otherwise
    CLIPSeg relevance heatmap → adaptive threshold → connected components →
    SAM 2 box+peak-point refinement of the top components → matte cleanup.

    Form fields:
        image:     JPEG/PNG/WebP, max MAX_UPLOAD_MB, longest side GROUND_MAX_SIDE.
        phrases:   JSON array of 1..GROUND_MAX_PHRASES strings.
        threshold: optional float 0..1 overriding GROUND_THRESHOLD — the
                   RELATIVE fraction of the heatmap peak a pixel must reach.
        refine:    "0" to skip SAM 2 refinement (faster, coarser).

    Response:
        {
          "width": int, "height": int, "model": str, "refine": bool,
          "results": [{
              "phrase": str,
              "found": bool,
              "score": float,        # heatmap peak 0..1
              "coverage": float,     # mask area / frame area
              "bbox": [x,y,w,h] | null,
              "components": int,
              "refined": bool,       # SAM 3 used, or SAM 2 fallback refined
              "maskPng": str | null  # base64 greyscale PNG, white = selected
          }]
        }

    A phrase that doesn't bind (peak < GROUND_MIN_PEAK) returns found=false
    with score — callers decide whether to fall back or surface it.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(415, f"unsupported content-type: {image.content_type}")

    try:
        phrase_list = json.loads(phrases)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"invalid phrases JSON: {e}")
    if not isinstance(phrase_list, list) or not phrase_list:
        raise HTTPException(400, "phrases must be a non-empty JSON array of strings")
    phrase_list = [str(p).strip() for p in phrase_list if str(p).strip()]
    if not phrase_list:
        raise HTTPException(400, "phrases contained no usable text")
    if len(phrase_list) > GROUND_MAX_PHRASES:
        raise HTTPException(400, f"too many phrases: {len(phrase_list)} > {GROUND_MAX_PHRASES}")

    thr = GROUND_THRESHOLD if threshold is None else max(0.05, min(0.95, float(threshold)))
    do_refine = str(refine).strip() not in ("0", "false", "False")

    contents = await _read_limited(image)
    if not contents:
        raise HTTPException(400, "empty upload")
    try:
        img = Image.open(io.BytesIO(contents))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise HTTPException(400, f"could not decode image: {e}")
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.width, img.height) > GROUND_MAX_SIDE:
        raise HTTPException(
            413,
            f"image too large ({img.width}x{img.height}); max longest side {GROUND_MAX_SIDE}px",
        )

    from scipy import ndimage

    W, H = img.size
    frame_area = float(W * H)
    rgb = np.asarray(img, dtype=np.uint8)

    t0 = time.perf_counter()

    if await run_in_threadpool(_ensure_sam3, app):
        try:
            results = await run_in_threadpool(
                _sam3_ground_results, app, img, phrase_list, rgb
            )
            elapsed = time.perf_counter() - t0
            log.info(
                "ground/text %s (%dx%d) engine=sam3 phrases=%s in %.2fs",
                image.filename or "<unnamed>", W, H, phrase_list, elapsed,
            )
            return JSONResponse({
                "width": W,
                "height": H,
                "model": app.state.sam3_model_id,
                "engine": "sam3",
                "refine": True,
                "elapsed_ms": int(elapsed * 1000),
                "results": results,
            }, headers={"Cache-Control": "no-store"})
        except Exception:
            log.exception("SAM 3 text grounding failed; using CLIPSeg fallback")

    if not await run_in_threadpool(_ensure_ground, app):
        raise HTTPException(
            501,
            "Text grounding not available on this server. Install SAM 3, or "
            "install torch + transformers and set GROUND_MODEL_ID to a CLIPSeg "
            "model in your HF cache.",
        )

    heatmaps = await run_in_threadpool(_ground_heatmaps, app, img, phrase_list)

    results = []
    any_refined = False
    for pi, phrase in enumerate(phrase_list):
        heat = heatmaps[pi]
        peak = float(heat.max())
        if peak < GROUND_MIN_PEAK:
            results.append({
                "phrase": phrase, "found": False, "score": round(peak, 4),
                "coverage": 0.0, "bbox": None, "components": 0,
                "refined": False, "maskPng": None,
            })
            continue

        # Peak-relative threshold (CLIPSeg sigmoids are range-compressed —
        # see GROUND_THRESHOLD comment), clamped to an absolute noise floor.
        eff_thr = max(GROUND_FLOOR, thr * peak)
        binary = heat >= eff_thr
        labeled, n = ndimage.label(binary)
        comps = []
        for ci in range(1, n + 1):
            comp = labeled == ci
            area = int(comp.sum())
            if area < GROUND_MIN_AREA_FRAC * frame_area:
                continue
            mean_heat = float(heat[comp].mean())
            comps.append((mean_heat * math.sqrt(area), comp, area))
        comps.sort(key=lambda t: -t[0])
        comps = comps[:GROUND_MAX_COMPONENTS]

        if not comps:
            results.append({
                "phrase": phrase, "found": False, "score": round(peak, 4),
                "coverage": 0.0, "bbox": None, "components": 0,
                "refined": False, "maskPng": None,
            })
            continue

        union = np.zeros((H, W), dtype=bool)
        used_refine = False
        for rank, (_, comp, area) in enumerate(comps):
            refined_mask = None
            if do_refine and rank < GROUND_REFINE_TOP:
                bbox = _bbox_from_mask(comp)
                # Positive point at the component's confidence peak.
                comp_heat = np.where(comp, heat, 0.0)
                py_, px_ = np.unravel_index(int(comp_heat.argmax()), comp_heat.shape)
                refined_mask = await run_in_threadpool(
                    _sam2_refine_box, app, img, bbox, (float(px_), float(py_))
                )
                if refined_mask is not None:
                    # Sanity-gate the refinement: a mask that exploded far
                    # beyond the prompt region (>4x area with little overlap)
                    # means SAM grabbed the wrong object — keep the coarse one.
                    r_area = float(refined_mask.sum())
                    inter = float((refined_mask & comp).sum())
                    if r_area > 4.0 * area and inter / float(area or 1) < 0.3:
                        refined_mask = None
            if refined_mask is not None:
                union |= refined_mask
                used_refine = True
            else:
                union |= comp

        alpha = (union.astype(np.uint8)) * 255
        if _MATTE_CLEANUP:
            try:
                alpha = await run_in_threadpool(clean_matte, alpha, rgb)
            except Exception:
                log.exception("clean_matte failed in /ground/text; using raw mask")
        any_refined = any_refined or used_refine

        results.append({
            "phrase": phrase,
            "found": True,
            "score": round(peak, 4),
            "coverage": round(float((alpha > 127).sum()) / frame_area, 4),
            "bbox": list(_bbox_from_mask(alpha > 127) or ()) or None,
            "components": len(comps),
            "refined": used_refine,
            "maskPng": _mask_png_b64(alpha),
        })

    elapsed = time.perf_counter() - t0
    log.info(
        "ground/text %s (%dx%d) phrases=%s refine=%s in %.2fs",
        image.filename or "<unnamed>", W, H,
        [r["phrase"] for r in results], any_refined, elapsed,
    )
    return JSONResponse({
        "width": W,
        "height": H,
        "model": app.state.ground_model_id,
        "engine": "clipseg",
        "refine": any_refined,
        "elapsed_ms": int(elapsed * 1000),
        "results": results,
    }, headers={"Cache-Control": "no-store"})


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


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )
