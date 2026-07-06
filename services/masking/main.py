"""Phosmith masking service.

Focused FastAPI service for the editor's AI selection tools: subject saliency
(rembg/BiRefNet), SAM 3.1 concept/box/point segmentation, monocular depth
(Depth Anything V2) and open-vocab text grounding. Split out of the original
`services/segment` service so it can deploy to its own Hugging Face Space,
separate from the erase/inpaint/auto-crop service.

No GPU required; auto-uses CUDA (NVIDIA) or MPS (Apple Silicon) when present.
SAM 3.1 is optional (gated checkpoint) — every SAM-3 route degrades to a
saliency/None fallback when the package or weights are unavailable.
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
from typing import List

import numpy as np

# OpenCV + SciPy power the saliency-matte cleanup. Optional: clean_matte()
# returns the matte untouched if either is missing.
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
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from rembg import new_session, remove
from starlette.concurrency import run_in_threadpool

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("masking-service")

# Load services/masking/.env (if present) before reading config.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except Exception:  # pragma: no cover - optional
    pass

# ─── Config ──────────────────────────────────────────────────────────────────

# Persistent model cache (HF_HOME + rembg U2NET_HOME). Set to a dir that
# survives restarts (HF Spaces persistent storage /data/models, Docker volume).
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "").strip() or None
if MODEL_CACHE_DIR:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)
    _u2net_dir = os.path.join(MODEL_CACHE_DIR, "u2net")
    os.makedirs(_u2net_dir, exist_ok=True)
    os.environ.setdefault("U2NET_HOME", _u2net_dir)
    log.info("model cache pinned to %r (HF_HOME + U2NET_HOME)", MODEL_CACHE_DIR)

MODEL_NAME = os.getenv("SEGMENT_MODEL", "isnet-general-use").strip()
DEPTH_MODEL_ID = os.getenv("DEPTH_MODEL_ID", "depth-anything/Depth-Anything-V2-Small-hf").strip()
DEPTH_CACHE_MAX = int(os.getenv("DEPTH_CACHE_MAX", "20").strip())
DEPTH_CACHE_MAX_PIXELS = int(os.getenv("DEPTH_CACHE_MAX_PIXELS", str(2048 * 2048)).strip())
DEPTH_MAX_SIDE = int(os.getenv("DEPTH_MAX_SIDE", "2048").strip())
PORT = int(os.getenv("PORT", "8002").strip())
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "24").strip())
SEGMENT_MAX_SIDE = int(os.getenv("SEGMENT_MAX_SIDE", "2048").strip())
SEGMENT_EAGER_MODELS = os.getenv("SEGMENT_EAGER_MODELS", "0").strip() not in ("0", "false", "False", "")

# SAM 3.1 — preferred for subject/concept/box/point masks. Optional (gated).
SAM3_ENABLE = os.getenv("SAM3_ENABLE", "1").strip() not in ("0", "false", "False", "")
SAM3_MODEL_ID = os.getenv("SAM3_MODEL_ID", "facebook/sam3.1").strip()
SAM3_CHECKPOINT_PATH = os.getenv("SAM3_CHECKPOINT_PATH", "").strip() or None
SAM3_CONFIDENCE = float(os.getenv("SAM3_CONFIDENCE", "0.25").strip())
SAM3_SUBJECT_PROMPT = os.getenv("SAM3_SUBJECT_PROMPT", "main subject").strip() or "main subject"
SAM3_INSTANCES_MAX = int(os.getenv("SAM3_INSTANCES_MAX", "24").strip())
SAM3_EAGER = os.getenv("SAM3_EAGER", "0").strip() not in ("0", "false", "False", "")
SAM2_MAX_CLICKS = int(os.getenv("SAM2_MAX_CLICKS", "50").strip())

# High-precision matting refinement of SAM 3 masks: turn the coarse binary SAM
# mask into a hair-accurate soft alpha via a trimap + alpha-matting solve guided
# by the RGB image (PyMatting closed-form, with a pure-cv2 guided-filter
# fallback). Off by default (adds CPU latency); enable with SAM3_REFINE_MATTING=1.
SAM3_REFINE_MATTING = os.getenv("SAM3_REFINE_MATTING", "0").strip() not in ("0", "false", "False", "")
SAM3_REFINE_METHOD = os.getenv("SAM3_REFINE_METHOD", "pymatting_cf").strip()  # pymatting_cf | pymatting_knn | guided
SAM3_REFINE_MAX_SIDE = int(os.getenv("SAM3_REFINE_MAX_SIDE", "1024").strip())  # cap matting resolution (CPU perf/RAM)
SAM3_REFINE_ERODE = int(os.getenv("SAM3_REFINE_ERODE", "6").strip())          # sure-fg shrink (px @ full res)
SAM3_REFINE_DILATE = int(os.getenv("SAM3_REFINE_DILATE", "18").strip())       # unknown-band width — room for hair

# Matte-cleanup tuning (see clean_matte).
MATTE_HOLE_FILL_MAX_FRAC = float(os.getenv("MATTE_HOLE_FILL_MAX_FRAC", "0.02").strip())
MATTE_FAINT_RECOVER_MAX = int(os.getenv("MATTE_FAINT_RECOVER_MAX", "96").strip())
MATTE_FAINT_MIN_SOLID_FRAC = float(os.getenv("MATTE_FAINT_MIN_SOLID_FRAC", "0.50").strip())

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]

# Text grounding (/ground/text).
GROUND_MAX_SIDE = int(os.getenv("GROUND_MAX_SIDE", "2048").strip())
GROUND_MAX_PHRASES = int(os.getenv("GROUND_MAX_PHRASES", "4").strip())

# rembg model registry. isnet-*/u2net*/silueta = MIT; bria-rmbg = CC BY-NC.
# birefnet-* need rembg >= 2.0.57 (this env: 2.0.75); birefnet-general is the
# quality pick for backlit/complex mattes (~930 MB first download).
ALLOWED_MODELS = {
    "isnet-general-use", "u2net", "u2netp", "u2net_human_seg",
    "u2net_cloth_seg", "silueta", "bria-rmbg",
    "birefnet-general", "birefnet-general-lite", "birefnet-portrait",
    "birefnet-dis", "birefnet-hrsod", "birefnet-massive",
}
if MODEL_NAME not in ALLOWED_MODELS:
    log.warning("unknown SEGMENT_MODEL=%r; falling back to isnet-general-use", MODEL_NAME)
    MODEL_NAME = "isnet-general-use"


# ─── Execution providers / torch device ─────────────────────────────────────

def detect_providers() -> List[str]:
    """ONNX providers for the rembg session: CUDA when present, else CPU.

    CoreML is deliberately NOT preferred on Apple Silicon. This process also
    loads the torch/MPS models (SAM 3, Depth Anything), and the rembg CoreML
    session intermittently HARD-FAILS at inference under that Neural-Engine
    contention — ONNXRuntime error -1, "Unable to compute the prediction using a
    neural network model" — which 500s /segment/instances (the editor's Select
    Subject saliency-seed path). CoreML is also ~6× slower than CPU for
    isnet-general-use here (≈30s vs ≈5s). Opt back in explicitly with
    SEGMENT_PROVIDERS=CoreMLExecutionProvider,CPUExecutionProvider."""
    override = os.getenv("SEGMENT_PROVIDERS", "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]
    providers: List[str] = ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort  # type: ignore
        if "CUDAExecutionProvider" in set(ort.get_available_providers()):
            providers.insert(0, "CUDAExecutionProvider")
    except Exception as e:  # pragma: no cover
        log.debug("onnxruntime provider probe failed: %s", e)
    return providers


def _rembg_matte(img: "Image.Image") -> "Image.Image":
    """rembg saliency matte (single-channel 'L'), self-healing to a CPU session.

    If the active ONNX provider fails at inference (e.g. a CoreML session
    configured via SEGMENT_PROVIDERS dying under MPS contention), rebuild the
    session on CPU once and retry rather than 500-ing the request. CPU is the
    default provider, so on a healthy box this is a plain passthrough."""
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


def detect_torch_device():
    """Best torch device: CUDA > MPS > CPU. Returns (device, label)."""
    try:
        import torch  # type: ignore
    except ImportError:
        return None, "torch-missing"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


# ─── Depth Anything V2 ───────────────────────────────────────────────────────

DEPTH_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _image_hash(img: Image.Image) -> str:
    """Stable 64-bit hash of pixel data for the depth cache key."""
    return hashlib.sha256(img.tobytes()).hexdigest()[:16]


def _depth_predict(app, img: Image.Image) -> np.ndarray:
    """Depth Anything V2 → uint8 HxW (white=near). Cached by image hash."""
    key = _image_hash(img)
    cached = DEPTH_CACHE.get(key)
    if cached is not None:
        DEPTH_CACHE.move_to_end(key)
        return cached

    import torch  # type: ignore
    processor = app.state.depth_processor
    model = app.state.depth_model
    device = app.state.depth_device

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(pixel_values=inputs.pixel_values)
    depth = outputs.predicted_depth.squeeze(0).cpu().numpy()
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-6:
        normalised = np.zeros_like(depth, dtype=np.uint8)
    else:
        normalised = ((depth - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)

    if normalised.size <= DEPTH_CACHE_MAX_PIXELS:
        DEPTH_CACHE[key] = normalised
        while len(DEPTH_CACHE) > DEPTH_CACHE_MAX:
            DEPTH_CACHE.popitem(last=False)
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
    """Clean a saliency matte into a solid subject mask while preserving soft
    edges: fill small interior holes (model drop-outs), keep large ones (genuine
    see-through gaps), drop specks, recover deep faint pixels. HxW uint8 in/out.
    Returns the input untouched when OpenCV/SciPy are unavailable."""
    if not _MATTE_CLEANUP:
        return matte_u8
    if matte_u8.ndim != 2:
        matte_u8 = matte_u8[..., 0]
    h, w = matte_u8.shape
    matte = matte_u8

    k = max(3, min(9, int(round(min(h, w) * 0.0035))))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    binary = (matte > 24).astype(np.uint8)
    if not binary.any():
        return matte

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_area = max(int(0.0005 * h * w), 64)
    keep = np.zeros(num, dtype=bool)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep[1:] = areas[1:] >= min_area
    core = keep[labels].astype(np.uint8)
    if not core.any():
        core = binary

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

    region = cv2.dilate(clean_bin, kernel, iterations=1)
    gated = np.where(region > 0, matte, 0).astype(np.uint8)
    out = gated.copy()
    out[small_holes] = 255

    solid_frac = float(((matte >= 128) & (clean_bin > 0)).sum()) / subject_area
    if solid_frac >= MATTE_FAINT_MIN_SOLID_FRAC:
        dist = cv2.distanceTransform(clean_bin, cv2.DIST_L2, 3)
        deep = dist > (3.0 * k)
        faint_core = deep & (matte > 24) & (matte <= MATTE_FAINT_RECOVER_MAX)
        out[faint_core] = 255
    return out


# ─── SAM 3 helpers ───────────────────────────────────────────────────────────

_SAM3_LOCK = threading.Lock()
_SAM3_INFER_LOCK = threading.Lock()


def _sam3_loadable() -> bool:
    try:
        return bool(importlib.util.find_spec("sam3") and importlib.util.find_spec("torch"))
    except Exception:
        return False


def _sam3_device_label() -> str:
    """SAM 3's builder handles CUDA/CPU reliably; avoid MPS mismatches → CPU on Apple."""
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sam3_checkpoint_kwargs():
    if SAM3_CHECKPOINT_PATH:
        return {"checkpoint_path": SAM3_CHECKPOINT_PATH, "load_from_HF": False}
    if SAM3_MODEL_ID.lower().replace("_", ".") in {"sam3.1", "facebook/sam3.1"}:
        from sam3.model_builder import download_ckpt_from_hf  # type: ignore

        return {"checkpoint_path": download_ckpt_from_hf(version="sam3.1"), "load_from_HF": False}
    return {"checkpoint_path": None, "load_from_HF": True}


_SAM3_CPU_PATCHED = False


def _patch_sam3_fused_mlp_for_cpu() -> None:
    """Patch two upstream sam3/torch CPU-on-Apple-Silicon issues once: the
    bf16-casting fused addmm_act (dtype mismatch on CPU) and a pin_memory()
    call that crashes when MPS is merely available."""
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
    """Load SAM 3 into app.state (blocking; caller holds _SAM3_LOCK)."""
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
        # Instance interactivity = point/box click-select prompts (/sam2/click).
        model = build_sam3_image_model(
            device=device,
            enable_segmentation=True,
            enable_inst_interactivity=True,
            **kwargs,
        )
        processor = Sam3Processor(model, device=device, confidence_threshold=SAM3_CONFIDENCE)
        app.state.sam3_model = model
        app.state.sam3_processor = processor
        app.state.sam3_device = device
        app.state.sam3_model_id = SAM3_MODEL_ID
        app.state.sam3_available = True
        log.info("SAM 3 ready in %.1fs on %s", time.perf_counter() - t0, device)
        return True
    except ImportError:
        log.warning(
            "SAM 3 package not installed; SAM 3 routes will fall back. Install "
            "facebookresearch/sam3 and authenticate with HF for the gated checkpoint."
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
    """Tight [x, y, w, h] of a non-empty boolean mask."""
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _bbox_from_mask(mask: np.ndarray) -> "tuple[int, int, int, int] | None":
    """Tight (x, y, w, h) of True pixels, or None if empty."""
    if mask is None or mask.size == 0 or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _mask_png_b64(alpha: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(alpha, "L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _soft_alpha_from_mask(mask: np.ndarray) -> np.ndarray:
    alpha = (mask.astype(np.uint8)) * 255
    return np.asarray(
        Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(1.0)), dtype=np.uint8
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
            "class_id": -1, "label": label,
            "confidence": float(scores[i]) if i < len(scores) else 1.0,
            "mask": mb, "area": area, "source": "sam3",
        })
    out.sort(key=lambda inst: -inst["area"])
    return out[:SAM3_INSTANCES_MAX]


def _sam3_instances_for_prompt(app, img: Image.Image, prompt: str) -> "list[dict]":
    processor = app.state.sam3_processor
    with _SAM3_INFER_LOCK:
        state = processor.set_image(img)
        output = processor.set_text_prompt(state=state, prompt=prompt)
    return _instances_from_sam3_output(output, prompt)


def _best_mask_from_output(out: dict, W: int, H: int):
    """Pick the highest-score binary mask from a SAM 3 geometric-prompt output.
    Returns (uint8 HxW white=selected, score) — all-zero when nothing found."""
    masks = out.get("masks") if out else None
    scores = out.get("scores") if out else None
    if masks is None or len(masks) == 0:
        return np.zeros((H, W), dtype=np.uint8), 0.0
    s = scores.detach().cpu().numpy().reshape(-1) if scores is not None else np.array([0.0])
    best = int(s.argmax())
    m = np.squeeze(masks[best].detach().cpu().numpy())
    binary = m > (0.0 if float(m.min()) < 0.0 else 0.5)
    return (binary.astype(np.uint8) * 255), float(s[best])


def _sam3_box_mask(app, img: Image.Image, box_xyxy: "tuple[float, float, float, float]"):
    """SAM 3.1 box-prompted single-object selection. box_xyxy in pixels →
    (mask uint8 HxW, score). add_geometric_prompt wants [cx,cy,w,h] in 0..1."""
    W, H = img.width, img.height
    x0, y0, x1, y1 = box_xyxy
    box_norm = [((x0 + x1) / 2.0) / W, ((y0 + y1) / 2.0) / H, (x1 - x0) / W, (y1 - y0) / H]
    processor = app.state.sam3_processor
    with _SAM3_INFER_LOCK:
        state = processor.set_image(img)
        out = processor.add_geometric_prompt(box=box_norm, label=True, state=state)
    return _best_mask_from_output(out, W, H)


def _box_norm_from_points(points, labels, W, H):
    """Box [cx,cy,w,h] in 0..1 around the positive click points — the click-select
    fallback when SAM 3 doesn't accept raw point geometry. A single point gets a
    ~16%-of-frame box centred on it; multiple positives use their padded bbox."""
    pos = [(x, y) for (x, y), l in zip(points, labels) if l]
    if not pos:
        pos = points
    xs = [p[0] for p in pos]
    ys = [p[1] for p in pos]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 - x0 < 1 and y1 - y0 < 1:
        side = 0.16 * min(W, H)
        x0, x1 = xs[0] - side / 2, xs[0] + side / 2
        y0, y1 = ys[0] - side / 2, ys[0] + side / 2
    else:
        padx, pady = 0.12 * (x1 - x0 + 1), 0.12 * (y1 - y0 + 1)
        x0, x1, y0, y1 = x0 - padx, x1 + padx, y0 - pady, y1 + pady
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(W), x1), min(float(H), y1)
    return [((x0 + x1) / 2.0) / W, ((y0 + y1) / 2.0) / H, max(1.0, x1 - x0) / W, max(1.0, y1 - y0) / H]


def _sam3_point_mask(app, img: Image.Image, points, labels):
    """SAM 3.1 point-click selection. Tries raw point geometry across the likely
    add_geometric_prompt signatures; on any failure, falls back to a box around
    the positive clicks (the proven box prompt). Returns (mask uint8, score)."""
    W, H = img.width, img.height
    npts = [[x / W, y / H] for x, y in points]
    processor = app.state.sam3_processor
    with _SAM3_INFER_LOCK:
        state = processor.set_image(img)
        out = None
        for kwargs in (
            {"points": npts, "labels": list(labels)},
            {"point_coords": npts, "point_labels": list(labels)},
            {"points": npts, "point_labels": list(labels)},
        ):
            try:
                out = processor.add_geometric_prompt(state=state, **kwargs)
                if out and out.get("masks") is not None and len(out.get("masks")):
                    break
                out = None
            except (TypeError, KeyError):
                out = None
            except Exception:
                out = None
        if out is None:
            box_norm = _box_norm_from_points(points, labels, W, H)
            out = processor.add_geometric_prompt(box=box_norm, label=True, state=state)
    return _best_mask_from_output(out, W, H)


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
        "class_id": -1, "label": label, "confidence": 1.0,
        "mask": mb, "area": int(mb.sum()), "source": "saliency",
    }


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
                    "phrase": phrase, "found": False, "score": 0.0, "coverage": 0.0,
                    "bbox": None, "components": 0, "refined": True, "source": "sam3", "maskPng": None,
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
                    log.exception("clean_matte failed after SAM 3 grounding; using raw mask")
            bbox = _bbox_from_mask(alpha > 127)
            results.append({
                "phrase": phrase, "found": True,
                "score": round(max(float(i["confidence"]) for i in instances), 4),
                "coverage": round(float((alpha > 127).sum()) / frame_area, 4),
                "bbox": list(bbox) if bbox else None,
                "components": len(instances), "refined": True, "source": "sam3",
                "maskPng": _mask_png_b64(alpha),
            })
    return results


# ─── Depth lazy loader ───────────────────────────────────────────────────────

_DEPTH_LOCK = threading.Lock()


def _torch_stack_loadable() -> bool:
    try:
        return bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))
    except Exception:  # pragma: no cover
        return False


def _load_depth(app: FastAPI) -> bool:
    """Load Depth Anything V2 into app.state (blocking; caller holds _DEPTH_LOCK)."""
    try:
        import torch  # type: ignore  # noqa: F401
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # type: ignore

        device, device_label = detect_torch_device()
        if device is None:
            raise ImportError("torch not available")
        log.info("loading Depth model %r onto %s ...", DEPTH_MODEL_ID, device_label)
        t0 = time.perf_counter()
        app.state.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        app.state.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(device)
        app.state.depth_model.eval()
        app.state.depth_device = device
        app.state.depth_model_id = DEPTH_MODEL_ID
        app.state.depth_available = True
        log.info("Depth ready in %.1fs on %s", time.perf_counter() - t0, device_label)
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
    """Lazily load Depth; return True/'cold' if ready, False if failed."""
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


# ─── App ─────────────────────────────────────────────────────────────────────

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
        app.state.session = new_session("u2net", providers=["CPUExecutionProvider"])
        app.state.model_name = "u2net"
        app.state.providers = ["CPUExecutionProvider"]

    app.state.sam3_available = False
    app.state.sam3_load_failed = not SAM3_ENABLE
    app.state.sam3_model_id = SAM3_MODEL_ID
    app.state.depth_available = False
    app.state.depth_load_failed = False
    app.state.depth_model_id = DEPTH_MODEL_ID

    if SAM3_EAGER:
        await run_in_threadpool(_ensure_sam3, app)
    if SEGMENT_EAGER_MODELS:
        await run_in_threadpool(_ensure_depth, app)
    else:
        log.info("SAM 3 and Depth load lazily on first use (SAM3_EAGER/SEGMENT_EAGER_MODELS=1 to preload).")

    yield
    DEPTH_CACHE.clear()
    log.info("shutting down masking service")


app = FastAPI(
    title="Phosmith Masking Service",
    version="1.0.0",
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


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Reject oversize POSTs by Content-Length before the body is read."""
    if request.method == "POST":
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
            return Response(content=f"file too large (> {MAX_UPLOAD_MB}MB)", status_code=413)
    return await call_next(request)


async def _read_limited(image: UploadFile) -> bytes:
    """Stream an UploadFile into memory, aborting past the upload cap (defends
    against chunked uploads that bypass the Content-Length middleware)."""
    contents = bytearray()
    while True:
        chunk = await run_in_threadpool(image.file.read, 64 * 1024)
        if not chunk:
            break
        contents.extend(chunk)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file too large (> {MAX_UPLOAD_MB}MB)")
    return bytes(contents)


def _decode_image(contents: bytes, *, max_side: int, rgb_only: bool = False) -> Image.Image:
    """Decode + validate an upload into a PIL image, enforcing the side cap."""
    if not contents:
        raise HTTPException(400, "empty upload")
    try:
        img = Image.open(io.BytesIO(contents))
        img.load()
        # Camera JPEGs store the rotation in EXIF; every model here works on the
        # raw pixels, so without this the mask comes back rotated relative to
        # what any EXIF-aware viewer (or the browser) shows. No-op without EXIF.
        img = ImageOps.exif_transpose(img)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise HTTPException(400, f"could not decode image: {e}")
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N"):
        # 16/32-bit greyscale: PIL's convert("RGB") truncates instead of
        # rescaling, which blacks the frame out and every model sees nothing.
        arr = np.asarray(img, dtype=np.float32)
        img = Image.fromarray(np.clip(arr / 257.0, 0, 255).astype(np.uint8), "L")
    if rgb_only:
        if img.mode != "RGB":
            img = img.convert("RGB")
    elif img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    if max(img.width, img.height) > max_side:
        raise HTTPException(
            413, f"image too large ({img.width}x{img.height}); max longest side is {max_side}px"
        )
    return img


def _require_image_ct(image: UploadFile) -> None:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(415, f"unsupported content-type: {image.content_type}")


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> dict:
    return {"status": "ok", "message": "Phosmith Masking Service is running"}


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
        "sam3_available": getattr(app.state, "sam3_available", False)
        or (SAM3_ENABLE and _sam3_loadable() and not getattr(app.state, "sam3_load_failed", False)),
        "sam3_loaded": getattr(app.state, "sam3_available", False),
        "sam3_model": app.state.sam3_model_id,
        "sam3_confidence": SAM3_CONFIDENCE,
        "depth_available": app.state.depth_available
        or (_torch_stack_loadable() and not app.state.depth_load_failed),
        "depth_loaded": app.state.depth_available,
        "depth_model": app.state.depth_model_id,
    }


@app.post("/warmup")
async def warmup() -> dict:
    """Pre-load SAM 3 + Depth so the first real request is fast."""
    results = {}
    sam3_was = getattr(app.state, "sam3_available", False)
    sam3_status = await run_in_threadpool(_ensure_sam3, app)
    results["sam3"] = "already_loaded" if sam3_was else "loaded" if sam3_status else "failed"
    depth_was = app.state.depth_available
    depth_status = await run_in_threadpool(_ensure_depth, app)
    results["depth"] = "already_loaded" if depth_was else "loaded" if depth_status else "failed"
    log.info("warmup results: %s", results)
    return {"status": "ok", "models": results}


@app.post("/segment")
async def segment(image: UploadFile = File(..., alias="image")) -> Response:
    """Subject background-removal → RGBA PNG (alpha = subject). SAM 3 concept
    first, rembg saliency fallback."""
    _require_image_ct(image)
    contents = await _read_limited(image)
    t0 = time.perf_counter()
    img = _decode_image(contents, max_side=SEGMENT_MAX_SIDE)
    img_rgb = img.convert("RGB")
    np_rgb = np.asarray(img_rgb)

    subject_mode = "sam3"
    subjects = 0
    final_alpha = None
    matte = None  # cleaned saliency matte, computed at most once

    async def _saliency_matte():
        nonlocal matte
        if matte is None:
            try:
                matte_img = await run_in_threadpool(_rembg_matte, img)
            except Exception as e:
                log.exception("rembg.remove failed")
                raise HTTPException(500, f"segmentation failed: {e}")
            raw_matte = np.asarray(matte_img.convert("L"), dtype=np.uint8)
            matte = await run_in_threadpool(clean_matte, raw_matte)
        return matte

    if await run_in_threadpool(_ensure_sam3, app):
        if SAM3_SUBJECT_PROMPT == "main subject":
            # SAM 3's open-vocab detector cannot ground the abstract default
            # prompt — a concept pass always comes back empty after a full
            # detection sweep (~50 s/image on CPU). Take the same saliency-bbox
            # → SAM 3 box-prompt path as /segment/instances' subject_box mode:
            # one cheap rembg pass seeds SAM 3's strongest single-object prompt.
            try:
                m8 = await _saliency_matte()
                bbox = _bbox_from_mask(m8 > 127)
                if bbox is not None:
                    bx, by, bw, bh = bbox
                    m, _score = await run_in_threadpool(
                        _sam3_box_mask, app, img_rgb, (bx, by, bx + bw, by + bh)
                    )
                    mb = m > 127
                    if mb.any():
                        subjects = 1
                        final_alpha = _soft_alpha_from_mask(mb)
                        if SAM3_REFINE_MATTING:
                            final_alpha = await run_in_threadpool(
                                _refine_alpha_with_matting, final_alpha, np_rgb, mb
                            )
            except HTTPException:
                raise
            except Exception:
                log.exception("SAM 3 box-seed subject failed; using saliency fallback")
        else:
            # A deployment-specific CONCRETE prompt — the concept pass is real.
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
        final_alpha = await _saliency_matte()
        subjects = 1 if final_alpha.any() else 0

    rgba = np.dstack([np_rgb, final_alpha]).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    elapsed = time.perf_counter() - t0
    log.info("segmented %dx%d mode=%s subjects=%d in %.2fs", img.width, img.height, subject_mode, subjects, elapsed)
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
    subject_box: bool = Form(False),
) -> JSONResponse:
    """SAM 3-first concept instance detection: one greyscale mask PER instance
    (label, confidence, bbox, centroid, area) plus a union mask. `subject_box`
    takes the fast saliency-bbox → single SAM 3 box-prompt path for the generic
    'main subject' concept (SAM 3's detector won't ground that abstract noun)."""
    _require_image_ct(image)
    contents = await _read_limited(image)
    t0 = time.perf_counter()
    img = _decode_image(contents, max_side=SEGMENT_MAX_SIDE)
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    concept = str(prompt or SAM3_SUBJECT_PROMPT).strip() or SAM3_SUBJECT_PROMPT

    mode = "sam3"
    source_model = app.state.sam3_model_id
    instances = []

    if subject_box:
        try:
            matte_img = await run_in_threadpool(_rembg_matte, img)
        except Exception as e:
            log.exception("rembg.remove failed")
            raise HTTPException(500, f"segmentation failed: {e}")
        raw_matte = np.asarray(matte_img.convert("L"), dtype=np.uint8)
        matte = await run_in_threadpool(clean_matte, raw_matte)
        bbox = _bbox_from_mask(matte > 127)
        if bbox is not None and await run_in_threadpool(_ensure_sam3, app):
            bx, by, bw, bh = bbox
            try:
                m, score = await run_in_threadpool(
                    _sam3_box_mask, app, img_rgb, (bx, by, bx + bw, by + bh)
                )
                mb = m > 127
                if mb.any():
                    instances = [{
                        "class_id": -1, "label": concept, "confidence": float(score),
                        "mask": mb, "area": int(mb.sum()), "source": "sam3-box",
                    }]
            except Exception:
                log.exception("SAM 3 box-seed failed; using saliency matte")
        if not instances:
            mode = "saliency"
            source_model = app.state.model_name
            inst = _saliency_instance(concept, matte)
            instances = [inst] if inst else []
    else:
        if await run_in_threadpool(_ensure_sam3, app):
            try:
                instances = await run_in_threadpool(_sam3_instances_for_prompt, app, img_rgb, concept)
            except Exception:
                log.exception("SAM 3 instance detection failed; falling back to saliency")
                instances = []
        if not instances:
            mode = "saliency"
            source_model = app.state.model_name
            try:
                matte_img = await run_in_threadpool(_rembg_matte, img)
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
                "index": idx, "label": inst["label"], "class_id": inst["class_id"],
                "confidence": round(float(inst["confidence"]), 4), "source": inst["source"],
                "bbox": _bbox_of(mb), "area": int(inst["area"]),
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
    log.info("segment/instances (%dx%d) mode=%s prompt=%r count=%d in %.2fs", w, h, mode, concept, len(items), elapsed)
    return JSONResponse(
        {
            "width": w, "height": h, "model": source_model, "prompt": concept,
            "mode": mode, "count": len(items), "truncated": truncated,
            "instances": items, "union_png": union_b64, "elapsed_ms": int(elapsed * 1000),
        },
        headers={"Cache-Control": "no-store"},
    )


def _parse_box(raw: str, W: int, H: int) -> "tuple[float, float, float, float]":
    """Validate a JSON [x0,y0,x1,y1] box against the image bounds."""
    try:
        box_data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"invalid box JSON: {e}")
    if not (isinstance(box_data, list) and len(box_data) == 4):
        raise HTTPException(400, f"box must be [x0, y0, x1, y1]; got {box_data!r}")
    for i, v in enumerate(box_data):
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise HTTPException(400, f"box[{i}] must be a finite number; got {v!r}")
    x0, y0, x1, y1 = box_data
    if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
        raise HTTPException(400, f"box {box_data} is degenerate or outside bounds ({W}x{H})")
    return float(x0), float(y0), float(x1), float(y1)


@app.post("/segment/box")
async def segment_box(
    image: UploadFile = File(..., alias="image"),
    box: str = Form(...),
) -> Response:
    """Box-prompted single-object selection with SAM 3.1 (strongest single-object
    prompt). box: JSON [x0,y0,x1,y1] in pixels. → greyscale PNG (white=object)."""
    _sam3_status = await run_in_threadpool(_ensure_sam3, app)
    if not _sam3_status:
        raise HTTPException(501, "SAM 3 not available. Install facebookresearch/sam3 + HF auth for facebook/sam3.1.")
    _sam3_cold = _sam3_status == "cold"
    _require_image_ct(image)
    contents = await _read_limited(image)
    img = _decode_image(contents, max_side=SEGMENT_MAX_SIDE, rgb_only=True)
    x0, y0, x1, y1 = _parse_box(box, img.width, img.height)

    t0 = time.perf_counter()
    try:
        best_mask, score = await run_in_threadpool(_sam3_box_mask, app, img, (x0, y0, x1, y1))
    except Exception as e:
        log.exception("segment/box failed")
        raise HTTPException(500, f"sam3 box inference failed: {e}")
    elapsed = time.perf_counter() - t0
    log.info("SAM 3 box select %dx%d in %.2fs (score=%.3f)", img.width, img.height, elapsed, score)

    buf = io.BytesIO()
    Image.fromarray(best_mask, mode="L").save(buf, format="PNG", optimize=True)
    resp_headers = {
        "Cache-Control": "no-store", "X-Model": app.state.sam3_model_id or "sam3.1",
        "X-Score": f"{score:.4f}", "X-Elapsed-Ms": str(int(elapsed * 1000)),
    }
    if _sam3_cold:
        resp_headers["X-Cold-Load"] = "true"
    return Response(content=buf.getvalue(), media_type="image/png", headers=resp_headers)


def _parse_clicks(clicks_raw, points_raw, labels_raw, W, H):
    """Accept either `clicks` ([[x,y,label],...]) or `points`+`labels`. Returns
    (points[(x,y)], labels[int]) validated against the image bounds."""
    points: "list[tuple[float, float]]" = []
    labels: "list[int]" = []
    if clicks_raw:
        try:
            data = json.loads(clicks_raw)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"invalid clicks JSON: {e}")
        if not isinstance(data, list):
            raise HTTPException(400, "clicks must be a JSON array of [x, y, label]")
        for c in data:
            if not (isinstance(c, list) and len(c) >= 2):
                raise HTTPException(400, f"each click must be [x, y, label?]; got {c!r}")
            points.append((float(c[0]), float(c[1])))
            labels.append(1 if len(c) < 3 else int(bool(c[2])))
    elif points_raw:
        try:
            pts = json.loads(points_raw)
            lbls = json.loads(labels_raw) if labels_raw else [1] * len(pts)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"invalid points/labels JSON: {e}")
        for p, l in zip(pts, lbls):
            points.append((float(p[0]), float(p[1])))
            labels.append(int(bool(l)))
    if not points:
        raise HTTPException(400, "no click points supplied")
    if len(points) > SAM2_MAX_CLICKS:
        raise HTTPException(400, f"too many clicks: {len(points)} > {SAM2_MAX_CLICKS}")
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise HTTPException(400, "click coords must be finite")
        if not (-1 <= x <= W + 1 and -1 <= y <= H + 1):
            raise HTTPException(400, f"click ({x}, {y}) outside bounds ({W}x{H})")
    return points, labels


@app.post("/sam2/click")
async def sam2_click(
    image: UploadFile = File(..., alias="image"),
    clicks: str = Form(None),
    points: str = Form(None),
    labels: str = Form(None),
    box: str = Form(None),
) -> Response:
    """Point-click (and optional box) selection via SAM 3.1 interactivity.

    Form fields (either click format works):
        clicks: JSON [[x, y, label], ...]  (label 1=include, 0=exclude)
        points + labels: JSON [[x,y],...] + JSON [1,0,...]
        box (optional): JSON [x0,y0,x1,y1] — used alone or to seed the prompt.
    → greyscale PNG (white = selected). Point geometry is best-effort: when the
    installed SAM 3.1 doesn't accept raw points, a box around the positive
    clicks is used; the browser also keeps an on-device SlimSAM fallback."""
    _sam3_status = await run_in_threadpool(_ensure_sam3, app)
    if not _sam3_status:
        raise HTTPException(501, "SAM 3 not available. Install facebookresearch/sam3 + HF auth for facebook/sam3.1.")
    _sam3_cold = _sam3_status == "cold"
    _require_image_ct(image)
    contents = await _read_limited(image)
    img = _decode_image(contents, max_side=SEGMENT_MAX_SIDE, rgb_only=True)
    W, H = img.width, img.height

    t0 = time.perf_counter()
    try:
        if box and not (clicks or points):
            x0, y0, x1, y1 = _parse_box(box, W, H)
            best_mask, score = await run_in_threadpool(_sam3_box_mask, app, img, (x0, y0, x1, y1))
        else:
            pts, lbls = _parse_clicks(clicks, points, labels, W, H)
            best_mask, score = await run_in_threadpool(_sam3_point_mask, app, img, pts, lbls)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("sam2/click failed")
        raise HTTPException(500, f"sam3 click inference failed: {e}")
    elapsed = time.perf_counter() - t0
    log.info("SAM 3 click select %dx%d in %.2fs (score=%.3f)", W, H, elapsed, score)

    buf = io.BytesIO()
    Image.fromarray(best_mask, mode="L").save(buf, format="PNG", optimize=True)
    resp_headers = {
        "Cache-Control": "no-store", "X-Model": app.state.sam3_model_id or "sam3.1",
        "X-Score": f"{score:.4f}", "X-Elapsed-Ms": str(int(elapsed * 1000)),
    }
    if _sam3_cold:
        resp_headers["X-Cold-Load"] = "true"
    return Response(content=buf.getvalue(), media_type="image/png", headers=resp_headers)


@app.post("/depth")
async def depth(image: UploadFile = File(..., alias="image")) -> Response:
    """Monocular depth (Depth Anything V2) → greyscale PNG at input resolution
    (white=near). Per-image min-max normalised; LRU-cached by image hash."""
    _depth_status = await run_in_threadpool(_ensure_depth, app)
    if not _depth_status:
        raise HTTPException(501, "Depth Anything V2 not available. Install torch + transformers and restart.")
    _depth_cold = _depth_status == "cold"
    _require_image_ct(image)
    contents = await _read_limited(image)
    img = _decode_image(contents, max_side=DEPTH_MAX_SIDE, rgb_only=True)

    t0 = time.perf_counter()
    try:
        depth_arr = _depth_predict(app, img)
    except Exception as e:
        log.exception("depth predict failed")
        raise HTTPException(500, f"depth inference failed: {e}")
    elapsed = time.perf_counter() - t0

    # The model runs at ~518² internally; resize back to input resolution (Lanczos
    # preserves depth edges) so the map drops onto the canvas 1:1.
    if depth_arr.shape != (img.height, img.width):
        depth_resized = Image.fromarray(depth_arr, mode="L").resize((img.width, img.height), Image.LANCZOS)
        depth_arr = np.array(depth_resized, dtype=np.uint8)
    log.info("Depth %dx%d in %.2fs", img.width, img.height, elapsed)

    buf = io.BytesIO()
    Image.fromarray(depth_arr, mode="L").save(buf, format="PNG", optimize=True)
    resp_headers = {
        "Cache-Control": "no-store", "X-Model": app.state.depth_model_id or "depth",
        "X-Width": str(img.width), "X-Height": str(img.height),
        "X-Elapsed-Ms": str(int(elapsed * 1000)),
    }
    if _depth_cold:
        resp_headers["X-Cold-Load"] = "true"
    return Response(content=buf.getvalue(), media_type="image/png", headers=resp_headers)


@app.post("/ground/text")
async def ground_text(
    image: UploadFile = File(..., alias="image"),
    phrases: str = Form(...),
) -> JSONResponse:
    """Text-grounded masking: free-text phrase(s) → soft mask(s) via SAM 3
    open-vocab concept grounding. phrases: JSON array of 1..GROUND_MAX_PHRASES."""
    _require_image_ct(image)
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

    contents = await _read_limited(image)
    img = _decode_image(contents, max_side=GROUND_MAX_SIDE, rgb_only=True)
    W, H = img.size
    rgb = np.asarray(img, dtype=np.uint8)

    t0 = time.perf_counter()
    if not await run_in_threadpool(_ensure_sam3, app):
        raise HTTPException(501, "Text grounding requires SAM 3 (facebookresearch/sam3 + HF auth for facebook/sam3.1).")
    try:
        results = await run_in_threadpool(_sam3_ground_results, app, img, phrase_list, rgb)
    except Exception as e:
        log.exception("SAM 3 text grounding failed")
        raise HTTPException(500, f"text grounding failed: {e}")
    elapsed = time.perf_counter() - t0
    log.info("ground/text (%dx%d) phrases=%s in %.2fs", W, H, phrase_list, elapsed)
    return JSONResponse(
        {
            "width": W, "height": H, "model": app.state.sam3_model_id, "engine": "sam3",
            "refine": True, "elapsed_ms": int(elapsed * 1000), "results": results,
        },
        headers={"Cache-Control": "no-store"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
