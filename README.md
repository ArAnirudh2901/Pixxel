<p align="center">
  <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 32 32">
    <rect width="32" height="32" rx="7" fill="#0B0D12"/>
    <path d="M16 3 C 16.7 9.6, 22.4 15.3, 29 16 C 22.4 16.7, 16.7 22.4, 16 29 C 15.3 22.4, 9.6 16.7, 3 16 C 9.6 15.3, 15.3 9.6, 16 3 Z" fill="#06B8D4"/>
  </svg>
</p>

<h1 align="center">Phosmith — AI Image Studio</h1>

<p align="center">
  <strong>A professional-grade, AI-powered image editor built for the browser.</strong><br />
  Combines Photoshop-class editing tools with cutting-edge AI models — all running locally in WebGL2 and on-device Python services.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-16-black?style=flat-square&logo=nextdotjs" alt="Next.js 16" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black" alt="React 19" />
  <img src="https://img.shields.io/badge/Bun-1.3-F9F1E1?style=flat-square&logo=bun&logoColor=black" alt="Bun 1.3" />
  <img src="https://img.shields.io/badge/WebGL2-GLSL%20ES%203.0-06B8D4?style=flat-square" alt="WebGL2" />
  <img src="https://img.shields.io/badge/Fabric.js-7-green?style=flat-square" alt="Fabric.js 7" />
  <img src="https://img.shields.io/badge/FastAPI-PyTorch-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square" alt="AGPL-3.0 License" />
</p>

---

<p align="center">
  <img src="docs/screenshots/hero.png" alt="Phosmith — Shape Light, Forge Photos" width="100%" />
</p>

---

## Table of Contents

- [Overview](#overview)
- [Screenshots](#-screenshots)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Editor Tools](#-editor-tools)
- [AI Capabilities](#-ai-capabilities)
- [The Megashader Engine](#-the-megashader-engine)
- [Agent Command System](#-agent-command-system)
- [Dashboard & Project Management](#-dashboard--project-management)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Local AI Services](#-local-ai-services)
- [AI Agent (Agentic Editing)](#-ai-agent-agentic-editing)
- [Scripts Reference](#-scripts-reference)
- [Deployment](#-deployment)
- [Learn More](#-learn-more)

---

## Overview

Phosmith is a state-of-the-art web-based image editor that seamlessly blends professional-grade adjustment tools with advanced AI capabilities. It features a custom WebGL2 compositing engine (the **Megashader**), non-destructive mask layers, an AI agentic editing assistant with a collage command registry, and support for local large language models and computer vision models.

The editor runs entirely in the browser — image processing is handled by the GPU via WebGL2 shaders, and AI inference is offloaded to a local Python FastAPI service. No cloud GPU required.

This project uses **[Bun](https://bun.sh)** as the sole package manager and runtime. Do not use npm, yarn, or pnpm.

---

## 📸 Screenshots

### Landing Page

The marketing site showcases the neobrutalist design system with a bold hero, scrolling feature ticker, and stats bar.

<p align="center">
  <img src="docs/screenshots/hero.png" alt="Hero Section — Shape Light, Forge Photos" width="100%" />
</p>

### Features & Pricing

The toolkit section highlights nine core tools (AI Agent, AI Extend, Upscale, etc.) while the pricing section offers a clear Free vs Pro comparison.

<table>
  <tr>
    <td><img src="docs/screenshots/features.png" alt="Features — The Full Toolkit" width="100%" /></td>
    <td><img src="docs/screenshots/pricing.png" alt="Pricing — Two Tiers, No Fluff" width="100%" /></td>
  </tr>
  <tr>
    <td align="center"><em>Features — The Toolkit</em></td>
    <td align="center"><em>Pricing — Free & Pro Tiers</em></td>
  </tr>
</table>

### Dashboard

The project dashboard provides a grid view of all saved projects with live canvas thumbnails, creation timestamps, and bulk selection.

<p align="center">
  <img src="docs/screenshots/dashboard.png" alt="Dashboard — Project Grid with New Project and Select buttons" width="100%" />
</p>

### Editor

The full-featured editor with a 13-tool topbar, a left-hand property panel (shown: Resize tool), the WebGL2 canvas with selection handles, and a zoom slider at the bottom.

<p align="center">
  <img src="docs/screenshots/editor.png" alt="Editor — Resize tool with canvas, topbar, and property panel" width="100%" />
</p>

---

## ✨ Key Features

| Category | Highlights |
|---|---|
| **Non-Destructive Editing** | 100+ procedural mask layers composited in real-time on the GPU |
| **AI Selection & Masking** | Subject detection (BiRefNet), point-based segmentation (SAM 2), depth estimation (Depth Anything V2), NL mask descriptions |
| **Professional Adjustments** | 15+ parameters — Exposure, Curves, Temperature, Vibrance, Film Grain, and more |
| **AI Agent Chat** | Type any edit or collage prompt — the agent executes it autonomously with a full command registry |
| **Collage Builder** | Stylish multi-image templates with rounded/circle frames, drop shadows, AI-generated backgrounds, per-cell replace/edit, and an auto-template generator |
| **AI Background** | Generate, replace, or remove backgrounds using AI inpainting/outpainting |
| **AI Extender** | Expand canvas boundaries with AI-generated content (outpainting) |
| **AI Object Remover** | Click any object → SAM 2 segments it → LaMa fills the hole seamlessly |
| **NL Masking** | Describe a region in plain text ("the dog on the left", "everything except the sky") and the agent masks it |
| **Rich Text Engine** | Google Fonts integration, text effects, shadows, outlines, curved text |
| **Export** | PNG / JPEG / WebP at 1×, 2×, or 3× resolution, plus clipboard copy |

---

## 🏛 Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Next.js 16 (App Router)             │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │ Dashboard │  │  Editor  │  │   API Routes       │  │
│  │  (React)  │  │ (Fabric  │  │  /api/ai/*         │  │
│  │           │  │  + WebGL) │  │  /api/canvas/*     │  │
│  └──────────┘  └────┬─────┘  │  /api/imagekit/*    │  │
│                     │        │  /api/neon/*         │  │
│                     ▼        └────────────────────┘  │
│            ┌────────────┐                            │
│            │ Megashader  │    ┌────────────────────┐  │
│            │  (WebGL2    │    │  Local Mask Service │  │
│            │   GLSL)     │    │  (FastAPI + PyTorch) │  │
│            └────────────┘    │  BiRefNet, SAM2,     │  │
│                              │  Depth Anything V2,  │  │
│                              │  LaMa (inpaint),     │  │
│                              │  YOLO (instances),   │  │
│                              │  CLIPSeg (grounding) │  │
│                              └────────────────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │  Clerk   │  │  Neon    │  │  ImageKit CDN      │  │
│  │  (Auth)  │  │ (Postgres│  │  (Image Storage    │  │
│  │          │  │  + Prisma│  │   + Transforms)    │  │
│  └──────────┘  └──────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

---

## 🛠 Editor Tools

The editor topbar exposes **13 tools**, each with its own property panel:

### Core Editing

| Tool | Description |
|---|---|
| **Resize** | Change canvas and image dimensions. Includes Original, 50%, Fit, and Fill presets. Maintains aspect ratio with linked width/height fields. |
| **Crop** | Freeform and preset ratio cropping (1:1, 4:3, 16:9, 3:2, etc.) with a live preview overlay. Includes flip horizontal/vertical. AI Auto-Crop suggests 4 strategies: subject-aware, aspect-ratio, content-fill, and depth-guided. |
| **Images** | Multi-layer image management. Upload, reorder, rename, merge, show/hide, and duplicate layers. Drag-and-drop support. |
| **Adjust** | Professional-grade color and tone adjustments — see [Adjustments](#adjustments) below. |
| **Draw** | Freehand drawing with configurable brush size, color, and opacity. Supports pen/marker styles. |
| **Erase** | Smart eraser with AI Object Remover mode: click an object → SAM 2 segments it → LaMa fills the hole with background content. |
| **Mask** | Comprehensive selection and masking system — see [Masking Tools](#masking-tools) below. |
| **Text** | Rich text engine with 50+ Google Fonts, text shadows, outlines, letter spacing, and alignment controls. |

### AI Tools (Pro)

| Tool | Description |
|---|---|
| **AI Background** | Generate new backgrounds from text prompts, remove backgrounds entirely, or replace them with AI-generated scenes. |
| **AI Extender** | Expand canvas boundaries (outpaint) in any direction with contextually coherent AI-generated content. |
| **AI Edit** | Describe edits in natural language. The AI plan engine generates an edit sequence applied via ImageKit transforms. |
| **Agent** | A fully autonomous agentic editing assistant — see [AI Agent](#-ai-agent-agentic-editing). Also handles collage commands via natural language. |
| **Collage** | Stylish multi-image collage layouts with 14 grid templates, rounded/circle photo frames, drop shadows, per-cell Replace & Edit actions, AI-generated themed backgrounds, and an auto-template generator. |

---

### Adjustments

The **Adjust** tool provides 15+ parameters, all computed in real-time via WebGL2 shaders:

| Parameter | Range | Description |
|---|---|---|
| Brightness | −100 → +100 | Overall light level |
| Contrast | −100 → +100 | Tonal range expansion/compression |
| Saturation | −100 → +100 | Color intensity |
| Vibrance | −100 → +100 | Selective saturation (protects skin tones) |
| Exposure | −2.0 → +2.0 | Simulated f-stop exposure shift |
| Temperature | −100 → +100 | Warm ↔ Cool white balance |
| Tint | −100 → +100 | Green ↔ Magenta shift |
| Hue | −180° → +180° | Global hue rotation |
| Highlights | −100 → +100 | Recover or boost highlight detail |
| Shadows | −100 → +100 | Recover or crush shadow detail |
| Sharpness | 0 → +100 | Unsharp mask sharpening |
| Blur | 0 → +100 | Gaussian blur |
| Noise | 0 → +100 | Add luminance noise |
| Film Grain | 0 → +100 | Cinematic grain simulation |
| Curves | Per-channel | RGB and per-channel curves with control points |

---

### Masking Tools

The mask system is organized into four categories:

#### AI Tools
- **Select Subject** — One-click subject isolation using **BiRefNet**. Falls back to HuggingFace APIs if the local mask service is unavailable.
- **Click to Select** — Point-based semantic segmentation powered by **SAM 2**. Click anywhere on the image and SAM 2 predicts the object boundary. Supports box prompts ("Draw box" drag mode) for strong whole-object selection.
- **Detect All Subjects** — Runs YOLO instance detection + BiRefNet refinement to enumerate every subject in the image as individual clickable chips (person 1, the dog, etc.).
- **Depth Range** — Select objects based on 3D depth estimation using **Depth Anything V2**.
- **Natural Language** — Describe the region in plain text ("the red jacket", "everything except the sky") — routes through the agent's `mask.fromDescription` command which chains CLIPSeg grounding + SAM 2 refinement.

#### Draw Selection
- **Selection Brush** — Paint a selection mask with adjustable size and feather. Includes *Edge Snapping* mode (bilateral filter) that snaps brush strokes to detected edges.
- **Lasso Select** — Three modes: freehand lasso, polygonal lasso, and *Magnetic Lasso* that automatically snaps to edges.

#### Range Selection
- **Color Range** — Eyedropper-based selection. Pick a color, adjust tolerance, and select all pixels within that range.
- **Luminance Range** — Select pixels based on brightness thresholds.
- **Linear Gradient** — Create a gradient mask with configurable angle, spread, and falloff.
- **Radial Gradient** — Create a radial gradient mask from a center point.

#### Destructive
- **Quick Erase** — One-click background removal that permanently modifies the image layer.

All non-destructive selections are composited as **mask layers** in the Megashader engine — reorderable, toggleable, invertible, and adjustable at any time.

#### Mask Boundary Extension
Every mask layer (including AI-detected subjects) supports a **Boundary** slider that grows or shrinks the mask edge by an absolute pixel amount. The pristine texture is preserved under `baseTextureKey` so setting boundary to 0 always restores the original edge.

---

## 🤖 AI Capabilities

### Computer Vision Models

| Model | Task | Provider |
|---|---|---|
| **BiRefNet** | Subject segmentation (background removal) | Local FastAPI / HuggingFace fallback |
| **SAM 2** | Point and box-based semantic segmentation | Local FastAPI |
| **Depth Anything V2** | Monocular depth estimation | Local FastAPI |
| **YOLO26n-seg** | Multi-instance subject detection | Local FastAPI |
| **CLIPSeg** | Open-vocabulary text grounding | Local FastAPI (lazy) / in-browser fallback |
| **LaMa** | Object removal inpainting | Local FastAPI (lazy) |
| **RMBG-1.4** | Client-side background removal | In-browser (WASM/WebGPU via transformers.js) |
| **FLUX.1** | AI background / collage background generation | HuggingFace Inference API |

### AI Transform Pipeline

- **Background generation** — Text-to-image via `/api/ai/background` (FLUX.1; `raw:true` mode for illustration/watercolor prompts)
- **Inpainting** — Fill masked regions via `/api/ai/inpaint` (LaMa local-first, HF Stable Diffusion fallback; crops to padded mask bounds)
- **Outpainting** — Expand canvas boundaries via `/api/ai/extend`
- **Edit planning** — Natural language → edit parameter mapping via `/api/ai/edit-plan` (Gemini, heuristic fallback, grade loop with Gemini judge + critic)
- **Mask planning** — Natural language → mask step plan via `/api/ai/mask-plan`
- **Auto-crop** — 4-strategy crop via `/api/ai/auto-crop` (subject-aware, aspect-ratio, content-fill, depth-guided)

### AI Routing

`src/lib/ai-routing.js` is the single source of truth for where each capability runs. Each capability has a user preference (`auto | client | server`) stored in localStorage. `auto` = server-first with runtime fallback. The Mask tool's "AI Processing" section exposes per-capability toggles.

### Image Analysis

- **Color extraction** — Dominant color palette via `fast-average-color` (used to seed AI background prompts)
- **Histogram analysis** — Per-channel RGB and luminance histograms
- **Image fingerprinting** — Perceptual hashing (dHash) for cache keying
- **Feature extraction** — Edge density, contrast, saturation, scene analysis for the AI agent

---

## ⚡ The Megashader Engine

At the core of the editing experience is a custom **WebGL2** compositing engine:

- **Non-destructive workflows** — Supports 100+ procedural mask layers with zero lag. Each mask layer is a GLSL program that runs entirely on the GPU.
- **Real-time preview** — All adjustments, masks, and blends are computed per-frame.
- **Blend modes** — Photoshop-parity blend modes: Normal, Screen, Multiply, Overlay, Soft Light, Hard Light, Darken, Lighten, Color Dodge, Color Burn, Difference, Exclusion, Add, Subtract, Divide.
- **Mask chain composition** — Multiple mask layers compose via union, intersection, subtraction, and XOR.
- **230 GLSL invariant tests** — The shader pipeline is validated by `bun run verify`.

### How It Works

```
Image Layer → [Adjustment Shaders] → [Mask Chain (GLSL)] → [Blend Modes] → Final Composite
                                           ↑
                                    Mask Layers (N):
                                    ├─ Brush strokes
                                    ├─ AI segmentation (BiRefNet / SAM 2 / YOLO)
                                    ├─ CLIPSeg text grounding
                                    ├─ Lasso / Color Range / Luminance
                                    ├─ Gradient masks
                                    └─ Depth maps
```

Each mask layer stores its parameters (not pixels) so they remain fully editable. The Megashader recompiles and re-renders the entire chain on every parameter change at 60 fps.

---

## 🧩 Agent Command System

All editor capabilities are exposed through a **command registry** at `src/lib/agent/command-registry.js`. Domains are registered by `canvas.jsx` on editor mount and are accessible as `window.__phosmith.agent`.

| Domain | Commands |
|---|---|
| `mask.*` | `selectSubject`, `clickSelect`, `addSubjectBox`, `detectSubjects`, `selectSubjects`, `fromDescription`, `expandLayer` |
| `crop.*` | `auto`, `subjectAware`, `fitAspect`, `contentFill`, `applyBox` |
| `collage.*` | `createTemplate`, `fromDescription`, `autoTemplate`, `generateBackground`, `setBackground`, `listLayouts`, `listStyles` |

The AI agent chat routes typed prompts to the appropriate domain:
- **Edit prompts** → `edit-plan` (Gemini grade loop)
- **Collage prompts** → `collage.fromDescription` (detected via `COLLAGE_INTENT_RE`)
- **Mask prompts** → `mask.fromDescription` (NL → CLIPSeg + SAM 2)

Every command run is logged in the **History panel** (pinned in the editor sidebar) with a cyan Bot badge, separate from user-driven changes.

---

## 📋 Dashboard & Project Management

The dashboard provides a grid view of all projects with:

- **Project cards** with live canvas thumbnails (pixel-art disintegration animation on delete)
- **Create new projects** from uploaded images or blank canvases
- **Auto-save** — Debounced writes to Neon/Postgres with a Redis-backed snapshot cache for fast loads
- **Canvas state persistence** — Full Fabric.js canvas state (objects, filters, mask layers, collage cell data, viewport) serialized and restored on reload
- **Image storage** via ImageKit CDN with on-the-fly transformations

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Framework** | Next.js 16 (App Router, React 19, React Compiler) |
| **Runtime** | Bun 1.3 |
| **Canvas** | Fabric.js 7 + custom WebGL2 shaders |
| **GPU Compute** | WebGL2 GLSL ES 3.0 (Megashader engine) |
| **UI** | Tailwind CSS 4, Framer Motion, Radix UI, Lucide Icons |
| **Auth** | Clerk (email, OAuth, org support) |
| **Database** | Neon (serverless Postgres) + Prisma ORM |
| **Caching** | Redis (Upstash) for canvas snapshot caching |
| **Image CDN** | ImageKit (storage, transforms, AI pipeline) |
| **AI Models** | BiRefNet, SAM 2, Depth Anything V2, YOLO, CLIPSeg, LaMa, RMBG-1.4, FLUX.1 |
| **AI Service** | Python FastAPI + PyTorch (local inference, lazy model loading) |
| **In-Browser AI** | RMBG-1.4 via transformers.js (WebGPU → WASM fallback) |
| **Billing** | Clerk Billing (Pro tier for AI tools) |

---

## 🚀 Getting Started

### Prerequisites

- [Bun](https://bun.sh) ≥ 1.3
- [Node.js](https://nodejs.org/) ≥ 18 (required by some Next.js internals)
- [Python](https://www.python.org/) ≥ 3.10 (for the local AI mask service)

### Installation

```bash
# Clone the repo
git clone https://github.com/ArAnirudh2901/phosmith.git
cd phosmith

# Install dependencies
bun install

# Set up environment variables
cp .env.example .env.local
# → Edit .env.local with your keys (see Environment Variables below)

# Generate Prisma client
bun run prisma:generate

# Push database schema
bun run db:push

# Start development server
bun dev
```

Open [http://localhost:3000](http://localhost:3000) to launch the editor.

---

## 🔑 Environment Variables

Create a `.env.local` file in the project root:

```env
# ── Authentication (Clerk) ──
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up

# ── Database (Neon/Postgres) ──
DATABASE_URL=postgresql://...
DIRECT_URL=postgresql://...

# ── Image Storage (ImageKit) ──
NEXT_PUBLIC_IMAGEKIT_PUBLIC_KEY=public_...
IMAGEKIT_PRIVATE_KEY=private_...
NEXT_PUBLIC_IMAGEKIT_URL_ENDPOINT=https://ik.imagekit.io/your-id

# ── AI (Gemini — edit planning & mask planning) ──
GEMINI_API_KEY=AIza...

# ── Caching (Redis / Upstash — optional) ──
UPSTASH_REDIS_REST_URL=https://...
UPSTASH_REDIS_REST_TOKEN=...

# ── Local AI Mask Service (optional) ──
MASK_SERVICE_URL=http://127.0.0.1:8001
```

> **Note:** The editor works without the optional services — AI mask tools fall back to HuggingFace APIs, and the agent uses a local visual-metrics planner instead of Gemini.

---

## 🧠 Local AI Services

### Mask Service (BiRefNet · SAM 2 · Depth · YOLO · CLIPSeg · LaMa)

The editor's heavy AI tools run via a local **Python FastAPI** service to avoid blocking the Node.js thread.

```bash
# One-time: install Python dependencies (PyTorch, rembg, transformers, ultralytics)
bun run mask:install

# Start the FastAPI service on port 8001
bun run mask:dev
```

Add the service URL to `.env.local`:
```env
MASK_SERVICE_URL=http://127.0.0.1:8001
```

> **Without this service:** SAM 2, Depth, YOLO, CLIPSeg, and LaMa tools return 501. Select Subject falls back to HuggingFace APIs. See `services/segment/.env.example` for model configuration.

### Verification

Run end-to-end tests for each AI endpoint:

```bash
bun run verify:segment      # BiRefNet subject segmentation
bun run verify:semantic     # SAM 2 point-based segmentation
bun run verify:instances    # YOLO multi-instance detection
bun run verify:depth        # Depth Anything V2 depth estimation
bun run verify:depth:full   # Comprehensive depth tests
bun run verify:auto-crop    # Auto-crop 4-strategy pipeline
bun run verify:inpaint      # LaMa object removal inpainting
bun run verify:nl-mask      # NL mask parser + CLIPSeg grounding
bun run verify:client-ai    # In-browser RMBG-1.4 (Playwright harness)
bun run verify:agent        # Agent grade loop invariants
bun run verify:mask         # Mask edge-snap bilateral filter
```

---

## 🤖 AI Agent (Agentic Editing)

The **AI Agent** is a fully autonomous editing assistant accessible via the Agent tool panel. It can analyze images, apply professional transformations, create collages, and mask regions from natural language descriptions.

### How It Works

1. **Prompt routing** — The agent detects whether the prompt is a collage intent (`COLLAGE_INTENT_RE`) or an edit/mask intent, and routes accordingly.
2. **Edit path** — Captures canvas state → Gemini analyzes and generates an edit plan → grade loop (executor → Gemini judge → critic → corrective re-plan, max 3 iterations) → applies changes.
3. **Collage path** — `collage.fromDescription` heuristic parser maps free text to layout/style/background/theme → executes the collage command → optionally generates an AI background.
4. **Mask path** — `mask.fromDescription` → `/api/ai/mask-plan` (Gemini, heuristic fallback) → executor chains subjects + CLIPSeg grounding + depth/luminance/color layers.

### Agent Capabilities

| Category | Actions |
|---|---|
| **Color grading** | Cinematic grades, portrait retouching, auto-enhance, style transfer |
| **Collage** | Create templates from description, auto-generate stylish templates, set or generate AI backgrounds |
| **Masking** | Select subjects, click-select, box-select, detect all subjects, natural language region masking |
| **Crop** | AI auto-crop with 4 strategies (subject-aware, aspect-ratio, content-fill, depth-guided) |

The agent has a persistent chat interface with conversation history stored per-project. Every agent action is logged with a cyan Bot badge in the pinned History panel.

---

## 📜 Scripts Reference

| Command | Description |
|---|---|
| `bun dev` | Start Next.js dev server with HMR |
| `bun run build` | Production build |
| `bun start` | Start production server |
| `bun run lint` | Run ESLint |
| `bun run prisma:generate` | Generate Prisma Client |
| `bun run prisma:migrate` | Apply database schema migrations |
| `bun run db:push` | Push Prisma schema to Neon/Postgres |
| `bun run imagekit:docs` | Crawl ImageKit docs into a local JSON knowledge base |
| `bun run mask:install` | Install Python deps for the local mask service |
| `bun run mask:dev` | Start the FastAPI mask service on port 8001 |
| `bun run verify` | Run all 230 Megashader GLSL invariant tests |
| `bun run verify:mask` | Mask edge-snap bilateral filter |
| `bun run verify:segment` | BiRefNet subject segmentation |
| `bun run verify:semantic` | SAM 2 point-based segmentation |
| `bun run verify:instances` | YOLO multi-instance detection |
| `bun run verify:depth` | Depth Anything V2 depth estimation |
| `bun run verify:depth:full` | Comprehensive depth estimation tests |
| `bun run verify:auto-crop` | Auto-crop 4-strategy pipeline |
| `bun run verify:inpaint` | LaMa object removal inpainting |
| `bun run verify:nl-mask` | NL mask parser + CLIPSeg grounding |
| `bun run verify:client-ai` | In-browser RMBG-1.4 (Playwright harness) |
| `bun run verify:agent` | Agent grade loop invariants |

---

## ☁️ Production Deployment

To run Phosmith in production with all AI features active, you need to deploy the Next.js frontend to **Vercel** and the Python FastAPI Mask Service to a **Hugging Face Space (Docker SDK)**.

### 1. Backend: Deploy Mask Service to Hugging Face Spaces

Hugging Face Spaces provides a free 16 GB CPU container tier, which is comfortable for running this FastAPI service.

1. **Create a Space**:
   - Log into [Hugging Face](https://huggingface.co/) and click **New Space**.
   - Set the SDK option to **Docker** (Blank template).
   - Set Space visibility to **Public** (so Vercel's backend can query it without OAuth credentials).
2. **Push the Code**:
   - Initialize a Git repo or upload files directly. Push the contents of the `/services/segment` directory directly to the root of your Hugging Face Space repository.
   - Ensure the following files are present at the root of the Space repository:
     - `Dockerfile` (already configured to expose port `7860`)
     - `requirements.txt`
     - `main.py`
     - `yolo26n-seg.pt` (optional, will download on start if missing)
3. **Build & Direct URL**:
   - Hugging Face will automatically read the `Dockerfile`, install OpenCV/PyTorch dependencies, expose port `7860`, and run the uvicorn server.
   - Once the build finishes and shows **Running**, note your public Space URL:
     `https://<your-username>-<your-space-name>.hf.space`

### 2. Frontend: Deploy Next.js to Vercel

1. Connect your repository to **Vercel**.
2. Configure the build settings:
   - **Framework Preset**: Next.js
   - **Install Command**: `bun install`
   - **Build Command**: `bun run build`
3. Add your production environment variables in the Vercel dashboard:
   - Auth variables (`NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`, etc.)
   - Database and cache variables (`DATABASE_URL`, `UPSTASH_REDIS_REST_URL`, etc.)
   - Image CDN variables (`NEXT_PUBLIC_IMAGEKIT_PUBLIC_KEY`, `IMAGEKIT_PRIVATE_KEY`, etc.)
   - LLM variables (`GEMINI_API_KEY`)
   - **Mask Service variable**: Set `MASK_SERVICE_URL` to your Hugging Face Space URL (e.g. `https://<your-username>-<your-space-name>.hf.space`). *Ensure there is no trailing slash.*
4. Deploy the project.

> 💡 **Pro-Tip (Preventing Cold Starts):** Free-tier Hugging Face Spaces automatically sleep after 48 hours of inactivity. The first API request after a sleep period will wake the space up, which takes about 30–60 seconds. To prevent timeouts, you can set up a simple uptime monitor/cron to regularly ping the `/health` endpoint of your Space to keep it active.


---

## 📚 Learn More

- [Next.js Documentation](https://nextjs.org/docs) — Framework reference
- [Bun Documentation](https://bun.sh/docs) — Runtime and package manager
- [Fabric.js Documentation](http://fabricjs.com/docs/) — Canvas library
- [Prisma Documentation](https://www.prisma.io/docs) — Database ORM
- [Clerk Documentation](https://clerk.com/docs) — Authentication
- [ImageKit Documentation](https://docs.imagekit.io/) — Image CDN and AI transforms

---

## 📄 License

**GNU AGPL v3 or later** — see [LICENSE](LICENSE). Copyright © 2026 Anirudh
Aravalli.

Phosmith is also a build input to
[Mask Studio](https://github.com/ArAnirudh2901/Image-Masking-test), which bundles
`src/lib/megashader/` and the editor mask UI into its output. Mask Studio is
AGPL-3.0 for its own reason (a YOLOE-derived detector), and this license is what
lets the two combine.

**If you deploy this.** AGPL §13 means anyone who uses a hosted Phosmith over a
network must be offered the source of the version they are using. A "Source"
link in the app shell pointing at this repo — or at your fork, if you modified
it — is the usual way to satisfy that. The Vercel instructions above do not add
one for you.

---

<p align="center">
  <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#0B0D12"/><path d="M16 3 C 16.7 9.6, 22.4 15.3, 29 16 C 22.4 16.7, 16.7 22.4, 16 29 C 15.3 22.4, 9.6 16.7, 3 16 C 9.6 15.3, 15.3 9.6, 16 3 Z" fill="#06B8D4"/></svg><br />
  Built with ❤️ using Next.js, Fabric.js, WebGL2, and a lot of GLSL.
</p>
