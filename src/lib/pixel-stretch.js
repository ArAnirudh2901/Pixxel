/**
 * Pixel Stretch — pure rendering core (no Fabric, no React).
 *
 * The classic "pixel stretch / streak" effect (and the one in the reference
 * video) is, mechanically, a single move: take ONE thin line of pixels (the
 * "seed line") and sweep it along a path. Sweeping a horizontal seed line
 * straight down gives the familiar vertical color streaks; bending the path
 * sweeps that same line into a flowing ribbon / arch.
 *
 * Photoshop makes you do this in four disconnected steps — marquee a 1px strip,
 * Free-Transform-scale it to smear, Edit ▸ Transform ▸ Warp the mesh by hand,
 * then flatten. We collapse all of that into one continuous, live model:
 *
 *      ribbon = sweep( seedLine, path )
 *
 *   • path is a cubic Bézier that starts at the band's seed edge and travels
 *     `length × bandExtent` in the stretch direction, bowed by `bend` and
 *     S-shaped by `twist`.
 *   • length 1 + bend 0  →  a straight ribbon that exactly fills the band
 *     (plain pixel stretch).
 *   • length > 1 / bend ≠ 0  →  the ribbon sweeps out into an arch (the warp).
 *
 * The SAME function renders the live preview (at on-screen resolution) and the
 * final committed bitmap (at full image resolution) — only W/H differ. All
 * geometry params are normalised to [0..1] in image space so they're
 * resolution-independent.
 */

// ─── Parameters ──────────────────────────────────────────────────────────────

/**
 * @typedef {Object} StretchParams
 * @property {'vertical'|'horizontal'} axis  Streak direction.
 * @property {1|-1} direction  Which way streaks travel from the seed edge.
 * @property {{x:number,y:number,w:number,h:number}} band  Source region, normalised image coords.
 * @property {Array<{x:number,y:number}>|null} polygon  Optional lasso polygon.
 * @property {number} seed  0..1 position of the sampled line within the band.
 * @property {number} length  Ribbon length multiple (>= 1).
 * @property {number} bend  -1..1 perpendicular bow.
 * @property {number} twist  -1..1 S-curve.
 * @property {number} fade  0..1 opacity taper.
 * @property {number} taper 0..1 width taper.
 * @property {boolean} mirror  Symmetric ribbon.
 * @property {number} opacity  0..1 overall strength.
 * @property {Array<Array<{x:number,y:number}>>|null} warpGrid  R×C control points
 *   (R rows, C cols; R,C ≥ 2, default 4×4 = Photoshop's default warp) for the
 *   Photoshop-style warp mesh, in normalised image coords. When set, the
 *   stretched pixels are rendered through a smooth interpolating (Catmull-Rom)
 *   bicubic surface defined by these points — every control point lies ON the
 *   surface, so dragging one bends the streaks through it. Split the grid to add
 *   rows/columns and shape as many curves as you like. null = simple slider mode.
 * @property {{x:number,y:number,w:number,h:number}|null} warpRest  The grid's
 *   undeformed (rest) rectangle in normalised image coords. The warp samples the
 *   stretched buffer from this rectangle and maps it onto the deformed grid, so
 *   only the stretched band — not the whole image — is warped.
 */

/** @type {StretchParams} */
export const DEFAULT_STRETCH = {
  axis: 'vertical',
  direction: 1,
  band: { x: 0.32, y: 0.08, w: 0.36, h: 0.7 },
  polygon: null,
  seed: 0,
  length: 1,
  bend: 0,
  twist: 0,
  fade: 0,
  taper: 0,
  mirror: false,
  opacity: 1,
  warpGrid: null,
  warpRest: null,
  flowPath: null,
}

// Warp is a composite bicubic Bézier surface — exactly Photoshop's Warp. The
// control net has (3·patches + 1) points per axis: 4 = one patch (default warp),
// 7 = two patches, 13 = four. Anchors sit at indices divisible by 3 and are
// interpolated; the points between them are the bezier tangent / interior
// handles that pull the surface (and never lie on it) — the Photoshop feel.
export const WARP_MIN_DIM = 4   // one Bézier patch
export const WARP_MAX_DIM = 13  // four patches per axis (dense mesh)
export const WARP_DEFAULT_ROWS = 4
export const WARP_DEFAULT_COLS = 4

/** True when a net index is an anchor (interpolated patch corner). */
const isWarpAnchor = (i) => i % 3 === 0
/** Patch counts for a net: (dim-1)/3 per axis. */
const warpPatches = (grid) => ({ pr: (grid.length - 1) / 3, pc: (grid[0].length - 1) / 3 })

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi)
const clamp01 = (v) => clamp(v, 0, 1)
// clamp() is NaN-transparent (Math.min/max propagate NaN) — sanitizers must
// coerce non-finite input to a fallback first.
const num = (v, d) => (Number.isFinite(v) ? v : d)

// ─── Warp Grid (Photoshop-style 4×4 control point mesh) ─────────────────────

/**
 * The grid's undeformed footprint: the band region extended by `length` in the
 * stretch direction, clamped to the image. This rectangle is the warp's rest
 * shape — the warp samples the stretched buffer from here.
 *
 * @param {StretchParams} params
 * @returns {{x:number,y:number,w:number,h:number}}
 */
export function getWarpRest(params = DEFAULT_STRETCH) {
  const p = { ...DEFAULT_STRETCH, ...params }
  const b = p.band
  const vertical = p.axis === 'vertical'
  let x0, y0, gw, gh
  if (vertical) {
    x0 = b.x
    gw = b.w
    if (p.direction > 0) { y0 = b.y; gh = Math.min(b.h * p.length, 1 - b.y) }
    else { gh = Math.min(b.h * p.length, b.y + b.h); y0 = b.y + b.h - gh }
  } else {
    y0 = b.y
    gh = b.h
    if (p.direction > 0) { x0 = b.x; gw = Math.min(b.w * p.length, 1 - b.x) }
    else { gw = Math.min(b.w * p.length, b.x + b.w); x0 = b.x + b.w - gw }
  }
  return { x: x0, y: y0, w: Math.max(0.001, gw), h: Math.max(0.001, gh) }
}

/** Snap a requested dimension to the nearest valid net size (3·patches+1). */
const snapNetDim = (n) => clamp(Math.round((Math.round(num(n, WARP_MIN_DIM)) - 1) / 3) * 3 + 1, WARP_MIN_DIM, WARP_MAX_DIM)

/**
 * Create a default warp control net — points evenly spaced over the rest
 * footprint. A 4×4 net (the default) is exactly Photoshop's default warp (one
 * bicubic patch). Evenly-spaced controls describe a flat patch, so it's an
 * identity warp until a control point is dragged. `rows`/`cols` snap to the
 * nearest valid net size.
 *
 * @param {StretchParams} params  Uses band, axis, direction, length.
 * @param {number} [rows=4]
 * @param {number} [cols=4]
 * @returns {Array<Array<{x:number,y:number}>>}
 */
export function createDefaultWarpGrid(params = DEFAULT_STRETCH, rows = WARP_DEFAULT_ROWS, cols = WARP_DEFAULT_COLS) {
  const R = snapNetDim(rows)
  const C = snapNetDim(cols)
  const rest = getWarpRest(params)
  const grid = []
  for (let row = 0; row < R; row++) {
    const r = []
    for (let col = 0; col < C; col++) {
      r.push({ x: rest.x + (col / (C - 1)) * rest.w, y: rest.y + (row / (R - 1)) * rest.h })
    }
    grid.push(r)
  }
  return grid
}

/**
 * Validate and clamp a warp control net. Must be rectangular with each axis a
 * valid Bézier-net size (4, 7, 10 or 13 — i.e. 3·patches+1) so the renderer can
 * always carve it into whole bicubic patches. Returns null if malformed.
 */
function sanitizeWarpGrid(grid) {
  if (!Array.isArray(grid) || grid.length < WARP_MIN_DIM || grid.length > WARP_MAX_DIM) return null
  if ((grid.length - 1) % 3 !== 0) return null
  const C = Array.isArray(grid[0]) ? grid[0].length : 0
  if (C < WARP_MIN_DIM || C > WARP_MAX_DIM || (C - 1) % 3 !== 0) return null
  const out = []
  for (let r = 0; r < grid.length; r++) {
    if (!Array.isArray(grid[r]) || grid[r].length !== C) return null
    const row = []
    for (let c = 0; c < C; c++) {
      const pt = grid[r][c]
      if (!pt || !Number.isFinite(pt.x) || !Number.isFinite(pt.y)) return null
      row.push({ x: clamp(pt.x, -0.6, 1.6), y: clamp(pt.y, -0.6, 1.6) })
    }
    out.push(row)
  }
  return out
}

/** Clamp a rest rectangle to a sane normalised range, or null. */
function sanitizeWarpRest(rest) {
  if (!rest || !Number.isFinite(rest.x) || !Number.isFinite(rest.y) ||
      !Number.isFinite(rest.w) || !Number.isFinite(rest.h)) return null
  return {
    x: clamp(rest.x, -0.5, 1.5),
    y: clamp(rest.y, -0.5, 1.5),
    w: clamp(rest.w, 0.001, 2),
    h: clamp(rest.h, 0.001, 2),
  }
}


/** Sanitise a freeform polygon to clamped normalised points, or null if invalid. */
function sanitizePolygon(poly) {
  if (!Array.isArray(poly) || poly.length < 3) return null
  const pts = []
  for (const pt of poly) {
    if (!pt || typeof pt.x !== 'number' || typeof pt.y !== 'number') continue
    pts.push({ x: clamp01(pt.x), y: clamp01(pt.y) })
  }
  return pts.length >= 3 ? pts : null
}

/** Axis-aligned bounding box (normalised) of a polygon — use it as the `band`. */
export function getPolygonBBox(poly) {
  const pts = sanitizePolygon(poly)
  if (!pts) return null
  let minX = 1, minY = 1, maxX = 0, maxY = 0
  for (const { x, y } of pts) {
    if (x < minX) minX = x
    if (y < minY) minY = y
    if (x > maxX) maxX = x
    if (y > maxY) maxY = y
  }
  return {
    x: minX,
    y: minY,
    w: Math.max(0.01, maxX - minX),
    h: Math.max(0.01, maxY - minY),
  }
}

/** Clamp any (possibly model/UI-supplied) params into the safe, renderable range. */
export function clampStretchParams(p = {}) {
  const base = { ...DEFAULT_STRETCH, ...p }
  const band = { ...DEFAULT_STRETCH.band, ...(p.band || {}) }
  const D = DEFAULT_STRETCH
  // Keep the band inside the image and give it a non-zero footprint.
  // NaN/Infinity (model/UI-supplied) fall back to defaults before clamping.
  band.w = clamp(num(band.w, D.band.w), 0.01, 1)
  band.h = clamp(num(band.h, D.band.h), 0.01, 1)
  band.x = clamp(num(band.x, D.band.x), 0, 1 - band.w)
  band.y = clamp(num(band.y, D.band.y), 0, 1 - band.h)
  return {
    axis: base.axis === 'horizontal' ? 'horizontal' : 'vertical',
    direction: base.direction < 0 ? -1 : 1,
    band,
    polygon: sanitizePolygon(base.polygon),
    seed: clamp01(num(base.seed, D.seed)),
    length: clamp(num(base.length, D.length), 1, 8),
    bend: clamp(num(base.bend, D.bend), -1, 1),
    twist: clamp(num(base.twist, D.twist), -1, 1),
    fade: clamp01(num(base.fade, D.fade)),
    taper: clamp(num(base.taper, D.taper), -1, 1),   // <0 flares the tip wider, >0 narrows it
    mirror: Boolean(base.mirror),
    opacity: clamp01(num(base.opacity, D.opacity)),
    warpGrid: sanitizeWarpGrid(base.warpGrid),
    warpRest: sanitizeWarpRest(base.warpRest),
    flowPath: sanitizeFlowPath(base.flowPath),
  }
}

// ─── Presets ─────────────────────────────────────────────────────────────────
// Curated looks that map onto the reference video and beyond. The band itself
// is left to the user; presets only set the stretch/warp character.

export const PIXEL_STRETCH_PRESETS = [
  { id: 'straight', label: 'Straight', hint: 'Clean pixel streaks', params: { length: 1, bend: 0, twist: 0, fade: 0, taper: 0, mirror: false } },
  { id: 'tall', label: 'Tall Smear', hint: 'Streaks shoot past the band', params: { length: 2.4, bend: 0, twist: 0, fade: 0.15, taper: 0, mirror: false } },
  { id: 'arch', label: 'Arch', hint: 'Bow into a single arc', params: { length: 2.2, bend: 0.6, twist: 0, fade: 0.1, taper: 0.1, mirror: false } },
  { id: 'sweep', label: 'Ribbon Sweep', hint: 'Long flowing ribbon', params: { length: 4, bend: 0.85, twist: 0.15, fade: 0.25, taper: 0.25, mirror: false } },
  { id: 'scurve', label: 'S-Curve', hint: 'Serpentine flow', params: { length: 3, bend: 0.7, twist: 1, fade: 0.2, taper: 0.15, mirror: false } },
  { id: 'mirror', label: 'Mirror Arc', hint: 'Symmetric double arch', params: { length: 2.2, bend: 0.7, twist: 0, fade: 0.15, taper: 0.1, mirror: true } },
]

// ─── Geometry ────────────────────────────────────────────────────────────────

const cubic = (p0, c1, c2, p1, t) => {
  const u = 1 - t
  const a = u * u * u
  const b = 3 * u * u * t
  const c = 3 * u * t * t
  const d = t * t * t
  return {
    x: a * p0.x + b * c1.x + c * c2.x + d * p1.x,
    y: a * p0.y + b * c1.y + c * c2.y + d * p1.y,
  }
}

const cubicTangent = (p0, c1, c2, p1, t) => {
  const u = 1 - t
  const x = 3 * u * u * (c1.x - p0.x) + 6 * u * t * (c2.x - c1.x) + 3 * t * t * (p1.x - c2.x)
  const y = 3 * u * u * (c1.y - p0.y) + 6 * u * t * (c2.y - c1.y) + 3 * t * t * (p1.y - c2.y)
  const len = Math.hypot(x, y) || 1
  return { x: x / len, y: y / len }
}

/**
 * Resolve normalised params into concrete pixel-space geometry for a W×H frame.
 * `dir` lets the caller flip the whole thing for the mirror pass.
 */
function resolveGeometry(p, W, H, dir = p.direction) {
  const vertical = p.axis === 'vertical'
  const bx = p.band.x * W
  const by = p.band.y * H
  const bw = p.band.w * W
  const bh = p.band.h * H

  // Cross-axis span of the seed line, and the axis extent that sets ribbon length.
  const stripLen = vertical ? bw : bh
  const axisExtent = vertical ? bh : bw
  const total = axisExtent * p.length

  // Unit vectors: d = travel direction, n = perpendicular (bend) direction.
  const d = vertical ? { x: 0, y: dir } : { x: dir, y: 0 }
  const n = vertical ? { x: 1, y: 0 } : { x: 0, y: 1 }

  // Start = centre of the band's seed edge (the edge opposite to travel).
  const start = vertical
    ? { x: bx + bw / 2, y: dir > 0 ? by : by + bh }
    : { x: dir > 0 ? bx : bx + bw, y: by + bh / 2 }

  // Cubic control points. `bend` bows the path perpendicular to travel; `twist`
  // controls the SECOND control point's offset: twist 0 → both controls bow the
  // same way (a clean arch); twist 1 → the second bows the opposite way
  // (S-curve); twist -1 → it bows further the same way (loop / hook).
  const bow = p.bend * total * 0.6
  const c1Off = bow
  const c2Off = bow * (1 - 2 * p.twist) // twist∈[-1,1] → multiplier∈[3,-1]
  const c1 = {
    x: start.x + d.x * (total * 0.33) + n.x * c1Off,
    y: start.y + d.y * (total * 0.33) + n.y * c1Off,
  }
  const c2 = {
    x: start.x + d.x * (total * 0.66) + n.x * c2Off,
    y: start.y + d.y * (total * 0.66) + n.y * c2Off,
  }

  const end = { x: start.x + d.x * total, y: start.y + d.y * total }

  // Seed sampling line within the band (independent of where the ribbon starts).
  const seedLine = vertical
    ? { x0: bx, y0: by + p.seed * bh, x1: bx + bw, y1: by + p.seed * bh }
    : { x0: bx + p.seed * bw, y0: by, x1: bx + p.seed * bw, y1: by + bh }

  return { vertical, bx, by, bw, bh, stripLen, axisExtent, total, d, n, start, c1, c2, end, seedLine }
}

/**
 * On-canvas handle anchor points (in normalised image coords) so the overlay
 * can draw draggable controls without re-deriving the geometry.
 */
export function getStretchAnchors(params, W = 1, H = 1) {
  W = num(W, 1) > 0 ? num(W, 1) : 1
  H = num(H, 1) > 0 ? num(H, 1) : 1
  const p = clampStretchParams(params)
  const g = resolveGeometry(p, W, H)
  const mid = cubic(g.start, g.c1, g.c2, g.end, 0.5)
  return {
    start: { x: g.start.x / W, y: g.start.y / H },
    end: { x: g.end.x / W, y: g.end.y / H },
    mid: { x: mid.x / W, y: mid.y / H },
    seedLine: {
      x0: g.seedLine.x0 / W, y0: g.seedLine.y0 / H,
      x1: g.seedLine.x1 / W, y1: g.seedLine.y1 / H,
    },
  }
}

/**
 * Sample the ribbon centerline as normalised points — used to draw the on-canvas
 * guide curve. Aspect ratio (H/W) must match the render target.
 */
export function getStretchPath(params, W = 1, H = 1, samples = 48) {
  W = num(W, 1) > 0 ? num(W, 1) : 1
  H = num(H, 1) > 0 ? num(H, 1) : 1
  const p = clampStretchParams(params)
  const g = resolveGeometry(p, W, H)
  const pts = []
  for (let i = 0; i <= samples; i++) {
    const t = i / samples
    const c = cubic(g.start, g.c1, g.c2, g.end, t)
    pts.push({ x: c.x / W, y: c.y / H })
  }
  return pts
}

// ─── Rendering ───────────────────────────────────────────────────────────────

/**
 * Allocate an intermediate raster buffer. Prefers OffscreenCanvas — it's
 * faster (no DOM node), works inside Web Workers, and exposes the async
 * `convertToBlob` encoder — and falls back to a detached <canvas> when
 * OffscreenCanvas is unavailable (older Safari).
 */
export function createStretchBuffer(w, h) {
  const W = Math.max(1, Math.round(w))
  const H = Math.max(1, Math.round(h))
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(W, H)
  const c = (typeof document !== 'undefined' ? document.createElement('canvas') : null)
  if (!c) throw new Error('No canvas implementation available')
  c.width = W
  c.height = H
  return c
}

/** Draw an element into a fresh W×H buffer (used to snapshot the source). */
export function makeSampleCanvas(sourceEl, W, H) {
  const c = createStretchBuffer(W, H)
  const ctx = c.getContext('2d')
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(sourceEl, 0, 0, c.width, c.height)
  return c
}

// A reusable scratch buffer for the seed strip — re-sized in place instead of
// re-allocated every frame, so dragging doesn't churn the GC.
let _stripBuf = null
let _stripCtx = null

/** Extract the seed line as a stripLen×1 strip of colours (reused buffer). */
function buildSeedStrip(sample, g) {
  const len = Math.max(1, Math.round(g.stripLen))
  if (!_stripBuf) {
    _stripBuf = createStretchBuffer(len, 1)
    _stripCtx = _stripBuf.getContext('2d')
  } else if (_stripBuf.width !== len) {
    _stripBuf.width = len
    _stripBuf.height = 1
  }
  const ctx = _stripCtx
  ctx.imageSmoothingEnabled = true
  ctx.clearRect(0, 0, len, 1)
  if (g.vertical) {
    // one horizontal row → horizontal strip
    ctx.drawImage(sample, g.seedLine.x0, g.seedLine.y0, g.bw, 1, 0, 0, len, 1)
  } else {
    // one vertical column → rotate it into a horizontal strip
    ctx.setTransform(0, 1, -1, 0, len, 0)
    ctx.drawImage(sample, g.seedLine.x0, g.seedLine.y0, 1, g.bh, 0, 0, len, 1)
    ctx.setTransform(1, 0, 0, 1, 0, 0)
  }
  return _stripBuf
}

/**
 * Pick how many slices to sweep. Straight ribbons need almost none (the fast
 * path handles them); curvier ribbons need more to stay smooth. Always capped
 * by `maxSlices` (the perf lever the caller sets per quality tier).
 */
function sliceCount(g, p, maxSlices) {
  const curviness = Math.min(1.5, Math.abs(p.bend) + Math.abs(p.twist) * 0.75)
  const want = Math.ceil(g.total * (0.45 + curviness))
  return clamp(want, 24, Math.max(24, Math.round(maxSlices)))
}

// ── Gap-free ribbon sweep (continuous textured triangle strip) ───────────────
//
// Stamping rotated rectangular slices along a curved or flared path leaves
// comb-teeth GAPS: a slice is a rigid rectangle placed at arc-length-even spacing
// along the CENTRELINE, but on a bend (or a widening flare) the strip's OUTER
// edge travels farther than its centre per step, so consecutive stamps stop
// overlapping at the outer radius — exactly the fan-of-rays artifact. The cure is
// to sweep the seed strip as ONE continuous triangle strip: consecutive
// cross-sections SHARE an edge, so the ribbon is gap-free by construction at any
// curvature or width, the same way the Warp mesh never seams.
//
// A `section` is { cx, cy, nx, ny, hw } — centre point, unit normal (perpendicular
// to travel) and half-width, in pixel space. The 1-px seed strip's full width maps
// across L = centre − n·hw  →  R = centre + n·hw.

let _ribbonBuf = { canvas: null, ctx: null }
function getRibbonBuf(W, H) {
  const r = _ribbonBuf
  if (!r.canvas) { r.canvas = createStretchBuffer(W, H); r.ctx = r.canvas.getContext('2d') }
  else if (r.canvas.width !== W || r.canvas.height !== H) { r.canvas.width = W; r.canvas.height = H }
  return r
}

/**
 * Sweep `strip` (a stripLen×1 seed line) through `sections` as a gap-free textured
 * ribbon. Rendered OPAQUELY into a reused scratch buffer first — drawing the
 * seam-overlapped triangles straight onto a <1 globalAlpha would let the sub-pixel
 * seam margins double-darken into faint ribs — then composited once with `opacity`
 * and an optional tip `fade` (alpha taper along travel). Respects ctx's clip.
 */
function sweepStripMesh(ctx, strip, sections, W, H, opts = {}) {
  if (!sections || sections.length < 2) return
  const seedLen = strip.width
  const opacity = opts.opacity == null ? 1 : clamp01(opts.opacity)
  const fade = opts.fade || 0
  const seam = opts.quality === 'max' ? 0.7 : opts.quality === 'low' ? 0.5 : 0.6

  const rb = getRibbonBuf(W, H)
  const b = rb.ctx
  b.setTransform(1, 0, 0, 1, 0, 0)
  b.clearRect(0, 0, W, H)
  b.globalAlpha = 1
  b.imageSmoothingEnabled = true
  b.imageSmoothingQuality = 'high'
  for (let i = 0; i < sections.length - 1; i++) {
    const s0 = sections[i], s1 = sections[i + 1]
    const l0x = s0.cx - s0.nx * s0.hw, l0y = s0.cy - s0.ny * s0.hw
    const r0x = s0.cx + s0.nx * s0.hw, r0y = s0.cy + s0.ny * s0.hw
    const l1x = s1.cx - s1.nx * s1.hw, l1y = s1.cy - s1.ny * s1.hw
    const r1x = s1.cx + s1.nx * s1.hw, r1y = s1.cy + s1.ny * s1.hw
    // Source quad is the whole strip [0..seedLen] × [0..1] (1-px tall ⇒ the colour
    // is constant along travel — that IS the stretch).
    drawTexturedTriangle(b, strip, l0x, l0y, r0x, r0y, l1x, l1y, 0, 0, seedLen, 0, 0, 1, seam)
    drawTexturedTriangle(b, strip, r0x, r0y, r1x, r1y, l1x, l1y, seedLen, 0, seedLen, 1, 0, 1, seam)
  }

  if (fade > 0) {
    const a = sections[0], z = sections[sections.length - 1]
    const grad = b.createLinearGradient(a.cx, a.cy, z.cx, z.cy)
    grad.addColorStop(0, 'rgba(0,0,0,1)')
    grad.addColorStop(1, `rgba(0,0,0,${clamp01(1 - fade)})`)
    b.globalCompositeOperation = 'destination-in'
    b.fillStyle = grad
    b.fillRect(0, 0, W, H)
    b.globalCompositeOperation = 'source-over'
  }

  ctx.save()
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.globalAlpha = opacity
  ctx.imageSmoothingEnabled = true
  ctx.drawImage(rb.canvas, 0, 0)
  ctx.restore()
  ctx.globalAlpha = 1
}

/** Sweep one ribbon (one direction) onto ctx. */
function sweepRibbon(ctx, strip, g, p, W, H, maxSlices, quality) {
  const stripLen = strip.width
  const straight = Math.abs(p.bend) < 0.002 && Math.abs(p.twist) < 0.002

  // Fast, perfectly-crisp path for an un-bent ribbon with no taper/fade — a single
  // rotated quad is already gap-free.
  if (straight && p.fade <= 0 && Math.abs(p.taper) <= 0.002) {
    const a = Math.atan2(g.d.y, g.d.x) - Math.PI / 2
    const cos = Math.cos(a), sin = Math.sin(a)
    ctx.globalAlpha = p.opacity
    // rotate about the start point, then draw in local space (+y = travel)
    ctx.setTransform(cos, sin, -sin, cos, g.start.x, g.start.y)
    ctx.drawImage(strip, 0, 0, stripLen, 1, -g.stripLen / 2, 0, g.stripLen, g.total)
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.globalAlpha = 1
    return
  }

  // Curved / tapered / faded: sweep the strip as one continuous gap-free ribbon.
  // `taper` < 0 widens the tip (a flare/trumpet); > 0 narrows it.
  const slices = sliceCount(g, p, maxSlices)
  const sections = new Array(slices + 1)
  for (let i = 0; i <= slices; i++) {
    const t = i / slices
    const c = cubic(g.start, g.c1, g.c2, g.end, t)
    const tan = cubicTangent(g.start, g.c1, g.c2, g.end, t)
    const hw = Math.max(0.25, (g.stripLen * (1 - p.taper * t)) / 2)
    sections[i] = { cx: c.x, cy: c.y, nx: tan.y, ny: -tan.x, hw }
  }
  sweepStripMesh(ctx, strip, sections, W, H, { opacity: p.opacity, fade: p.fade, quality })
}

/**
 * Build (into the current path) the region swept out by dragging a freeform
 * polygon along the ribbon's path — the Minkowski sum of the polygon with the
 * stretch path. We lay down the polygon (rotated to follow the tangent and
 * translated to each sample) repeatedly; overlapping same-wound copies union
 * under nonzero winding, so `ctx.clip('nonzero')` yields exactly the area the
 * streaks should occupy. `samples` is chosen so consecutive copies overlap.
 */
function addSweptPolygon(ctx, polyPx, g, samples) {
  const a0 = Math.atan2(g.d.y, g.d.x)
  const start = g.start
  for (let i = 0; i <= samples; i++) {
    const t = i / samples
    const c = cubic(g.start, g.c1, g.c2, g.end, t)
    const tan = cubicTangent(g.start, g.c1, g.c2, g.end, t)
    const rot = Math.atan2(tan.y, tan.x) - a0
    const cos = Math.cos(rot)
    const sin = Math.sin(rot)
    for (let j = 0; j < polyPx.length; j++) {
      const ox = polyPx[j].x - start.x
      const oy = polyPx[j].y - start.y
      const x = c.x + ox * cos - oy * sin
      const y = c.y + ox * sin + oy * cos
      if (j === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.closePath()
  }
}

/** How many polygon copies to lay down so the swept clip stays gap-free. */
function sweptClipSamples(polyPx, g, p) {
  // Travel-extent of the polygon (projection onto the travel axis): copies must
  // step by less than this to overlap. Curvier paths need more for smoothness.
  let lo = Infinity
  let hi = -Infinity
  for (const pt of polyPx) {
    const proj = pt.x * g.d.x + pt.y * g.d.y
    if (proj < lo) lo = proj
    if (proj > hi) hi = proj
  }
  const extent = Math.max(2, hi - lo)
  const overlap = Math.ceil(g.total / (extent * 0.6))
  const curve = Math.ceil((Math.abs(p.bend) + Math.abs(p.twist)) * 90)
  return clamp(Math.max(overlap, curve, 24), 24, 400)
}

const QUALITY_SLICES = { low: 220, high: 1600, max: 6000 }

/**
 * Render the pixel-stretch ribbon(s) onto `ctx` (a W×H 2D context). The caller
 * supplies `sample`, a W×H snapshot of the source image to draw seed colours
 * from, and is responsible for clearing/pre-painting the base image.
 *
 * `opts.quality` ('low' during drag, 'high' when settled, 'max' for the bake)
 * or an explicit `opts.maxSlices` tunes the cost/quality trade-off.
 *
 * Returns false if there's nothing to draw (degenerate band), true otherwise.
 */
export function renderPixelStretch(ctx, sample, params, W, H, opts = {}) {
  const p = clampStretchParams(params)
  // Mode dispatch (most expressive first): a multi-anchor Flow Path smear, then
  // the Photoshop-style Warp mesh, otherwise the single-arch simple sweep.
  if (p.flowPath) return renderFlowStretch(ctx, sample, p, W, H, opts)
  if (p.warpGrid) return renderWarpMesh(ctx, sample, p, W, H, opts)
  const maxSlices = opts.maxSlices || QUALITY_SLICES[opts.quality] || QUALITY_SLICES.high
  return paintSweep(ctx, sample, p, W, H, maxSlices, opts.quality)
}

/**
 * Paint the ribbon sweep (+ optional lasso clip + mirror pass) onto `ctx`.
 * `p` must already be clamped. Shared by the simple renderer and the warp
 * buffer builder. Returns false for a degenerate band.
 */
function paintSweep(ctx, sample, p, W, H, maxSlices, quality) {
  const g = resolveGeometry(p, W, H)
  if (g.stripLen < 1 || g.total < 1) return false

  const strip = buildSeedStrip(sample, g)
  const prevAlpha = ctx.globalAlpha
  const prevSmoothing = ctx.imageSmoothingEnabled
  ctx.imageSmoothingEnabled = true

  // Freeform "lasso": clip the sweep to the polygon dragged along the path, so
  // the streaks take the selection's silhouette instead of the band rectangle.
  const polyPx = p.polygon ? p.polygon.map((pt) => ({ x: pt.x * W, y: pt.y * H })) : null
  const clipped = polyPx && polyPx.length >= 3
  if (clipped) {
    ctx.save()
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.beginPath()
    addSweptPolygon(ctx, polyPx, g, sweptClipSamples(polyPx, g, p))
    if (p.mirror) {
      const pm = { ...p, bend: -p.bend }
      const gm = resolveGeometry(pm, W, H, -p.direction)
      addSweptPolygon(ctx, polyPx, gm, sweptClipSamples(polyPx, gm, pm))
    }
    ctx.clip('nonzero')
  }

  sweepRibbon(ctx, strip, g, p, W, H, maxSlices, quality)

  if (p.mirror) {
    // Same seed colours, swept the opposite way with the bow mirrored.
    const pm = { ...p, bend: -p.bend }
    const gm = resolveGeometry(pm, W, H, -p.direction)
    sweepRibbon(ctx, strip, gm, pm, W, H, maxSlices, quality)
  }

  if (clipped) ctx.restore()
  ctx.globalAlpha = prevAlpha
  ctx.imageSmoothingEnabled = prevSmoothing
  return true
}

/**
 * Re-composite the original subject (the selected band or polygon region) from
 * `sample` on top of ctx. Call this AFTER `renderPixelStretch` to produce the
 * classic "subject pops out from behind the color streaks" effect seen in the
 * reference videos.
 *
 * When a polygon (lasso) selection is active, clips to the polygon path so only
 * the subject silhouette is drawn. Otherwise clips to the band rectangle.
 */
export function renderSubjectOverlay(ctx, sample, params, W, H) {
  const p = clampStretchParams(params)
  const polyPx = p.polygon ? p.polygon.map((pt) => ({ x: pt.x * W, y: pt.y * H })) : null
  const hasPolygon = polyPx && polyPx.length >= 3

  ctx.save()
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.imageSmoothingEnabled = true

  // Clip to the selected region (polygon or band rectangle).
  ctx.beginPath()
  if (hasPolygon) {
    ctx.moveTo(polyPx[0].x, polyPx[0].y)
    for (let i = 1; i < polyPx.length; i++) ctx.lineTo(polyPx[i].x, polyPx[i].y)
    ctx.closePath()
  } else {
    const bx = p.band.x * W
    const by = p.band.y * H
    const bw = p.band.w * W
    const bh = p.band.h * H
    ctx.rect(bx, by, bw, bh)
  }
  ctx.clip()

  // Draw the original image within the clipped region — this puts the subject
  // back on top of the stretched colour ribbons.
  ctx.drawImage(sample, 0, 0, W, H)
  ctx.restore()
}

// ─── Warp Mesh Rendering (composite bicubic Bézier surface — Photoshop's Warp) ─
//
// Photoshop's Warp is a grid of bicubic Bézier patches. The control net carries
// ANCHORS (patch corners, at indices divisible by 3) which the surface passes
// through, and BEZIER HANDLES between them (the tangent / interior controls)
// which pull the sheet without lying on it — that pull is what produces the
// signature direction-handle curves. Splitting subdivides a patch (de Casteljau)
// into more patches, so you can sculpt as many independent curves as you want.

/** 1-D cubic Bézier (Bernstein basis) over 4 control values. */
const bezier1D = (p0, p1, p2, p3, t) => {
  const u = 1 - t
  return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3
}

/**
 * Evaluate the composite Bézier surface at (u,v) ∈ [0,1]². The net is carved
 * into (pr×pc) bicubic patches; we find the patch under (u,v), take its 4×4
 * sub-net, and evaluate the bicubic Bernstein blend. Anchors (patch corners)
 * are interpolated exactly; the in-between controls pull the surface.
 */
function evalWarpSurface(grid, u, v) {
  const { pr, pc } = warpPatches(grid)
  // Locate the patch and the local (lu,lv) within it.
  const fu = Math.min(pc, Math.max(0, u * pc))
  const pj = Math.min(pc - 1, Math.floor(fu === pc ? fu - 1e-9 : fu))
  const lu = fu - pj
  const fv = Math.min(pr, Math.max(0, v * pr))
  const pi = Math.min(pr - 1, Math.floor(fv === pr ? fv - 1e-9 : fv))
  const lv = fv - pi
  const r0 = pi * 3, c0 = pj * 3
  // Bézier in u for each of the 4 patch rows, then Bézier in v across the results.
  const cx = [0, 0, 0, 0], cy = [0, 0, 0, 0]
  for (let k = 0; k < 4; k++) {
    const row = grid[r0 + k]
    cx[k] = bezier1D(row[c0].x, row[c0 + 1].x, row[c0 + 2].x, row[c0 + 3].x, lu)
    cy[k] = bezier1D(row[c0].y, row[c0 + 1].y, row[c0 + 2].y, row[c0 + 3].y, lu)
  }
  return {
    x: bezier1D(cx[0], cx[1], cx[2], cx[3], lv),
    y: bezier1D(cy[0], cy[1], cy[2], cy[3], lv),
  }
}

// ── Stretched-buffer cache ───────────────────────────────────────────────────
// The warp deforms an already-stretched buffer (base image + straight sweep).
// During a warp-handle drag ONLY the grid changes, so we rebuild that buffer
// once and reuse it across every drag frame — the expensive sweep runs once, not
// 60×/sec. Keyed by everything that affects the buffer EXCEPT the grid.
let _warpBuf = { canvas: null, ctx: null, sig: '' }
let _sampleId = 0

function warpBufSignature(sample, straight, W, H, quality) {
  if (!sample.__psId) sample.__psId = ++_sampleId
  const b = straight.band
  return [
    sample.__psId, W, H, quality || 'high', straight.axis, straight.direction,
    b.x.toFixed(4), b.y.toFixed(4), b.w.toFixed(4), b.h.toFixed(4),
    straight.seed.toFixed(4), straight.length.toFixed(3), straight.fade.toFixed(3),
    straight.taper.toFixed(3), straight.opacity.toFixed(3), straight.mirror ? 1 : 0,
    straight.polygon ? straight.polygon.length : 0,
  ].join('|')
}

/** Build (or reuse) the base+sweep buffer the warp mesh samples from. */
function buildStretchedBuffer(sample, p, W, H, opts) {
  // Straighten the sweep: in warp mode the mesh is the sole source of curvature,
  // matching Photoshop's "Free-Transform scale, THEN Warp" workflow.
  const straight = { ...p, bend: 0, twist: 0 }
  const sig = warpBufSignature(sample, straight, W, H, opts.quality)
  let c = _warpBuf.canvas
  const sizeOk = c && c.width === W && c.height === H
  if (sizeOk && _warpBuf.sig === sig && !opts.freshBuffer) return c

  if (!sizeOk) {
    c = createStretchBuffer(W, H)
    _warpBuf.canvas = c
    _warpBuf.ctx = c.getContext('2d')
  }
  const bctx = _warpBuf.ctx
  bctx.setTransform(1, 0, 0, 1, 0, 0)
  bctx.clearRect(0, 0, W, H)
  bctx.imageSmoothingEnabled = true
  bctx.drawImage(sample, 0, 0, W, H)
  const maxSlices = opts.maxSlices || QUALITY_SLICES[opts.quality] || QUALITY_SLICES.high
  // Only pre-stretch the buffer when the user has EXPLICITLY lengthened the strip
  // (length > 1). At rest the warp box must show the ORIGINAL pixels so no streaks
  // appear until a control handle is dragged — the mesh deformation alone smears
  // the selected pixels into streaks, exactly like Photoshop's Free-Transform →
  // Warp. (Pre-fix layers saved with length > 1 keep their baked sweep.)
  if (straight.length > 1.001) paintSweep(bctx, sample, straight, W, H, maxSlices, opts.quality)
  _warpBuf.sig = sig
  return c
}

/**
 * Render the stretched image through the warp mesh. The base image + straight
 * pixel-stretch are drawn into a cached buffer; the buffer region under the
 * grid's rest rectangle is then texture-mapped onto the deformed interpolating
 * surface. Tessellation density adapts to quality (low=drag, high=settle,
 * max=bake) and to grid size, so a split grid stays smooth without over-drawing.
 *
 * @param {CanvasRenderingContext2D} ctx  Output context (W×H), base already painted
 * @param {HTMLCanvasElement|OffscreenCanvas} sample  Source image snapshot (W×H)
 * @param {StretchParams} params
 * @param {number} W
 * @param {number} H
 * @param {Object} [opts]
 * @param {'low'|'high'|'max'} [opts.quality]
 * @returns {boolean}
 */
export function renderWarpMesh(ctx, sample, params, W, H, opts = {}) {
  const p = clampStretchParams(params)
  const grid = p.warpGrid
  if (!grid) return false
  const rest = p.warpRest || getWarpRest(p)

  // The already-stretched content the mesh deforms (reused across warp drags).
  const buf = buildStretchedBuffer(sample, p, W, H, opts)

  const { pr, pc } = warpPatches(grid)
  // Segments PER PATCH — each bicubic patch needs enough subdivision to stay
  // smooth when a handle is pulled hard. Denser on settle/bake than during drag.
  const perPatch = opts.quality === 'low' ? 14 : opts.quality === 'max' ? 44 : 26
  const nu = Math.max(3, Math.min(240, pc * perPatch))
  const nv = Math.max(3, Math.min(240, pr * perPatch))
  // Overdraw each triangle by a sub-pixel margin so antialiased edges of
  // neighbouring triangles overlap instead of leaving hairline seams.
  const seam = opts.quality === 'max' ? 0.75 : 0.6

  // Precompute the dest+src lattice once, then stitch triangles from it. This
  // keeps the hot loop free of surface evals and object allocation.
  const cells = (nu + 1) * (nv + 1)
  const dX = new Float64Array(cells), dY = new Float64Array(cells)
  const sX = new Float64Array(cells), sY = new Float64Array(cells)
  for (let j = 0; j <= nv; j++) {
    const v = j / nv
    const sy = (rest.y + v * rest.h) * H
    for (let i = 0; i <= nu; i++) {
      const u = i / nu
      const d = evalWarpSurface(grid, u, v)
      const idx = j * (nu + 1) + i
      dX[idx] = d.x * W
      dY[idx] = d.y * H
      sX[idx] = (rest.x + u * rest.w) * W
      sY[idx] = sy
    }
  }

  ctx.save()
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  const stride = nu + 1
  for (let j = 0; j < nv; j++) {
    for (let i = 0; i < nu; i++) {
      const a = j * stride + i, b = a + 1, c = a + stride, e = c + 1
      drawTexturedTriangle(ctx, buf,
        dX[a], dY[a], dX[b], dY[b], dX[c], dY[c],
        sX[a], sY[a], sX[b], sY[b], sX[c], sY[c], seam)
      drawTexturedTriangle(ctx, buf,
        dX[b], dY[b], dX[e], dY[e], dX[c], dY[c],
        sX[b], sY[b], sX[e], sY[e], sX[c], sY[c], seam)
    }
  }
  ctx.restore()
  return true
}

/**
 * Draw a textured triangle via an affine transform that maps the source
 * triangle onto the destination triangle, clipped to the destination. `seam`
 * grows the clip outward from the centroid by that many pixels so adjacent
 * triangles overlap and antialiased seams disappear (the texture mapping itself
 * is unchanged — only the clip region is enlarged). Degenerate / collinear
 * triangles (a folded mesh) are skipped to avoid NaN transforms.
 */
function drawTexturedTriangle(ctx, img,
  dx0, dy0, dx1, dy1, dx2, dy2, // destination triangle
  sx0, sy0, sx1, sy1, sx2, sy2, // source triangle
  seam = 0,
) {
  // Compute the affine transform that maps the source triangle to the
  // destination. Bail first if the SOURCE triangle is degenerate.
  const denom = (sx0 * (sy1 - sy2) + sx1 * (sy2 - sy0) + sx2 * (sy0 - sy1))
  if (Math.abs(denom) < 1e-9) return

  // Clip to the destination triangle, inflated by `seam` to hide cracks.
  let cx0 = dx0, cy0 = dy0, cx1 = dx1, cy1 = dy1, cx2 = dx2, cy2 = dy2
  if (seam > 0) {
    const gx = (dx0 + dx1 + dx2) / 3, gy = (dy0 + dy1 + dy2) / 3
    const push = (x, y) => {
      const vx = x - gx, vy = y - gy
      const len = Math.hypot(vx, vy)
      if (len < 1e-6) return [x, y]
      const s = (len + seam) / len
      return [gx + vx * s, gy + vy * s]
    }
    ;[cx0, cy0] = push(dx0, dy0);[cx1, cy1] = push(dx1, dy1);[cx2, cy2] = push(dx2, dy2)
  }
  ctx.save()
  ctx.beginPath()
  ctx.moveTo(cx0, cy0)
  ctx.lineTo(cx1, cy1)
  ctx.lineTo(cx2, cy2)
  ctx.closePath()
  ctx.clip()

  // [sx0,sy0]→[dx0,dy0], [sx1,sy1]→[dx1,dy1], [sx2,sy2]→[dx2,dy2].
  const inv = 1 / denom
  const a = (dx0 * (sy1 - sy2) + dx1 * (sy2 - sy0) + dx2 * (sy0 - sy1)) * inv
  const b = (dx0 * (sx2 - sx1) + dx1 * (sx0 - sx2) + dx2 * (sx1 - sx0)) * inv
  const c = (dx0 * (sx1 * sy2 - sx2 * sy1) + dx1 * (sx2 * sy0 - sx0 * sy2) + dx2 * (sx0 * sy1 - sx1 * sy0)) * inv
  const d = (dy0 * (sy1 - sy2) + dy1 * (sy2 - sy0) + dy2 * (sy0 - sy1)) * inv
  const e = (dy0 * (sx2 - sx1) + dy1 * (sx0 - sx2) + dy2 * (sx1 - sx0)) * inv
  const f = (dy0 * (sx1 * sy2 - sx2 * sy1) + dy1 * (sx2 * sy0 - sx0 * sy2) + dy2 * (sx0 * sy1 - sx1 * sy0)) * inv

  ctx.setTransform(a, d, b, e, c, f)
  ctx.drawImage(img, 0, 0)
  ctx.restore()
}

/**
 * Get the warp control net as flat pixel-space handles for the overlay.
 * `kind` is 'anchor' (interpolated patch corner — drag to move the sheet),
 * 'handle' (a bezier direction handle — drag to bend the curve; `anchorRow/Col`
 * point to the anchor it belongs to so the UI can draw the tangent line) or
 * 'interior' (a patch's inner pull control).
 *
 * @returns {Array<{row:number,col:number,x:number,y:number,kind:'anchor'|'handle'|'interior',anchorRow:number,anchorCol:number}>|null}
 */
export function getWarpGridHandles(params, W = 1, H = 1) {
  const p = clampStretchParams(params)
  const grid = p.warpGrid
  if (!grid) return null
  const R = grid.length, C = grid[0].length
  const handles = []
  for (let r = 0; r < R; r++) {
    for (let c = 0; c < C; c++) {
      const ar = isWarpAnchor(r), ac = isWarpAnchor(c)
      let kind, anchorRow = -1, anchorCol = -1
      if (ar && ac) {
        kind = 'anchor'
      } else if (ar !== ac) {
        kind = 'handle'
        if (ar) { anchorRow = r; anchorCol = c % 3 === 1 ? c - 1 : c + 1 }
        else { anchorCol = c; anchorRow = r % 3 === 1 ? r - 1 : r + 1 }
      } else {
        kind = 'interior'
      }
      handles.push({ row: r, col: c, x: grid[r][c].x * W, y: grid[r][c].y * H, kind, anchorRow, anchorCol })
    }
  }
  return handles
}

/**
 * Sample the warp mesh as smooth polylines — one per patch-boundary row and
 * column (the visible Photoshop grid). Each polyline is pixel-space points.
 *
 * @returns {{rows:Array<Array<{x:number,y:number}>>, cols:Array<Array<{x:number,y:number}>>}|null}
 */
export function getWarpGridCurves(params, W = 1, H = 1, segPerPatch = 16) {
  const p = clampStretchParams(params)
  const grid = p.warpGrid
  if (!grid) return null
  const { pr, pc } = warpPatches(grid)
  const R = grid.length, C = grid[0].length
  const seg = clamp(Math.round(segPerPatch), 4, 48)
  const rows = []
  for (let r = 0; r < R; r += 3) {            // anchor rows = patch boundaries
    const v = r / (R - 1)
    const line = []
    const n = pc * seg
    for (let i = 0; i <= n; i++) { const d = evalWarpSurface(grid, i / n, v); line.push({ x: d.x * W, y: d.y * H }) }
    rows.push(line)
  }
  const cols = []
  for (let c = 0; c < C; c += 3) {
    const u = c / (C - 1)
    const line = []
    const n = pr * seg
    for (let j = 0; j <= n; j++) { const d = evalWarpSurface(grid, u, j / n); line.push({ x: d.x * W, y: d.y * H }) }
    cols.push(line)
  }
  return { rows, cols }
}

const _mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 })

/** de Casteljau split of a cubic [p0..p3] at t=0.5 → 7 points (shared midpoint). */
function splitCubicHalf(p0, p1, p2, p3) {
  const m01 = _mid(p0, p1), m12 = _mid(p1, p2), m23 = _mid(p2, p3)
  const l = _mid(m01, m12), r = _mid(m12, m23)
  const mid = _mid(l, r)
  return [p0, m01, l, mid, r, m23, p3]
}

/** Subdivide one composite-Bézier polyline (pc cubic segments) at every midpoint
 *  → 2× the patches, identical shape. Length 3n+1 → 6n+1. */
function subdivideBezierLine(line) {
  const segs = (line.length - 1) / 3
  const out = []
  for (let s = 0; s < segs; s++) {
    const sub = splitCubicHalf(line[s * 3], line[s * 3 + 1], line[s * 3 + 2], line[s * 3 + 3])
    if (s === 0) out.push(sub[0])
    out.push(sub[1], sub[2], sub[3], sub[4], sub[5], sub[6])
  }
  return out
}

/**
 * Split-warp: subdivide every patch along one axis (de Casteljau, so the shape
 * is byte-for-byte preserved) — doubling the patches there and giving you new
 * control points to sculpt. Capped at WARP_MAX_DIM. axis: 'row' | 'col'.
 */
export function addWarpSplit(grid, axis) {
  const g = sanitizeWarpGrid(grid)
  if (!g) return grid
  if (axis === 'row') {
    if (g.length >= WARP_MAX_DIM) return g
    const C = g[0].length
    const cols = []
    for (let c = 0; c < C; c++) {
      const col = []
      for (let r = 0; r < g.length; r++) col.push(g[r][c])
      cols.push(subdivideBezierLine(col))
    }
    const out = []
    for (let r = 0; r < cols[0].length; r++) {
      const row = []
      for (let c = 0; c < C; c++) row.push({ ...cols[c][r] })
      out.push(row)
    }
    return out
  }
  if (g[0].length >= WARP_MAX_DIM) return g
  return g.map((row) => subdivideBezierLine(row).map((pt) => ({ ...pt })))
}

// ─── Warp presets ────────────────────────────────────────────────────────────
// Each fn returns a displacement {dx,dy} in rest-normalised units for a control
// point at normalised cell coords (cu across the band, cv along the stretch:
// cv=0 = seed edge, cv=1 = streak tips). `applyWarpPreset` adds it to the flat
// grid — instant Photoshop-style warp shapes the user can then refine by hand.

const _falloff = (cu, cv) => Math.max(0, 1 - 2 * Math.hypot(cu - 0.5, cv - 0.5))

const PRESET_FNS = {
  flat: () => ({ dx: 0, dy: 0 }),
  arc: (cu, cv, a) => ({ dx: 0, dy: -0.45 * a * (1 - (2 * cu - 1) ** 2) }),
  arch: (cu, cv, a) => ({ dx: 0, dy: -0.6 * a * (1 - (2 * cu - 1) ** 2) * cv }),
  fan: (cu, cv, a) => ({ dx: (cu - 0.5) * 0.7 * a * cv, dy: -0.15 * a * (1 - (2 * cu - 1) ** 2) * cv }),
  wave: (cu, cv, a) => ({ dx: 0.4 * a * Math.sin(cv * Math.PI * 2) * cv, dy: 0 }),
  flag: (cu, cv, a) => ({ dx: 0, dy: 0.3 * a * Math.sin(cu * Math.PI * 3) * cv }),
  bulge: (cu, cv, a) => {
    const f = _falloff(cu, cv) * 0.6 * a
    return { dx: (cu - 0.5) * f, dy: (cv - 0.5) * f }
  },
  twist: (cu, cv, a) => {
    const ang = a * Math.PI * 0.6 * _falloff(cu, cv)
    const rx = cu - 0.5, ry = cv - 0.5
    const cos = Math.cos(ang), sin = Math.sin(ang)
    return { dx: rx * cos - ry * sin - rx, dy: rx * sin + ry * cos - ry }
  },
}

export const WARP_PRESETS = [
  { id: 'flat', label: 'Flat', hint: 'Straight grid (reset)' },
  { id: 'arc', label: 'Arc', hint: 'Bow into a single arc' },
  { id: 'arch', label: 'Arch', hint: 'Fan the tips into an arch' },
  { id: 'fan', label: 'Fan', hint: 'Splay the streaks outward' },
  { id: 'wave', label: 'Wave', hint: 'Growing serpentine wave' },
  { id: 'flag', label: 'Flag', hint: 'Rippling flag flutter' },
  { id: 'bulge', label: 'Bulge', hint: 'Push out from the centre' },
  { id: 'twist', label: 'Twist', hint: 'Spiral around the centre' },
]

/**
 * Build a warp grid from a named preset over the current rest footprint.
 * Returns `{ grid, rest }` — commit both to params.
 *
 * @param {StretchParams} params
 * @param {string} presetId
 * @param {number} [amount=1]  Intensity multiplier.
 * @param {number} [rows]  Override grid rows (defaults to current/4).
 * @param {number} [cols]  Override grid cols (defaults to current/4).
 */
export function applyWarpPreset(params, presetId, amount = 1, rows, cols) {
  amount = clamp(num(amount, 1), -2, 2)
  const p = clampStretchParams(params)
  const R = snapNetDim(rows || (p.warpGrid ? p.warpGrid.length : WARP_DEFAULT_ROWS))
  const C = snapNetDim(cols || (p.warpGrid ? p.warpGrid[0].length : WARP_DEFAULT_COLS))
  const rest = getWarpRest(p)
  const fn = PRESET_FNS[presetId] || PRESET_FNS.flat
  const grid = []
  for (let r = 0; r < R; r++) {
    const row = []
    for (let c = 0; c < C; c++) {
      const cu = C > 1 ? c / (C - 1) : 0
      const cv = R > 1 ? r / (R - 1) : 0
      const d = fn(cu, cv, amount)
      row.push({ x: rest.x + (cu + d.dx) * rest.w, y: rest.y + (cv + d.dy) * rest.h })
    }
    grid.push(row)
  }
  return { grid, rest }
}

// ─── Flow Path (multi-anchor directional spline smear) ───────────────────────
//
// The reference "pixel stretch" trend routes the smear through a freeform,
// multi-point curve: you drop a handful of anchors and the seed line is dragged
// along the WHOLE spline, following every bend. This is the directional cousin
// of the Warp mesh — a 1-D flow line (with per-anchor Bézier tangent handles and
// a width profile) instead of a 2-D cage — and it's the most expressive mode.
//
// Anchor model (all coords normalised to image space):
//   { x, y, hix, hiy, hox, hoy, w }
//     • (hix,hiy) IN-tangent handle  — offset ADDED to the anchor to get the
//        control point the curve approaches this anchor through.
//     • (hox,hoy) OUT-tangent handle — offset for the control point the curve
//        leaves this anchor through.
//     • w  width multiplier at this anchor (the ribbon thickness profile).
//   A smooth anchor keeps hin = -hout; breaking that yields a corner (Pen-tool).
// The segment between anchors i and i+1 is the cubic
//   p0 = A_i,  c1 = A_i + hout_i,  c2 = A_{i+1} + hin_{i+1},  p3 = A_{i+1}.

export const FLOW_MIN_ANCHORS = 2
export const FLOW_MAX_ANCHORS = 24
export const FLOW_DEFAULT_WIDTH = 0.18

const lerpPt = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t })

/** Validate/clamp a flow-path object to the renderable range, or null. */
function sanitizeFlowPath(fp) {
  if (!fp || !Array.isArray(fp.anchors) || fp.anchors.length < FLOW_MIN_ANCHORS) return null
  const anchors = []
  for (const a of fp.anchors) {
    if (!a || !Number.isFinite(a.x) || !Number.isFinite(a.y)) continue
    const fin = (v, d = 0) => (Number.isFinite(v) ? v : d)
    anchors.push({
      x: clamp(a.x, -0.6, 1.6), y: clamp(a.y, -0.6, 1.6),
      hix: clamp(fin(a.hix), -1.5, 1.5), hiy: clamp(fin(a.hiy), -1.5, 1.5),
      hox: clamp(fin(a.hox), -1.5, 1.5), hoy: clamp(fin(a.hoy), -1.5, 1.5),
      w: clamp(fin(a.w, 1), 0.05, 4),
    })
    if (anchors.length >= FLOW_MAX_ANCHORS) break
  }
  if (anchors.length < FLOW_MIN_ANCHORS) return null
  return { anchors, width: clamp(Number.isFinite(fp.width) ? fp.width : FLOW_DEFAULT_WIDTH, 0.01, 1.5) }
}

/** Cubic control points for the segment leaving anchor i (null past the end). */
function flowSegment(anchors, i) {
  const a = anchors[i], b = anchors[i + 1]
  if (!a || !b) return null
  return {
    p0: { x: a.x, y: a.y },
    c1: { x: a.x + (a.hox || 0), y: a.y + (a.hoy || 0) },
    c2: { x: b.x + (b.hix || 0), y: b.y + (b.hiy || 0) },
    p3: { x: b.x, y: b.y },
    w0: a.w ?? 1, w1: b.w ?? 1,
  }
}

/**
 * Build an arc-length lookup table of points along the whole flow spline, in
 * PIXEL space (W×H). Each entry carries the unit tangent, the interpolated width
 * multiplier, the cumulative arc length `s`, and its (seg, local-t) origin so
 * callers can split a segment. Pure math — no canvas, so it's unit-testable.
 *
 * @returns {{pts:Array<{x,y,tx,ty,w,s,seg,lt}>, total:number}}
 */
export function buildFlowLUT(anchors, W = 1, H = 1, perSeg = 48) {
  const pts = []
  if (!Array.isArray(anchors) || anchors.length < 2) return { pts, total: 0 }
  const seg = clamp(Math.round(perSeg), 6, 240)
  let total = 0, prev = null
  for (let s = 0; s < anchors.length - 1; s++) {
    const cs = flowSegment(anchors, s)
    for (let i = (s === 0 ? 0 : 1); i <= seg; i++) {
      const t = i / seg
      const c = cubic(cs.p0, cs.c1, cs.c2, cs.p3, t)
      const tan = cubicTangent(cs.p0, cs.c1, cs.c2, cs.p3, t)
      const x = c.x * W, y = c.y * H
      if (prev) total += Math.hypot(x - prev.x, y - prev.y)
      pts.push({ x, y, tx: tan.x, ty: tan.y, w: cs.w0 + (cs.w1 - cs.w0) * t, s: total, seg: s, lt: t })
      prev = { x, y }
    }
  }
  return { pts, total }
}

/** Binary-search the LUT for the sample at arc length `s` (px), lerping fields. */
function flowAt(lut, s) {
  const pts = lut.pts
  if (pts.length === 0) return null
  if (s <= 0) return pts[0]
  if (s >= lut.total) return pts[pts.length - 1]
  let lo = 0, hi = pts.length - 1
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1
    if (pts[mid].s < s) lo = mid; else hi = mid
  }
  const a = pts[lo], b = pts[hi]
  const span = b.s - a.s || 1
  const t = (s - a.s) / span
  return {
    x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t,
    tx: a.tx + (b.tx - a.tx) * t, ty: a.ty + (b.ty - a.ty) * t,
    w: a.w + (b.w - a.w) * t,
  }
}

/**
 * Sample the seed line — a strip of colours taken PERPENDICULAR to the path at
 * its first anchor — into a `seedLen`×1 buffer. The affine maps image-pixel
 * coords so the perpendicular line lands along the strip's x-axis (bilinear), so
 * the seed is correct for any path orientation (not just axis-aligned).
 */
function buildFlowSeedStrip(sample, lut, fp, W, H) {
  const start = lut.pts[0]
  const tl = Math.hypot(start.tx, start.ty) || 1
  const tHatx = start.tx / tl, tHaty = start.ty / tl   // travel direction
  const perpx = -tHaty, perpy = tHatx                  // seed line direction
  const seedLen = Math.max(2, Math.round(fp.width * Math.min(W, H)))
  const half = seedLen / 2
  const W0x = start.x - perpx * half, W0y = start.y - perpy * half  // strip x=0 world pt
  const strip = createStretchBuffer(seedLen, 1)
  const sctx = strip.getContext('2d')
  sctx.imageSmoothingEnabled = true
  sctx.clearRect(0, 0, seedLen, 1)
  // image P → strip: strip_x = perp·(P-W0), strip_y = tHat·(P-W0) + 0.5
  const a = perpx, c = perpy, b = tHatx, d = tHaty
  const e = -(W0x * a + W0y * c)
  const f = -(W0x * b + W0y * d) + 0.5
  sctx.setTransform(a, b, c, d, e, f)
  sctx.drawImage(sample, 0, 0)
  sctx.setTransform(1, 0, 0, 1, 0, 0)
  return { strip, seedLen }
}

/**
 * Render the flow-path smear: sweep the perpendicular seed strip along the spline
 * at arc-length-even spacing, rotating it to follow the tangent, with a width
 * profile (per-anchor width × global taper) and an alpha fade toward the tip.
 * `opts.quality` ('low' drag / 'high' settle / 'max' bake) tunes slice density.
 */
export function renderFlowStretch(ctx, sample, params, W, H, opts = {}) {
  const p = clampStretchParams(params)
  const fp = p.flowPath
  if (!fp) return false
  const perSeg = opts.quality === 'low' ? 24 : opts.quality === 'max' ? 96 : 56
  const lut = buildFlowLUT(fp.anchors, W, H, perSeg)
  if (lut.total < 2 || lut.pts.length < 2) return false

  const { strip, seedLen } = buildFlowSeedStrip(sample, lut, fp, W, H)
  const maxSlices = opts.maxSlices || QUALITY_SLICES[opts.quality] || QUALITY_SLICES.high
  // Cross-sections only need enough density to follow the spline's curve smoothly
  // — the ribbon mesh is gap-free at ANY spacing (unlike the old slice stamping,
  // which needed thousands of overlapping rectangles and still gapped on bends).
  const slices = clamp(Math.ceil(lut.total * 0.8), 32, maxSlices)
  const sections = new Array(slices + 1)
  for (let i = 0; i <= slices; i++) {
    const t = i / slices
    const at = flowAt(lut, t * lut.total)
    if (!at) { sections[i] = sections[i - 1]; continue }
    const al = Math.hypot(at.tx, at.ty) || 1
    const tx = at.tx / al, ty = at.ty / al
    const hw = Math.max(0.25, (seedLen * at.w * (1 - p.taper * t)) / 2)
    sections[i] = { cx: at.x, cy: at.y, nx: ty, ny: -tx, hw }
  }
  sweepStripMesh(ctx, strip, sections, W, H, { opacity: p.opacity, fade: p.fade, quality: opts.quality })
  return true
}

/** The spline as a normalised-or-pixel polyline (W=H=1 → normalised) for overlay. */
export function getFlowPathCurve(params, W = 1, H = 1, perSeg = 32) {
  const p = clampStretchParams(params)
  if (!p.flowPath) return null
  return buildFlowLUT(p.flowPath.anchors, W, H, perSeg).pts.map((pt) => ({ x: pt.x, y: pt.y }))
}

/**
 * Flat list of draggable controls (pixel space) for the overlay/hit-testing:
 * one 'anchor' per point plus its 'in'/'out' tangent handles (endpoints expose
 * only the handle that has a segment).
 */
export function getFlowPathHandles(params, W = 1, H = 1) {
  const p = clampStretchParams(params)
  const fp = p.flowPath
  if (!fp) return null
  const A = fp.anchors
  const out = []
  for (let i = 0; i < A.length; i++) {
    const a = A[i]
    out.push({ idx: i, kind: 'anchor', x: a.x * W, y: a.y * H })
    if (i > 0) out.push({ idx: i, kind: 'in', x: (a.x + (a.hix || 0)) * W, y: (a.y + (a.hiy || 0)) * H })
    if (i < A.length - 1) out.push({ idx: i, kind: 'out', x: (a.x + (a.hox || 0)) * W, y: (a.y + (a.hoy || 0)) * H })
  }
  return out
}

/**
 * Build a flow path from a bare list of points (e.g. an AI/heuristic plan),
 * deriving smooth Catmull-Rom tangent handles. `opts.tension` (default 1) scales
 * the handles; `opts.width` sets the ribbon width.
 */
export function createFlowPathFromPoints(points, opts = {}) {
  const pts = (points || [])
    .filter((q) => q && Number.isFinite(q.x) && Number.isFinite(q.y))
    .map((q) => ({ x: clamp(q.x, -0.6, 1.6), y: clamp(q.y, -0.6, 1.6) }))
  if (pts.length < 2) return null
  const k = Number.isFinite(opts.tension) ? opts.tension : 1
  const anchors = pts.map((pt, i) => {
    const prev = pts[i - 1] || pt
    const next = pts[i + 1] || pt
    const tx = ((next.x - prev.x) / 6) * k
    const ty = ((next.y - prev.y) / 6) * k
    return { x: pt.x, y: pt.y, hix: -tx, hiy: -ty, hox: tx, hoy: ty, w: 1 }
  })
  return sanitizeFlowPath({ anchors, width: opts.width ?? FLOW_DEFAULT_WIDTH })
}

/**
 * Default flow path for the current params — samples the simple-mode arch so
 * toggling Simple → Flow Path inherits the look, then hands you full multi-point
 * control. Falls back to a centred vertical-ish flow when geometry is degenerate.
 */
export function createDefaultFlowPath(params = DEFAULT_STRETCH, nAnchors = 4) {
  const p = clampStretchParams(params)
  const g = resolveGeometry(p, 1, 1)
  const n = clamp(Math.round(nAnchors), 2, 8)
  const pts = []
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1)
    const c = cubic(g.start, g.c1, g.c2, g.end, t)
    pts.push({ x: c.x, y: c.y })
  }
  const widthBase = clamp((g.vertical ? p.band.w : p.band.h) || FLOW_DEFAULT_WIDTH, 0.04, 1)
  return createFlowPathFromPoints(pts, { width: widthBase }) ||
    createFlowPathFromPoints([{ x: 0.5, y: 0.2 }, { x: 0.5, y: 0.8 }], { width: FLOW_DEFAULT_WIDTH })
}

/** Re-derive smooth Catmull-Rom handles from the current anchor POSITIONS. */
export function smoothFlowPath(fp) {
  const path = sanitizeFlowPath(fp)
  if (!path) return fp
  const re = createFlowPathFromPoints(path.anchors.map((a) => ({ x: a.x, y: a.y })), { width: path.width })
  // keep per-anchor widths
  if (re) re.anchors.forEach((a, i) => { a.w = path.anchors[i]?.w ?? 1 })
  return re || path
}

/**
 * Insert an anchor on the spline at the point nearest (nx,ny) (normalised),
 * splitting the underlying cubic with de Casteljau so the curve SHAPE is exactly
 * preserved and the neighbours' handles are updated. Returns the new path (or the
 * input untouched when already at the anchor cap).
 */
export function insertFlowAnchor(fp, nx, ny, W = 1, H = 1) {
  const path = sanitizeFlowPath(fp)
  if (!path || path.anchors.length >= FLOW_MAX_ANCHORS) return path
  const lut = buildFlowLUT(path.anchors, W, H, 40)
  if (lut.pts.length < 2) return path
  const px = nx * W, py = ny * H
  let best = lut.pts[0], bd = Infinity
  for (const pt of lut.pts) {
    const d = (pt.x - px) ** 2 + (pt.y - py) ** 2
    if (d < bd) { bd = d; best = pt }
  }
  const seg = best.seg
  const t = clamp(best.lt, 0.02, 0.98)
  const cs = flowSegment(path.anchors, seg)
  const m0 = lerpPt(cs.p0, cs.c1, t)
  const m1 = lerpPt(cs.c1, cs.c2, t)
  const m2 = lerpPt(cs.c2, cs.p3, t)
  const q0 = lerpPt(m0, m1, t)
  const q1 = lerpPt(m1, m2, t)
  const split = lerpPt(q0, q1, t)
  const a0 = path.anchors[seg], a1 = path.anchors[seg + 1]
  const anchors = path.anchors.map((a) => ({ ...a }))
  anchors[seg].hox = m0.x - a0.x; anchors[seg].hoy = m0.y - a0.y
  anchors[seg + 1].hix = m2.x - a1.x; anchors[seg + 1].hiy = m2.y - a1.y
  anchors.splice(seg + 1, 0, {
    x: split.x, y: split.y,
    hix: q0.x - split.x, hiy: q0.y - split.y,
    hox: q1.x - split.x, hoy: q1.y - split.y,
    w: (a0.w + a1.w) / 2,
  })
  return sanitizeFlowPath({ anchors, width: path.width })
}

/** Remove the anchor at `idx` (never drops below FLOW_MIN_ANCHORS). */
export function removeFlowAnchor(fp, idx) {
  const path = sanitizeFlowPath(fp)
  if (!path || path.anchors.length <= FLOW_MIN_ANCHORS) return path
  if (idx < 0 || idx >= path.anchors.length) return path
  return sanitizeFlowPath({ anchors: path.anchors.filter((_, i) => i !== idx), width: path.width })
}

// ── Flow presets ─────────────────────────────────────────────────────────────
// Each generator returns a list of normalised points routed relative to the
// selected band — instant flowing shapes the user can then refine by hand.

const _flowBandCenter = (band) => ({
  cx: band.x + band.w / 2, cy: band.y + band.h / 2,
  bw: band.w, bh: band.h,
})

const FLOW_PRESET_FNS = {
  ribbon: (b) => {
    const { cx, cy, bh } = _flowBandCenter(b)
    const span = clamp(bh * 2.6, 0.4, 1.3)
    const y0 = clamp(cy - span / 2, 0.02, 0.98)
    return [
      { x: cx - 0.18, y: y0 }, { x: cx + 0.14, y: y0 + span * 0.33 },
      { x: cx - 0.12, y: y0 + span * 0.66 }, { x: cx + 0.16, y: clamp(y0 + span, 0.02, 0.98) },
    ]
  },
  scurve: (b) => {
    const { cx, cy, bh } = _flowBandCenter(b)
    const span = clamp(bh * 2.4, 0.4, 1.2)
    const y0 = clamp(cy - span / 2, 0.02, 0.98)
    return [
      { x: cx, y: y0 }, { x: cx + 0.22, y: y0 + span * 0.3 },
      { x: cx - 0.22, y: y0 + span * 0.7 }, { x: cx, y: clamp(y0 + span, 0.02, 0.98) },
    ]
  },
  spiral: (b) => {
    const { cx, cy } = _flowBandCenter(b)
    const pts = []
    const turns = 1.5, steps = 7
    for (let i = 0; i <= steps; i++) {
      const t = i / steps
      const r = 0.06 + t * 0.34
      const a = t * turns * Math.PI * 2
      pts.push({ x: clamp(cx + Math.cos(a) * r, 0.02, 0.98), y: clamp(cy + Math.sin(a) * r, 0.02, 0.98) })
    }
    return pts
  },
  cascade: (b) => {
    const { cx, cy, bw } = _flowBandCenter(b)
    const x0 = clamp(cx - bw, 0.04, 0.6)
    return [
      { x: x0, y: clamp(cy - 0.28, 0.02, 0.98) }, { x: x0 + 0.26, y: cy - 0.04 },
      { x: x0 + 0.5, y: cy + 0.14 }, { x: clamp(x0 + 0.8, 0.1, 0.98), y: clamp(cy + 0.3, 0.02, 0.98) },
    ]
  },
}

export const FLOW_PRESETS = [
  { id: 'ribbon', label: 'Ribbon', hint: 'Flowing woven ribbon' },
  { id: 'scurve', label: 'S-Flow', hint: 'Serpentine S-curve' },
  { id: 'spiral', label: 'Spiral', hint: 'Vortex spiral inward' },
  { id: 'cascade', label: 'Cascade', hint: 'Diagonal cascading sweep' },
]

/** Build a flow path from a named preset over the current band. */
export function applyFlowPreset(params, presetId) {
  const p = clampStretchParams(params)
  const fn = FLOW_PRESET_FNS[presetId] || FLOW_PRESET_FNS.ribbon
  const pts = fn(p.band)
  return createFlowPathFromPoints(pts, { width: clamp((p.axis === 'vertical' ? p.band.w : p.band.h) || FLOW_DEFAULT_WIDTH, 0.05, 0.6) })
}

// ─── Subject-aware placement (layer compositing) ─────────────────────────────
//
// A pixel-stretch is committed as its OWN layer above the source photo. To let
// the streaks sit "above / partially on / below the subject", we use the subject
// matte: knock the subject out of the ribbon layer (so the subject on the layer
// below shows through) by an adjustable amount. `clientSubjectMask` returns a
// LUMINANCE matte (white = subject, fully opaque), so these helpers first turn it
// into an ALPHA matte before compositing.

/**
 * Convert a luminance subject matte (white = subject, opaque) into an ALPHA matte
 * canvas (white fill, alpha = subject coverage) at W×H, with optional edge
 * feather (px blur). Returns null if the matte can't be read (tainted).
 */
export function matteToAlphaCanvas(matte, W, H, feather = 0) {
  if (!matte) return null
  const mw = matte.width, mh = matte.height
  if (!mw || !mh) return null
  const src = createStretchBuffer(mw, mh)
  const sctx = src.getContext('2d', { willReadFrequently: true })
  sctx.drawImage(matte, 0, 0)
  let img
  try { img = sctx.getImageData(0, 0, mw, mh) } catch { return null }
  const d = img.data
  for (let i = 0; i < d.length; i += 4) {
    const a = d[i]                       // grayscale → R == coverage
    d[i] = 255; d[i + 1] = 255; d[i + 2] = 255; d[i + 3] = a
  }
  sctx.putImageData(img, 0, 0)
  const out = createStretchBuffer(W, H)
  const octx = out.getContext('2d')
  octx.imageSmoothingEnabled = true
  if (feather > 0) octx.filter = `blur(${feather}px)`
  octx.drawImage(src, 0, 0, W, H)
  octx.filter = 'none'
  return out
}

/**
 * Knock the subject out of an already-painted ribbon layer (destination-out) so
 * the subject on the layer BELOW shows through. `alphaMatte` is from
 * matteToAlphaCanvas; `strength` 0..1 scales removal (1 = subject fully behind
 * the ribbons; 0.5 = the streaks half-cover the subject; 0 = on top of it).
 */
export function applySubjectKnockout(ctx, alphaMatte, W, H, strength = 1) {
  if (!alphaMatte || strength <= 0) return
  const prevOp = ctx.globalCompositeOperation
  const prevAlpha = ctx.globalAlpha
  ctx.globalCompositeOperation = 'destination-out'
  ctx.globalAlpha = clamp01(strength)
  ctx.imageSmoothingEnabled = true
  ctx.drawImage(alphaMatte, 0, 0, W, H)
  ctx.globalCompositeOperation = prevOp
  ctx.globalAlpha = prevAlpha
}

/**
 * Build the subject cutout from the source — the subject pixels where the matte
 * is opaque, transparent elsewhere — so the live preview can re-composite the
 * subject ON TOP of the ribbons (matching the layered bake: subject-below + a
 * subject-shaped hole in the ribbons reads the same as subject-on-top).
 */
export function buildSubjectCutout(sample, alphaMatte, W, H) {
  if (!alphaMatte) return null
  const out = createStretchBuffer(W, H)
  const octx = out.getContext('2d')
  octx.imageSmoothingEnabled = true
  octx.drawImage(sample, 0, 0, W, H)
  octx.globalCompositeOperation = 'destination-in'
  octx.drawImage(alphaMatte, 0, 0, W, H)
  octx.globalCompositeOperation = 'source-over'
  return out
}

// ─── On-device heuristic planner ─────────────────────────────────────────────

/**
 * Analyse a sample canvas and pick a strong pixel-stretch plan WITHOUT any API
 * call — the "think like an editor" fallback when the AI route is unavailable.
 *
 * It scores every candidate seed line by colourfulness + tonal variety (vivid,
 * multi-colour lines make the best streaks), picks the axis with the strongest
 * line, sweeps toward the emptiest (lowest-detail) side so the streaks have room
 * to breathe, and sizes the ribbon to the available space — then dresses it with
 * a tasteful editorial arch. Returns a plan shaped like the AI route's plan.
 *
 * @param {HTMLCanvasElement|OffscreenCanvas} canvas
 * @returns {{region:{x,y,w,h},axis,direction,seed,length,bend,twist,fade,taper,mirror,opacity,reasoning}|null}
 */
export function analyzeStretchPlan(canvas) {
  const W = canvas.width | 0, H = canvas.height | 0
  if (W < 4 || H < 4) return null
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) return null
  let img
  try { img = ctx.getImageData(0, 0, W, H) } catch { return null } // tainted → bail
  const d = img.data

  const nx = Math.min(W, 180), ny = Math.min(H, 180)
  const sx = W / nx, sy = H / ny
  const luma = new Float32Array(nx * ny)
  const sat = new Float32Array(nx * ny)
  for (let gy = 0; gy < ny; gy++) {
    const py = Math.min(H - 1, (gy * sy) | 0)
    for (let gx = 0; gx < nx; gx++) {
      const px = Math.min(W - 1, (gx * sx) | 0)
      const o = (py * W + px) * 4
      const r = d[o], g = d[o + 1], b = d[o + 2]
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b)
      const k = gy * nx + gx
      luma[k] = 0.299 * r + 0.587 * g + 0.114 * b
      sat[k] = mx > 4 ? (mx - mn) / mx : 0
    }
  }

  // Score every row (vertical-stretch seed) and column (horizontal-stretch seed)
  // by mean saturation + tonal spread, and measure detail to find empty space.
  const lineStats = (count, span, at) => {
    const score = new Float32Array(count)
    const detail = new Float32Array(count)
    for (let i = 0; i < count; i++) {
      let sSum = 0, lSum = 0, lSq = 0, eSum = 0, prev = 0
      for (let j = 0; j < span; j++) {
        const k = at(i, j)
        const L = luma[k]
        sSum += sat[k]; lSum += L; lSq += L * L
        if (j > 0) eSum += Math.abs(L - prev)
        prev = L
      }
      const mean = lSum / span
      const variance = Math.max(0, lSq / span - mean * mean)
      score[i] = sSum / span + 1.3 * (Math.sqrt(variance) / 255)
      detail[i] = eSum / span
    }
    return { score, detail }
  }
  const rows = lineStats(ny, nx, (i, j) => i * nx + j)
  const cols = lineStats(nx, ny, (i, j) => j * nx + i)

  const argmax = (arr) => { let bi = 0, bv = -Infinity; for (let i = 0; i < arr.length; i++) if (arr[i] > bv) { bv = arr[i]; bi = i } return { i: bi, v: bv } }
  const bestRow = argmax(rows.score)
  const bestCol = argmax(cols.score)

  const aspect = W / H
  const vAppeal = bestRow.v * (aspect < 1 ? 1.15 : 0.95)
  const hAppeal = bestCol.v * (aspect > 1 ? 1.15 : 0.95)
  const vertical = vAppeal >= hAppeal

  // Detail on each side of the seed → sweep toward the emptier side.
  const seedIdx = vertical ? bestRow.i : bestCol.i
  const detail = vertical ? rows.detail : cols.detail
  const count = detail.length
  let before = 0, after = 0
  for (let i = 0; i < count; i++) (i < seedIdx ? (before += detail[i]) : (after += detail[i]))
  const beforeAvg = before / Math.max(1, seedIdx)
  const afterAvg = after / Math.max(1, count - seedIdx)
  const direction = afterAvg <= beforeAvg ? 1 : -1
  const seedNorm = seedIdx / (count - 1)

  // Room available toward the sweep direction → ribbon length.
  const room = direction > 0 ? 1 - seedNorm : seedNorm
  const length = clamp(1.6 + room * 5, 1.8, 5.5)

  // A thin source slab straddling the seed line, leaving cross-axis breathing room.
  const slab = 0.16
  const region = vertical
    ? { x: 0.08, y: clamp(seedNorm - slab / 2, 0, 1 - slab), w: 0.84, h: slab }
    : { x: clamp(seedNorm - slab / 2, 0, 1 - slab), y: 0.08, w: slab, h: 0.84 }

  // ── Intelligent flow-path routing (on-device "model") ──────────────────────
  // Build a 2-D energy map (local luma gradient) and route a smooth multi-point
  // path from the most colourful seed cell, marching along the stretch axis while
  // drifting toward the CALMEST cross-position at each step — so the streak flows
  // through negative space like a pro would draw it, no API needed.
  const energy = new Float32Array(nx * ny)
  for (let gy = 0; gy < ny; gy++) {
    for (let gx = 0; gx < nx; gx++) {
      const lx = luma[gy * nx + Math.min(nx - 1, gx + 1)] - luma[gy * nx + Math.max(0, gx - 1)]
      const ly = luma[Math.min(ny - 1, gy + 1) * nx + gx] - luma[Math.max(0, gy - 1) * nx + gx]
      energy[gy * nx + gx] = Math.hypot(lx, ly)
    }
  }
  const startCross = (() => {
    let bi = 0, bv = -Infinity
    if (vertical) {
      const gy = Math.min(ny - 1, Math.round(seedNorm * (ny - 1)))
      for (let gx = 0; gx < nx; gx++) { const v = sat[gy * nx + gx]; if (v > bv) { bv = v; bi = gx } }
      return bi / (nx - 1)
    }
    const gx = Math.min(nx - 1, Math.round(seedNorm * (nx - 1)))
    for (let gy = 0; gy < ny; gy++) { const v = sat[gy * nx + gx]; if (v > bv) { bv = v; bi = gy } }
    return bi / (ny - 1)
  })()
  const K = 6
  const endAxis = direction > 0
    ? clamp(seedNorm + room, seedNorm + 0.15, 0.98)
    : clamp(seedNorm - room, 0.02, seedNorm - 0.15)
  const flowPath = []
  let cross = startCross
  const crossSpan = vertical ? nx : ny
  const lateralWin = Math.max(1, Math.round(0.16 * (crossSpan - 1)))
  for (let i = 0; i <= K; i++) {
    const tt = i / K
    const axisPos = seedNorm + (endAxis - seedNorm) * tt
    if (i > 0) {
      const center = Math.round(cross * (crossSpan - 1))
      const axisIdx = vertical
        ? Math.min(ny - 1, Math.round(axisPos * (ny - 1)))
        : Math.min(nx - 1, Math.round(axisPos * (nx - 1)))
      let bestC = center, bestE = Infinity
      for (let c = Math.max(0, center - lateralWin); c <= Math.min(crossSpan - 1, center + lateralWin); c++) {
        const k = vertical ? axisIdx * nx + c : c * nx + axisIdx
        const e = energy[k] + Math.abs(c - center) * 0.4   // bias toward smooth drift
        if (e < bestE) { bestE = e; bestC = c }
      }
      cross = bestC / (crossSpan - 1)
    }
    flowPath.push(vertical ? { x: cross, y: axisPos } : { x: axisPos, y: cross })
  }

  return {
    region,
    axis: vertical ? 'vertical' : 'horizontal',
    direction,
    seed: 0.5,
    length,
    bend: 0.5,
    twist: 0.15,
    fade: 0.18,
    taper: 0.12,
    mirror: false,
    opacity: 1,
    flowPath,
    flowWidth: clamp(slab * 1.1, 0.08, 0.4),
    reasoning: `On-device: ${vertical ? 'vertical' : 'horizontal'} flow from the most colourful ${vertical ? 'row' : 'column'}, routed ${direction > 0 ? 'forward' : 'back'} through the calmest negative space.`,
  }
}
