#!/usr/bin/env bun
// Hostile-input fuzz for phosmith's pure libs. Failure = throw, non-finite
// numbers in output, or broken shape contracts.
import { boxBlur, growCoverage, MAX_GROW_PX } from '../src/lib/mask-grow-core.js'
import { computeGradientMagnitude, snapToEdgePoint } from '../src/lib/mask-edge-snap.js'
import { buildLut, buildPackedLutFromCurves, buildCurveSvgPath, LUT_SIZE } from '../src/lib/curve-lut.js'
import {
  clampStretchParams, getPolygonBBox, getStretchAnchors, getStretchPath,
  createDefaultWarpGrid, getWarpRest, getWarpGridHandles, getWarpGridCurves,
  addWarpSplit, applyWarpPreset, WARP_MIN_DIM, WARP_MAX_DIM, DEFAULT_STRETCH,
} from '../src/lib/pixel-stretch.js'
import { parseMaskDescription } from '../src/lib/agent/nl-mask-parser.js'

let pass = 0, fail = 0
const fails = []
const ok = (label, cond, detail = '') => {
  if (cond) { pass += 1 } else { fail += 1; fails.push(`${label} — ${detail}`); console.log(`✗ ${label} — ${detail}`) }
}

// Walk any structure; return path of first NaN/±Infinity number.
const nonFinite = (v, path = '$', depth = 0, seen = new Set()) => {
  if (depth > 8 || v == null) return null
  if (typeof v === 'number') return Number.isFinite(v) ? null : `${path}=${v}`
  if (typeof v !== 'object') return null
  if (seen.has(v)) return null
  seen.add(v)
  if (ArrayBuffer.isView(v)) {
    for (let i = 0; i < v.length; i += 1) if (!Number.isFinite(v[i])) return `${path}[${i}]=${v[i]}`
    return null
  }
  if (Array.isArray(v)) {
    for (let i = 0; i < v.length; i += 1) { const r = nonFinite(v[i], `${path}[${i}]`, depth + 1, seen); if (r) return r }
    return null
  }
  for (const k of Object.keys(v)) { const r = nonFinite(v[k], `${path}.${k}`, depth + 1, seen); if (r) return r }
  return null
}
const attempt = (label, fn, { scan = true, shape } = {}) => {
  try {
    const out = fn()
    if (scan) { const nf = nonFinite(out); ok(`${label}: finite output`, !nf, nf || '') }
    if (shape) { const s = shape(out); ok(`${label}: shape`, !s, s || '') }
  } catch (e) { ok(`${label}: no throw`, false, String(e).slice(0, 120)) }
}

const HOSTILE_NUM = [NaN, Infinity, -Infinity, 0, -1, 1e9, -1e9, 0.5, 2.5]

/* ── mask-grow-core ── */
{
  const W = 16, H = 16
  const cover = new Uint8ClampedArray(W * H).fill(0)
  for (let y = 4; y < 12; y += 1) for (let x = 4; x < 12; x += 1) cover[y * W + x] = 255
  for (const px of HOSTILE_NUM) {
    attempt(`growCoverage px=${px}`, () => growCoverage(cover, W, H, px), {
      shape: (o) => (o.length === W * H ? null : `len=${o.length}`),
    })
  }
  attempt('growCoverage w=0', () => growCoverage(cover, 0, H, 5))
  attempt('growCoverage h=0', () => growCoverage(cover, W, 0, 5))
  attempt('growCoverage cover SHORT (w*h mismatch)', () => growCoverage(cover.slice(0, 50), W, H, 5), {
    shape: (o) => (o.length === W * H ? null : `len=${o.length}`),
  })
  attempt('growCoverage Float32 input', () => growCoverage(Float32Array.from(cover), W, H, 3))
  for (const r of [0, 1, 500, 2.5]) {
    attempt(`boxBlur r=${r}`, () => boxBlur(Float32Array.from(cover, (v) => v / 255), W, H, r))
  }
  attempt('boxBlur 1x1', () => boxBlur(new Float32Array([0.5]), 1, 1, 3))
}

/* ── mask-edge-snap ── */
{
  const W = 16, H = 16
  const rgba = new Uint8ClampedArray(W * H * 4)
  for (let y = 0; y < H; y += 1) for (let x = 0; x < W; x += 1) {
    const v = x < 8 ? 20 : 230
    rgba.set([v, v, v, 255], (y * W + x) * 4)
  }
  const map = computeGradientMagnitude(rgba, W, H)
  attempt('computeGradientMagnitude 0x0', () => computeGradientMagnitude(new Uint8ClampedArray(0), 0, 0))
  attempt('computeGradientMagnitude 1x1', () => computeGradientMagnitude(new Uint8ClampedArray(4), 1, 1))
  attempt('computeGradientMagnitude SHORT rgba', () => computeGradientMagnitude(rgba.slice(0, 40), W, H))
  attempt('snap null map', () => snapToEdgePoint(null, 5, 5, 4, 0.2))
  for (const v of HOSTILE_NUM) {
    attempt(`snap px=${v}`, () => snapToEdgePoint(map, v, 8, 4, 0.2), { scan: false })
    attempt(`snap radius=${v}`, () => snapToEdgePoint(map, 8, 8, v, 0.2), { scan: false })
    attempt(`snap contrast=${v}`, () => snapToEdgePoint(map, 8, 8, 4, v))
  }
  attempt('snap map w/h mismatch', () => snapToEdgePoint({ mag: map.mag, w: 64, h: 64 }, 60, 60, 5, 0.1), { scan: false })
}

/* ── curve-lut ── */
{
  const lutShape = (o) => (o && o.length === LUT_SIZE ? null : `len=${o?.length}`)
  attempt('buildLut undefined', () => buildLut(undefined), { shape: lutShape })
  attempt('buildLut []', () => buildLut([]), { shape: lutShape })
  attempt('buildLut NaN pts', () => buildLut([{ x: NaN, y: NaN }, { x: 0.5, y: Infinity }]), { shape: lutShape })
  attempt('buildLut out-of-range', () => buildLut([{ x: -5, y: 9 }, { x: 7, y: -3 }]), { shape: lutShape })
  attempt('buildLut dup x', () => buildLut([{ x: 0.5, y: 0.1 }, { x: 0.5, y: 0.9 }]), { shape: lutShape })
  attempt('buildLut 200 random', () => buildLut(Array.from({ length: 200 }, () => ({ x: Math.random() * 3 - 1, y: Math.random() * 3 - 1 }))), { shape: lutShape })
  attempt('buildPackedLut garbage', () => buildPackedLutFromCurves({ r: 'x', g: 5, b: [{ x: NaN }], master: [{}] }), {
    shape: (o) => (o?.packed?.length === LUT_SIZE * 4 ? null : `packed len=${o?.packed?.length}`),
  })
  attempt('buildCurveSvgPath NaN viewBox', () => {
    const d = buildCurveSvgPath([{ x: 0, y: 0 }, { x: 1, y: 1 }], { left: NaN, right: 100, top: 0, bottom: 100 })
    return d.includes('NaN') ? { nanInPath: NaN } : d  // surface as non-finite
  })
}

/* ── pixel-stretch (pure geometry) ── */
{
  const garbage = { cx: NaN, cy: Infinity, angleDeg: 'up', length: -5, width: 1e9, bend: NaN, twistDeg: Infinity, taper: null, fade: {}, mirror: 'yes', mode: 42, flowPoints: [{ x: NaN, y: 2 }, 'p'], warpGrid: { rows: NaN, cols: 99, points: [{ x: NaN, y: NaN }] }, lassoPoly: [{ x: NaN, y: -3 }] }
  attempt('clampStretchParams garbage', () => clampStretchParams(garbage))
  attempt('clampStretchParams {}', () => clampStretchParams({}))
  const safe = clampStretchParams(garbage)
  attempt('getStretchAnchors(garbage-clamped)', () => getStretchAnchors(safe, 200, 100))
  for (const [w, hh] of [[0, 0], [NaN, 100], [200, -1]]) {
    attempt(`getStretchPath W=${w},H=${hh}`, () => getStretchPath(clampStretchParams({}), w, hh))
  }
  attempt('getPolygonBBox null', () => getPolygonBBox(null))
  attempt('getPolygonBBox []', () => getPolygonBBox([]))
  attempt('getPolygonBBox NaN pts', () => getPolygonBBox([{ x: NaN, y: NaN }, { x: 2, y: -1 }]))
  for (const rc of [0, 100, NaN, 2.5]) {
    attempt(`createDefaultWarpGrid rows=cols=${rc}`, () => createDefaultWarpGrid(DEFAULT_STRETCH, rc, rc), {
      // 2D array, each axis a valid net size (3·patches+1 within [4,13])
      shape: (g) => {
        const R = Array.isArray(g) ? g.length : 0
        const C = Array.isArray(g?.[0]) ? g[0].length : 0
        const okDim = (n) => n >= WARP_MIN_DIM && n <= WARP_MAX_DIM && (n - 1) % 3 === 0
        return okDim(R) && okDim(C) && g.every((row) => row.length === C) ? null : `R=${R} C=${C}`
      },
    })
  }
  attempt('getWarpRest garbage', () => getWarpRest(garbage))
  const corrupt = { rows: 3, cols: 3, points: [{ x: NaN, y: NaN }] } // short + NaN
  attempt('getWarpGridHandles corrupt grid', () => getWarpGridHandles({ ...DEFAULT_STRETCH, warpGrid: corrupt }, 200, 100))
  attempt('getWarpGridCurves corrupt grid', () => getWarpGridCurves({ ...DEFAULT_STRETCH, warpGrid: corrupt }, 200, 100))
  attempt('addWarpSplit at MAX', () => {
    let g = createDefaultWarpGrid(DEFAULT_STRETCH, WARP_MAX_DIM, WARP_MAX_DIM)
    return addWarpSplit(g, 'row')
  })
  // contract: invalid grid is returned unchanged (caller keeps state) — no scan
  attempt('addWarpSplit garbage grid', () => addWarpSplit({ rows: NaN, cols: 2, points: null }, 'col'), { scan: false })
  attempt('addWarpSplit bad axis', () => addWarpSplit(createDefaultWarpGrid(), 'diagonal'))
  attempt('applyWarpPreset unknown id', () => applyWarpPreset(clampStretchParams({}), 'nope', 1))
  for (const amt of [NaN, -5, 1e9]) {
    attempt(`applyWarpPreset amount=${amt}`, () => applyWarpPreset(clampStretchParams({}), 'bulge', amt))
  }
}

/* ── nl-mask-parser ── */
{
  const inputs = ['', null, undefined, 42, 'a'.repeat(100000), '🐕🔥 émöjí ünïcode', 'everything except everything',
    'the the the the the dog dog dog', String.fromCharCode(0, 1, 2, 3), 'mask the '.repeat(5000),
    'select the reddish-blue transparent invisible 47th elephant behind the front']
  for (const inp of inputs) {
    const tag = typeof inp === 'string' ? JSON.stringify(inp.slice(0, 30)) : String(inp)
    attempt(`parseMaskDescription ${tag}`, () => parseMaskDescription(inp))
  }
}

console.log(`\n${pass} passed, ${fail} failed`)
if (fails.length) { console.log('FAILURES:'); fails.forEach((f) => console.log(` - ${f}`)) }
process.exit(fail ? 1 : 0)
