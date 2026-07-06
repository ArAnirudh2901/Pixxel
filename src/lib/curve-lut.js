/**
 * Tone-curve LUT math — PURE, no Fabric, no DOM.
 *
 * Mirrors the LUT math in `curves-filter.js` (the production Fabric filter) so
 * the megashader masking engine and the standalone Mask Studio can build the
 * same curve LUTs WITHOUT pulling Fabric into those bundles. The algorithm
 * (Fritsch–Carlson monotonic spline + RGBA packing) is identical and must stay
 * in sync; a later refactor can make `curves-filter.js` import from here so
 * there is a single source of truth. Keep changes mirrored until then.
 */

export const LUT_SIZE = 256

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v)

export const identityLut = () => {
    const lut = new Uint8ClampedArray(LUT_SIZE)
    for (let i = 0; i < LUT_SIZE; i++) lut[i] = i
    return lut
}

export const isIdentityLut = (lut) => {
    if (!lut || lut.length !== LUT_SIZE) return true
    for (let i = 0; i < LUT_SIZE; i++) if (lut[i] !== i) return false
    return true
}

// Sort points by x, ensure first and last are clamped to the edges
const normalizePoints = (points) => {
    const arr = (Array.isArray(points) ? points : [])
        .filter((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y))
        .map((p) => ({ x: clamp01(p.x), y: clamp01(p.y) }))
        .sort((a, b) => a.x - b.x)
    if (arr.length === 0) return [{ x: 0, y: 0 }, { x: 1, y: 1 }]
    if (arr[0].x > 0) arr.unshift({ x: 0, y: arr[0].y })
    if (arr[arr.length - 1].x < 1) arr.push({ x: 1, y: arr[arr.length - 1].y })
    return arr
}

// Monotonic cubic (Fritsch-Carlson) interpolation — won't overshoot, ideal for LUTs.
const monotonicSlopes = (pts) => {
    const n = pts.length
    const dx = new Array(n - 1)
    const dy = new Array(n - 1)
    const slope = new Array(n - 1)
    for (let i = 0; i < n - 1; i++) {
        dx[i] = pts[i + 1].x - pts[i].x
        dy[i] = pts[i + 1].y - pts[i].y
        slope[i] = dx[i] === 0 ? 0 : dy[i] / dx[i]
    }
    const m = new Array(n)
    m[0] = slope[0] || 0
    m[n - 1] = slope[n - 2] || 0
    for (let i = 1; i < n - 1; i++) {
        if (slope[i - 1] * slope[i] <= 0) {
            m[i] = 0
        } else {
            m[i] = (slope[i - 1] + slope[i]) / 2
        }
    }
    for (let i = 0; i < n - 1; i++) {
        if (slope[i] === 0) {
            m[i] = 0
            m[i + 1] = 0
        } else {
            const a = m[i] / slope[i]
            const b = m[i + 1] / slope[i]
            const s = a * a + b * b
            if (s > 9) {
                const t = 3 / Math.sqrt(s)
                m[i] = t * a * slope[i]
                m[i + 1] = t * b * slope[i]
            }
        }
    }
    return m
}

const evalSegment = (p0, p1, m0, m1, x) => {
    const h = p1.x - p0.x
    if (h === 0) return p0.y
    const t = (x - p0.x) / h
    const t2 = t * t
    const t3 = t2 * t
    const h00 = 2 * t3 - 3 * t2 + 1
    const h10 = t3 - 2 * t2 + t
    const h01 = -2 * t3 + 3 * t2
    const h11 = t3 - t2
    return h00 * p0.y + h10 * h * m0 + h01 * p1.y + h11 * h * m1
}

export const buildLut = (points) => {
    const pts = normalizePoints(points)
    const isIdentity = pts.length === 2 && pts[0].x === 0 && pts[0].y === 0 && pts[1].x === 1 && pts[1].y === 1
    if (isIdentity) return identityLut()
    const slopes = monotonicSlopes(pts)
    const lut = new Uint8ClampedArray(LUT_SIZE)
    let seg = 0
    for (let i = 0; i < LUT_SIZE; i++) {
        const x = i / (LUT_SIZE - 1)
        while (seg < pts.length - 2 && x > pts[seg + 1].x) seg++
        const y = evalSegment(pts[seg], pts[seg + 1], slopes[seg], slopes[seg + 1], x)
        lut[i] = Math.round(clamp01(y) * 255)
    }
    return lut
}

// SVG path data through N points for visual rendering — uses the same Hermite spline.
export const buildCurveSvgPath = (points, viewBox) => {
    const pts = normalizePoints(points)
    const { left, right, top, bottom } = viewBox
    // A non-finite viewBox would emit "NaN" into the path data.
    if (![left, right, top, bottom].every(Number.isFinite)) return ''
    const w = right - left
    const h = bottom - top
    const X = (x) => left + x * w
    const Y = (y) => bottom - y * h
    const slopes = monotonicSlopes(pts)
    let d = `M ${X(pts[0].x).toFixed(2)} ${Y(pts[0].y).toFixed(2)}`
    for (let i = 0; i < pts.length - 1; i++) {
        const p0 = pts[i]
        const p1 = pts[i + 1]
        const dx = (p1.x - p0.x) / 3
        const c1x = p0.x + dx
        const c1y = p0.y + slopes[i] * dx
        const c2x = p1.x - dx
        const c2y = p1.y - slopes[i + 1] * dx
        d += ` C ${X(c1x).toFixed(2)} ${Y(c1y).toFixed(2)}, ${X(c2x).toFixed(2)} ${Y(c2y).toFixed(2)}, ${X(p1.x).toFixed(2)} ${Y(p1.y).toFixed(2)}`
    }
    return d
}

export const DEFAULT_CURVE_POINTS = [
    { x: 0, y: 0 },
    { x: 1, y: 1 },
]

export const arePointsIdentity = (points) => {
    if (!Array.isArray(points) || points.length !== 2) return false
    const [a, b] = points
    return a?.x === 0 && a?.y === 0 && b?.x === 1 && b?.y === 1
}

// Pack the 4 LUTs into a single 256×1 RGBA buffer (R=lutR, G=lutG, B=lutB, A=lutMaster).
// Uploaded as a WebGL texture so a fragment shader can do four LUT lookups per pixel from a
// single sampler — used by PhosmithCurvesFilter AND the megashader per-layer tone curves.
export const packLutsRgba = (lutR, lutG, lutB, lutM) => {
    const packed = new Uint8Array(LUT_SIZE * 4)
    for (let i = 0; i < LUT_SIZE; i++) {
        packed[i * 4 + 0] = lutR[i]
        packed[i * 4 + 1] = lutG[i]
        packed[i * 4 + 2] = lutB[i]
        packed[i * 4 + 3] = lutM[i]
    }
    return packed
}

/**
 * Convenience for non-Fabric consumers (megashader, Mask Studio): build the
 * packed 256×1 RGBA LUT directly from curve POINTS for each channel. Pass the
 * per-channel point arrays ({x,y} in 0..1); omitted channels are identity.
 * Returns a Uint8Array(256*4) ready for texImage2D, plus an `identity` flag.
 *
 * @param {{ master?: Array, r?: Array, g?: Array, b?: Array }} curves
 * @returns {{ packed: Uint8Array, identity: boolean }}
 */
export const buildPackedLutFromCurves = (curves = {}) => {
    const lutR = buildLut(curves.r)
    const lutG = buildLut(curves.g)
    const lutB = buildLut(curves.b)
    const lutM = buildLut(curves.master)
    const identity = isIdentityLut(lutR) && isIdentityLut(lutG) && isIdentityLut(lutB) && isIdentityLut(lutM)
    return { packed: packLutsRgba(lutR, lutG, lutB, lutM), identity }
}
