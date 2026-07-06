/**
 * Magnetic-lasso edge-snapping engine (pure, DOM-free)
 * ----------------------------------------------------
 * The Photoshop-style magnetic lasso snaps a rough path to the nearest
 * high-contrast edge. Both the interactive tool (`mask.jsx`) and the headless
 * agent command (`agent/mask-commands.js`) need the SAME math, so it lives here
 * as two pure functions that take raw RGBA pixels — the callers own the canvas
 * extraction, scaling, and caching.
 *
 * Keeping the core pure means it is unit-testable in Node/bun without a browser
 * or a WebGL context (see `scripts/verify-mask-edge-snap.mjs`).
 *
 * @module mask-edge-snap
 */

/**
 * Compute a normalised (0..1) Sobel gradient-magnitude map from RGBA pixels.
 * Strong edges → ~1, flat regions → ~0. The 1px border is left at 0 (the Sobel
 * kernel needs a full 3×3 neighbourhood).
 *
 * @param {Uint8ClampedArray|Uint8Array|number[]} rgba  Row-major RGBA, length w*h*4.
 * @param {number} w
 * @param {number} h
 * @returns {{ mag: Float32Array, w: number, h: number }}
 */
export const computeGradientMagnitude = (rgba, w, h) => {
    const n = Math.max(0, w * h)
    const lum = new Float32Array(n)
    // `|| 0` guards a short rgba buffer (undefined reads → NaN in the map).
    for (let i = 0, p = 0; p < n; i += 4, p += 1) {
        lum[p] = 0.2126 * (rgba[i] || 0) + 0.7152 * (rgba[i + 1] || 0) + 0.0722 * (rgba[i + 2] || 0)
    }
    const mag = new Float32Array(n)
    let max = 1e-6
    for (let y = 1; y < h - 1; y += 1) {
        for (let x = 1; x < w - 1; x += 1) {
            const i = y * w + x
            const gx = (lum[i - w + 1] + 2 * lum[i + 1] + lum[i + w + 1])
                - (lum[i - w - 1] + 2 * lum[i - 1] + lum[i + w - 1])
            const gy = (lum[i + w - 1] + 2 * lum[i + w] + lum[i + w + 1])
                - (lum[i - w - 1] + 2 * lum[i - w] + lum[i - w + 1])
            const m = Math.sqrt(gx * gx + gy * gy)
            mag[i] = m
            if (m > max) max = m
        }
    }
    const inv = 1 / max
    for (let i = 0; i < n; i += 1) mag[i] *= inv
    return { mag, w, h }
}

/**
 * Snap a point to the strongest edge within `radius` px. The score is
 * **proximity-dominant** (`gradient × proximity²`) so the path hugs the NEAREST
 * edge instead of jumping to a far stronger one inside the search window — the
 * fix that keeps the magnetic lasso from skipping across the image. Coordinates
 * are in the gradient map's own pixel space (callers scale in/out).
 *
 * @param {{ mag: Float32Array, w: number, h: number } | null | undefined} map
 * @param {number} px
 * @param {number} py
 * @param {number} radius     Search radius in map px.
 * @param {number} contrast   Edge threshold 0..1 — gradients below it are ignored.
 * @returns {{ x: number, y: number }}  Snapped point (input unchanged if no edge clears the threshold).
 */
export const snapToEdgePoint = (map, px, py, radius, contrast) => {
    if (!map || !map.mag) return { x: px, y: py }
    const { mag, w, h } = map
    const r = Math.max(1, radius)
    const x0 = Math.max(0, Math.floor(px - r))
    const x1 = Math.min(w - 1, Math.ceil(px + r))
    const y0 = Math.max(0, Math.floor(py - r))
    const y1 = Math.min(h - 1, Math.ceil(py + r))
    const thr = Math.max(0, Math.min(1, contrast))
    const r2 = r * r
    let best = 0
    let bx = px
    let by = py
    for (let yy = y0; yy <= y1; yy += 1) {
        for (let xx = x0; xx <= x1; xx += 1) {
            const dx = xx - px
            const dy = yy - py
            const d2 = dx * dx + dy * dy
            if (d2 > r2) continue
            const g = mag[yy * w + xx]
            if (g < thr) continue
            const prox = 1 - Math.sqrt(d2) / r
            const score = g * prox * prox
            if (score > best) {
                best = score
                bx = xx
                by = yy
            }
        }
    }
    return best > 0 ? { x: bx, y: by } : { x: px, y: py }
}
