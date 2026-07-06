/**
 * mask-grow-core (pure — no DOM, no megashader imports)
 * ------------------------------------------------------
 * The morphology math behind mask boundary extension, kept dependency-free so
 * the verify scripts can pin it. The canvas/layer wrappers live in
 * mask-grow.js.
 *
 * Morphology is approximated with a single-pass separable box blur followed
 * by a smoothstep re-threshold: blurring a step edge by radius R produces a
 * linear ramp over [-R, +R], so cutting that ramp at the value it takes `px`
 * beyond the original edge dilates (or erodes, for negative `px`) by almost
 * exactly `px` pixels — in O(n) and with a soft anti-aliased edge.
 */

export const MAX_GROW_PX = 200

/** Single-pass separable box blur (radius r) on a single channel. O(n). */
export const boxBlur = (src, w, h, r) => {
    // Non-integer r breaks the running-sum indexing; NaN passes `r < 1`.
    r = Number.isFinite(r) ? Math.floor(r) : 0
    if (r < 1) return Float32Array.from(src)
    const tmp = new Float32Array(w * h)
    const out = new Float32Array(w * h)
    const win = 2 * r + 1

    // Horizontal pass via running sum (clamped sampling at the borders).
    for (let y = 0; y < h; y += 1) {
        const row = y * w
        let sum = 0
        for (let x = -r; x <= r; x += 1) sum += src[row + Math.min(w - 1, Math.max(0, x))]
        for (let x = 0; x < w; x += 1) {
            tmp[row + x] = sum / win
            const add = Math.min(w - 1, x + r + 1)
            const sub = Math.max(0, x - r)
            sum += src[row + add] - src[row + sub]
        }
    }
    // Vertical pass.
    for (let x = 0; x < w; x += 1) {
        let sum = 0
        for (let y = -r; y <= r; y += 1) sum += tmp[Math.min(h - 1, Math.max(0, y)) * w + x]
        for (let y = 0; y < h; y += 1) {
            out[y * w + x] = sum / win
            const add = Math.min(h - 1, y + r + 1)
            const sub = Math.max(0, y - r)
            sum += tmp[add * w + x] - tmp[sub * w + x]
        }
    }
    return out
}

const smoothstep = (lo, hi, v) => {
    const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)))
    return t * t * (3 - 2 * t)
}

/**
 * Grow (px > 0) or shrink (px < 0) a coverage channel by ~|px| pixels.
 *
 * @param {Uint8ClampedArray|Uint8Array|Float32Array} cover  0..255 per pixel
 * @param {number} w
 * @param {number} h
 * @param {number} px  signed distance in pixels (clamped to ±MAX_GROW_PX)
 * @returns {Uint8ClampedArray} new 0..255 coverage
 */
export const growCoverage = (cover, w, h, px) => {
    const dist = Math.max(-MAX_GROW_PX, Math.min(MAX_GROW_PX, Math.round(px || 0)))
    if (dist === 0 || w < 1 || h < 1) return Uint8ClampedArray.from(cover)

    // Blur radius slightly past the requested distance so the target
    // threshold sits inside the ramp, not at its degenerate endpoint.
    const r = Math.max(1, Math.ceil(Math.abs(dist) * 1.25))
    const src = new Float32Array(w * h)
    for (let i = 0; i < src.length; i += 1) src[i] = cover[i] / 255

    const blurred = boxBlur(src, w, h, r)

    // A step edge blurred by radius r ramps linearly: value at signed
    // distance d OUTSIDE the edge ≈ (r - d) / (2r). Cut at d = ±dist with a
    // small smoothstep band for an anti-aliased boundary.
    const target = dist > 0 ? (r - dist) / (2 * r) : (r + Math.abs(dist)) / (2 * r)
    const band = Math.max(0.02, 0.5 / r)
    const out = new Uint8ClampedArray(w * h)
    for (let i = 0; i < out.length; i += 1) {
        out[i] = Math.round(smoothstep(target - band, target + band, blurred[i]) * 255)
    }
    return out
}
