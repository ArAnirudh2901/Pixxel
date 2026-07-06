/**
 * client-ai-core (pure — no DOM, no transformers.js)
 * ----------------------------------------------------
 * The validation and self-test-evaluation logic behind the in-browser AI
 * engine (client-ai.js), kept dependency-free so the bun verify suite can
 * pin it even though the models themselves only run in a browser.
 *
 * The split matters for production robustness: model OUTPUTS are validated
 * with the functions here before they're allowed to become masks, so a
 * broken backend (corrupt WASM build, WebGPU driver bug, truncated model
 * download) produces a loud, routable failure instead of a silently empty
 * or garbage selection.
 */

/** Reject a hung promise after `ms` — a stuck WebGPU context or stalled
 *  model download must fail the attempt (so routing can fall back), not
 *  freeze the mask step forever. */
export const withTimeout = (promise, ms, label = 'operation') =>
    new Promise((resolve, reject) => {
        const timer = setTimeout(
            () => reject(new Error(`${label} timed out after ${Math.round(ms / 1000)}s`)),
            ms,
        )
        promise.then(
            (v) => { clearTimeout(timer); resolve(v) },
            (e) => { clearTimeout(timer); reject(e) },
        )
    })

/**
 * Statistics over a 0..1 relevance/coverage map: peak, coverage fraction at
 * `threshold`, and the bbox of above-threshold pixels. Also flags non-finite
 * values — the canonical symptom of a broken inference backend.
 *
 * @param {Float32Array|number[]} map
 * @param {number} w
 * @param {number} h
 * @param {number} [threshold=0.5]
 * @returns {{ peak:number, coverage:number, bbox:[number,number,number,number]|null, finite:boolean }}
 */
export const analyzeCoverage = (map, w, h, threshold = 0.5) => {
    let peak = 0
    let count = 0
    let finite = true
    let x0 = w; let y0 = h; let x1 = -1; let y1 = -1
    for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
            const v = map[y * w + x]
            if (!Number.isFinite(v)) { finite = false; continue }
            if (v > peak) peak = v
            if (v >= threshold) {
                count += 1
                if (x < x0) x0 = x
                if (x > x1) x1 = x
                if (y < y0) y0 = y
                if (y > y1) y1 = y
            }
        }
    }
    return {
        peak,
        coverage: w * h > 0 ? count / (w * h) : 0,
        bbox: x1 >= x0 && y1 >= y0 ? [x0, y0, x1 - x0 + 1, y1 - y0 + 1] : null,
        finite,
    }
}

/**
 * Validate a grounding output before it becomes a mask. `found` results must
 * have real, finite, non-empty coverage; anything else is downgraded to a
 * miss (canvas dropped) or rejected as a backend failure.
 *
 * @param {{ peak:number, coverage:number, finite:boolean }} stats
 * @param {{ minPeak?: number, minCoverage?: number, maxCoverage?: number }} [opts]
 * @returns {{ usable:boolean, reason:string|null }}
 */
export const validateGroundOutput = (stats, { minPeak = 0.25, minCoverage = 0.0005, maxCoverage = 0.98 } = {}) => {
    if (!stats.finite) return { usable: false, reason: 'non-finite values in model output (broken backend)' }
    if (stats.peak < minPeak) return { usable: false, reason: `peak ${stats.peak.toFixed(3)} below ${minPeak}` }
    if (stats.coverage < minCoverage) return { usable: false, reason: 'mask is effectively empty' }
    if (stats.coverage > maxCoverage) return { usable: false, reason: 'mask covers the whole frame (degenerate output)' }
    return { usable: true, reason: null }
}

/**
 * Validate a depth-map output: correct dims and real tonal spread. A flat
 * map means the backend produced a constant — useless for depth planes.
 *
 * @param {{ width:number, height:number, min:number, max:number, finite:boolean }} stats
 * @param {{ width:number, height:number }} expected
 * @returns {{ usable:boolean, reason:string|null }}
 */
export const validateDepthOutput = (stats, expected) => {
    if (!stats.finite) return { usable: false, reason: 'non-finite values in depth output' }
    if (stats.width !== expected.width || stats.height !== expected.height) {
        return { usable: false, reason: `depth dims ${stats.width}x${stats.height} ≠ expected ${expected.width}x${expected.height}` }
    }
    if (stats.max - stats.min < 0.05) return { usable: false, reason: 'depth map is flat (constant output)' }
    return { usable: true, reason: null }
}

/** Min/max/finite over a 0..1 map — input to validateDepthOutput. */
export const analyzeRange = (map) => {
    let min = Infinity
    let max = -Infinity
    let finite = true
    for (let i = 0; i < map.length; i += 1) {
        const v = map[i]
        if (!Number.isFinite(v)) { finite = false; continue }
        if (v < min) min = v
        if (v > max) max = v
    }
    if (min === Infinity) { min = 0; max = 0 }
    return { min, max, finite }
}

/**
 * Evaluate a self-test run against the known synthetic scene (a red disc at
 * `disc` on a flat grey field). Pure: takes measured results, returns the
 * check list the UI / harness renders.
 *
 * @param {object} results
 * @param {{ found:boolean, score:number, bbox:[number,number,number,number]|null }} results.ground
 * @param {{ width:number, height:number, spread:number }|null} results.depth
 * @param {{ width:number, height:number, coverage:number, bbox:number[]|null }|null} [results.segment]
 *        Background-removal matte stats. OPTIONAL: omit the key entirely to
 *        skip the segmentation checks; null means it ran and failed.
 * @param {{ width:number, height:number, coverage:number, bbox:number[]|null }|null} [results.sam]
 *        Click-select mask stats (same optional/null contract as segment).
 * @param {{ cx:number, cy:number, w:number, h:number }} disc  expected target
 * @returns {{ ok:boolean, checks:Array<{label:string, ok:boolean, detail:string}> }}
 */
export const evaluateSelfTest = ({ ground, depth, segment, sam }, disc) => {
    const checks = []

    const gFound = Boolean(ground?.found)
    checks.push({
        label: 'Grounding binds "the red circle"',
        ok: gFound,
        detail: `score ${ground?.score ?? 'n/a'}`,
    })

    let centered = false
    if (gFound && Array.isArray(ground.bbox)) {
        const [x, y, w, h] = ground.bbox
        centered = disc.cx >= x && disc.cx <= x + w && disc.cy >= y && disc.cy <= y + h
    }
    checks.push({
        label: 'Mask covers the disc center',
        ok: centered,
        detail: ground?.bbox ? `bbox ${ground.bbox.join(',')}` : 'no bbox',
    })

    const dOk = Boolean(depth) && depth.width === disc.w && depth.height === disc.h
    checks.push({
        label: 'Depth map at expected size',
        ok: dOk,
        detail: depth ? `${depth.width}x${depth.height}` : 'no output',
    })
    checks.push({
        label: 'Depth map has tonal spread',
        ok: Boolean(depth) && depth.spread >= 0.05,
        detail: depth ? `spread ${depth.spread.toFixed(3)}` : 'no output',
    })

    if (segment !== undefined) {
        // The disc is the scene's only subject: a sane matte selects a real
        // but partial region, and its bbox contains the disc center.
        const coverageOk = Boolean(segment)
            && segment.coverage > 0.01 && segment.coverage < 0.9
        checks.push({
            label: 'Background removal isolates a subject',
            ok: coverageOk,
            detail: segment ? `coverage ${segment.coverage.toFixed(3)}` : 'no output',
        })
        let segCentered = false
        if (segment && Array.isArray(segment.bbox)) {
            const [x, y, w, h] = segment.bbox
            segCentered = disc.cx >= x && disc.cx <= x + w && disc.cy >= y && disc.cy <= y + h
        }
        checks.push({
            label: 'Subject matte covers the disc center',
            ok: segCentered,
            detail: segment?.bbox ? `bbox ${segment.bbox.join(',')}` : 'no bbox',
        })
    }

    if (sam !== undefined) {
        // A click on the disc center must select a real partial region…
        const coverageOk = Boolean(sam) && sam.coverage > 0.01 && sam.coverage < 0.9
        checks.push({
            label: 'Click-select isolates a region',
            ok: coverageOk,
            detail: sam ? `coverage ${sam.coverage.toFixed(3)}` : 'no output',
        })
        // …whose bbox contains the clicked point.
        let samCentered = false
        if (sam && Array.isArray(sam.bbox)) {
            const [x, y, w, h] = sam.bbox
            samCentered = disc.cx >= x && disc.cx <= x + w && disc.cy >= y && disc.cy <= y + h
        }
        checks.push({
            label: 'Click-select mask covers the click point',
            ok: samCentered,
            detail: sam?.bbox ? `bbox ${sam.bbox.join(',')}` : 'no bbox',
        })
    }

    return { ok: checks.every((c) => c.ok), checks }
}
