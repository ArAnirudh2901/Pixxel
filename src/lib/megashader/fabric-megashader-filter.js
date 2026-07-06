/**
 * Fabric Megashader Filter
 * ------------------------
 * A `fabric.Image.filters.BaseFilter` subclass that wraps the private-WebGL
 * megashader renderer (`megashader-renderer.js`). It implements `applyTo2d`
 * to integrate with Fabric v7's Canvas2D filter pipeline (Fabric's WebGL
 * filtering is disabled globally — see `canvas.jsx`). The filter receives
 * a `pipelineState` object from Fabric, renders the megashader via its own
 * private WebGL2 context, and writes the result back into `pipelineState`
 * so downstream filters (curves, adjustment, etc.) see the megashader's
 * output.
 * Lifecycle:
 *   1. Caller (the canvas component) creates a `new MegashaderFilter({ ... })`
 *      and pushes it into `image.filters`. The constructor stores the layer
 *      stack but does NOT compile a shader yet.
 *   2. Fabric calls `filter.applyTo(pipelineState)` for each filter in the
 *      chain. Our `applyTo2d` renders the megashader via `renderMegashader`
 *      and writes the result back into `pipelineState.imageData`.
 *   3. `toObject` produces a Fabric-friendly plain object for persistence
 *      via `loadFromJSON`. `fromObject` is the inverse.
 *
 * Why the filter does not subclass the existing `BaseFilter` directly:
 *   The codebase already has a custom `PhosmithCurvesFilter` extending
 *   `fabric.Image.filters.BaseFilter`. We mirror that pattern but keep this
 *   filter in its own file so the curves work and the megashader work can
 *   evolve independently.
 *
 * @module megashader/fabric-megashader-filter
 */

import { filters, classRegistry } from 'fabric'
import { renderMegashader, disposeRenderer } from './megashader-renderer'
import { getMaskTexture, setMaskTexture } from './mask-types'
import { buildPackedLutFromCurves } from '../curve-lut'

const MEGASHADER_FILTER_TYPE = 'Megashader'

// Field names that carry an opaque mask-texture-cache key, by kind. Used to
// serialise/restore the per-layer textures (semantic / depth / smartBrush /
// lasso) so a persisted chain survives save → reload.
const TEXTURE_KEY_FIELDS = ['maskTextureKey', 'brushTextureKey', 'depthMapKey']

// Longest-side cap for PERSISTED mask textures. Masks are soft alpha maps
// sampled at normalized UV with LINEAR filtering (see megashader-renderer /
// glsl-mask-kinds), so a downscaled texture restores resolution-independently
// — it just upsamples bilinearly at render time. Capping keeps a multi-layer
// chain's PNG data URLs well under the Neon canvasState budget
// (MAX_NEON_STATE_CHARS in canvas.jsx, whose oversize trim only sheds
// history). In-session rendering keeps the full-resolution texture — this cap
// applies at serialize time only.
const MAX_PERSIST_TEXTURE_DIM = 1024

/**
 * Convert a cached texture (ImageData | HTMLCanvasElement | HTMLImageElement |
 * ImageBitmap) to a PNG data URL for persistence, downscaled so its longest
 * side is at most MAX_PERSIST_TEXTURE_DIM. Returns null if it can't be
 * rasterised (non-browser, tainted canvas, zero-size).
 */
const textureToDataUrl = (data) => {
    try {
        if (!data || typeof document === 'undefined') return null
        // ImageData isn't drawable — blit it onto a scratch canvas first so
        // the single scale-draw below covers every source type.
        let source = data
        if (typeof ImageData !== 'undefined' && data instanceof ImageData) {
            const scratch = document.createElement('canvas')
            scratch.width = data.width
            scratch.height = data.height
            scratch.getContext('2d')?.putImageData(data, 0, 0)
            source = scratch
        }
        const w = source.width || source.naturalWidth || 0
        const h = source.height || source.naturalHeight || 0
        if (!w || !h) return null
        const scale = Math.min(1, MAX_PERSIST_TEXTURE_DIM / Math.max(w, h))
        if (
            scale === 1 &&
            typeof HTMLCanvasElement !== 'undefined' &&
            source instanceof HTMLCanvasElement
        ) {
            return source.toDataURL('image/png')
        }
        const canvas = document.createElement('canvas')
        canvas.width = Math.max(1, Math.round(w * scale))
        canvas.height = Math.max(1, Math.round(h * scale))
        const ctx = canvas.getContext('2d')
        if (!ctx) return null
        ctx.imageSmoothingEnabled = true
        ctx.imageSmoothingQuality = 'high'
        ctx.drawImage(source, 0, 0, canvas.width, canvas.height)
        return canvas.toDataURL('image/png')
    } catch {
        return null
    }
}

/**
 * Decode a PNG data URL back into the texture cache under `key`. Resolves
 * once the texture is registered (or on error) so callers can await it BEFORE
 * the filter renders (otherwise the layer would sample the null texture).
 */
const restoreTexture = (key, dataUrl) => new Promise((resolve) => {
    try {
        if (typeof document === 'undefined' || !dataUrl) { resolve(); return }
        const img = new Image()
        img.onload = () => {
            try {
                const c = document.createElement('canvas')
                c.width = img.naturalWidth || img.width
                c.height = img.naturalHeight || img.height
                c.getContext('2d')?.drawImage(img, 0, 0)
                setMaskTexture(key, c)
            } catch { /* ignore */ }
            resolve()
        }
        img.onerror = () => resolve()
        img.src = dataUrl
    } catch {
        resolve()
    }
})

/**
 * @typedef {Object} MegashaderFilterOptions
 * @property {import('./mask-types').MaskStack} [stack]    The current mask stack.
 * @property {number} [globalMaskAlpha]                    0..1 — extra UI fader.
 * @property {boolean} [enabled]                           When false, `applyTo`
 *                                                         is a no-op passthrough
 *                                                         (used during preview
 *                                                         and tool changes).
 */

export class MegashaderFilter extends filters.BaseFilter {
    // Fabric v7's BaseFilter exposes `type` as a GETTER-ONLY accessor that
    // reads `this.constructor.type` — assigning `this.type = ...` in the
    // constructor throws "Cannot set property type ... which has only a
    // getter", which killed every construction of this filter (and, caught
    // by a silent catch upstream, made the whole mask pipeline a no-op).
    // The static field is the v7-correct way to name the class.
    static type = MEGASHADER_FILTER_TYPE

    /**
     * @param {MegashaderFilterOptions} [options]
     */
    constructor(options = {}) {
        super()
        // No fragmentSource — the renderer compiles the real source via its
        // own private WebGL2 context. This filter implements applyTo2d() to
        // integrate with Fabric v7's Canvas2D filter pipeline.
        this.stack = options.stack || { chain: [] }
        this.globalMaskAlpha = typeof options.globalMaskAlpha === 'number' ? options.globalMaskAlpha : 1
        this.enabled = options.enabled !== false
        // Chain-wide render options. globalInvert is persisted (a real mask
        // property); maskOverlay/overlayColor are transient view state.
        this.globalInvert = options.globalInvert === true
        this.maskOverlay = options.maskOverlay === true
        this.overlayColor = options.overlayColor || null
    }

    /**
     * Fabric v7 calls `applyTo(pipelineState)` for each filter in the
     * chain. The base class dispatches to `applyTo2d(pipelineState)` when
     * WebGL filtering is disabled (our case — see `canvas.jsx`).
     *
     * We override `applyTo` with a defensive wrapper so that a megashader
     * failure never kills the entire filter chain (which would leave the
     * canvas blank — the original bug).
     *
     * @param {object} options  Fabric's pipelineState object.
     */
    applyTo(options) {
        try {
            this.applyTo2d(options)
        } catch (e) {
            console.warn('[megashader] applyTo failed, passing through:', e)
            // Don't touch pipelineState — the downstream filters and
            // Fabric's putImageData will use whatever imageData is
            // already there (the pre-megashader state), which is the
            // correct passthrough behavior.
        }
    }

    /**
     * 2D filter implementation. Receives Fabric's pipelineState:
     *   { imageData, ctx, canvasEl, sourceWidth, sourceHeight, originalEl, ... }
     *
     * We render the megashader from the current canvas state (canvasEl,
     * which holds the accumulated output of all prior filters in the
     * chain) and write the result back into `pipelineState.imageData`
     * so downstream filters see the megashader's output.
     *
     * @param {object} options  Fabric's pipelineState object.
     */
    applyTo2d(options) {
        if (!this.enabled) return

        const { canvasEl, ctx, sourceWidth, sourceHeight } = options || {}
        if (!canvasEl || !ctx) return

        // The canvasEl holds the current filtered pixels (accumulated from
        // all prior filters in the chain). We need to flush the current
        // imageData to it first, because prior filters may have modified
        // imageData in-place without drawing it back to the canvas.
        if (options.imageData) {
            ctx.putImageData(options.imageData, 0, 0)
        }

        let result
        try {
            result = renderMegashader(canvasEl, this.stack, {
                globalMaskAlpha: this.globalMaskAlpha,
                globalInvert: this.globalInvert,
                maskOverlay: this.maskOverlay,
                overlayColor: this.overlayColor,
            })
        } catch (e) {
            console.warn('[megashader] render failed, passing through source:', e)
            return  // Leave pipelineState unchanged — passthrough.
        }

        if (!result) return  // Passthrough — renderMegashader returned null.

        // Draw the megashader result back onto the pipeline canvas and
        // re-read imageData so downstream filters see the megashader's
        // output.
        const w = sourceWidth || canvasEl.width
        const h = sourceHeight || canvasEl.height
        ctx.clearRect(0, 0, w, h)
        ctx.drawImage(result, 0, 0)
        options.imageData = ctx.getImageData(0, 0, w, h)
    }

    /**
     * Update the mask stack + global alpha. Triggers a redraw of the owning
     * image via the caller (we don't have a Fabric `image` ref here).
     *
     * @param {MegashaderFilterOptions} next
     */
    update(next = {}) {
        if (next.stack) this.stack = next.stack
        if (typeof next.globalMaskAlpha === 'number') this.globalMaskAlpha = next.globalMaskAlpha
        if (typeof next.enabled === 'boolean') this.enabled = next.enabled
    }

    /**
     * Fabric serialises filters via `toObject`/`toJSON`. Persist the full
     * stack (incl. per-layer grade params) plus every texture-backed
     * layer's mask bitmap as a size-capped PNG data URL, so a persisted
     * chain survives save → reload. Tone-curve LUTs are the one exception:
     * they are rebuilt from `layer.curves` in `fromObject` instead (the
     * packed LUT keeps the master curve in the ALPHA channel, and a canvas
     * PNG round-trip premultiplies RGB by alpha, corrupting it).
     *
     * @returns {object}
     */
    toObject() {
        // Serialise every texture-backed layer's texture (semantic / depth /
        // smartBrush / lasso) as a PNG data URL keyed by its cache key, so a
        // reloaded project re-registers them and the selection survives.
        const textures = {}
        const chain = (this.stack && Array.isArray(this.stack.chain)) ? this.stack.chain : []
        for (const entry of chain) {
            const layer = entry && entry.layer
            if (!layer) continue
            for (const field of TEXTURE_KEY_FIELDS) {
                const key = layer[field]
                if (typeof key === 'string' && key && !textures[key]) {
                    const url = textureToDataUrl(getMaskTexture(key))
                    if (url) textures[key] = url
                }
            }
        }
        const out = {
            ...super.toObject(),
            type: MEGASHADER_FILTER_TYPE,
            stack: this.stack,
            globalMaskAlpha: this.globalMaskAlpha,
            globalInvert: this.globalInvert,
            enabled: this.enabled,
        }
        if (Object.keys(textures).length > 0) out.textures = textures
        return out
    }

    /**
     * Inverse of `toObject`. Fabric's `loadFromJSON` calls this on the
     * registered class. We restore the persisted textures into the cache
     * BEFORE resolving the filter, so the first render samples real masks
     * instead of the null texture.
     *
     * @param {object} [object]
     * @param {{ target: any }} [options]
     */
    static async fromObject(object, options = {}) {
        const textures = object && object.textures
        if (textures && typeof textures === 'object') {
            await Promise.all(
                Object.entries(textures).map(([key, url]) => restoreTexture(key, url)),
            )
        }
        // Rebuild per-layer tone-curve LUTs from their control points
        // (mirrors `applyCurve` in mask.jsx). Without this the persisted
        // `curveLutKey` points at a missing cache entry after reload, the
        // shader's `curveOn` stays 0, and the curve grade silently
        // disappears. Rebuilding (not PNG-round-tripping) is deliberate —
        // see the `toObject` docblock. Runs before the filter resolves so
        // the first render samples a real LUT, even if the Mask panel is
        // never opened.
        const chain = Array.isArray(object?.stack?.chain) ? object.stack.chain : []
        for (const entry of chain) {
            const layer = entry && entry.layer
            if (!layer || !layer.curveLutKey || !layer.curves) continue
            try {
                const { packed, identity } = buildPackedLutFromCurves(layer.curves)
                if (!identity) {
                    setMaskTexture(
                        layer.curveLutKey,
                        new ImageData(new Uint8ClampedArray(packed), 256, 1),
                    )
                }
            } catch { /* curve stays identity for this layer — curveOn 0 */ }
        }
        const filter = new MegashaderFilter({
            stack: object?.stack,
            globalMaskAlpha: object?.globalMaskAlpha,
            globalInvert: object?.globalInvert,
            enabled: object?.enabled,
        })
        const target = options?.target
        if (target) filter.target = target
        return filter
    }
}

// Register with Fabric's classRegistry so `loadFromJSON` can rehydrate
// `type: "Megashader"` filters. Importing this file is the side effect
// that wires it up — the same pattern the codebase uses for the curves
// filter (see `import "../../../../../lib/curves-filter"` in canvas.jsx).
classRegistry.setClass(MegashaderFilter, MEGASHADER_FILTER_TYPE)

/**
 * Tear down the renderer's GL cache. Call this from a React cleanup
 * effect if the editor is fully unmounting.
 */
export const disposeMegashader = () => disposeRenderer()
