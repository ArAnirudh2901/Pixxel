/**
 * Megashader Mask Types
 * ----------------------
 * JSDoc @typedef definitions for the Advanced Image Masking System.
 *
 * This file is the single source of truth for the shape of every mask layer
 * the megashader can composite. New mask kinds (Step 2–6) will add their own
 * variant to the `MaskLayer` union and their own GLSL function builder in
 * `glsl-fragments.js`. The compiler + renderer are driven purely by these
 * types, so no other file needs to be touched when a new mask kind is added.
 *
 * Why JSDoc and not TypeScript:
 * The phosmith codebase is pure JavaScript (see AGENTS.md / jsconfig.json).
 * JSDoc gives editor-time IntelliSense in any TS-aware IDE without forcing
 * a build-tooling migration. If the project later moves to TS, these typedefs
 * are the only place that needs renaming.
 *
 * @module megashader/mask-types
 */

/**
 * Boolean operations for compositing mask layers together. The first layer in
 * a chain defines the base; each subsequent layer is combined with the running
 * alpha using one of these operations.
 *
 * The original four (`replace` / `add` / `subtract` / `intersect`) cover the
 * basic boolean set. The four added in Sub-task 1 bring Photoshop parity for
 * the most common 2D blending modes. All eight are computed on the running
 * alpha only — colour is always blended proportionally to the new layer's
 * alpha via `mix(runningColor, c_i, a_i)`.
 *
 * - `replace`:    first layer only — overwrites any prior alpha. The compiler
 *                 enforces this only at slot 0; a stray `replace` later in
 *                 the chain is treated as a wholesale overwrite so a misuse
 *                 is visible (see `buildBooleanChain`).
 * - `add`:        union — `clamp(a + b, 0, 1)`. Photoshop "linear dodge".
 * - `subtract`:   difference — `max(a - b, 0)`. Photoshop "linear burn" on
 *                 the dark side.
 * - `intersect`:  multiply — `a * b`. Photoshop "multiply". Identical
 *                 formula to `intersect`; kept distinct so the layer panel
 *                 can read in either Photoshop-language or boolean-language.
 * - `screen`:     1 - (1 - a) * (1 - b). Photoshop "screen". Always lifts
 *                 the running alpha — never darker than either input.
 * - `lighten`:    `max(a, b)`. Photoshop "lighten".
 * - `darken`:     `min(a, b)`. Photoshop "darken".
 * - `overlay`:    Photoshop "overlay" — multiply on dark halves, screen on
 *                 light halves, with the running alpha as the base. Cheap
 *                 edge-enhancement blend; good for sharpening mask edges.
 *                 Formula: `a < 0.5 ? 2*a*b : 1 - 2*(1-a)*(1-b)`.
 *
 * @typedef {'replace' | 'add' | 'subtract' | 'intersect'
 *         | 'screen' | 'lighten' | 'darken' | 'overlay'} BlendOp
 */

/**
 * Base fields present on every mask layer. Discriminated by `kind`.
 *
 * @typedef {Object} MaskLayerBase
 * @property {string} id            Stable id (used as React key, history ref).
 * @property {string} label         Human-readable name for the layer panel.
 * @property {number} opacity       0..1 multiplier applied after the layer's
 *                                  intrinsic alpha. Lets users soften a layer
 *                                  without re-tuning its parameters.
 * @property {boolean} visible      When false, the layer is skipped in the
 *                                  compositing chain. Cheap UI-level toggle.
 * @property {boolean} inverted     When true, the layer's output alpha is
 *                                  replaced with `1 - alpha` before boolean
 *                                  compositing. Useful for "select everything
 *                                  except X" workflows.
 * @property {boolean} lock         When true, the layer is rendered but its
 *                                  parameters are read-only in the layer panel
 *                                  (sliders / inputs become disabled). Cheap
 *                                  UI-level guard against accidental edits.
 *                                  Photoshop parity: matches the padlock icon
 *                                  in the Layers panel. Sub-task 1.
 * @property {number} exposure      EV stops, -3..+3. Multiplies the layer's
 *                                  pixel colour by 2^stops. Applied only in
 *                                  the region the layer selects. Step 8.
 * @property {number} contrast      -100..+100. Pull toward 0.5 (positive) or
 *                                  toward 0.5 from the same side (negative —
 *                                  same formula). Applied only in the region
 *                                  the layer selects. Step 8.
 * @property {number} saturation    -100..+100. Lerp toward (-100) or away
 *                                  from (+100) luminance. Applied only in the
 *                                  region the layer selects. Step 8.
 * @property {number} vibrance      -100..+100. Like saturation, but the boost
 *                                  is scaled by (1 - currentSaturation) so
 *                                  already-vivid pixels (and skin tones) are
 *                                  protected. Applied only in the masked region.
 * @property {number} brightness    -100..+100. Additive shift in 0..1 RGB
 *                                  space (× 0.01). Applied only in the region
 *                                  the layer selects. Step 8.
 */

/**
 * Step 3 — Procedural Linear Gradient mask.
 * Projects each pixel onto the line P1→P2 and returns a 0..1 alpha based on
 * position along that line, with smooth feathering near the start/end.
 *
 * @typedef {Object} LinearMaskLayer
 * @property {'linear'} kind
 * @property {string} id
 * @property {string} label
 * @property {number} opacity
 * @property {boolean} visible
 * @property {boolean} inverted
 * @property {{x:number,y:number}} p1       Image-space start point (px).
 * @property {{x:number,y:number}} p2       Image-space end point (px).
 * @property {number} feather               0..1 — width of the soft falloff
 *                                          (0 = hard edge, 1 = full gradient).
 * @property {number} position              0..1 — midpoint of the gradient along
 *                                          the P1→P2 line (0.5 = centered).
 */

/**
 * Step 3 — Procedural Radial / Elliptical Gradient mask.
 * Uses an inverse 2×3 affine transform to warp UV space into a unit ellipse.
 *
 * @typedef {Object} RadialMaskLayer
 * @property {'radial'} kind
 * @property {string} id
 * @property {string} label
 * @property {number} opacity
 * @property {boolean} visible
 * @property {boolean} inverted
 * @property {{x:number,y:number}} center   Image-space center point (px).
 * @property {number} rotation              Radians — the ellipse's major-axis angle.
 * @property {{x:number,y:number}} radius   Half-extents along the rotated axes.
 * @property {number} feather               0..1 — width of the soft falloff.
 */

/**
 * Step 2 — Luminance range mask. Selects pixels whose WCAG sRGB luminosity
 * falls within [min, max].
 *
 * @typedef {Object} LuminanceMaskLayer
 * @property {'luminance'} kind
 * @property {string} id
 * @property {string} label
 * @property {number} opacity
 * @property {boolean} visible
 * @property {boolean} inverted
 * @property {number} min                   0..1 (luminance floor).
 * @property {number} max                   0..1 (luminance ceiling).
 * @property {number} softness              0..1 — width of the smoothstep edge
 *                                          on both thresholds.
 */

/**
 * Step 2 — Color (HSB hue) range mask. Selects pixels whose hue is within
 * `tolerance` of `target` (cyclical distance, so red wraps to red).
 *
 * @typedef {Object} ColorMaskLayer
 * @property {'color'} kind
 * @property {string} id
 * @property {string} label
 * @property {number} opacity
 * @property {boolean} visible
 * @property {boolean} inverted
 * @property {{h:number,s:number,b:number}} target   Target HSB color.
 *                                                  h is degrees 0..360.
 * @property {number} tolerance                     0..180 — max cyclical hue
 *                                                  distance (degrees).
 * @property {number} softness                      0..1 — width of the smooth
 *                                                  edge around the tolerance.
 */

/**
 * Step 7 — Manual precision Smart Brush (edge-preserving filter).
 *
 * The user's painted alpha texture (a free-hand stroke on an offscreen
 * canvas) is uploaded to a sampler2D and filtered through a bilateral
 * filter in the GLSL body. The bilateral filter preserves edges by
 * weighting neighbour samples by both spatial distance AND colour
 * similarity — across a high-contrast edge, the colour weight is tiny
 * so the alpha doesn't bleed past the edge.
 *
 * @typedef {Object} SmartBrushMaskLayer
 * @property {'smartBrush'} kind
 * @property {string} id
 * @property {string} label
 * @property {number} opacity
 * @property {boolean} visible
 * @property {boolean} inverted
 * @property {string} brushTextureKey      Opaque handle to the offscreen
 *                                        alpha canvas. Resolved at apply-time
 *                                        to a sampler2D uniform.
 * @property {number} filterRadius         1..8 — kernel radius in image px.
 * @property {number} sigmaColor           0.01..1 — colour Gaussian sigma.
 * @property {number} sigmaSpace           0.01..8 — spatial Gaussian sigma.
 */

/**
 * Step 5 — Semantic AI mask (SAM 2). Click-to-select segmentation; the
 * decoder returns a binary mask uploaded to a sampler2D.
 *
 * @typedef {Object} SemanticMaskLayer
 * @property {'semantic'} kind
 * @property {string} id
 * @property {string} label
 * @property {number} opacity
 * @property {boolean} visible
 * @property {boolean} inverted
 * @property {string} maskTextureKey        Opaque handle to the decoder output
 *                                          ImageData/HTMLCanvasElement.
 * @property {number} feather                0..1 — soft-edge width.
 */

/**
 * Step 6 — Depth-based mask (Depth Anything V2). Selects pixels whose depth
 * (normalised 0..1, 0 = far) falls within [min, max].
 *
 * @typedef {Object} DepthMaskLayer
 * @property {'depth'} kind
 * @property {string} id
 * @property {string} label
 * @property {number} opacity
 * @property {boolean} visible
 * @property {boolean} inverted
 * @property {string} depthMapKey           Opaque handle to the depth-map
 *                                          ImageData (white = near).
 * @property {number} min                    0..1 (near floor).
 * @property {number} max                    0..1 (far ceiling).
 * @property {number} softness               0..1 — smoothstep width.
 */

/**
 * @typedef {LinearMaskLayer | RadialMaskLayer | LuminanceMaskLayer
 *           | ColorMaskLayer | SmartBrushMaskLayer | SemanticMaskLayer
 *           | DepthMaskLayer} MaskLayer
 */

/**
 * A single compositing entry — pairs a layer with the operation used to
 * combine it into the running alpha. The first entry's `op` is always
 * 'replace' (the compiler enforces this).
 *
 * @typedef {Object} MaskChainEntry
 * @property {MaskLayer} layer
 * @property {BlendOp} op
 */

/**
 * The full mask stack: the array of layers and the order in which to
 * composite them. Per-image adjustments (exposure/contrast/saturation/
 * brightness) used to live on the stack as a top-level `adjust` field
 * (pre-Step 8), but Step 8 moved them onto each layer so different
 * regions of the image can have different adjustments. The renderer
 * short-circuits to identity when every layer's adjustments are 0
 * (see `isAdjustmentsIdentity` in this module).
 *
 * @typedef {Object} MaskStack
 * @property {MaskChainEntry[]} chain
 */

/**
 * Runtime shape of the compiled megashader. `frag` and `vert` are complete
 * GLSL ES 1.00 strings ready to feed into gl.createShader/Program. `cacheKey`
 * is the deterministic hash used to dedupe compiled WebGLPrograms.
 *
 * @typedef {Object} CompiledShader
 * @property {string} frag
 * @property {string} vert
 * @property {string} cacheKey
 * @property {boolean} passthrough          true if the chain has no layers
 *                                          and no adjustments — caller may
 *                                          skip installing the filter entirely.
 */

/**
 * Identifier kind for the public MegashaderFilter Fabric class. Use this as
 * `fabric.Image.filters.MegashaderFilter` after `import { MegashaderFilter }`.
 *
 * @typedef {typeof import('./fabric-megashader-filter').MegashaderFilter} MegashaderFilterClass
 */

export const BLEND_OPS = /** @type {const} */ ([
    'replace',
    'add',
    'subtract',
    'intersect',
    // Sub-task 1 — Photoshop-parity blend modes
    'screen',
    'lighten',
    'darken',
    'overlay',
])
export const MASK_KINDS = /** @type {const} */ ([
    'linear',
    'radial',
    'luminance',
    'color',
    'smartBrush',
    'semantic',
    'depth',
    'lasso',
    'brush',
    'path',
])

/**
 * Per-layer output modes (root-cause #1 fix). A mask layer SELECTS a
 * region; `fillMode` decides what visible effect that selection has:
 *
 * - `adjust`: recolour the masked region via the per-layer exposure /
 *             contrast / saturation / brightness adjustments. A layer with
 *             all-zero adjustments is colour-identity (invisible) — this is
 *             the historical behaviour, now opt-in.
 * - `fill`:   tint/overlay the masked region with `fillColor` at
 *             `fillStrength`. Makes a pure selection ALWAYS visible without
 *             dragging an adjustment slider (Quick-Mask style). The default
 *             for freshly-added selection layers and the lasso.
 * - `erase`:  knock the masked region out of the image (alpha → 0). The
 *             "cut" half of the lasso's cut/erase mode.
 *
 * @typedef {'adjust' | 'fill' | 'erase'} FillMode
 */
export const FILL_MODES = /** @type {const} */ (['adjust', 'fill', 'erase'])

/** Default selection tint (magenta) used by `fill` mode. RGB 0..1. */
export const DEFAULT_FILL_COLOR = { r: 1, g: 0, b: 0.6 }

/**
 * Map a `FillMode` string to the float the shader's `uLayer_<slot>_fillMode`
 * uniform expects: adjust → 0, fill → 1, erase → 2.
 *
 * @param {string} mode
 * @returns {number}
 */
export const fillModeToFloat = (mode) => (mode === 'fill' ? 1 : (mode === 'erase' ? 2 : 0))

/**
 * Coerce a partial layer's fill fields into the canonical shape. Used by
 * `sanitiseLayer` and the layer factories so every layer carries a valid
 * `fillMode` / `fillColor` / `fillStrength` regardless of where it came
 * from (factory, deserialised project state, or a raw `addLayer` patch).
 *
 * @param {{ fillMode?: string, fillColor?: {r:number,g:number,b:number}, fillStrength?: number }} layer
 * @returns {{ fillMode: string, fillColor: {r:number,g:number,b:number}, fillStrength: number }}
 */
export const sanitiseFill = (layer = {}) => {
    const fillMode = FILL_MODES.includes(/** @type {any} */ (layer.fillMode)) ? layer.fillMode : 'adjust'
    const c = layer.fillColor || DEFAULT_FILL_COLOR
    const ch = (v, fb) => (typeof v === 'number' && Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : fb)
    return {
        fillMode,
        fillColor: { r: ch(c.r, DEFAULT_FILL_COLOR.r), g: ch(c.g, DEFAULT_FILL_COLOR.g), b: ch(c.b, DEFAULT_FILL_COLOR.b) },
        fillStrength: ch(layer.fillStrength, 0.5),
    }
}

/**
 * Step 9 — Identity fast-path detection. Returns `true` if the stack's
 * every-layer per-layer image adjustment is the identity transform
 * (all four fields exactly 0). The GLSL chain is mathematically
 * identity in that case:
 *
 *   1. `applyLayerAdjust_<slot>(srcRgb)` early-returns `srcRgb` when
 *      all four uniforms are 0, so every `c_i` equals `srcRgb`.
 *   2. The boolean chain keeps `runningColor = mix(srcRgb, srcRgb, a_i) = srcRgb`
 *      for every op (replace/add/subtract/intersect).
 *   3. The final `mix(srcRgb, runningColor, runningAlpha * uMaskAlpha)`
 *      is `mix(srcRgb, srcRgb, x) = srcRgb` for any x.
 *
 *   So the rendered output is bit-for-bit the source canvas. The
 *   renderer can short-circuit and return the source as-is, skipping
 *   the WebGL upload / draw / readback entirely.
 *
 * Note: this check is about *colour math identity*, not about whether
 * layers exist. A stack with 5 layers, all with default adjustments,
 * is still identity — the chain doesn't change `srcRgb` even though
 * the boolean composition is non-trivial. The alpha side of the
 * chain is irrelevant to the colour output when adjustments are 0.
 *
 * @param {import('./mask-types').MaskStack | null | undefined} stack
 * @returns {boolean}
 */
/**
 * Every per-layer image-adjustment field. The original four plus the
 * pro-parity tonal + white-balance set. Used by the identity/visibility
 * predicates and the sanitiser so adding a field is a one-line change.
 */
export const ADJUST_FIELDS = /** @type {const} */ ([
    'exposure', 'contrast', 'saturation', 'vibrance', 'brightness',
    'highlights', 'shadows', 'whites', 'blacks', 'temperature', 'tint',
    'texture', 'dehaze',
])

/** True if a 3-way colour-wheel offset is non-neutral. */
const wheelActive = (w) => Array.isArray(w) && w.some((n) => Number.isFinite(n) && n !== 0)

/** True if any adjustment field on the layer is non-zero. Includes the
 *  colour-grading extensions — gamma (identity is 1, not 0), the 3-way colour
 *  wheels, and a tone-curve LUT — so a grade-only layer is NOT mistaken for a
 *  no-op and skipped by the renderer's short-circuit. */
const layerHasAdjustment = (l) => {
    if (!l) return false
    if (ADJUST_FIELDS.some((f) => l[f])) return true
    if (typeof l.gamma === 'number' && l.gamma !== 1) return true
    if (l.curveLutKey) return true
    return wheelActive(l.wheelShadows) || wheelActive(l.wheelMidtones) || wheelActive(l.wheelHighlights)
}

export const isAdjustmentsIdentity = (stack) => {
    if (!stack || !Array.isArray(stack.chain)) return true
    for (const entry of stack.chain) {
        if (!entry || !entry.layer) continue
        if (layerHasAdjustment(entry.layer)) return false
    }
    return true
}

/**
 * Root-cause #1 fix — the real "is this stack a no-op?" predicate that
 * replaces `isAdjustmentsIdentity` at the render/apply short-circuits.
 *
 * Unlike `isAdjustmentsIdentity` (which only checks colour math), this
 * returns `false` whenever ANY visible layer produces an output:
 *   - a non-zero per-layer adjustment (recolour), OR
 *   - `fillMode === 'fill'` with a non-zero `fillStrength` (tint), OR
 *   - `fillMode === 'erase'` (alpha knockout).
 *
 * Invisible layers (`visible === false`) and zero-opacity layers
 * contribute nothing, so they don't count. An empty chain is a no-op.
 * This is why a freshly-added selection (fill mode, default strength)
 * now renders instead of short-circuiting to the untouched source.
 *
 * @param {import('./mask-types').MaskStack | null | undefined} stack
 * @returns {boolean}
 */
export const stackHasNoVisibleEffect = (stack) => {
    if (!stack || !Array.isArray(stack.chain) || stack.chain.length === 0) return true
    for (const entry of stack.chain) {
        const l = entry && entry.layer
        if (!l) continue
        if (l.visible === false) continue
        if (typeof l.opacity === 'number' && l.opacity <= 0) continue
        if (layerHasAdjustment(l)) return false
        if (l.fillMode === 'erase') return false
        if (l.fillMode === 'fill') {
            const strength = typeof l.fillStrength === 'number' ? l.fillStrength : 0.5
            if (strength > 0) return false
        }
    }
    return true
}

/**
 * Factory for an empty/identity MaskStack. Used as the `useMaskLayers` initial
 * state and as the no-op value when the megashader is bypassed.
 *
 * @returns {MaskStack}
 */
export const createEmptyStack = () => ({ chain: [] })

/**
 * Type guard: is the given object a BlendOp string?
 *
 * @param {unknown} op
 * @returns {op is BlendOp}
 */
export const isBlendOp = (op) => typeof op === 'string' && BLEND_OPS.includes(/** @type {BlendOp} */ (op))

/**
 * Clamp a number into [lo, hi]. Non-finite values (NaN, ±Infinity,
 * undefined, non-numbers) fall back to `fallback` — the caller picks
 * the safe default (e.g. 0 for adjustments, the range's natural zero).
 * Used by `sanitiseLayer` to normalise Step 8's per-layer adjustment
 * fields where a missing value should mean "neutral" (0), not the
 * range's minimum.
 *
 * @param {unknown} v
 * @param {number} lo
 * @param {number} hi
 * @param {number} [fallback=lo]
 * @returns {number}
 */
const clampFinite = (v, lo, hi, fallback = lo) => (
    typeof v === 'number' && Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : fallback
)

/** Normalise a 3-way colour-wheel offset to a clamped [r,g,b] in -1..1.
 *  Accepts an array or {x,y,z}; anything else → neutral [0,0,0]. */
const sanitiseWheel = (w) => {
    const a = Array.isArray(w) ? w : (w && typeof w === 'object' ? [w.x, w.y, w.z] : [])
    return [0, 1, 2].map((i) => clampFinite(a[i], -1, 1, 0))
}

/**
 * Sanitise a MaskLayer into the canonical shape: ensures all `MaskLayerBase`
 * fields are present with safe defaults. Unknown `kind` values throw so the
 * compiler fails loud rather than silently producing a broken shader.
 *
 * @param {Partial<MaskLayer> & { kind: string }} layer
 * @returns {MaskLayer}
 */
export const sanitiseLayer = (layer) => {
    if (!layer || typeof layer !== 'object' || !layer.kind) {
        throw new Error('[megashader] sanitiseLayer: layer must be an object with a `kind`')
    }
    if (!MASK_KINDS.includes(/** @type {MASK_KINDS[number]} */ (layer.kind))) {
        throw new Error(`[megashader] sanitiseLayer: unknown mask kind "${layer.kind}"`)
    }
    return /** @type {MaskLayer} */ ({
        // Defaults that should be re-asserted after the spread. The spread
        // would otherwise overwrite them if `layer.id`/`layer.label` are
        // empty strings (an empty string is falsy, so the `||` fallback
        // runs, but the spread then writes back the original empty
        // string). Re-asserting after the spread makes the fallback
        // actually take effect.
        id: layer.id || `layer-${Math.random().toString(36).slice(2, 10)}`,
        label: layer.label || 'Untitled mask',
        // Step 8: per-layer image adjustments. Default to identity (0) so
        // a freshly created layer is visually identical to the global-
        // adjust era. Clamped in their own canonical ranges, not the
        // [0, 1] opacity range.
        exposure: 0,
        contrast: 0,
        saturation: 0,
        vibrance: 0,
        brightness: 0,
        opacity: 1,
        // Spread AFTER the defaults so a layer-supplied opacity wins; we
        // clamp last so a malicious or out-of-range value (e.g. 1.5 from
        // an upstream tool) still ends up in [0, 1] regardless of where
        // it came from.
        ...layer,
        // Re-assert id + label fallbacks (see comment at top of the
        // returned object). Also re-assert visible/inverted to keep the
        // JSDoc boolean contract clean — the renderer defends against
        // undefined with `=== false ? 0 : 1` so this is cosmetic, but
        // it makes the layer object self-consistent and removes the
        // surprise of `layer.visible === undefined`.
        id: layer.id || `layer-${Math.random().toString(36).slice(2, 10)}`,
        label: layer.label || 'Untitled mask',
        visible: layer.visible !== false,
        inverted: layer.inverted === true,
        opacity: typeof layer.opacity === 'number'
            ? Math.max(0, Math.min(1, layer.opacity))
            : 1,
        exposure: clampFinite(layer.exposure, -3, 3, 0),
        contrast: clampFinite(layer.contrast, -100, 100, 0),
        saturation: clampFinite(layer.saturation, -100, 100, 0),
        vibrance: clampFinite(layer.vibrance, -100, 100, 0),
        brightness: clampFinite(layer.brightness, -100, 100, 0),
        // Pro-parity tonal + white-balance adjustments (default identity).
        highlights: clampFinite(layer.highlights, -100, 100, 0),
        shadows: clampFinite(layer.shadows, -100, 100, 0),
        whites: clampFinite(layer.whites, -100, 100, 0),
        blacks: clampFinite(layer.blacks, -100, 100, 0),
        temperature: clampFinite(layer.temperature, -100, 100, 0),
        tint: clampFinite(layer.tint, -100, 100, 0),
        // Detail — local-contrast ops (sample the source neighbourhood).
        texture: clampFinite(layer.texture, -100, 100, 0),
        dehaze: clampFinite(layer.dehaze, -100, 100, 0),
        // Colour-grading extensions (per mask): gamma (1 = identity), 3-way
        // colour wheels (vec3 offsets -1..1 for shadows/midtones/highlights),
        // and a tone-curve LUT key (renderer uploads the packed 256×1 RGBA LUT
        // stored under it). curves = the raw points, kept for the UI/rebuild.
        gamma: clampFinite(layer.gamma, 0.2, 2.2, 1),
        wheelShadows: sanitiseWheel(layer.wheelShadows),
        wheelMidtones: sanitiseWheel(layer.wheelMidtones),
        wheelHighlights: sanitiseWheel(layer.wheelHighlights),
        ...(typeof layer.curveLutKey === 'string' && layer.curveLutKey ? { curveLutKey: layer.curveLutKey } : {}),
        ...(layer.curves && typeof layer.curves === 'object' ? { curves: layer.curves } : {}),
        lock: layer.lock === true,
        // Per-layer output mode (fill/adjust/erase) + fill colour/strength.
        ...sanitiseFill(layer),
    })
}

/* ─── Luminance + Color factories (Step 2) ────────────────────────────── */

/**
 * Build a fully-formed LuminanceMaskLayer with safe defaults. The layer
 * is ready to push into `useMaskLayers` or the megashader compiler — no
 * additional sanitisation needed.
 *
 * @param {object} [opts]
 * @param {number} [opts.min=0]        0..1 — luminance floor.
 * @param {number} [opts.max=0.5]      0..1 — luminance ceiling.
 * @param {number} [opts.softness=0.1] 0..1 — smoothstep edge width.
 * @param {string} [opts.label]        Human-readable label.
 * @returns {LuminanceMaskLayer}
 */
export const luminanceLayer = ({ min = 0, max = 0.5, softness = 0.1, label } = {}) => {
    const lo = Math.max(0, Math.min(1, min))
    const hi = Math.max(lo, Math.min(1, max))
    return /** @type {LuminanceMaskLayer} */ ({
        kind: 'luminance',
        id: `lum-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        label: label || `Luminance ${(lo * 255).toFixed(0)}–${(hi * 255).toFixed(0)}`,
        opacity: 1,
        visible: true,
        lock: false,
        inverted: false,
        min: lo,
        max: hi,
        softness: Math.max(0, Math.min(1, softness)),
    })
}

/**
 * Build a fully-formed ColorMaskLayer with safe defaults.
 *
 * @param {object} [opts]
 * @param {{h:number,s:number,b:number}} [opts.target={h:0,s:1,b:1}]  Target HSB colour.
 * @param {number} [opts.tolerance=0.15]  0..1 — combined-metric tolerance.
 * @param {number} [opts.softness=0.1]    0..1 — smoothstep edge width.
 * @param {string} [opts.label]           Human-readable label.
 * @returns {ColorMaskLayer}
 */
export const colorLayer = ({ target = { h: 0, s: 1, b: 1 }, tolerance = 0.15, softness = 0.1, label } = {}) => {
    const safeTarget = {
        h: Math.max(0, Math.min(360, Number(target?.h) || 0)),
        s: Math.max(0, Math.min(1, Number(target?.s) || 0)),
        b: Math.max(0, Math.min(1, Number(target?.b) || 0)),
    }
    return /** @type {ColorMaskLayer} */ ({
        kind: 'color',
        id: `col-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        label: label || 'Color range',
        opacity: 1,
        visible: true,
        lock: false,
        inverted: false,
        target: safeTarget,
        tolerance: Math.max(0, Math.min(1, tolerance)),
        softness: Math.max(0, Math.min(1, softness)),
    })
}

/* ─── Linear + Radial factories (Step 3) ──────────────────────────────── */

const TWO_PI = Math.PI * 2

/**
 * Build a fully-formed LinearMaskLayer. The `p1`/`p2` endpoints are in
 * image-space pixels. If `imageSize` is provided and the caller didn't
 * override p1/p2, the line defaults to a horizontal line across the
 * full image at vertical centre — the most common starting point and
 * easy to drag from.
 *
 * @param {object} [opts]
 * @param {{x:number,y:number}} [opts.p1]   Start point (image-space pixels).
 * @param {{x:number,y:number}} [opts.p2]   End point (image-space pixels).
 * @param {number} [opts.position=0.5]      0..1 — midpoint of the band along
 *                                          P1→P2 (0.5 = centred).
 * @param {number} [opts.feather=0.1]      0..1 — band half-width.
 * @param {{width:number,height:number}} [opts.imageSize]
 *                                          Source dimensions; used only
 *                                          when p1/p2 are not supplied,
 *                                          to pick sensible defaults.
 * @param {string} [opts.label]
 * @returns {LinearMaskLayer}
 */
export const linearLayer = ({
    p1, p2,
    position = 0.5,
    feather = 0.1,
    imageSize,
    label,
} = {}) => {
    const defP1 = imageSize ? { x: 0, y: imageSize.height / 2 } : { x: 0, y: 0 }
    const defP2 = imageSize ? { x: imageSize.width, y: imageSize.height / 2 } : { x: 100, y: 0 }
    const finalP1 = p1 || defP1
    const finalP2 = p2 || defP2
    return /** @type {LinearMaskLayer} */ ({
        kind: 'linear',
        id: `lin-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        label: label || 'Linear gradient',
        opacity: 1,
        visible: true,
        lock: false,
        inverted: false,
        p1: {
            x: clampFinite(finalP1.x, 0, 100000),
            y: clampFinite(finalP1.y, 0, 100000),
        },
        p2: {
            x: clampFinite(finalP2.x, 0, 100000),
            y: clampFinite(finalP2.y, 0, 100000),
        },
        position: clampFinite(position, 0, 1),
        feather: clampFinite(feather, 0, 1),
    })
}

/**
 * Build a fully-formed RadialMaskLayer. The `center` and `radius` are
 * in image-space pixels. If `imageSize` is provided and the caller
 * didn't override, the ellipse defaults to a circle of half-image-size
 * centred on the image. `rotation` is in radians.
 *
 * @param {object} [opts]
 * @param {{x:number,y:number}} [opts.center]   Centre point (image-space pixels).
 * @param {number} [opts.rotation=0]           Radians — ellipse major-axis angle.
 * @param {{x:number,y:number}} [opts.radius]  Half-extents along rotated axes.
 * @param {number} [opts.feather=0.1]          0..1 — soft-edge width.
 * @param {{width:number,height:number}} [opts.imageSize]
 * @param {string} [opts.label]
 * @returns {RadialMaskLayer}
 */
export const radialLayer = ({
    center, rotation = 0, radius,
    feather = 0.1,
    imageSize,
    label,
} = {}) => {
    const defCenter = imageSize ? { x: imageSize.width / 2, y: imageSize.height / 2 } : { x: 50, y: 50 }
    const defRadius = imageSize ? { x: imageSize.width / 2, y: imageSize.height / 2 } : { x: 50, y: 50 }
    const finalCenter = center || defCenter
    const finalRadius = radius || defRadius
    return /** @type {RadialMaskLayer} */ ({
        kind: 'radial',
        id: `rad-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        label: label || 'Radial gradient',
        opacity: 1,
        visible: true,
        lock: false,
        inverted: false,
        center: {
            x: clampFinite(finalCenter.x, 0, 100000),
            y: clampFinite(finalCenter.y, 0, 100000),
        },
        // Normalise to [0, 2π) so the renderer can rely on a stable range
        // (the JS `%` operator preserves sign, so we add then mod).
        rotation: ((rotation % TWO_PI) + TWO_PI) % TWO_PI,
        radius: {
            x: clampFinite(finalRadius.x, 0.001, 100000),
            y: clampFinite(finalRadius.y, 0.001, 100000),
        },
        feather: clampFinite(feather, 0, 1),
    })
}

/* ─── Semantic AI mask (Step 5) ─────────────────────────────────────── */

/**
 * Module-scoped texture cache for kinds that ship a per-layer image
 * (semantic masks, smart brush strokes, depth maps). The cache is keyed
 * by an opaque string the layer carries in `maskTextureKey`, and stores
 * an `ImageData` (or any object GL.texImage2D accepts — `HTMLCanvasElement`,
 * `ImageBitmap`, `HTMLImageElement` all work). The renderer pulls the
 * data out by key when it builds the GLSL samplers.
 *
 * Why module-scoped: the cache survives React re-renders and lives
 * outside the hook's state. `useMaskLayers` is responsible for clearing
 * entries when layers are removed (see its `removeLayer` callback),
 * so the cache can't grow without bound even with aggressive layer
 * churn.
 *
 * @type {Map<string, ImageData | HTMLCanvasElement | HTMLImageElement | ImageBitmap>}
 */
const maskTextureCache = new Map()

/**
 * Store a texture handle under an opaque key. Callers should generate
 * a unique key (e.g. with `crypto.randomUUID()`) and pass the same
 * key to `semanticLayer({ maskTextureKey: key, ... })`. Storing the
 * same key twice overwrites — that's intentional, but rare.
 *
 * @param {string} key
 * @param {ImageData | HTMLCanvasElement | HTMLImageElement | ImageBitmap} data
 */
/**
 * Per-key content VERSION for the persistent GL texture cache in
 * `megashader-renderer.js`. `setMaskTexture` is the single write path for
 * mask/LUT pixels (brush stroke, boundary grow, AI mask, curve LUT,
 * undo/redo restore), so bumping the version here — on EVERY write — is the
 * exact, complete signal that a key's pixels changed. The renderer re-uploads
 * only on a version change and reuses the cached GL texture otherwise, so a
 * stale texture can never be served: any pixel change goes through here and
 * increments the counter. The counter is monotonic (never reset, even across
 * clearMaskTexture) so a cleared-then-reused key can't collide with a version
 * the GL cache still holds.
 * @type {Map<string, number>}
 */
const maskTextureVersions = new Map()

export const setMaskTexture = (key, data) => {
    if (typeof key !== 'string' || !key) return
    if (data) {
        maskTextureCache.set(key, data)
        maskTextureVersions.set(key, (maskTextureVersions.get(key) || 0) + 1)
    }
}

/**
 * Current content version for a mask-texture key. 0 when the key has never
 * been written. The renderer compares this against the version its cached GL
 * texture was uploaded from; a mismatch forces a re-upload.
 *
 * @param {string} key
 * @returns {number}
 */
export const getMaskTextureVersion = (key) => {
    if (typeof key !== 'string' || !key) return 0
    return maskTextureVersions.get(key) || 0
}

/**
 * Look up a previously-stored texture. Returns `undefined` when the
 * key is unknown so the renderer can short-circuit instead of uploading
 * a missing texture.
 *
 * @param {string} key
 * @returns {ImageData | HTMLCanvasElement | HTMLImageElement | ImageBitmap | undefined}
 */
export const getMaskTexture = (key) => {
    if (typeof key !== 'string' || !key) return undefined
    return maskTextureCache.get(key)
}

/**
 * Remove a texture from the cache. Called from `useMaskLayers.removeLayer`
 * when a layer carrying a `maskTextureKey` is removed. The GL texture
 * itself is per-render in the renderer (created+uploaded+deleted each
 * frame) so no GL cleanup is needed here.
 *
 * @param {string} [key]
 */
export const clearMaskTexture = (key) => {
    if (typeof key !== 'string' || !key) return
    maskTextureCache.delete(key)
}

/**
 * Build a fully-formed SemanticMaskLayer. The layer is a thin handle to
 * an entry in the mask texture cache — `maskTextureKey` is the lookup
 * the renderer uses at draw time. The texture data MUST already be
 * in the cache before the layer is rendered (callers usually do this
 * by calling `setMaskTexture` right after fetching the mask from
 * the AI endpoint, and before calling `addLayer('semantic', ...)`).
 *
 * @param {object} [opts]
 * @param {string} [opts.maskTextureKey]      Required — the opaque handle
 *                                            into the mask texture cache.
 * @param {number} [opts.feather=0.1]        0..1 — smoothstep edge width.
 * @param {string} [opts.label]
 * @returns {SemanticMaskLayer}
 */
export const semanticLayer = ({ maskTextureKey, feather = 0.1, label } = {}) => {
    if (typeof maskTextureKey !== 'string' || !maskTextureKey) {
        throw new Error('[megashader] semanticLayer: `maskTextureKey` is required')
    }
    return /** @type {SemanticMaskLayer} */ ({
        kind: 'semantic',
        id: `sem-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        label: label || 'AI Subject',
        opacity: 1,
        visible: true,
        lock: false,
        inverted: false,
        maskTextureKey,
        feather: clampFinite(feather, 0, 1),
    })
}

/* ─── Lasso selection (freehand / polygonal) ──────────────────────────── */

/**
 * Build a fully-formed lasso selection layer. Like the semantic layer it's
 * a thin handle into the module-scoped texture cache (`maskTextureKey`),
 * resolved to a sampler2D at draw time. The closed polygon is rasterised
 * to an alpha canvas by the caller (white inside, transparent outside) and
 * stored with `setMaskTexture(maskTextureKey, canvas)` BEFORE the layer is
 * added, so the renderer never samples a missing texture.
 *
 * Defaults to `fillMode: 'fill'` so the selection is visible the instant
 * it's created (root-cause #1) — the lasso tool switches it to `erase` for
 * the cut/erase sink.
 *
 * @param {object} [opts]
 * @param {string} [opts.maskTextureKey]   Required — opaque cache handle.
 * @param {number} [opts.feather=0.05]    0..1 — soft-edge width.
 * @param {string} [opts.label]
 * @param {string} [opts.fillMode='fill']  'fill' | 'adjust' | 'erase'.
 * @param {{r:number,g:number,b:number}} [opts.fillColor]
 * @param {number} [opts.fillStrength=0.5]
 * @returns {object}
 */
export const lassoLayer = ({ maskTextureKey, feather = 0.05, label, fillMode = 'fill', fillColor, fillStrength = 0.5 } = {}) => {
    if (typeof maskTextureKey !== 'string' || !maskTextureKey) {
        throw new Error('[megashader] lassoLayer: `maskTextureKey` is required')
    }
    return {
        kind: 'lasso',
        id: `las-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        label: label || 'Lasso selection',
        opacity: 1,
        visible: true,
        lock: false,
        inverted: false,
        maskTextureKey,
        feather: clampFinite(feather, 0, 1),
        ...sanitiseFill({ fillMode, fillColor, fillStrength }),
    }
}

/**
 * Build a fully-formed Pen / Bézier path selection layer. Engine-identical to
 * the lasso (a thin handle into the module texture cache via `maskTextureKey`)
 * — the caller rasterises the closed Bézier path to an alpha canvas with
 * `rasterisePath`, registers it via `setMaskTexture`, then pushes this layer.
 * Defaults to `fill` mode so the path selection is visible the instant it's
 * added; switch to `adjust` to drive per-layer local adjustments inside it.
 *
 * @param {object} [opts]
 * @param {string} [opts.maskTextureKey]   Required — opaque cache handle.
 * @param {number} [opts.feather=0.04]
 * @param {string} [opts.label]
 * @param {string} [opts.fillMode='fill']  'fill' | 'adjust' | 'erase'.
 * @param {{r:number,g:number,b:number}} [opts.fillColor]
 * @param {number} [opts.fillStrength=0.5]
 * @returns {object}
 */
export const pathLayer = ({ maskTextureKey, feather = 0.04, label, fillMode = 'fill', fillColor, fillStrength = 0.5 } = {}) => {
    if (typeof maskTextureKey !== 'string' || !maskTextureKey) {
        throw new Error('[megashader] pathLayer: `maskTextureKey` is required')
    }
    return {
        kind: 'path',
        id: `pth-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        label: label || 'Pen path',
        opacity: 1,
        visible: true,
        lock: false,
        inverted: false,
        maskTextureKey,
        feather: clampFinite(feather, 0, 1),
        ...sanitiseFill({ fillMode, fillColor, fillStrength }),
    }
}

/**
 * Build a fully-formed plain Brush selection layer. Like the lasso it's a thin
 * handle into the module-scoped texture cache (`maskTextureKey`) — the caller
 * paints an alpha canvas, registers it with `setMaskTexture(maskTextureKey,
 * canvas)`, then pushes this layer. The GLSL samples the painted alpha verbatim
 * (no bilateral filter; use `smartBrushLayer` for edge-snapping). Defaults to
 * `fill` mode so the painted selection is visible the instant it's added.
 *
 * @param {object} [opts]
 * @param {string} [opts.maskTextureKey]   Required — opaque cache handle.
 * @param {string} [opts.label]
 * @param {string} [opts.fillMode='fill']  'fill' | 'adjust' | 'erase'.
 * @param {{r:number,g:number,b:number}} [opts.fillColor]
 * @param {number} [opts.fillStrength=0.5]
 * @returns {object}
 */
export const brushLayer = ({ maskTextureKey, label, fillMode = 'fill', fillColor, fillStrength = 0.5 } = {}) => {
    if (typeof maskTextureKey !== 'string' || !maskTextureKey) {
        throw new Error('[megashader] brushLayer: `maskTextureKey` is required')
    }
    return {
        kind: 'brush',
        id: `brush-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        label: label || 'Brush selection',
        opacity: 1,
        visible: true,
        lock: false,
        inverted: false,
        maskTextureKey,
        ...sanitiseFill({ fillMode, fillColor, fillStrength }),
    }
}

/* ─── Depth map (Step 6) ────────────────────────────────────────────────── */

/**
 * Build a fully-formed DepthMaskLayer. Like the semantic layer, the
 * depth map data is a thin handle to the mask texture cache, fetched
 * by the renderer at draw time.
 *
 * The depth map is the *whole-image* output of Depth Anything V2,
 * resized to the original image's resolution (the Python service
 * does the resize). The user then picks a `[min, max]` range on a
 * 0..1 slider to select the depth band they want.
 *
 * @param {object} [opts]
 * @param {string} [opts.depthMapKey]        Required — opaque handle into
 *                                            the mask texture cache. Same
 *                                            cache as semantic layers use
 *                                            (`setMaskTexture`).
 * @param {number} [opts.min=0]             0..1 — near floor (depth >= min).
 * @param {number} [opts.max=0.5]           0..1 — far ceiling (depth <= max).
 * @param {number} [opts.softness=0.1]      0..1 — smoothstep half-width on
 *                                            each edge.
 * @param {string} [opts.label]
 * @returns {DepthMaskLayer}
 */
export const depthLayer = ({
    depthMapKey,
    min = 0,
    max = 0.5,
    softness = 0.1,
    label,
} = {}) => {
    if (typeof depthMapKey !== 'string' || !depthMapKey) {
        throw new Error('[megashader] depthLayer: `depthMapKey` is required')
    }
    const dMin = clampFinite(min, 0, 1)
    const dMax = clampFinite(max, 0, 1)
    return /** @type {DepthMaskLayer} */ ({
        kind: 'depth',
        id: `dep-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        label: label || 'Depth range',
        opacity: 1,
        visible: true,
        lock: false,
        inverted: false,
        depthMapKey,
        min: Math.min(dMin, dMax),
        max: Math.max(dMin, dMax),
        softness: clampFinite(softness, 0, 1),
    })
}

/* ─── Smart Brush (Step 7) ────────────────────────────────────────────────── */

/**
 * Build a fully-formed SmartBrushMaskLayer. Like the semantic + depth
 * layers, the painted alpha is a thin handle to the same module-scoped
 * mask texture cache; the renderer pulls it by `brushTextureKey` at draw
 * time.
 *
 * The GLSL body runs a bilateral filter over a window of `filterRadius`
 * around the current pixel. The bilateral filter preserves edges by
 * weighting neighbour samples by both spatial distance and colour
 * similarity — across a sharp colour transition, the colour weight is
 * tiny so the brush stroke doesn't bleed past the edge.
 *
 * `filterRadius` is clamped to 1..8 (the schema's loop bound is a
 * compile-time constant). `sigmaColor` is the colour Gaussian's
 * standard deviation in 0..1 RGB space — small = strict edges.
 * `sigmaSpace` is the spatial Gaussian's sigma in image pixels.
 *
 * @param {object} [opts]
 * @param {string} [opts.brushTextureKey]     Required — opaque handle into
 *                                            the mask texture cache (callers
 *                                            usually pair this with
 *                                            `setMaskTexture` right after the
 *                                            user finishes painting).
 * @param {number} [opts.filterRadius=3]     1..8 — kernel radius (image px).
 * @param {number} [opts.sigmaColor=0.15]   0.01..1 — colour Gaussian sigma.
 * @param {number} [opts.sigmaSpace=2]      0.01..8 — spatial Gaussian sigma.
 * @param {string} [opts.label]
 * @returns {SmartBrushMaskLayer}
 */
export const smartBrushLayer = ({
    brushTextureKey,
    filterRadius = 3,
    sigmaColor = 0.15,
    sigmaSpace = 2,
    label,
} = {}) => {
    if (typeof brushTextureKey !== 'string' || !brushTextureKey) {
        throw new Error('[megashader] smartBrushLayer: `brushTextureKey` is required')
    }
    return /** @type {SmartBrushMaskLayer} */ ({
        kind: 'smartBrush',
        id: `brs-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        label: label || 'Smart brush',
        opacity: 1,
        visible: true,
        lock: false,
        inverted: false,
        brushTextureKey,
        filterRadius: clampFinite(filterRadius, 1, 8),
        sigmaColor: clampFinite(sigmaColor, 0.01, 1),
        sigmaSpace: clampFinite(sigmaSpace, 0.01, 8),
    })
}
