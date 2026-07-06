/**
 * Megashader Public API
 * ---------------------
 * Single import surface for the rest of the app. Hides the compiler /
 * renderer / Fabric-filter modules behind a small, stable facade.
 *
 * Typical usage from the canvas component:
 *
 *   import {
 *     applyMegashaderFilter,
 *     MegashaderFilter,
 *     hasMegashaderWebGL2,
 *     createEmptyStack,
 *   } from '@/lib/megashader'
 *
 *   useEffect(() => {
 *     if (!canvasEditor || !image) return
 *     const filter = new MegashaderFilter({ stack, globalMaskAlpha: 1 })
 *     image.filters = [...nonMegashaderFilters, filter]
 *     image.applyFilters()
 *     return () => {
 *       // Remove the filter on cleanup; the renderer disposes its GL cache
 *       // lazily (only when fully unmounted).
 *     }
 *   }, [image, stack])
 *
 * Side-effect: importing this file registers the MegashaderFilter class
 * with Fabric's `classRegistry` (via the same pattern as
 * `src/lib/curves-filter.js`).
 *
 * @module megashader
 */

export { MegashaderFilter } from './fabric-megashader-filter'
export {
    applyMegashaderFilter,
    hasMegashaderWebGL2,
    disposeMegashader,
} from './apply-megashader'

export {
    compileMegashader,
    computeCacheKey,
    MAX_LAYERS,
} from './megashader-compiler'

export {
    createEmptyStack,
    sanitiseLayer,
    isAdjustmentsIdentity,
    stackHasNoVisibleEffect,
    isBlendOp,
    BLEND_OPS,
    MASK_KINDS,
    FILL_MODES,
    DEFAULT_FILL_COLOR,
    fillModeToFloat,
    sanitiseFill,
    luminanceLayer,
    colorLayer,
    linearLayer,
    radialLayer,
    smartBrushLayer,
    semanticLayer,
    depthLayer,
    lassoLayer,
    brushLayer,
    pathLayer,
    setMaskTexture,
    getMaskTexture,
    clearMaskTexture,
} from './mask-types'

export { rasterisePath, smoothToBezier } from './path-raster'

export { KIND_SCHEMAS, KIND_BUILDERS, getKindBuilder, getKindSchema } from './glsl-mask-kinds'

// Step 10.3: render perf metrics. Read-only snapshot via
// `getRenderMetrics`, zero via `resetRenderMetrics`. Exposed for
// the dev test panel ("Perf" badge) so engineers can see compile
// time, cache hit rate, draw count, and identity-short-circuit
// count without opening devtools.
export { getRenderMetrics, resetRenderMetrics, renderMegashader } from './megashader-renderer'
