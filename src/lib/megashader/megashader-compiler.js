/**
 * Megashader Compiler
 * -------------------
 * Pure function: takes a `MaskStack`, returns a `CompiledShader`. No GL
 * context is touched here — this is the piece that runs in Node for unit
 * tests (`scripts/verify-megashader.mjs`) and on the client when the
 * *shape* of the layer set changes (not on per-layer parameter edits).
 *
 * Cache key strategy:
 *   The cache key is `kindSet|opChain|maxOpacity` — it's a coarse hash of
 *   the structural properties that force a recompile. Changing a single
 *   layer's threshold does NOT bust the cache, because that's just a
 *   uniform update. Adding/removing a layer OR changing an op DOES bust it.
 *
 *   The actual `WebGLProgram` cache lives in `megashader-renderer.js` and
 *   is keyed by this same string. Keeping the cache key derivation in the
 *   compiler means the renderer never has to re-derive it.
 *
 * @module megashader/megashader-compiler
 */

import {
    buildLayerFunction,
    buildLayerAdjustFunction,
    buildVertexShader,
    buildFragmentTemplate,
    buildBooleanChain,
    buildEvalDispatcher,
} from './glsl-fragments'
import { BLEND_OPS, MASK_KINDS } from './mask-types'

/** Hard cap so the megashader doesn't grow unbounded. 8 layers × 5 bytes
 *  per op is well under any GPU's uniform array limit. */
export const MAX_LAYERS = 8

const truncateChain = (chain) => (Array.isArray(chain) ? chain.slice(0, MAX_LAYERS) : [])

/**
 * Structural compile cache (Cluster A change 3). `compileMegashader` builds
 * the full GLSL source on every call, but the result is a pure function of
 * `cacheKey` (kinds|ops|length): layer PARAMETER values are uniforms, never
 * baked into the source (see `buildLayerFunction` / `buildLayerAdjustFunction`,
 * which read only `slotIndex`/`kind`). So a given `cacheKey` always yields a
 * byte-identical `frag`/`vert`, and memoising the whole CompiledShader is
 * lossless. This is the SAME key the renderer's WebGLProgram cache uses, so
 * the two caches stay perfectly aligned. LRU-capped to bound memory.
 * @type {Map<string, import('./mask-types').CompiledShader>}
 */
const compiledCache = new Map()
const MAX_COMPILED_CACHE = 64

/**
 * Insert a compiled shader into `compiledCache` under `cacheKey`, evicting the
 * oldest entry when the cap is reached, and return it (so callers can
 * `return rememberCompiled(key, {...})`).
 * @param {string} cacheKey
 * @param {import('./mask-types').CompiledShader} compiled
 * @returns {import('./mask-types').CompiledShader}
 */
const rememberCompiled = (cacheKey, compiled) => {
    if (compiledCache.size >= MAX_COMPILED_CACHE) {
        const oldest = compiledCache.keys().next().value
        if (oldest !== undefined) compiledCache.delete(oldest)
    }
    compiledCache.set(cacheKey, compiled)
    return compiled
}

/**
 * Normalise a MaskStack. Throws on shape errors (unknown kind, missing
 * fields, wrong types) so a malformed stack never reaches the GPU.
 *
 * @param {import('./mask-types').MaskStack | null | undefined} stack
 * @returns {import('./mask-types').MaskStack}
 */
const normaliseStack = (stack) => {
    if (!stack || typeof stack !== 'object') return { chain: [] }
    const chain = truncateChain(stack.chain)
    const normalised = chain.map((entry, i) => {
        if (!entry || typeof entry !== 'object' || !entry.layer) {
            throw new Error(`[megashader] normaliseStack: chain[${i}] is not a MaskChainEntry`)
        }
        const layer = entry.layer
        if (!layer.kind) {
            throw new Error(`[megashader] normaliseStack: chain[${i}].layer has no \`kind\``)
        }
        // Validate kind here so callers don't have to remember to call
        // sanitiseLayer first. The full sanitiser (defaults + clamping) is
        // optional and lives in mask-types.js. MASK_KINDS is the single
        // source of truth, so a new kind (e.g. 'lasso') is accepted here
        // automatically once it's registered there.
        if (!MASK_KINDS.includes(layer.kind)) {
            throw new Error(`[megashader] normaliseStack: chain[${i}].layer.kind "${layer.kind}" is not a known mask kind`)
        }
        const op = i === 0 ? 'replace' : (entry.op || 'add')
        // BLEND_OPS is the single source of truth shared with the layer
        // panel (mask-types.js) and the GLSL `buildBooleanChain`. Validating
        // against it here means picking a Photoshop blend mode (screen /
        // lighten / darken / overlay) no longer throws and silently blanks
        // the whole filter — it compiles to its real GLSL case.
        if (!BLEND_OPS.includes(op)) {
            throw new Error(`[megashader] normaliseStack: chain[${i}].op "${op}" is not a valid BlendOp`)
        }
        return { layer, op }
    })
    return { chain: normalised }
}

/**
 * Derive the cache key for a given MaskStack. Two stacks with the same key
 * can share a compiled WebGLProgram. The key is intentionally short and
 * stable across parameter changes.
 *
 * @param {import('./mask-types').MaskStack} stack
 * @returns {string}
 */
export const computeCacheKey = (stack) => {
    const kinds = stack.chain.map((e) => e.layer.kind || 'unknown').join(',')
    const ops = stack.chain.map((e) => e.op).join(',')
    // Adjustments don't change the *shader* (only the uniforms), but they
    // DO change the final colour math. Since the shader template always
    // declares the adjust uniforms, two stacks differing only in `adjust`
    // can still share a program. We use '0' as a placeholder.
    return `mk|${kinds}|${ops}|${stack.chain.length}`
}

/**
 * Build the complete fragment + vertex shader source for a MaskStack.
 *
 * @param {import('./mask-types').MaskStack | null | undefined} rawStack
 * @returns {import('./mask-types').CompiledShader}
 */
export const compileMegashader = (rawStack) => {
    const stack = normaliseStack(rawStack)
    // Structural compile cache (Cluster A change 3): the CompiledShader is a
    // pure function of `cacheKey`, so return a memoised result before doing
    // any GLSL string building on a cache hit.
    const cacheKey = computeCacheKey(stack)
    const memoised = compiledCache.get(cacheKey)
    if (memoised) return memoised
    const vert = buildVertexShader()
    const fragmentTemplate = buildFragmentTemplate()

    if (stack.chain.length === 0) {
        // Passthrough — no mask functions, no adjust functions, no chain.
        // The template's empty-chain branch initialises `runningColor` to
        // `srcRgb` and `runningAlpha` to 0, so the final mix is identity.
        // The dispatcher is also emitted (returns 0 unconditionally) so
        // the fragment compiles cleanly even though it's never reached.
        return rememberCompiled(cacheKey, {
            frag: fragmentTemplate
                .replace('{{MASK_FUNCTIONS}}', '// passthrough — no layers')
                .replace('{{ADJUST_FUNCTIONS}}', '// passthrough — no adjustments')
                .replace('{{EVAL_DISPATCHER}}', 'float evalLayer(int idx) { return 0.0; }')
                .replace('{{BOOLEAN_CHAIN}}', 'vec3 runningColor = srcRgb;\n        float runningAlpha = 0.0;\n        float eraseAlpha = 0.0;'),
            vert,
            cacheKey,
            passthrough: true,
        })
    }

    const layerFns = stack.chain
        .map((entry, i) => buildLayerFunction(i, entry.layer.kind, entry.layer))
        .join('\n')

    // Step 8: one per-layer adjust function per layer. Each reads its own
    // `uLayer_<slot>_adjust_*` uniforms and applies them in sequence.
    // The renderer writes 0 for any field a layer doesn't supply, so the
    // function's early-out fires (returns srcRgb unchanged) for layers
    // that don't need adjustments.
    const adjustFns = stack.chain
        .map((entry, i) => buildLayerAdjustFunction(i, entry.layer))
        .join('\n')

    const booleanChain = buildBooleanChain(stack.chain)
    const evalDispatcher = buildEvalDispatcher(stack.chain)

    const frag = fragmentTemplate
        .replace('{{MASK_FUNCTIONS}}', layerFns)
        .replace('{{ADJUST_FUNCTIONS}}', adjustFns)
        .replace('{{EVAL_DISPATCHER}}', evalDispatcher)
        .replace('{{BOOLEAN_CHAIN}}', booleanChain)

    return rememberCompiled(cacheKey, { frag, vert, cacheKey, passthrough: false })
}
