/**
 * Megashader Renderer
 * -------------------
 * Owns a *private* WebGL2 context and a program cache. The renderer is
 * deliberately decoupled from Fabric's filter chain — the codebase already
 * disables Fabric's WebGL filtering (`fabricConfig.enableGLFiltering = false`
 * in `canvas.jsx`) because the curves LUT filter had subtle interaction
 * bugs. Routing the megashader through Fabric's GL pipeline would inherit
 * that problem, so we manage our own context here.
 *
 * The contract with `fabric-megashader-filter.js` is:
 *   1. Caller calls `renderMegashader(sourceCanvas, compiledShader, stack)`.
 *   2. Renderer binds the source 2D canvas as a texture, runs the megashader
 *      fragment shader across a fullscreen quad, reads the result back to a
 *      2D canvas, and returns it. The Fabric filter then `drawImage`s this
 *      result onto the chain's output canvas.
 *   3. If WebGL2 is unavailable, the renderer falls back to a CPU path that
 *      draws the source canvas to the output (no-op for Step 1's stub
 *      layers, but the integration stays correct so later steps can plug in
 *      real CPU math without touching the caller).
 *
 * This module also owns the WebGLProgram cache. The key is the
 * `compiledShader.cacheKey` produced by `megashader-compiler.js`.
 *
 * @module megashader/megashader-renderer
 */

import { compileMegashader } from './megashader-compiler'
import { getKindSchema, normaliseUniformValue } from './glsl-mask-kinds'
import { getMaskTexture, getMaskTextureVersion, stackHasNoVisibleEffect, fillModeToFloat } from './mask-types'

const MAX_PROGRAM_CACHE = 64

// Step 10.3: lightweight perf metrics. The renderer tracks compile
// count, cache hit/miss, eviction count, and the most recent compile
// wall-clock time. Exposed via `getRenderMetrics()` (read-only
// snapshot) so the dev test panel can display them. Reset by
// `resetRenderMetrics()` (also exposed) — useful when taking a
// "before/after" measurement around a code change.
//
// All counters are module-scoped (not exposed for write) so the
// renderer is the only writer. The metrics are best-effort and have
// no perf cost in the hot path — they're only touched on a cache
// miss (compile) and on every getOrCreateProgram call (hit/miss
// branch).
const renderMetrics = {
    /** Total number of unique programs compiled across the lifetime. */
    compileCount: 0,
    /** Cache hits (program was already compiled and is still in the LRU). */
    cacheHits: 0,
    /** Cache misses (program had to be compiled and linked). */
    cacheMisses: 0,
    /** Number of programs evicted from the LRU (size capped at MAX_PROGRAM_CACHE). */
    evictions: 0,
    /** Wall-clock time (ms) of the most recent compile+link. */
    lastCompileMs: 0,
    /** Cumulative wall-clock time (ms) spent compiling across the lifetime. */
    totalCompileMs: 0,
    /** Number of `renderMegashader` calls that actually drew a frame. */
    drawCount: 0,
    /** Number of `renderMegashader` calls that took the identity short-circuit. */
    identityShortCircuits: 0,
}

/**
 * Read-only snapshot of the current render metrics. The returned
 * object is a fresh copy, so callers can compare snapshots across
 * code paths without aliasing issues.
 *
 * @returns {typeof renderMetrics}
 */
const getRenderMetrics = () => ({ ...renderMetrics })

/**
 * Reset all render metrics to zero. Mainly for tests and for the
 * "Reset metrics" button in the test panel — useful when taking a
 * measurement around a specific user action (e.g. dragging a slider)
 * to isolate that action's cost.
 */
const resetRenderMetrics = () => {
    renderMetrics.compileCount = 0
    renderMetrics.cacheHits = 0
    renderMetrics.cacheMisses = 0
    renderMetrics.evictions = 0
    renderMetrics.lastCompileMs = 0
    renderMetrics.totalCompileMs = 0
    renderMetrics.drawCount = 0
    renderMetrics.identityShortCircuits = 0
}

/**
 * Minimal fullscreen-quad vertex shader. Compiled once into the quad program
 * (see `getQuadProgram`) so the VAO can be set up against a real, linked
 * program. We also `bindAttribLocation(program, 0, 'aPosition')` before
 * linking on every program (quad + megashader) so `aPosition` is always at
 * location 0 — that lets the same VAO bind to any megashader program.
 */
const QUAD_VERT = `attribute vec2 aPosition;
varying vec2 vUV;
void main() {
    vUV = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`

/** Fragment counterpart — outputs solid black, never actually drawn. */
const QUAD_FRAG = `precision mediump float;
void main() {
    gl_FragColor = vec4(0.0);
}
`

let glContext = null
let glCanvas = null
let programCache = /** @type {Map<string, WebGLProgram>} */ (new Map())
// Persistent GL texture cache for mask/LUT uploads (Cluster A change 1).
// Keyed by the mask-texture cache key; each entry records the GL texture and
// the data VERSION it was uploaded from (see getMaskTextureVersion). Reused
// across frames; re-uploaded only on a version bump. LRU-capped well above
// the <=16 textures any single frame can bind, so eviction never removes a
// texture that was bound earlier in the current frame.
let maskGlTextureCache = /** @type {Map<string, { tex: WebGLTexture, version: number }>} */ (new Map())
const MAX_MASK_GL_TEXTURES = 64
let quadProgram = /** @type {WebGLProgram | null} */ (null)
let quadVbo = /** @type {WebGLBuffer | null} */ (null)
let quadVao = /** @type {WebGLVertexArrayObject | null} */ (null)

const isBrowser = () => typeof window !== 'undefined' && typeof document !== 'undefined'

/**
 * Lazily create (and cache) the private WebGL2 context. Returns `null` if
 * the browser doesn't support WebGL2 — the renderer then runs in CPU mode.
 *
 * @returns {WebGL2RenderingContext | null}
 */
const ensureGl = () => {
    if (!isBrowser()) return null
    if (glContext) return glContext
    if (typeof document.createElement !== 'function') return null

    glCanvas = document.createElement('canvas')
    glCanvas.width = 1
    glCanvas.height = 1
    const ctx = /** @type {any} */ (glCanvas.getContext('webgl2', {
        premultipliedAlpha: false,
        preserveDrawingBuffer: false,
    })) || /** @type {any} */ (glCanvas.getContext('webgl'))
    if (!ctx) return null

    glContext = ctx
    return glContext
}

/**
 * Compile + link a GLSL program and cache it by `cacheKey`. Programs are
 * stored module-scope so the same shader source reuses the same program
 * across calls. LRU eviction keeps memory bounded.
 *
 * Step 10.3: updates `renderMetrics` on every call. A hit increments
 * `cacheHits` and returns early. A miss runs the compile+link and
 * increments `cacheMisses` + `compileCount` + `lastCompileMs` +
 * `totalCompileMs`. An eviction (LRU overflow) increments `evictions`.
 *
 * @param {import('./mask-types').CompiledShader} compiled
 * @returns {WebGLProgram | null}
 */
const getOrCreateProgram = (compiled) => {
    const gl = ensureGl()
    if (!gl) return null

    if (programCache.has(compiled.cacheKey)) {
        renderMetrics.cacheHits += 1
        return programCache.get(compiled.cacheKey) ?? null
    }
    renderMetrics.cacheMisses += 1
    const t0 = (typeof performance !== 'undefined' && typeof performance.now === 'function')
        ? performance.now()
        : Date.now()

    const vert = compileShader(gl, gl.VERTEX_SHADER, compiled.vert)
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, compiled.frag)
    if (!vert || !frag) return null

    const program = gl.createProgram()
    if (!program) return null
    gl.attachShader(program, vert)
    gl.attachShader(program, frag)
    // Pin `aPosition` to attribute location 0 so the quad VAO (set up in
    // `ensureQuadBuffers`) binds to the same slot on every program.
    gl.bindAttribLocation(program, 0, 'aPosition')
    gl.linkProgram(program)
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        // Surface the link error to the console (one-line, no exception
        // throw — the renderer is best-effort and falls back to CPU).
        const info = gl.getProgramInfoLog(program) || '(no info log)'
        console.warn('[megashader] program link failed:', info)
        gl.deleteShader(vert)
        gl.deleteShader(frag)
        gl.deleteProgram(program)
        return null
    }
    gl.deleteShader(vert)
    gl.deleteShader(frag)

    if (programCache.size >= MAX_PROGRAM_CACHE) {
        // Drop the oldest entry. Map iteration order is insertion order, so
        // the first key is the oldest.
        const oldest = programCache.keys().next().value
        if (oldest) {
            const oldProgram = programCache.get(oldest)
            if (oldProgram) gl.deleteProgram(oldProgram)
            programCache.delete(oldest)
            renderMetrics.evictions += 1
        }
    }
    programCache.set(compiled.cacheKey, program)
    renderMetrics.compileCount += 1
    const t1 = (typeof performance !== 'undefined' && typeof performance.now === 'function')
        ? performance.now()
        : Date.now()
    const dt = t1 - t0
    renderMetrics.lastCompileMs = dt
    renderMetrics.totalCompileMs += dt
    return program
}

/**
 * Compile a single GLSL shader stage.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {GLenum} type
 * @param {string} source
 * @returns {WebGLShader | null}
 */
const compileShader = (gl, type, source) => {
    const shader = gl.createShader(type)
    if (!shader) return null
    gl.shaderSource(shader, source)
    gl.compileShader(shader)
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader) || '(no info log)'
        console.warn(`[megashader] shader compile failed (${type === gl.VERTEX_SHADER ? 'vertex' : 'fragment'}):`, info)
        gl.deleteShader(shader)
        return null
    }
    return shader
}

/**
 * Check whether WebGL2 is available. Public so the Fabric filter can short-
 * circuit on the CPU path without allocating anything.
 *
 * @returns {boolean}
 */
export const hasWebGL2 = () => {
    if (!isBrowser()) return false
    return Boolean(ensureGl())
}

/**
 * Compile + link a one-off program whose only job is to give the VAO a
 * real, linked program to query attribute locations against. Cached
 * module-scope so we pay the compile cost exactly once per renderer
 * lifetime.
 *
 * @param {WebGL2RenderingContext} gl
 * @returns {WebGLProgram | null}
 */
const getQuadProgram = (gl) => {
    if (quadProgram) return quadProgram
    const program = gl.createProgram()
    if (!program) return null
    const vert = compileShader(gl, gl.VERTEX_SHADER, QUAD_VERT)
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, QUAD_FRAG)
    if (!vert || !frag) {
        if (vert) gl.deleteShader(vert)
        if (frag) gl.deleteShader(frag)
        gl.deleteProgram(program)
        return null
    }
    gl.attachShader(program, vert)
    gl.attachShader(program, frag)
    gl.bindAttribLocation(program, 0, 'aPosition')
    gl.linkProgram(program)
    gl.deleteShader(vert)
    gl.deleteShader(frag)
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const info = gl.getProgramInfoLog(program) || '(no info log)'
        console.warn('[megashader] quad program link failed:', info)
        gl.deleteProgram(program)
        return null
    }
    quadProgram = program
    return quadProgram
}

/**
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLProgram} program
 * @param {object} layer
 * @param {number} slotIndex
 * @param {{ kindUnits?: Map<number, number> }} textureBindings
 */
const writeKindSamplers = (gl, program, layer, slotIndex, textureBindings) => {
    let schema
    try {
        schema = getKindSchema(layer.kind)
    } catch {
        return
    }
    const samplers = schema.samplers
    if (!Array.isArray(samplers) || samplers.length === 0) return
    const unit = textureBindings.kindUnits?.get(slotIndex)
    for (const sampler of samplers) {
        const glslName = sampler.glsl.replace('<S>', String(slotIndex))
        const loc = gl.getUniformLocation(program, glslName)
        if (!loc) continue
        if (unit !== undefined) gl.uniform1i(loc, unit)
    }
}

/**
 * Bind the per-layer tone-curve LUT sampler to its texture unit and flip
 * `curveOn` on. When the layer has no curve LUT bound (the common case), force
 * `curveOn` to 0 so the GLSL skips the four LUT lookups and the early-out fires.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLProgram} program
 * @param {object} layer
 * @param {number} slotIndex
 * @param {{ curveUnits?: Map<number, number> }} textureBindings
 */
const writeCurveUniforms = (gl, program, layer, slotIndex, textureBindings) => {
    const onLoc = gl.getUniformLocation(program, `uLayer_${slotIndex}_curveOn`)
    const unit = textureBindings?.curveUnits?.get(slotIndex)
    if (unit === undefined) {
        if (onLoc) gl.uniform1f(onLoc, 0.0)
        return
    }
    const sLoc = gl.getUniformLocation(program, `uLayer_${slotIndex}_curveLut`)
    if (sLoc) gl.uniform1i(sLoc, unit)
    if (onLoc) gl.uniform1f(onLoc, 1.0)
}

/**
 * Write the per-kind UNIFORM fields (floats / vec2s) for one layer. Reads
 * the layer object's field name (matched against the schema's `name`),
 * normalises into the canonical shape the GLSL uniform writer expects
 * (via `normaliseUniformValue`), and pushes it to the GPU.
 *
 * Skips sampler fields — those are owned by `writeKindSamplers`.
 *
 * Step 8: adjustment fields (exposure/contrast/saturation/brightness)
 * are written separately by `writeLayerAdjustUniforms` because they're
 * not part of the per-kind schema.
 *
 * Bug history: pre-Step 8 this function only accepted `Array.isArray`
 * for vec2/vec3 values. The kind factories store those fields as
 * `{x, y}` objects (e.g. `linear.p1`, `radial.center`), so the
 * `Array.isArray` check failed and the value silently fell back to
 * the schema's `[0, 0]` default — every linear/radial mask rendered
 * at the origin with a zero radius regardless of the user's drag.
 * Step 8 routed vec2/vec3 through `normaliseUniformValue`, which
 * accepts both array AND `{x, y}` shapes and clamps each component
 * individually.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLProgram} program
 * @param {object} layer
 * @param {number} slotIndex
 */
const writeKindUniforms = (gl, program, layer, slotIndex) => {
    let schema
    try {
        schema = getKindSchema(layer.kind)
    } catch {
        return
    }
    const uniforms = schema.uniforms
    if (!Array.isArray(uniforms) || uniforms.length === 0) return
    for (const field of uniforms) {
        const glslName = field.glsl.replace('<S>', String(slotIndex))
        const loc = gl.getUniformLocation(program, glslName)
        if (!loc) continue
        // Bug #4: the color factory stores the picked colour nested as
        // `target: { h, s, b }`, but COLOR_SCHEMA declares flat
        // `targetH/targetS/targetB`. Without this remap, layer.targetH is
        // undefined → the schema default (H=0/red) is used → every colour
        // mask matched pure red regardless of the eyedropper pick.
        let raw = layer[field.name]
        if (layer.kind === 'color' && raw === undefined && layer.target && typeof layer.target === 'object') {
            if (field.name === 'targetH') raw = layer.target.h
            else if (field.name === 'targetS') raw = layer.target.s
            else if (field.name === 'targetB') raw = layer.target.b
        }
        const value = normaliseUniformValue(raw, field)
        if (field.type === 'float' && typeof value === 'number') {
            gl.uniform1f(loc, value)
        } else if (field.type === 'vec2' && Array.isArray(value) && value.length === 2) {
            gl.uniform2f(loc, value[0], value[1])
        } else if (field.type === 'vec3' && Array.isArray(value) && value.length === 3) {
            gl.uniform3f(loc, value[0], value[1], value[2])
        }
    }
}

/**
 * Write the universal COMMON uniforms for one layer: opacity (0..1),
 * inverted (0 or 1), visible (0 or 1). These are declared by
 * `buildLayerFunction` in glsl-fragments.js for every layer regardless
 * of kind.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLProgram} program
 * @param {object} layer
 * @param {number} slotIndex
 */
const writeLayerCommonUniforms = (gl, program, layer, slotIndex) => {
    const prefix = `uLayer_${slotIndex}`
    const opacityLoc = gl.getUniformLocation(program, `${prefix}_opacity`)
    if (opacityLoc) {
        const op = (typeof layer.opacity === 'number' && Number.isFinite(layer.opacity))
            ? Math.max(0, Math.min(1, layer.opacity))
            : 1
        gl.uniform1f(opacityLoc, op)
    }
    const invLoc = gl.getUniformLocation(program, `${prefix}_inverted`)
    if (invLoc) gl.uniform1f(invLoc, layer.inverted === true ? 1.0 : 0.0)
    const visLoc = gl.getUniformLocation(program, `${prefix}_visible`)
    if (visLoc) gl.uniform1f(visLoc, layer.visible === false ? 0.0 : 1.0)
}

/**
 * Step 8 — Write the per-layer image-adjustment uniforms: exposure
 * (EV stops, -3..+3), contrast (-100..+100), saturation (-100..+100),
 * brightness (-100..+100). All default to 0 so a freshly-created
 * layer's `applyLayerAdjust_<slot>(rgb)` early-outs to identity.
 *
 * Why the renderer writes 0 for missing fields rather than relying on
 * the GLSL function: every layer in the chain has its own function, so
 * every slot needs a uniform location set — leaving it unbound would
 * produce undefined behaviour on first draw.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLProgram} program
 * @param {object} layer
 * @param {number} slotIndex
 */
const writeLayerAdjustUniforms = (gl, program, layer, slotIndex) => {
    const prefix = `uLayer_${slotIndex}_adjust_`
    const set = (name, raw, lo, hi) => {
        const loc = gl.getUniformLocation(program, `${prefix}${name}`)
        if (!loc) return
        const value = (typeof raw === 'number' && Number.isFinite(raw))
            ? Math.max(lo, Math.min(hi, raw))
            : 0
        gl.uniform1f(loc, value)
    }
    set('exposure',    layer.exposure,    -3,   3)
    set('contrast',    layer.contrast,    -100, 100)
    set('saturation',  layer.saturation,  -100, 100)
    set('vibrance',    layer.vibrance,    -100, 100)
    set('brightness',  layer.brightness,  -100, 100)
    // Pro-parity tonal + white-balance adjustments (all default to 0).
    set('highlights',  layer.highlights,  -100, 100)
    set('shadows',     layer.shadows,     -100, 100)
    set('whites',      layer.whites,      -100, 100)
    set('blacks',      layer.blacks,      -100, 100)
    set('temperature', layer.temperature, -100, 100)
    set('tint',        layer.tint,        -100, 100)
    // Detail — local-contrast ops (sample the source neighbourhood in GLSL).
    set('texture',     layer.texture,     -100, 100)
    set('dehaze',      layer.dehaze,      -100, 100)

    // Gamma — per-channel power (identity 1.0); different default from the 0.0
    // fields above, so written directly rather than via `set`.
    const gammaLoc = gl.getUniformLocation(program, `${prefix}gamma`)
    if (gammaLoc) {
        const g = Number.isFinite(layer.gamma) ? Math.max(0.2, Math.min(2.2, layer.gamma)) : 1.0
        gl.uniform1f(gammaLoc, g)
    }
    // 3-way colour wheels — vec3 offsets (-1..1) on the non-adjust prefix.
    const setWheel = (name, raw) => {
        const loc = gl.getUniformLocation(program, `uLayer_${slotIndex}_${name}`)
        if (!loc) return
        const a = Array.isArray(raw) ? raw : (raw && typeof raw === 'object' ? [raw.x, raw.y, raw.z] : [])
        const cl = (v) => (Number.isFinite(v) ? Math.max(-1, Math.min(1, v)) : 0)
        gl.uniform3f(loc, cl(a[0]), cl(a[1]), cl(a[2]))
    }
    setWheel('wheel_shadows', layer.wheelShadows)
    setWheel('wheel_midtones', layer.wheelMidtones)
    setWheel('wheel_highlights', layer.wheelHighlights)
}

/**
 * Root-cause #1 — Write the per-layer fill-output uniforms: fillMode
 * (0 adjust / 1 fill / 2 erase), fillColor (vec3 0..1), fillStrength
 * (0..1). Declared by `buildLayerAdjustFunction` for every layer. A layer
 * that doesn't set these defaults to adjust mode (0) so behaviour is
 * unchanged from the pre-fillMode era.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLProgram} program
 * @param {object} layer
 * @param {number} slotIndex
 */
const writeLayerFillUniforms = (gl, program, layer, slotIndex) => {
    const prefix = `uLayer_${slotIndex}_`
    const modeLoc = gl.getUniformLocation(program, `${prefix}fillMode`)
    if (modeLoc) gl.uniform1f(modeLoc, fillModeToFloat(layer.fillMode))
    const colorLoc = gl.getUniformLocation(program, `${prefix}fillColor`)
    if (colorLoc) {
        const c = layer.fillColor || { r: 1, g: 0, b: 0.6 }
        const ch = (v, fb) => (typeof v === 'number' && Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : fb)
        gl.uniform3f(colorLoc, ch(c.r, 1), ch(c.g, 0), ch(c.b, 0.6))
    }
    const strengthLoc = gl.getUniformLocation(program, `${prefix}fillStrength`)
    if (strengthLoc) {
        const s = (typeof layer.fillStrength === 'number' && Number.isFinite(layer.fillStrength))
            ? Math.max(0, Math.min(1, layer.fillStrength))
            : 0.5
        gl.uniform1f(strengthLoc, s)
    }
}

/**
 * Write ALL uniforms for the megashader program in one go. Called by
 * `renderMegashader` after the source texture and kind textures are
 * bound. The function:
 *   1. Writes the chain-wide uniforms (`uImageSize`, `uMaskAlpha`).
 *   2. For each layer, writes the common (opacity/inverted/visible),
 *      kind-specific (luminance thresholds, radial radius, etc.), and
 *      Step 8 adjustment (exposure/contrast/saturation/brightness)
 *      uniforms.
 *   3. Binds each layer's samplers to its allocated texture unit (via
 *      `writeKindSamplers`).
 *
 * Pre-Step 8: this function was MISSING from the file — line 414 of
 * `renderMegashader` called it but it didn't exist, so any actual GL
 * draw would have thrown `writeUniforms is not defined`. The
 * pre-existing latent bug was only masked by the fact that nothing
 * actually exercised the GL path in CI. Step 8 adds the function as
 * part of the per-layer-adjustment work.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLProgram} program
 * @param {import('./mask-types').MaskStack} stack
 * @param {number} globalMaskAlpha   0..1 — UI fader on top of the chain.
 * @param {{ width: number, height: number }} imageSize
 * @param {{ kindUnits: Map<number, number> }} textureBindings
 */
const writeUniforms = (gl, program, stack, renderOpts, imageSize, textureBindings) => {
    const { globalMaskAlpha, globalInvert, maskOverlay, overlayColor } = renderOpts || {}
    // Chain-wide uniforms.
    const sizeLoc = gl.getUniformLocation(program, 'uImageSize')
    if (sizeLoc) gl.uniform2f(sizeLoc, imageSize.width, imageSize.height)
    const maskAlphaLoc = gl.getUniformLocation(program, 'uMaskAlpha')
    if (maskAlphaLoc) {
        const a = (typeof globalMaskAlpha === 'number' && Number.isFinite(globalMaskAlpha))
            ? Math.max(0, Math.min(1, globalMaskAlpha))
            : 1
        gl.uniform1f(maskAlphaLoc, a)
    }
    const invertLoc = gl.getUniformLocation(program, 'uGlobalInvert')
    if (invertLoc) gl.uniform1f(invertLoc, globalInvert ? 1.0 : 0.0)
    const overlayLoc = gl.getUniformLocation(program, 'uMaskOverlay')
    if (overlayLoc) gl.uniform1f(overlayLoc, maskOverlay ? 1.0 : 0.0)
    const overlayColLoc = gl.getUniformLocation(program, 'uMaskOverlayColor')
    if (overlayColLoc) {
        const c = overlayColor || { r: 1, g: 0, b: 0.25 }
        const ch = (v, fb) => (typeof v === 'number' && Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : fb)
        gl.uniform3f(overlayColLoc, ch(c.r, 1), ch(c.g, 0), ch(c.b, 0.25))
    }

    // Per-layer uniforms. Skip layers that aren't in the chain (the
    // shader always declares all 8 slots' adjust functions, but we only
    // need to set the active ones — the rest will read 0 from uninitialised
    // uniforms, which is fine because the chain never references them).
    if (!stack || !Array.isArray(stack.chain)) return
    for (let i = 0; i < stack.chain.length; i += 1) {
        const layer = stack.chain[i].layer
        if (!layer) continue
        writeLayerCommonUniforms(gl, program, layer, i)
        writeKindUniforms(gl, program, layer, i)
        writeLayerAdjustUniforms(gl, program, layer, i)
        writeLayerFillUniforms(gl, program, layer, i)
        writeKindSamplers(gl, program, layer, i, textureBindings)
        writeCurveUniforms(gl, program, layer, i, textureBindings)
    }
}

/**
 * Bind a cached GL texture for a mask/LUT `key` to the currently-active
 * texture unit, (re)uploading ONLY when the key's data VERSION changed since
 * the last upload. `setMaskTexture` bumps that version on every write (brush
 * stroke, boundary grow, AI mask, curve LUT, undo/redo restore), so a version
 * mismatch is the exact, complete signal that the pixels changed — a match
 * proves they are byte-identical and the cached texture is still correct.
 * `gl.isTexture` guards against a context-loss-invalidated handle (forces a
 * fresh upload). Returns true if a texture is bound to the active unit.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {string} key
 * @param {*} data            Cached texture source (already resolved by caller).
 * @param {boolean} flipY     UNPACK_FLIP_Y_WEBGL for the upload.
 * @param {GLenum} filter     gl.LINEAR or gl.NEAREST (min+mag).
 * @returns {boolean}
 */
const bindCachedMaskTexture = (gl, key, data, flipY, filter) => {
    const version = getMaskTextureVersion(key)
    const entry = maskGlTextureCache.get(key)
    if (entry && entry.version === version && gl.isTexture(entry.tex)) {
        // Reuse: pixels unchanged (version match) and the GL texture is valid.
        // Bind only — no re-upload. Refresh LRU recency (move to tail) so a
        // texture used this frame is never the eviction target.
        gl.bindTexture(gl.TEXTURE_2D, entry.tex)
        maskGlTextureCache.delete(key)
        maskGlTextureCache.set(key, entry)
        return true
    }
    // Miss or stale version → (re)upload. Delete the stale GL texture first.
    if (entry && entry.tex && gl.isTexture(entry.tex)) gl.deleteTexture(entry.tex)
    const tex = gl.createTexture()
    if (!tex) { maskGlTextureCache.delete(key); return false }
    gl.bindTexture(gl.TEXTURE_2D, tex)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, flipY)
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false)
    try {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, /** @type {any} */ (data))
    } catch {
        // texImage2D can throw on malformed data (e.g. ImageBitmap without the
        // right colour space). Drop the texture; caller binds the null texture.
        gl.deleteTexture(tex)
        maskGlTextureCache.delete(key)
        return false
    }
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    maskGlTextureCache.set(key, { tex, version })
    // Evict least-recently-used orphans beyond the cap. Never evict `key`
    // (just uploaded) or any key already bound this frame — with the cap
    // (64) far above the <=16 textures a frame can bind, the eviction front
    // only holds stale keys from earlier frames.
    if (maskGlTextureCache.size > MAX_MASK_GL_TEXTURES) {
        for (const oldKey of [...maskGlTextureCache.keys()]) {
            if (maskGlTextureCache.size <= MAX_MASK_GL_TEXTURES) break
            if (oldKey === key) continue
            const old = maskGlTextureCache.get(oldKey)
            if (old && old.tex && gl.isTexture(old.tex)) gl.deleteTexture(old.tex)
            maskGlTextureCache.delete(oldKey)
        }
    }
    return true
}

/**
 * Upload per-layer image textures (semantic masks, depth maps, future
 * kinds) and bind them to unique texture units. Returns a
 * `textureBindings` object the `writeUniforms` consumer uses to wire
 * each layer's sampler to its unit, plus an `ownedTextures` list of
 * `WebGLTexture` handles that MUST be deleted after the draw (per-frame
 * allocation — cheap, and the cache lives in mask-types.js keyed by
 * `maskTextureKey` / `depthMapKey` / `brushTextureKey`).
 *
 * Texture units:
 *   - 0 is reserved for the source image (bound by the caller)
 *   - 1..N are assigned in chain order across all texture-using kinds
 *     (semantic, smartBrush, depth) — the unit pool is shared because
 *     each layer only needs one texture, regardless of its kind
 *   - Layers whose texture is missing (cache miss) are skipped; the
 *     GLSL sampler for that layer is left at unit 0, so the shader
 *     samples the source image. That's a visible bug, but better than
 *     random colour. Callers should ensure the cache is populated
 *     before adding the layer.
 *
 * WebGL2 guarantees at least 16 texture units. With MAX_LAYERS = 8
 * and a one-texture-per-layer policy, we use at most 9 units in
 * practice. If a future kind needs more, the unit allocator will need
 * to wrap or share.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {import('./mask-types').MaskStack} stack
 * @returns {{ kindUnits: Map<number, number>, ownedTextures: WebGLTexture[] }}
 */
const bindKindTextures = (gl, stack) => {
    const ownedTextures = []
    const kindUnits = new Map()
    // Per-layer tone-curve LUT units (orthogonal to the kind mask — ANY kind can
    // carry a curve). Keyed slot → texture unit, filled after the kind loop.
    const curveUnits = new Map()
    let nextUnit = 1  // 0 is the source image
    let nullUnit = -1 // lazily-allocated 1×1 transparent texture unit

    // Bug #8: a texture-backed layer whose texture is MISSING (cache miss
    // after reload, malformed data, or no key) must NOT leave its sampler
    // bound to unit 0 — that's the source photo, and the layer would sample
    // the photo's luminance as its "mask", corrupting the whole composite.
    // Instead we bind a shared 1×1 transparent (alpha-0) texture so the
    // layer reads 0 and contributes nothing. Allocated once per draw and
    // reused for every miss.
    const ensureNullUnit = () => {
        if (nullUnit >= 0) return nullUnit
        if (nextUnit >= 16) return -1
        const tex = gl.createTexture()
        if (!tex) return -1
        const unit = nextUnit
        nextUnit += 1
        gl.activeTexture(gl.TEXTURE0 + unit)
        gl.bindTexture(gl.TEXTURE_2D, tex)
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false)
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 0]))
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
        ownedTextures.push(tex)
        nullUnit = unit
        return unit
    }

    if (!stack || !Array.isArray(stack.chain)) {
        return { kindUnits, curveUnits, ownedTextures }
    }
    for (let i = 0; i < stack.chain.length; i += 1) {
        const layer = stack.chain[i].layer
        // Only kinds that ship a per-layer image contribute a texture.
        // Each kind has its own field name on the layer object, so we
        // resolve explicitly per kind — using `||` would mix fields if
        // a future refactor renamed one (a kind shouldn't accidentally
        // pick up the wrong key). All three draw from the same module-
        // scoped `maskTextureCache`.
        let cacheKey = null
        if (layer.kind === 'semantic') {
            cacheKey = layer.maskTextureKey
        } else if (layer.kind === 'smartBrush') {
            cacheKey = layer.brushTextureKey
        } else if (layer.kind === 'depth') {
            cacheKey = layer.depthMapKey
        } else if (layer.kind === 'lasso') {
            cacheKey = layer.maskTextureKey
        } else if (layer.kind === 'brush') {
            cacheKey = layer.maskTextureKey
        } else if (layer.kind === 'path') {
            cacheKey = layer.maskTextureKey
        } else {
            continue
        }
        const data = (typeof cacheKey === 'string' && cacheKey) ? getMaskTexture(cacheKey) : undefined
        if (!data) {
            // Bug #8: missing texture → bind the shared null texture so this
            // layer reads alpha 0 instead of corrupting the composite with
            // the source photo on unit 0.
            const u = ensureNullUnit()
            if (u >= 0) kindUnits.set(i, u)
            continue
        }
        gl.activeTexture(gl.TEXTURE0 + nextUnit)
        // Persistent GL texture cache: (re)upload only when this key's data
        // VERSION changed (setMaskTexture bumps it on every write); otherwise
        // reuse the cached GL texture and just bind it. Kind textures are
        // Y-flipped (to match the source's UNPACK_FLIP_Y) and LINEAR-filtered.
        if (!bindCachedMaskTexture(gl, cacheKey, data, true, gl.LINEAR)) {
            // createTexture / texImage2D failed (malformed data) → bind the
            // shared null texture so the layer reads alpha 0 (Bug #8) rather
            // than sampling the source image on unit 0.
            const u = ensureNullUnit()
            if (u >= 0) kindUnits.set(i, u)
            continue
        }
        kindUnits.set(i, nextUnit)
        nextUnit += 1
        if (nextUnit >= 16) {
            // WebGL2 guarantees at least 16. We stop allocating rather
            // than overwrite an existing unit (each layer needs its own).
            // The next texture-using layer will silently render with a
            // default-bound sampler — visible, but the limit is well
            // past any realistic use case.
            break
        }
    }

    // ── Per-layer tone-curve LUTs ──────────────────────────────────────────
    // A layer with a non-identity curve carries `curveLutKey`; the UI builds a
    // 256×1 RGBA LUT (R/G/B + master in alpha — packLutsRgba) and registers it
    // via setMaskTexture. Upload one texture per such layer to its own unit and
    // record it so writeCurveUniforms can bind the sampler + flip curveOn on.
    for (let i = 0; i < stack.chain.length; i += 1) {
        if (nextUnit >= 16) break
        const layer = stack.chain[i].layer
        const key = layer && typeof layer.curveLutKey === 'string' ? layer.curveLutKey : null
        if (!key) continue
        const lut = getMaskTexture(key)
        if (!lut) continue
        gl.activeTexture(gl.TEXTURE0 + nextUnit)
        // Same persistent cache as the kind textures. The LUT is data (256×1):
        // no Y-flip, LINEAR so values between the 256 entries interpolate. The
        // key is the stable `curve-<id>`; setMaskTexture bumps its version each
        // time the user edits the curve, forcing a re-upload — otherwise reuse.
        if (!bindCachedMaskTexture(gl, key, lut, false, gl.LINEAR)) continue
        curveUnits.set(i, nextUnit)
        nextUnit += 1
    }

    return { kindUnits, curveUnits, ownedTextures }
}

/**
 * Lazily create the fullscreen-quad VBO + VAO used by every megashader
 * draw. Two triangles covering clip space [-1, 1]², with `aPosition`
 * pinned to attribute location 0 (see `bindAttribLocation` in
 * `getOrCreateProgram`) so the same VAO binds to any compiled program.
 *
 * Bug history: this function was REFERENCED at the draw site
 * (`const { vao } = ensureQuadBuffers(gl)`) but never defined, so the
 * very first real GL draw threw `ReferenceError: ensureQuadBuffers is
 * not defined`. That error propagated out of `applyTo`, was swallowed by
 * the try/catch in `apply-megashader.js`, and the megashader silently
 * rendered nothing — the single biggest reason "masking filters don't
 * work". The identity short-circuit hid it for all-zero-adjustment
 * stacks (which never reached the draw); any real effect crashed here.
 *
 * @param {WebGL2RenderingContext} gl
 * @returns {{ vao: WebGLVertexArrayObject | null, vbo: WebGLBuffer | null }}
 */
const ensureQuadBuffers = (gl) => {
    if (quadVao && quadVbo) return { vao: quadVao, vbo: quadVbo }
    // WebGL1 fallback contexts lack VAOs; the rest of the renderer assumes
    // WebGL2 (it calls gl.bindVertexArray unconditionally), so we guard and
    // bail to the CPU path if VAOs are unavailable.
    if (typeof gl.createVertexArray !== 'function') {
        return { vao: null, vbo: null }
    }
    // Ensure a linked program exists so attribute location 0 is valid for
    // the VAO's vertexAttribPointer call.
    getQuadProgram(gl)

    const verts = new Float32Array([
        -1, -1,
         1, -1,
        -1,  1,
        -1,  1,
         1, -1,
         1,  1,
    ])
    const vbo = gl.createBuffer()
    if (!vbo) return { vao: null, vbo: null }
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW)

    const vao = gl.createVertexArray()
    if (!vao) {
        gl.deleteBuffer(vbo)
        return { vao: null, vbo: null }
    }
    gl.bindVertexArray(vao)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    gl.bindVertexArray(null)
    gl.bindBuffer(gl.ARRAY_BUFFER, null)

    quadVbo = vbo
    quadVao = vao
    return { vao: quadVao, vbo: quadVbo }
}

/**
 * Main entry: render a `MaskStack` over a source 2D canvas and return a
 * 2D canvas containing the result. Always returns a same-sized 2D canvas
 * even if WebGL2 is unavailable (falls back to CPU passthrough).
 *
 * @param {HTMLCanvasElement | OffscreenCanvas} sourceCanvas
 * @param {import('./mask-types').MaskStack} stack
 * @param {{ globalMaskAlpha?: number }} [options]
 * @returns {HTMLCanvasElement}
 */
export const renderMegashader = (sourceCanvas, stack, options = {}) => {
    if (!sourceCanvas || typeof sourceCanvas.width !== 'number') {
        return renderCpuFallback({ width: 1, height: 1 }, stack)
    }

    const compiled = compileMegashader(stack)

    // Step 9 — Identity fast-path. When every layer's per-layer adjustments
    // are exactly 0, the GLSL chain is mathematically identity regardless
    // of how many layers are in the chain or what the boolean composition
    // produces — the colour-side math reduces to `mix(src, src, x) = src`
    // for any x. We can short-circuit and return the source canvas as-is,
    // skipping the WebGL upload / draw / readback / canvas creation.
    //
    // This was previously only triggered for the empty-chain case (the
    // `compiled.passthrough` branch). Now any chain with all-zero
    // adjustments short-circuits, which is the common case for users who
    // haven't touched the Step 8 adjustment sliders.
    //
    // We don't need to gate on globalMaskAlpha: when `runningColor` is
    // identically `srcRgb`, the final `mix(srcRgb, srcRgb, runningAlpha *
    // uMaskAlpha) = srcRgb` regardless of alpha. The mask's strength is
    // a no-op when the colour it would mix in equals the source.
    const overlayOn = options.maskOverlay === true
    if (compiled.passthrough || (!overlayOn && stackHasNoVisibleEffect(stack))) {
        // Root-cause #1: short-circuit only when the stack has NO visible
        // effect at all — no non-zero adjustment, no fill, no erase. A
        // freshly-added selection layer (fill mode) now fails this check
        // and actually renders, instead of returning the untouched source.
        // The "show mask" overlay must visualise the selection even for an
        // all-adjust-zero chain, so it bypasses this short-circuit (unless
        // the chain is genuinely empty → compiled.passthrough).
        // Step 10.3: count identity short-circuits. Distinct from
        // the `compiled.passthrough` branch (empty chain) — both
        // bypass the GL pipeline, so we count them together.
        renderMetrics.identityShortCircuits += 1
        if (typeof sourceCanvas.getContext === 'function' && sourceCanvas instanceof HTMLCanvasElement) {
            return sourceCanvas
        }
        return renderCpuFallback(sourceCanvas, stack)
    }

    const gl = ensureGl()
    if (!gl) return renderCpuFallback(sourceCanvas, stack)

    const program = getOrCreateProgram(compiled)
    if (!program) return renderCpuFallback(sourceCanvas, stack)

    const w = sourceCanvas.width
    const h = sourceCanvas.height
    if (glCanvas.width !== w) glCanvas.width = w
    if (glCanvas.height !== h) glCanvas.height = h
    gl.viewport(0, 0, w, h)

    const imageSize = { width: w, height: h }

    // Bind the source as a 2D texture.
    const texture = gl.createTexture()
    if (!texture) return renderCpuFallback(sourceCanvas, stack)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, /** @type {any} */ (sourceCanvas))
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

    gl.useProgram(program)
    const uImage = gl.getUniformLocation(program, 'uImage')
    if (uImage) gl.uniform1i(uImage, 0)

    // Identity matrix for the fullscreen quad — the megashader applies
    // pixel-space effects, not geometry transforms.
    const uMatrix = gl.getUniformLocation(program, 'uMatrix')
    if (uMatrix) {
        gl.uniformMatrix4fv(uMatrix, false, new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]))
    }

    // Upload kind-specific textures (semantic masks, depth maps etc.) and
    // capture the slot → texture-unit mapping. The textures live until the
    // gl.deleteTexture calls below.
    const { kindUnits, curveUnits, ownedTextures } = bindKindTextures(gl, stack || { chain: [] })

    writeUniforms(
        gl,
        program,
        stack || { chain: [] },
        {
            globalMaskAlpha: options.globalMaskAlpha ?? 1,
            globalInvert: options.globalInvert === true,
            maskOverlay: options.maskOverlay === true,
            overlayColor: options.overlayColor,
        },
        imageSize,
        { kindUnits, curveUnits },
    )

    // Bind quad + draw.
    const { vao } = ensureQuadBuffers(gl)
    if (vao) {
        gl.bindVertexArray(vao)
        gl.drawArrays(gl.TRIANGLES, 0, 6)
        gl.bindVertexArray(null)
    }

    // Read back to a 2D canvas. We use `gl.readPixels` rather than
    // `ctx.drawImage(glCanvas)` because the context was created with
    // `preserveDrawingBuffer: false` (for perf), and reading a WebGL canvas
    // via drawImage is undefined behaviour once the buffer has been
    // presented. readPixels is the spec-blessed way to grab framebuffer
    // contents, and we Y-flip manually because WebGL's origin is
    // bottom-left while Canvas2D's is top-left.
    const out = document.createElement('canvas')
    out.width = w
    out.height = h
    const ctx = out.getContext('2d')
    if (ctx) {
        const pixels = new Uint8Array(w * h * 4)
        try {
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
        } catch {
            gl.deleteTexture(texture)
            for (const tex of ownedTextures) gl.deleteTexture(tex)
            return renderCpuFallback(sourceCanvas, stack)
        }
        const imageData = ctx.createImageData(w, h)
        const rowBytes = w * 4
        for (let y = 0; y < h; y += 1) {
            const srcStart = (h - 1 - y) * rowBytes
            const dstStart = y * rowBytes
            imageData.data.set(pixels.subarray(srcStart, srcStart + rowBytes), dstStart)
        }
        ctx.putImageData(imageData, 0, 0)
    }

    gl.deleteTexture(texture)
    for (const tex of ownedTextures) gl.deleteTexture(tex)

    // Step 10.3: count successful draws (one per `renderMegashader`
    // call that reached the readback stage). Identity short-circuits
    // are counted separately via `identityShortCircuits`.
    renderMetrics.drawCount += 1
    return out
}

/**
 * CPU fallback path. Used when:
 *   - WebGL2 is unavailable (e.g. very old browser, or `webgl2` is
 *     blocked by the user's privacy settings);
 *   - the shader program fails to link (e.g. a new layer kind produced
 *     invalid GLSL — the renderer logs the link error and falls back);
 *   - `readPixels` throws (the GPU readback failed for some reason).
 *
 * The CPU fallback is a *passthrough*: it returns a 2D canvas that
 * contains a pixel-for-pixel copy of the source. This is intentional —
 * a real CPU-side implementation of the megashader would have to
 * re-port all 7 mask kinds in JavaScript, which is a lot of code for
 * a path that should never run on supported hardware. The user sees
 * their un-masked source image, with the megashader filter acting as
 * a no-op, which is the right behaviour for a degraded render path.
 *
 * Bug history: pre-Step 10, this function was called 6 times in
 * `renderMegashader` but never defined — any fallback path threw a
 * `ReferenceError: renderCpuFallback is not defined`. The bug was
 * latent because `hasMegashaderWebGL2` guards the typical entry point
 * and shaders usually link, so the fallback rarely fires. Step 10
 * adds the missing definition.
 *
 * @param {HTMLCanvasElement | { width: number, height: number }} source
 * @param {import('./mask-types').MaskStack} [stack]
 * @returns {HTMLCanvasElement}
 */
const renderCpuFallback = (source, stack) => {
    if (typeof document === 'undefined') {
        // Non-browser environment: return a minimal stub so callers
        // (tests, SSR) that reach this path get a non-null canvas. The
        // dimensions come from the `source` argument — either the real
        // canvas's width/height, or the synthetic `{ width, height }`
        // shape used in some error branches.
        const out = { width: 1, height: 1, getContext: () => null }
        return /** @type {HTMLCanvasElement} */ (out)
    }
    const out = document.createElement('canvas')
    if (source && typeof source.width === 'number' && typeof source.height === 'number') {
        out.width = source.width
        out.height = source.height
    } else {
        out.width = 1
        out.height = 1
        return out
    }
    // If `source` is a real canvas, drawImage it. Otherwise we have
    // no source pixels to copy, so the fallback canvas is blank.
    if (typeof source.getContext === 'function' && source instanceof HTMLCanvasElement) {
        const ctx = out.getContext('2d')
        if (ctx) {
            try {
                ctx.drawImage(source, 0, 0)
            } catch {
                // drawImage can throw for cross-origin canvases. The
                // best we can do is return a blank canvas; the caller
                // still gets a non-null result.
            }
        }
    }
    return out
}

/**
 * Reset all caches. Called by the Fabric filter on dispose and by tests.
 * Deletes every cached program, the quad program, the quad VBO/VAO, and
 * drops the GL context reference. The next `renderMegashader` call will
 * recreate everything from scratch.
 */
export const disposeRenderer = () => {
    const gl = glContext
    if (gl) {
        for (const program of programCache.values()) gl.deleteProgram(program)
        if (quadProgram) gl.deleteProgram(quadProgram)
        if (quadVbo) gl.deleteBuffer(quadVbo)
        if (quadVao) gl.deleteVertexArray(quadVao)
        for (const entry of maskGlTextureCache.values()) {
            if (entry && entry.tex && gl.isTexture(entry.tex)) gl.deleteTexture(entry.tex)
        }
    }
    maskGlTextureCache = new Map()
    programCache.clear()
    quadProgram = null
    quadVbo = null
    quadVao = null
    glContext = null
    glCanvas = null
}

// Step 10.3: re-export the metrics helpers so the test panel can
// read and reset them. Re-export (not just `export`) so a future
// barrel change doesn't accidentally shadow them.
export { getRenderMetrics, resetRenderMetrics }
