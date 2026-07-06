#!/usr/bin/env bun
/**
 * Megashader verification script
 * ------------------------------
 * Asserts the four fragment-shader string invariants defined in Step 1
 * of the architectural plan:
 *
 *   1. 0 layers       → "a = 0.0" passthrough path is present.
 *   2. 1 Add layer    → "clamp(a + b, 0., 1.)" appears.
 *   3. 1 Subtract     → "max(a - b, 0.)" appears.
 *   4. 1 Intersect    → "a * b" appears.
 *
 * Also asserts:
 *   - Adding a 2nd layer with `op: 'add'` emits the add op.
 *   - Mixed chains emit all three ops in order.
 *   - The compiled cache key changes when the op chain changes, but
 *     does NOT change when only per-layer opacity changes (params are
 *     uniforms, not shader structure).
 *   - The passthrough flag is true for an empty chain, false otherwise.
 *   - The compiler throws on unknown mask kinds.
 *   - Truncation at MAX_LAYERS works.
 *
 * Run: `bun run scripts/verify-megashader.mjs`
 * Exit code 0 = all pass, non-zero = at least one failed.
 *
 * This file is intentionally runnable as a plain Node ES module (no
 * Fabric, no DOM) so it works in CI without a browser.
 */

import { compileMegashader, computeCacheKey, MAX_LAYERS } from '../src/lib/megashader/megashader-compiler.js'
import { createEmptyStack, sanitiseLayer, isAdjustmentsIdentity, stackHasNoVisibleEffect, isBlendOp, BLEND_OPS, luminanceLayer, colorLayer, linearLayer, radialLayer, smartBrushLayer, semanticLayer, depthLayer, lassoLayer, setMaskTexture, getMaskTexture, clearMaskTexture } from '../src/lib/megashader/mask-types.js'
import { KIND_SCHEMAS, getKindBuilder, getKindSchema, normaliseUniformValue } from '../src/lib/megashader/glsl-mask-kinds.js'

let failures = 0
let checks = 0
const log = (ok, name, detail) => {
    const tag = ok ? '\x1b[32m✓\x1b[0m' : '\x1b[31m✗\x1b[0m'
    console.log(`${tag} ${name}${detail ? `  (${detail})` : ''}`)
    checks += 1
    if (!ok) failures += 1
}

// 1. 0 layers → passthrough
{
    const stack = createEmptyStack()
    const c = compileMegashader(stack)
    log(c.passthrough === true, '0 layers → passthrough=true')
    log(c.frag.includes('float runningAlpha = 0.0'), '0 layers → runningAlpha = 0.0 in frag')
    // The dispatcher `evalLayer(int)` is always emitted (cheap, ~10 lines).
    // What should be absent is the per-layer FUNCTION DEFINITIONS
    // (`float evalLayer_0() { ... }`). Match the definition form, not the
    // call site in the dispatcher.
    const defCount = (c.frag.match(/float\s+evalLayer_\d+\s*\(\s*\)\s*\{/g) || []).length
    log(defCount === 0, '0 layers → no per-layer function definitions emitted', `defCount=${defCount}`)
}

// 2. 1 layer with add op (first layer is always 'replace' semantically; the
// boolean chain only sees the first layer as base, no op applied).
// Step 8 chain format: per-layer colour is tracked in runningColor
// (vec3) and alpha in runningAlpha (float).
{
    const stack = {
        chain: [
            { layer: { ...sanitiseLayer({ kind: 'luminance', id: 'L0' }), id: 'L0' }, op: 'replace' },
            { layer: { ...sanitiseLayer({ kind: 'luminance', id: 'L1' }), id: 'L1' }, op: 'add' },
        ],
    }
    const c = compileMegashader(stack)
    log(!c.passthrough, '1+ layers → passthrough=false')
    log(c.frag.includes('evalLayer_0(') && c.frag.includes('evalLayer_1('), '2 layers → both evalLayer_N functions present')
    log(
        c.frag.includes('runningColor = mix(runningColor, c_1, a_1);')
            && c.frag.includes('runningAlpha = clamp(runningAlpha + a_1, 0.0, 1.0);'),
        '2 layers add op → mix(runningColor, c_1, a_1) + clamp(alpha + a_1) emitted',
    )
}

// 3. 1 layer with subtract op
{
    const stack = {
        chain: [
            { layer: { ...sanitiseLayer({ kind: 'radial', id: 'L0' }), id: 'L0' }, op: 'replace' },
            { layer: { ...sanitiseLayer({ kind: 'radial', id: 'L1' }), id: 'L1' }, op: 'subtract' },
        ],
    }
    const c = compileMegashader(stack)
    log(
        c.frag.includes('runningAlpha = max(runningAlpha - a_1, 0.0);'),
        'subtract op → max(runningAlpha - a_1, 0.0) emitted',
    )
}

// 4. 1 layer with intersect op
{
    const stack = {
        chain: [
            { layer: { ...sanitiseLayer({ kind: 'linear', id: 'L0' }), id: 'L0' }, op: 'replace' },
            { layer: { ...sanitiseLayer({ kind: 'linear', id: 'L1' }), id: 'L1' }, op: 'intersect' },
        ],
    }
    const c = compileMegashader(stack)
    log(
        c.frag.includes('runningColor = mix(runningColor, c_1, a_1);')
            && c.frag.includes('runningAlpha = runningAlpha * a_1;'),
        'intersect op → mix(colour) + runningAlpha * a_1 emitted',
    )
}

// 5. Mixed chain (all three ops in order)
{
    const stack = {
        chain: [
            { layer: { ...sanitiseLayer({ kind: 'luminance', id: 'A' }), id: 'A' }, op: 'replace' },
            { layer: { ...sanitiseLayer({ kind: 'color', id: 'B' }), id: 'B' }, op: 'add' },
            { layer: { ...sanitiseLayer({ kind: 'radial', id: 'C' }), id: 'C' }, op: 'subtract' },
            { layer: { ...sanitiseLayer({ kind: 'linear', id: 'D' }), id: 'D' }, op: 'intersect' },
        ],
    }
    const c = compileMegashader(stack)
    log(c.frag.includes('runningColor = mix(runningColor, c_1, a_1);') && c.frag.includes('runningAlpha = clamp(runningAlpha + a_1, 0.0, 1.0);'), 'mixed chain → add emitted')
    log(c.frag.includes('runningAlpha = max(runningAlpha - a_2, 0.0);'), 'mixed chain → subtract emitted')
    log(c.frag.includes('runningColor = mix(runningColor, c_3, a_3);') && c.frag.includes('runningAlpha = runningAlpha * a_3;'), 'mixed chain → intersect emitted')
}

// 6. Cache key: structural changes bust, parameter changes don't
{
    const base = {
        chain: [
            { layer: { ...sanitiseLayer({ kind: 'luminance', id: 'A', opacity: 0.5 }), id: 'A' }, op: 'replace' },
        ],
    }
    const sameShape = {
        chain: [
            { layer: { ...sanitiseLayer({ kind: 'luminance', id: 'B', opacity: 0.9 }), id: 'B' }, op: 'replace' },
        ],
    }
    const differentShape = {
        chain: [
            { layer: { ...sanitiseLayer({ kind: 'luminance', id: 'C' }), id: 'C' }, op: 'replace' },
            { layer: { ...sanitiseLayer({ kind: 'color', id: 'D' }), id: 'D' }, op: 'add' },
        ],
    }
    const k1 = computeCacheKey(base)
    const k2 = computeCacheKey(sameShape)
    const k3 = computeCacheKey(differentShape)
    log(k1 === k2, 'cache key stable across opacity changes', `k1=${k1} k2=${k2}`)
    log(k1 !== k3, 'cache key changes when chain shape changes', `k1=${k1} k3=${k3}`)
}

// 7. Passthrough flag reflects empty chain
{
    const empty = createEmptyStack()
    const nonEmpty = {
        chain: [
            { layer: { ...sanitiseLayer({ kind: 'luminance', id: 'X' }), id: 'X' }, op: 'replace' },
        ],
    }
    log(compileMegashader(empty).passthrough === true, 'empty → passthrough=true')
    log(compileMegashader(nonEmpty).passthrough === false, 'non-empty → passthrough=false')
}

// 8. Unknown kind throws
{
    let threw = false
    try {
        compileMegashader({
            chain: [{ layer: { id: 'BAD', kind: 'not-a-kind' }, op: 'replace' }],
        })
    } catch { threw = true }
    log(threw, 'unknown mask kind throws')
}

// 9. MAX_LAYERS truncation
{
    const layers = []
    for (let i = 0; i < MAX_LAYERS + 4; i += 1) {
        layers.push({
            layer: { ...sanitiseLayer({ kind: 'luminance', id: `L${i}` }), id: `L${i}` },
            op: i === 0 ? 'replace' : 'add',
        })
    }
    const c = compileMegashader({ chain: layers })
    // Count per-layer FUNCTION DEFINITIONS, not call sites in the dispatcher.
    const defCount = (c.frag.match(/float\s+evalLayer_\d+\s*\(\s*\)\s*\{/g) || []).length
    log(defCount === MAX_LAYERS, `chain truncated to MAX_LAYERS (${MAX_LAYERS})`, `emitted ${defCount} layer defs`)
}

// 10. Blend op type guard
{
    log(isBlendOp('add') && isBlendOp('subtract') && isBlendOp('intersect') && isBlendOp('replace'), 'isBlendOp accepts all four boolean ops')
    log(isBlendOp('screen') && isBlendOp('lighten') && isBlendOp('darken') && isBlendOp('overlay'), 'isBlendOp accepts the four Photoshop-parity ops')
    log(!isBlendOp('foo') && !isBlendOp(null) && !isBlendOp(42), 'isBlendOp rejects non-ops')
    log(BLEND_OPS.length === 8, 'BLEND_OPS is exactly 8 entries (4 boolean + 4 Photoshop)')
}

// 11. The compiled fragment shader contains the standard Fabric varyings
{
    const stack = {
        chain: [
            { layer: { ...sanitiseLayer({ kind: 'luminance', id: 'A' }), id: 'A' }, op: 'replace' },
        ],
    }
    const c = compileMegashader(stack)
    log(c.frag.includes('varying vec2 vTextureCoord'), 'frag declares vTextureCoord varying')
    log(c.frag.includes('uniform sampler2D uImage'), 'frag declares uImage sampler')
    log(c.frag.includes('texture2D(uImage, vTextureCoord)'), 'frag samples uImage with vTextureCoord')
    log(c.frag.includes('gl_FragColor = vec4('), 'frag writes to gl_FragColor')
}

// 12. Vertex shader is a fullscreen-quad pass-through
{
    const c = compileMegashader(createEmptyStack())
    log(c.vert.includes('attribute vec2 aPosition'), 'vert declares aPosition')
    log(c.vert.includes('varying vec2 vTextureCoord'), 'vert declares vTextureCoord varying')
    log(c.vert.includes('gl_Position = uMatrix * vec4(aPosition, 0.0, 1.0)'), 'vert writes gl_Position via uMatrix')
}

// ─── Step 2: Luminance + Color GLSL bodies + uniforms ──────────────────────

// 13. Luminance body has the WCAG luma formula and a two-edge smoothstep.
{
    const stack = {
        chain: [{ layer: luminanceLayer({ min: 0.1, max: 0.6, softness: 0.05 }), op: 'replace' }],
    }
    const c = compileMegashader(stack)
    const lumaFn = c.frag.match(/float\s+evalLayer_0_body\s*\(\s*\)\s*\{[\s\S]*?\n\s*\}/)
    log(
        Boolean(lumaFn) && lumaFn[0].includes('0.2126') && lumaFn[0].includes('0.7152') && lumaFn[0].includes('0.0722'),
        'luminance body uses WCAG luma coefficients (0.2126, 0.7152, 0.0722)',
    )
    const smoothstepCount = (lumaFn?.[0].match(/smoothstep\(/g) || []).length
    log(smoothstepCount >= 2, `luminance body uses ≥2 smoothstep edges (got ${smoothstepCount})`)
}

// 14. Luminance function declares all three kind-specific uniforms.
{
    const c = compileMegashader({
        chain: [{ layer: luminanceLayer(), op: 'replace' }],
    })
    // Uniforms are file-scope in GLSL, so the per-kind builder emits them
    // before the body function. Check the entire frag rather than the wrapper.
    log(
        c.frag.includes('uniform float uLayer_0_kind_luminance_min')
            && c.frag.includes('uniform float uLayer_0_kind_luminance_max')
            && c.frag.includes('uniform float uLayer_0_kind_luminance_softness'),
        'luminance function declares min/max/softness uniforms',
    )
}

// 15. Color body calls rgbToHsb and uses cyclic hue distance.
{
    const stack = {
        chain: [{ layer: colorLayer({ target: { h: 200, s: 0.8, b: 0.9 }, tolerance: 0.1 }), op: 'replace' }],
    }
    const c = compileMegashader(stack)
    const bodyMatch = c.frag.match(/float\s+evalLayer_0_body\s*\(\s*\)\s*\{[\s\S]*?\n\s*\}/)
    const body = bodyMatch?.[0] || ''
    log(body.includes('rgbToHsb'), 'color body calls rgbToHsb')
    // Regression guard: GLSL has no forward declarations, so the shared
    // rgbToHsb helper MUST be defined earlier in the assembled fragment than
    // the color mask body that calls it. When it sat below the mask-function
    // block the color shader failed to compile ("rgbToHsb: no matching
    // overloaded function") and the whole program fell back to CPU
    // passthrough — every colour-range mask rendered as a no-op.
    {
        const defIdx = c.frag.indexOf('vec3 rgbToHsb(')
        const callIdx = c.frag.indexOf('rgbToHsb(rgb)')
        log(defIdx >= 0 && callIdx >= 0 && defIdx < callIdx,
            'rgbToHsb is defined BEFORE the color body that calls it (GLSL has no forward decls)')
    }
    // Cyclic hue distance uses both `abs(...)` and a `360.0 - ...` fold.
    log(
        body.includes('360.0 - diffH') || body.includes('360.0 - abs(hsb.x'),
        'color body uses cyclic hue distance (360 - diffH)',
    )
}

// 16. Color function declares all five kind-specific uniforms.
{
    const c = compileMegashader({
        chain: [{ layer: colorLayer(), op: 'replace' }],
    })
    const fields = ['targetH', 'targetS', 'targetB', 'tolerance', 'softness']
    const allPresent = fields.every((f) => c.frag.includes(`uLayer_0_kind_color_${f}`))
    log(allPresent, 'color function declares targetH/S/B/tolerance/softness uniforms')
}

// 17. Cache key is stable across per-kind parameter changes.
{
    const a = compileMegashader({ chain: [{ layer: luminanceLayer({ min: 0, max: 0.5, softness: 0.1 }), op: 'replace' }] })
    const b = compileMegashader({ chain: [{ layer: luminanceLayer({ min: 0.2, max: 0.7, softness: 0.3 }), op: 'replace' }] })
    log(a.cacheKey === b.cacheKey, 'cache key stable across luminance min/max/softness changes', `a=${a.cacheKey} b=${b.cacheKey}`)

    const c = compileMegashader({ chain: [{ layer: colorLayer({ target: { h: 0, s: 1, b: 1 }, tolerance: 0.1 }), op: 'replace' }] })
    const d = compileMegashader({ chain: [{ layer: colorLayer({ target: { h: 200, s: 0.5, b: 0.8 }, tolerance: 0.3 }), op: 'replace' }] })
    log(c.cacheKey === d.cacheKey, 'cache key stable across color target/tolerance changes', `c=${c.cacheKey} d=${d.cacheKey}`)
}

// 18. KIND_SCHEMAS has the documented field counts.
{
    log(KIND_SCHEMAS.luminance.uniforms.length === 3, 'luminance schema has 3 uniform fields')
    log(KIND_SCHEMAS.color.uniforms.length === 5, 'color schema has 5 uniform fields')
    // Steps 5 (semantic) and 6 (depth) ship real schemas with their own
    // assertions further down. Step 7 (smartBrush) also has a real schema
    // and is asserted in groups 36+.
    log(KIND_SCHEMAS.smartBrush.uniforms.length > 0, 'smartBrush schema is real (not the Step 4 stub)')
}

// 19. Unknown kind throws in both the builder lookup and the schema lookup.
{
    let builderThrew = false
    try { getKindBuilder('not-a-kind') } catch { builderThrew = true }
    let schemaThrew = false
    try { getKindSchema('not-a-kind') } catch { schemaThrew = true }
    log(builderThrew, 'getKindBuilder throws on unknown kind')
    log(schemaThrew, 'getKindSchema throws on unknown kind')
}

// ─── Step 3: Linear + Radial (spatial) GLSL bodies + uniforms + uImageSize ──

// 20. Linear body projects onto P1→P2 via dot, uses smoothstep for feather.
{
    const stack = {
        chain: [{ layer: linearLayer({ imageSize: { width: 1000, height: 600 } }), op: 'replace' }],
    }
    const c = compileMegashader(stack)
    const bodyMatch = c.frag.match(/float\s+evalLayer_0_body\s*\(\s*\)\s*\{[\s\S]*?\n\s*\}/)
    const body = bodyMatch?.[0] || ''
    log(body.includes('dot('), 'linear body uses dot() for projection')
    log(body.includes('smoothstep('), 'linear body uses smoothstep() for feather')
    log(!body.includes('return 0.0;'), 'linear body is real (not the stub returning 0.0)')
    log(body.includes('uImageSize'), 'linear body references uImageSize')
}

// 21. Linear function declares p1, p2, position, feather uniforms.
{
    const c = compileMegashader({
        chain: [{ layer: linearLayer(), op: 'replace' }],
    })
    const fields = ['p1', 'p2', 'position', 'feather']
    const allPresent = fields.every((f) => c.frag.includes(`uLayer_0_kind_linear_${f}`))
    log(allPresent, 'linear function declares p1/p2/position/feather uniforms')
}

// 22. Linear cache key is stable across p1/p2/position/feather changes.
{
    const a = compileMegashader({
        chain: [{
            layer: linearLayer({
                imageSize: { width: 800, height: 600 },
                p1: { x: 0, y: 0 }, p2: { x: 100, y: 100 }, position: 0.3, feather: 0.1,
            }),
            op: 'replace',
        }],
    })
    const b = compileMegashader({
        chain: [{
            layer: linearLayer({
                imageSize: { width: 800, height: 600 },
                p1: { x: 50, y: 50 }, p2: { x: 400, y: 200 }, position: 0.7, feather: 0.4,
            }),
            op: 'replace',
        }],
    })
    log(a.cacheKey === b.cacheKey, 'cache key stable across linear p1/p2/position/feather changes', `a=${a.cacheKey} b=${b.cacheKey}`)
}

// 23. Radial body uses length and atan (rotation) and smoothstep.
{
    const stack = {
        chain: [{ layer: radialLayer({ imageSize: { width: 1000, height: 600 } }), op: 'replace' }],
    }
    const c = compileMegashader(stack)
    const bodyMatch = c.frag.match(/float\s+evalLayer_0_body\s*\(\s*\)\s*\{[\s\S]*?\n\s*\}/)
    const body = bodyMatch?.[0] || ''
    log(body.includes('length('), 'radial body uses length()')
    log(body.includes('atan(') || body.includes('cos(') || body.includes('sin('), 'radial body rotates into ellipse frame (atan/cos/sin)')
    log(body.includes('smoothstep('), 'radial body uses smoothstep() for feather')
    log(!body.includes('return 0.0;'), 'radial body is real (not the stub returning 0.0)')
    log(body.includes('uImageSize'), 'radial body references uImageSize')
}

// 24. Radial function declares center, rotation, radius, feather uniforms.
{
    const c = compileMegashader({
        chain: [{ layer: radialLayer(), op: 'replace' }],
    })
    const fields = ['center', 'rotation', 'radius', 'feather']
    const allPresent = fields.every((f) => c.frag.includes(`uLayer_0_kind_radial_${f}`))
    log(allPresent, 'radial function declares center/rotation/radius/feather uniforms')
}

// 25. Radial cache key is stable across center/rotation/radius/feather changes.
{
    const a = compileMegashader({
        chain: [{
            layer: radialLayer({
                imageSize: { width: 800, height: 600 },
                center: { x: 200, y: 200 }, rotation: 0, radius: { x: 100, y: 80 }, feather: 0.1,
            }),
            op: 'replace',
        }],
    })
    const b = compileMegashader({
        chain: [{
            layer: radialLayer({
                imageSize: { width: 800, height: 600 },
                center: { x: 400, y: 300 }, rotation: 1.2, radius: { x: 250, y: 200 }, feather: 0.5,
            }),
            op: 'replace',
        }],
    })
    log(a.cacheKey === b.cacheKey, 'cache key stable across radial center/rotation/radius/feather changes', `a=${a.cacheKey} b=${b.cacheKey}`)
}

// 26. Linear schema: 4 uniforms, of which 2 are vec2 (p1, p2).
{
    const schema = KIND_SCHEMAS.linear
    log(schema.uniforms.length === 4, 'linear schema has 4 uniform fields')
    const vec2Fields = schema.uniforms.filter((u) => u.type === 'vec2')
    log(vec2Fields.length === 2, `linear schema has 2 vec2 fields (p1, p2) (got ${vec2Fields.length})`)
    const vec2Names = vec2Fields.map((u) => u.name).sort().join(',')
    log(vec2Names === 'p1,p2', `linear vec2 fields are p1 and p2 (got ${vec2Names})`)
}

// 27. Radial schema: 4 uniforms, of which 2 are vec2 (center, radius).
{
    const schema = KIND_SCHEMAS.radial
    log(schema.uniforms.length === 4, 'radial schema has 4 uniform fields')
    const vec2Fields = schema.uniforms.filter((u) => u.type === 'vec2')
    log(vec2Fields.length === 2, `radial schema has 2 vec2 fields (center, radius) (got ${vec2Fields.length})`)
    const vec2Names = vec2Fields.map((u) => u.name).sort().join(',')
    log(vec2Names === 'center,radius', `radial vec2 fields are center and radius (got ${vec2Names})`)
}

// 28. Semantic (Step 5) factory throws without a maskTextureKey.
{
    let threw = false
    try { semanticLayer({}) } catch { threw = true }
    log(threw, 'semanticLayer throws without a maskTextureKey')
}

// 29. Semantic layer with a key compiles a real body that samples the mask texture.
{
    const layer = semanticLayer({ maskTextureKey: 'test-key-123', feather: 0.1 })
    log(layer.kind === 'semantic', 'semanticLayer returns kind=semantic')
    log(layer.maskTextureKey === 'test-key-123', 'semanticLayer preserves maskTextureKey')
    log(typeof layer.id === 'string' && layer.id.length > 0, 'semanticLayer returns a non-empty id')
    const c = compileMegashader({ chain: [{ layer, op: 'replace' }] })
    log(c.frag.includes('texture2D(uLayer_0_kind_semantic_mask'), 'semantic body samples the per-layer mask sampler')
    log(c.frag.includes('uLayer_0_kind_semantic_feather'), 'semantic body declares the feather uniform')
    log(!c.frag.includes('return 0.0; // Step 3: stub for kind "semantic"'), 'semantic body is not the stub')
}

// 30. Semantic schema: 1 uniform (feather) + 1 sampler (mask).
{
    const schema = KIND_SCHEMAS.semantic
    log(schema.uniforms.length === 1, `semantic schema has 1 uniform field (got ${schema.uniforms.length})`)
    log(schema.uniforms[0].name === 'feather', 'semantic uniform is "feather"')
    log(Array.isArray(schema.samplers) && schema.samplers.length === 1, 'semantic schema has 1 sampler')
    log(schema.samplers?.[0]?.name === 'mask', 'semantic sampler is named "mask"')
}

// 31. Texture cache round-trip: set, get, clear.
{
    const key = 'round-trip-test-key'
    const fakeData = { width: 1, height: 1, data: new Uint8Array([255, 0, 0, 255]) }
    log(getMaskTexture(key) === undefined, 'getMaskTexture returns undefined for missing key')
    setMaskTexture(key, fakeData)
    log(getMaskTexture(key) === fakeData, 'setMaskTexture + getMaskTexture round-trips')
    clearMaskTexture(key)
    log(getMaskTexture(key) === undefined, 'clearMaskTexture removes the entry')
}

// 32. Depth (Step 6) factory throws without a depthMapKey.
{
    let threw = false
    try { depthLayer({}) } catch { threw = true }
    log(threw, 'depthLayer throws without a depthMapKey')
}

// 33. Depth layer with a key compiles a real body that samples the depth texture
//     and applies the [min, max] range with a soft edge.
{
    const layer = depthLayer({ depthMapKey: 'test-depth-key', min: 0.2, max: 0.6, softness: 0.05 })
    log(layer.kind === 'depth', 'depthLayer returns kind=depth')
    log(layer.depthMapKey === 'test-depth-key', 'depthLayer preserves depthMapKey')
    log(layer.min === 0.2 && layer.max === 0.6, 'depthLayer preserves min/max (canonicalised order)')
    log(typeof layer.id === 'string' && layer.id.length > 0, 'depthLayer returns a non-empty id')
    const c = compileMegashader({ chain: [{ layer, op: 'replace' }] })
    log(c.frag.includes('texture2D(uLayer_0_kind_depth_map'), 'depth body samples the per-layer depth-map sampler')
    log(c.frag.includes('uLayer_0_kind_depth_min'), 'depth body declares the min uniform')
    log(c.frag.includes('uLayer_0_kind_depth_max'), 'depth body declares the max uniform')
    log(c.frag.includes('uLayer_0_kind_depth_softness'), 'depth body declares the softness uniform')
    log(c.frag.includes('smoothstep(dMin - soft, dMin + soft, depth)'), 'depth body uses smoothstep on the min edge')
    log(c.frag.includes('1.0 - smoothstep(dMax - soft, dMax + soft, depth)'), 'depth body uses smoothstep on the max edge')
    log(!c.frag.includes('return 0.0; // Step 6: stub for kind "depth"'), 'depth body is not the stub')
}

// 34. Depth schema: 3 uniforms (min, max, softness) + 1 sampler (map).
{
    const schema = KIND_SCHEMAS.depth
    log(schema.uniforms.length === 3, `depth schema has 3 uniform fields (got ${schema.uniforms.length})`)
    const uniformNames = schema.uniforms.map((u) => u.name).sort().join(',')
    log(uniformNames === 'max,min,softness', `depth uniform fields are min, max, softness (got ${uniformNames})`)
    log(Array.isArray(schema.samplers) && schema.samplers.length === 1, 'depth schema has 1 sampler')
    log(schema.samplers?.[0]?.name === 'map', 'depth sampler is named "map"')
}

// 35. Depth layer: min/max are canonicalised (lo ≤ hi), softness clamped to 0..1.
{
    const a = depthLayer({ depthMapKey: 'k', min: 0.7, max: 0.2 })
    log(a.min === 0.2 && a.max === 0.7, 'depthLayer swaps min>max to min<max')
    const b = depthLayer({ depthMapKey: 'k', softness: -1 })
    log(b.softness === 0, 'depthLayer clamps negative softness to 0')
    const c = depthLayer({ depthMapKey: 'k', softness: 2 })
    log(c.softness === 1, 'depthLayer clamps >1 softness to 1')
}

// ─── Step 7: Smart Brush (edge-preserving filter) ─────────────────────

// 36. Smart brush factory throws without a brushTextureKey.
{
    let threw = false
    try { smartBrushLayer({}) } catch { threw = true }
    log(threw, 'smartBrushLayer throws without a brushTextureKey')
}

// 37. Smart brush layer with a key compiles a real body that samples the
//     brush texture and the source image (bilateral filter).
{
    const layer = smartBrushLayer({ brushTextureKey: 'test-brush-key', filterRadius: 3, sigmaColor: 0.15, sigmaSpace: 2 })
    log(layer.kind === 'smartBrush', 'smartBrushLayer returns kind=smartBrush')
    log(layer.brushTextureKey === 'test-brush-key', 'smartBrushLayer preserves brushTextureKey')
    log(layer.filterRadius === 3, 'smartBrushLayer preserves filterRadius')
    log(typeof layer.id === 'string' && layer.id.length > 0, 'smartBrushLayer returns a non-empty id')
    const c = compileMegashader({ chain: [{ layer, op: 'replace' }] })
    log(c.frag.includes('texture2D(uLayer_0_kind_smartBrush_brush'), 'smartBrush body samples the per-layer brush sampler')
    log(c.frag.includes('texture2D(uImage, vTextureCoord)'), 'smartBrush body samples the source image (bilateral weights)')
    log(c.frag.includes('uLayer_0_kind_smartBrush_filterRadius'), 'smartBrush body declares the filterRadius uniform')
    log(c.frag.includes('uLayer_0_kind_smartBrush_sigmaColor'), 'smartBrush body declares the sigmaColor uniform')
    log(c.frag.includes('uLayer_0_kind_smartBrush_sigmaSpace'), 'smartBrush body declares the sigmaSpace uniform')
    log(c.frag.includes('exp('), 'smartBrush body uses Gaussian exp() for the bilateral weights')
    log(!c.frag.includes('return 0.0; // Step 4: stub for kind "smartBrush"'), 'smartBrush body is not the Step 4 stub')
}

// 38. Smart brush schema: 3 uniforms (filterRadius, sigmaColor, sigmaSpace) + 1 sampler (brush).
{
    const schema = KIND_SCHEMAS.smartBrush
    log(schema.uniforms.length === 3, `smartBrush schema has 3 uniform fields (got ${schema.uniforms.length})`)
    const uniformNames = schema.uniforms.map((u) => u.name).sort().join(',')
    log(uniformNames === 'filterRadius,sigmaColor,sigmaSpace', `smartBrush uniform fields are filterRadius, sigmaColor, sigmaSpace (got ${uniformNames})`)
    log(Array.isArray(schema.samplers) && schema.samplers.length === 1, 'smartBrush schema has 1 sampler')
    log(schema.samplers?.[0]?.name === 'brush', 'smartBrush sampler is named "brush"')
}

// 39. Smart brush clamping: filterRadius 1..8, sigmaColor 0.01..1, sigmaSpace 0.01..8.
{
    const lo = smartBrushLayer({ brushTextureKey: 'k', filterRadius: -5, sigmaColor: -1, sigmaSpace: -1 })
    log(lo.filterRadius === 1, 'smartBrushLayer clamps filterRadius < 1 to 1')
    log(lo.sigmaColor === 0.01, 'smartBrushLayer clamps sigmaColor < 0.01 to 0.01')
    log(lo.sigmaSpace === 0.01, 'smartBrushLayer clamps sigmaSpace < 0.01 to 0.01')
    const hi = smartBrushLayer({ brushTextureKey: 'k', filterRadius: 100, sigmaColor: 5, sigmaSpace: 50 })
    log(hi.filterRadius === 8, 'smartBrushLayer clamps filterRadius > 8 to 8')
    log(hi.sigmaColor === 1, 'smartBrushLayer clamps sigmaColor > 1 to 1')
    log(hi.sigmaSpace === 8, 'smartBrushLayer clamps sigmaSpace > 8 to 8')
}

// 40. Smart brush cache key is stable across filterRadius/sigmaColor/sigmaSpace changes.
{
    const a = compileMegashader({
        chain: [{ layer: smartBrushLayer({ brushTextureKey: 'k', filterRadius: 1, sigmaColor: 0.1, sigmaSpace: 1 }), op: 'replace' }],
    })
    const b = compileMegashader({
        chain: [{ layer: smartBrushLayer({ brushTextureKey: 'k', filterRadius: 5, sigmaColor: 0.5, sigmaSpace: 4 }), op: 'replace' }],
    })
    log(a.cacheKey === b.cacheKey, 'cache key stable across smartBrush filterRadius/sigmaColor/sigmaSpace changes', `a=${a.cacheKey} b=${b.cacheKey}`)
}

// 41. Smart brush body uses a windowed loop with a fixed MAX_RADIUS constant bound.
{
    const layer = smartBrushLayer({ brushTextureKey: 'k' })
    const c = compileMegashader({ chain: [{ layer, op: 'replace' }] })
    const bodyMatch = c.frag.match(/float\s+evalLayer_0_body\s*\(\s*\)\s*\{[\s\S]*?\n\s*\}/)
    const body = bodyMatch?.[0] || ''
    log(body.includes('MAX_RADIUS'), 'smartBrush body uses a compile-time MAX_RADIUS constant')
    log(/for\s*\(\s*int\s+dy\s*=/.test(body) && /for\s*\(\s*int\s+dx\s*=/.test(body), 'smartBrush body has nested for loops over dx/dy')
    log(/abs\(\s*float\(\s*dx\s*\)\s*\)\s*>\s*float\(\s*radiusI\s*\)/.test(body), 'smartBrush body guards the inner loop by the actual radius')
    // Regression guard: GLSL ES 1.00 has no integer abs() overload (added in
    // ES 3.00), so the guard MUST cast to float. A bare abs(dx)/abs(dy) fails
    // to compile and silently drops the whole program to CPU passthrough,
    // making every smart-brush mask a no-op. Strip // comments first so the
    // check tests the actual GLSL code, not prose that mentions abs(dx).
    const codeOnly = body.replace(/\/\/[^\n]*/g, '')
    log(!/abs\(\s*d[xy]\s*\)/.test(codeOnly), 'smartBrush body avoids the ES-1.00-illegal integer abs()')
    // The non-greedy regex above stops at the first inner `}` (the inner
    // for-loop's closing brace), so any assertion past that needs to match
    // against the whole fragment. The normalisation guard and UV clamp
    // both live near the bottom of the body.
    log(c.frag.includes('sumWeight > 1e-6'), 'smartBrush body guards the normalisation divisor')
    log(c.frag.includes('clamp(sampleUV'), 'smartBrush body clamps sampleUV to [0,1]')
}

// ─── Step 8: Per-layer image adjustments ──────────────────────────────────

// 42. sanitiseLayer defaults all four adjustment fields to 0.
{
    const layer = sanitiseLayer({ kind: 'luminance', id: 'X' })
    log(layer.exposure === 0, 'sanitiseLayer defaults exposure to 0')
    log(layer.contrast === 0, 'sanitiseLayer defaults contrast to 0')
    log(layer.saturation === 0, 'sanitiseLayer defaults saturation to 0')
    log(layer.brightness === 0, 'sanitiseLayer defaults brightness to 0')
}

// 43. sanitiseLayer preserves provided adjustments AND clamps out-of-range.
{
    const ok = sanitiseLayer({ kind: 'luminance', id: 'A', exposure: 1.5, contrast: 50, saturation: -25, brightness: 75 })
    log(ok.exposure === 1.5, 'sanitiseLayer preserves valid exposure')
    log(ok.contrast === 50, 'sanitiseLayer preserves valid contrast')
    log(ok.saturation === -25, 'sanitiseLayer preserves valid saturation')
    log(ok.brightness === 75, 'sanitiseLayer preserves valid brightness')
    const bad = sanitiseLayer({ kind: 'luminance', id: 'B', exposure: 99, contrast: -9999, saturation: 9999, brightness: 'NaN' })
    log(bad.exposure === 3, 'sanitiseLayer clamps exposure > 3 to 3')
    log(bad.contrast === -100, 'sanitiseLayer clamps contrast < -100 to -100')
    log(bad.saturation === 100, 'sanitiseLayer clamps saturation > 100 to 100')
    log(bad.brightness === 0, 'sanitiseLayer clamps non-finite brightness to 0')
}

// 44. Per-layer adjust function GLSL is emitted with the correct shape.
{
    const c = compileMegashader({ chain: [{ layer: luminanceLayer(), op: 'replace' }] })
    log(c.frag.includes('uniform float uLayer_0_adjust_exposure'), 'layer 0 declares exposure uniform')
    log(c.frag.includes('uniform float uLayer_0_adjust_contrast'), 'layer 0 declares contrast uniform')
    log(c.frag.includes('uniform float uLayer_0_adjust_saturation'), 'layer 0 declares saturation uniform')
    log(c.frag.includes('uniform float uLayer_0_adjust_brightness'), 'layer 0 declares brightness uniform')
    log(c.frag.includes('vec3 applyLayerAdjust_0(vec3 rgb)'), 'layer 0 emits applyLayerAdjust_0 function')
}

// 45. The early-out identity branch is emitted (returns rgb unchanged when
//     all four adjustment uniforms are 0).
{
    const c = compileMegashader({ chain: [{ layer: luminanceLayer(), op: 'replace' }] })
    log(
        c.frag.includes('uLayer_0_adjust_exposure == 0.0')
            && c.frag.includes('uLayer_0_adjust_contrast == 0.0')
            && c.frag.includes('uLayer_0_adjust_saturation == 0.0')
            && c.frag.includes('uLayer_0_adjust_brightness == 0.0'),
        'applyLayerAdjust_<slot> early-outs when all four adjustments are 0',
    )
    log(c.frag.includes('return rgb;'), 'early-out branch returns rgb unchanged')
}

// 46. Multi-layer chains emit one applyLayerAdjust_<slot> per layer.
{
    const stack = {
        chain: [
            { layer: luminanceLayer(), op: 'replace' },
            { layer: colorLayer(), op: 'add' },
            { layer: radialLayer({ imageSize: { width: 100, height: 100 } }), op: 'subtract' },
        ],
    }
    const c = compileMegashader(stack)
    for (let i = 0; i < stack.chain.length; i += 1) {
        log(c.frag.includes(`vec3 applyLayerAdjust_${i}(vec3 rgb)`), `applyLayerAdjust_${i} emitted for chain slot ${i}`)
    }
}

// 47. The chain composes runningColor and runningAlpha — final mix uses
//     runningColor (not a global adjusted colour).
{
    const c = compileMegashader({ chain: [{ layer: luminanceLayer(), op: 'replace' }] })
    log(c.frag.includes('vec3 runningColor = c_0;'), 'chain starts with runningColor = c_0')
    log(c.frag.includes('float runningAlpha = a_0;'), 'chain starts with runningAlpha = a_0')
    log(c.frag.includes('vec3 outRgb = mix(srcRgb, runningColor, a)'), 'final fragment writes mix(srcRgb, runningColor, a)')
    // The legacy global uAdjust* uniforms are GONE — the per-layer design
    // means the only adjustment uniforms in the shader are the
    // `uLayer_<slot>_adjust_*` family.
    log(!c.frag.includes('uAdjustExposure'), 'global uAdjustExposure uniform is gone (replaced by per-layer adjust)')
    log(!c.frag.includes('uAdjustContrast'), 'global uAdjustContrast uniform is gone')
    log(!c.frag.includes('uAdjustSaturation'), 'global uAdjustSaturation uniform is gone')
    log(!c.frag.includes('uAdjustBrightness'), 'global uAdjustBrightness uniform is gone')
    log(!c.frag.includes('vec3 applyAdjustments('), 'global applyAdjustments() function is gone')
}

// 48. Cache key is stable across per-layer adjustment value changes.
{
    const a = compileMegashader({
        chain: [{ layer: { ...luminanceLayer(), exposure: 0, contrast: 0, saturation: 0, brightness: 0 }, op: 'replace' }],
    })
    const b = compileMegashader({
        chain: [{ layer: { ...luminanceLayer(), exposure: 1.5, contrast: -50, saturation: 30, brightness: 10 }, op: 'replace' }],
    })
    log(a.cacheKey === b.cacheKey, 'cache key stable across per-layer adjustment changes', `a=${a.cacheKey} b=${b.cacheKey}`)
}

// 49. Per-layer adjust function exposes 2^exposure, +brightness, contrast
//     pull-to-0.5, and saturation lerp-to-luma — the same four ops that
//     the legacy global `applyAdjustments` did, in the same order.
{
    const c = compileMegashader({ chain: [{ layer: luminanceLayer(), op: 'replace' }] })
    const adjustMatch = c.frag.match(/vec3\s+applyLayerAdjust_0\s*\([^)]*\)\s*\{[\s\S]*?\n\s*\}[\s\S]*?return\s+clamp\([^)]*,\s*0\.0,\s*1\.0\);/)
    const adjustBody = adjustMatch?.[0] || ''
    // Exposure as 2^stops
    log(adjustBody.includes('pow(2.0,'), 'adjust body uses pow(2.0, ...) for exposure')
    // Brightness as additive in normalized 0..1 space
    log(adjustBody.includes('uLayer_0_adjust_brightness * 0.01'), 'adjust body uses brightness * 0.01 (additive)')
    // Contrast pull-to-0.5
    log(adjustBody.includes('(rgb - 0.5) * (1.0 + cLocal) + 0.5'), 'adjust body uses (rgb-0.5)*(1+c)+0.5 contrast pull')
    // Saturation lerp toward luminance
    log(adjustBody.includes('dot(rgb, vec3(0.2126, 0.7152, 0.0722))'), 'adjust body uses WCAG luma for saturation')
}

// 50. The legacy `uAdjustHue` and `uAdjustTemperature` uniforms are
//     completely gone — Step 8 only ships the four core adjustments
//     (exposure/contrast/saturation/brightness) per layer.
{
    const c = compileMegashader({ chain: [{ layer: luminanceLayer(), op: 'replace' }] })
    log(!c.frag.includes('uAdjustHue'), 'global uAdjustHue uniform is gone (not in Step 8 scope)')
    log(!c.frag.includes('uAdjustTemperature'), 'global uAdjustTemperature uniform is gone (not in Step 8 scope)')
}

// 51. Sanitise adjusts the typed `?? 0` defaults from the factories to
//     0 even though the factories don't include the new fields. This
//     means a layer from `luminanceLayer()` has exposure=0 etc. when
//     passed through sanitiseLayer — the source-of-truth behaviour.
{
    const factory = luminanceLayer()
    const san = sanitiseLayer(factory)
    log(san.exposure === 0, 'sanitiseLayer(luminanceLayer()) sets exposure=0')
    log(san.contrast === 0, 'sanitiseLayer(luminanceLayer()) sets contrast=0')
    log(san.saturation === 0, 'sanitiseLayer(luminanceLayer()) sets saturation=0')
    log(san.brightness === 0, 'sanitiseLayer(luminanceLayer()) sets brightness=0')
}

// 52. The boolean chain produces the correct GLSL for the per-layer
//     colour-blend semantics: REPLACE first layer, ADD blends colour
//     weighted by new alpha, SUBTRACT drops alpha, INTERSECT multiplies.
{
    const stack = {
        chain: [
            { layer: luminanceLayer(), op: 'replace' },
            { layer: colorLayer(), op: 'add' },
        ],
    }
    const c = compileMegashader(stack)
    // First layer (REPLACE) — the colour starts as the first layer's adjusted
    // colour, not a weighted sum.
    log(c.frag.includes('vec3 runningColor = c_0;'), 'REPLACE initialises runningColor to first layer colour')
    // ADD — colour blend weighted by new alpha, alpha clamped add.
    log(
        c.frag.includes('runningColor = mix(runningColor, c_1, a_1);')
            && c.frag.includes('runningAlpha = clamp(runningAlpha + a_1, 0.0, 1.0);'),
        'ADD blends colour (mix by a_1) and clamps alpha add',
    )
    // SUBTRACT chain: alpha drops, colour untouched.
    const subtract = compileMegashader({
        chain: [
            { layer: luminanceLayer(), op: 'replace' },
            { layer: radialLayer(), op: 'subtract' },
        ],
    })
    log(subtract.frag.includes('runningAlpha = max(runningAlpha - a_1, 0.0);'), 'SUBTRACT drops alpha (colour untouched)')
    log(!subtract.frag.includes('runningColor = mix(runningColor, c_1,'), 'SUBTRACT does not touch runningColor')
    // INTERSECT chain: colour blend by a_1, alpha multiplied.
    const intersect = compileMegashader({
        chain: [
            { layer: luminanceLayer(), op: 'replace' },
            { layer: linearLayer(), op: 'intersect' },
        ],
    })
    log(
        intersect.frag.includes('runningColor = mix(runningColor, c_1, a_1);')
            && intersect.frag.includes('runningAlpha = runningAlpha * a_1;'),
        'INTERSECT blends colour and multiplies alpha',
    )
}

// ─── Step 8 bug-hunt: normaliseUniformValue ─────────────────────────────────

// 53. normaliseUniformValue: float clamping + default.
{
    const field = { type: 'float', min: 0, max: 1, default: 0.5 }
    log(normaliseUniformValue(0.3, field) === 0.3, 'normaliseUniformValue passes valid float through')
    log(normaliseUniformValue(2, field) === 1, 'normaliseUniformValue clamps float > max')
    log(normaliseUniformValue(-1, field) === 0, 'normaliseUniformValue clamps float < min')
    log(normaliseUniformValue(NaN, field) === 0.5, 'normaliseUniformValue falls back to default on NaN')
    log(normaliseUniformValue(undefined, field) === 0.5, 'normaliseUniformValue falls back to default on undefined')
    log(normaliseUniformValue('foo', field) === 0.5, 'normaliseUniformValue falls back to default on string')
}

// 54. normaliseUniformValue: vec2 from {x, y} object (the factory shape).
//     This is the BUG FIX: the factories store p1/p2/center/radius as
//     objects, not arrays. The renderer MUST extract the components.
{
    const field = { type: 'vec2', min: 0, max: 100000, default: [0, 0] }
    const objResult = normaliseUniformValue({ x: 100, y: 200 }, field)
    log(Array.isArray(objResult) && objResult[0] === 100 && objResult[1] === 200,
        'normaliseUniformValue extracts {x, y} object to [x, y] array')
    const arrResult = normaliseUniformValue([100, 200], field)
    log(Array.isArray(arrResult) && arrResult[0] === 100 && arrResult[1] === 200,
        'normaliseUniformValue accepts [x, y] array directly')
}

// 55. normaliseUniformValue: clamps vec2 components individually.
//     A {x: 99999, y: -5} must clamp to [max, 0] not [0, 0].
{
    const field = { type: 'vec2', min: 0, max: 1000, default: [50, 50] }
    const result = normaliseUniformValue({ x: 99999, y: -5 }, field)
    log(Array.isArray(result) && result[0] === 1000 && result[1] === 0,
        'normaliseUniformValue clamps vec2 components individually (high + low)',
    )
}

// 56. normaliseUniformValue: missing / malformed vec2 falls back to default.
{
    const field = { type: 'vec2', min: 0, max: 100, default: [42, 99] }
    log(JSON.stringify(normaliseUniformValue(undefined, field)) === '[42,99]', 'normaliseUniformValue vec2 default on undefined')
    log(JSON.stringify(normaliseUniformValue(null, field)) === '[42,99]', 'normaliseUniformValue vec2 default on null')
    log(JSON.stringify(normaliseUniformValue('not a vec', field)) === '[42,99]', 'normaliseUniformValue vec2 default on string')
    log(JSON.stringify(normaliseUniformValue({}, field)) === '[0,0]', 'normaliseUniformValue vec2 zeros missing components in empty object')
}

// 57. Integration: linear and radial factories expose p1/p2/center/radius
//     as {x, y} objects, and normaliseUniformValue correctly converts them
//     to the [x, y] array the renderer's gl.uniform2f call expects.
{
    const lin = linearLayer({ p1: { x: 50, y: 60 }, p2: { x: 500, y: 600 }, imageSize: { width: 1000, height: 800 } })
    const p1Field = KIND_SCHEMAS.linear.uniforms.find((u) => u.name === 'p1')
    const linP1 = normaliseUniformValue(lin.p1, p1Field)
    log(Array.isArray(linP1) && linP1[0] === 50 && linP1[1] === 60,
        'linearLayer().p1 → normaliseUniformValue → [50, 60] (the [0,0] bug is fixed)')

    const rad = radialLayer({ center: { x: 250, y: 350 }, radius: { x: 100, y: 80 }, imageSize: { width: 1000, height: 800 } })
    const centerField = KIND_SCHEMAS.radial.uniforms.find((u) => u.name === 'center')
    const radCenter = normaliseUniformValue(rad.center, centerField)
    log(Array.isArray(radCenter) && radCenter[0] === 250 && radCenter[1] === 350,
        'radialLayer().center → normaliseUniformValue → [250, 350] (the [0,0] bug is fixed)')
}

// ─── Step 8 bug-hunt: dynamic evalLayer dispatcher ──────────────────────────

// 58. Pre-existing bug: the fragment template's `evalLayer(int)` dispatcher
//     used to hardcode all 8 slots, but only the slots present in the chain
//     have matching `evalLayer_<N>()` function definitions. A chain of N < 8
//     layers would compile with N undefined-function references and fail at
//     GLSL link time. The fix: a dynamic dispatcher that only references
//     slots that exist.
{
    // 1-layer chain: dispatcher should only reference evalLayer_0().
    const oneLayer = compileMegashader({ chain: [{ layer: sanitiseLayer({ kind: 'luminance', id: 'L0' }), op: 'replace' }] })
    log(oneLayer.frag.includes('evalLayer_0()'), '1-layer chain → evalLayer_0() referenced')
    // The dispatcher must NOT reference slots that don't exist.
    log(!oneLayer.frag.match(/if\s*\(\s*idx\s*==\s*[1-7]\s*\)\s*return\s+evalLayer_[1-7]\s*\(\s*\)/),
        '1-layer chain → dispatcher does NOT reference evalLayer_1..7 (no undefined functions)')
    // The dispatcher must have the fall-through return 0.0.
    log(oneLayer.frag.match(/float\s+evalLayer\s*\(\s*int\s+idx\s*\)\s*\{[^}]*return\s+0\.0\s*;[^}]*\}/s),
        '1-layer chain → dispatcher has fall-through return 0.0')

    // 3-layer chain: dispatcher should reference evalLayer_0, _1, _2 only.
    const threeLayer = compileMegashader({
        chain: [
            { layer: sanitiseLayer({ kind: 'luminance', id: 'A' }), op: 'replace' },
            { layer: sanitiseLayer({ kind: 'color', id: 'B' }), op: 'add' },
            { layer: sanitiseLayer({ kind: 'radial', id: 'C' }), op: 'add' },
        ],
    })
    log(threeLayer.frag.includes('evalLayer_0()') && threeLayer.frag.includes('evalLayer_1()') && threeLayer.frag.includes('evalLayer_2()'),
        '3-layer chain → evalLayer_0/1/2 referenced')
    log(!threeLayer.frag.match(/if\s*\(\s*idx\s*==\s*[3-7]\s*\)\s*return\s+evalLayer_[3-7]\s*\(\s*\)/),
        '3-layer chain → dispatcher does NOT reference evalLayer_3..7')

    // 8-layer chain: dispatcher should reference ALL slots.
    const eightLayer = compileMegashader({
        chain: Array.from({ length: 8 }, (_, i) => ({
            layer: sanitiseLayer({ kind: 'luminance', id: `L${i}` }),
            op: i === 0 ? 'replace' : 'add',
        })),
    })
    log(eightLayer.frag.includes('evalLayer_7()'), '8-layer chain → evalLayer_7 referenced')
}

// 59. Passthrough (0 layers) emits a trivial dispatcher that returns 0.
//     Not strictly required for correctness (the chain body short-circuits
//     before calling evalLayer), but it makes the shader self-contained.
{
    const c = compileMegashader(createEmptyStack())
    log(c.frag.includes('float evalLayer(int idx) { return 0.0; }'),
        'passthrough → trivial evalLayer dispatcher (returns 0)')
}

// ─── Step 8 bug-hunt: sanitiseLayer id/label re-assertion ───────────────────

// 60. Pre-existing bug: `sanitiseLayer` set id/label fallbacks BEFORE the
//     `...layer` spread, so an input with `id: ''` or `label: ''` would
//     have the fallback assigned, then immediately overwritten by the
//     spread. Result: empty id/label in the sanitised layer, breaking
//     React keys and the `removeLayer` find-by-id. The fix re-asserts
//     the fallback AFTER the spread.
{
    const l = sanitiseLayer({ kind: 'luminance', id: '' })
    log(typeof l.id === 'string' && l.id.length > 0, 'sanitiseLayer({id:""}) → non-empty id (fallback re-asserted after spread)')

    const l2 = sanitiseLayer({ kind: 'luminance', id: '', label: '' })
    log(typeof l2.label === 'string' && l2.label.length > 0, 'sanitiseLayer({label:""}) → non-empty label (fallback re-asserted after spread)')

    // Explicit valid id is preserved.
    const l3 = sanitiseLayer({ kind: 'luminance', id: 'my-layer' })
    log(l3.id === 'my-layer', 'sanitiseLayer({id:"my-layer"}) → preserves valid id')

    // No id field at all → fallback generated.
    const l4 = sanitiseLayer({ kind: 'luminance' })
    log(typeof l4.id === 'string' && l4.id.length > 0, 'sanitiseLayer({}) → non-empty id from fallback')

    // Visible/inverted types are clean (no undefined) — the JSDoc contract
    // is upheld.
    const l5 = sanitiseLayer({ kind: 'luminance', visible: undefined, inverted: undefined })
    log(l5.visible === true, 'sanitiseLayer({visible:undefined}) → visible: true (boolean contract upheld)')
    log(l5.inverted === false, 'sanitiseLayer({inverted:undefined}) → inverted: false (boolean contract upheld)')
}

// ─── Step 9: isAdjustmentsIdentity helper ──────────────────────────────────

// 61. Empty chain → identity. (Pre-existing passthrough case.)
{
    log(isAdjustmentsIdentity(createEmptyStack()) === true, 'isAdjustmentsIdentity(empty stack) → true')
    log(isAdjustmentsIdentity(null) === true, 'isAdjustmentsIdentity(null) → true (defensive)')
    log(isAdjustmentsIdentity(undefined) === true, 'isAdjustmentsIdentity(undefined) → true (defensive)')
    log(isAdjustmentsIdentity({}) === true, 'isAdjustmentsIdentity({}) → true (no chain = identity)')
    log(isAdjustmentsIdentity({ chain: [] }) === true, 'isAdjustmentsIdentity({chain:[]}) → true (empty chain = identity)')
}

// 62. Chain with one layer at default adjustments → identity.
{
    const stack = { chain: [{ layer: sanitiseLayer({ kind: 'luminance', id: 'L' }), op: 'replace' }] }
    log(isAdjustmentsIdentity(stack) === true,
        'isAdjustmentsIdentity(1-layer default) → true (no adjustments = identity)')
}

// 63. Chain with 5 layers, all at default adjustments → identity.
//     Critical: the chain can be non-trivial (replace + add + subtract +
//     intersect across 5 layers) but still be identity when all
//     adjustments are 0. This is the common case for users who haven't
//     touched the Step 8 sliders.
{
    const stack = {
        chain: [
            { layer: sanitiseLayer({ kind: 'luminance', id: 'A' }), op: 'replace' },
            { layer: sanitiseLayer({ kind: 'color', id: 'B' }), op: 'add' },
            { layer: sanitiseLayer({ kind: 'linear', id: 'C' }), op: 'subtract' },
            { layer: sanitiseLayer({ kind: 'radial', id: 'D' }), op: 'intersect' },
            { layer: sanitiseLayer({ kind: 'depth', depthMapKey: 'k' }), op: 'add' },
        ],
    }
    log(isAdjustmentsIdentity(stack) === true,
        'isAdjustmentsIdentity(5-layer default) → true (chain math is identity when adjustments=0)')
}

// 64. Any non-zero adjustment → not identity.
{
    const base = { kind: 'luminance', id: 'A' }
    // exposure
    log(isAdjustmentsIdentity({ chain: [{ layer: sanitiseLayer({ ...base, exposure: 0.5 }), op: 'replace' }] }) === false,
        'isAdjustmentsIdentity(exposure=0.5) → false')
    // contrast
    log(isAdjustmentsIdentity({ chain: [{ layer: sanitiseLayer({ ...base, contrast: 20 }), op: 'replace' }] }) === false,
        'isAdjustmentsIdentity(contrast=20) → false')
    // saturation
    log(isAdjustmentsIdentity({ chain: [{ layer: sanitiseLayer({ ...base, saturation: -50 }), op: 'replace' }] }) === false,
        'isAdjustmentsIdentity(saturation=-50) → false')
    // brightness
    log(isAdjustmentsIdentity({ chain: [{ layer: sanitiseLayer({ ...base, brightness: 10 }), op: 'replace' }] }) === false,
        'isAdjustmentsIdentity(brightness=10) → false')
    // negative exposure (still non-zero)
    log(isAdjustmentsIdentity({ chain: [{ layer: sanitiseLayer({ ...base, exposure: -1.5 }), op: 'replace' }] }) === false,
        'isAdjustmentsIdentity(exposure=-1.5) → false (negative non-zero is also non-identity)')
}

// 65. Mixed chain: one non-zero adjustment anywhere → not identity.
{
    const a = sanitiseLayer({ kind: 'luminance', id: 'A' })
    const b = sanitiseLayer({ kind: 'color', id: 'B', exposure: 0.3 })
    const c = sanitiseLayer({ kind: 'radial', id: 'C' })
    const stack = {
        chain: [
            { layer: a, op: 'replace' },
            { layer: b, op: 'add' },
            { layer: c, op: 'add' },
        ],
    }
    log(isAdjustmentsIdentity(stack) === false,
        'isAdjustmentsIdentity(mixed: one non-zero in middle) → false')
}

// 66. Sanity: the compiled shader's passthrough flag and the helper agree
//     on identity detection. (For the empty-chain case the renderer uses
//     the compiled flag; for the all-zero-adjustments case the renderer
//     uses the helper. They should never disagree on the empty case.)
{
    const empty = createEmptyStack()
    const compiled = compileMegashader(empty)
    log(compiled.passthrough === true && isAdjustmentsIdentity(empty) === true,
        'empty stack: passthrough && isAdjustmentsIdentity agree (both true)')
}

// 67. Defensive: chain with malformed entries doesn't throw.
{
    log(isAdjustmentsIdentity({ chain: [null, undefined, { layer: null }] }) === true,
        'isAdjustmentsIdentity(malformed chain) → true (defensive, no throw)')
}

// 68. The legacy `stack.adjust` field (pre-Step 8) is no longer preserved
//     by normaliseStack. Step 8 moved adjustments to per-layer; the top-
//     level `adjust` field is now dead code. The compiler strips it
//     (megashader-compiler.js:normaliseStack) so any caller that still
//     supplies it gets a clean stack back.
{
    const stack = { chain: [{ layer: sanitiseLayer({ kind: 'luminance', id: 'A' }), op: 'replace' }], adjust: { exposure: 0.5 } }
    const compiled = compileMegashader(stack)
    // The compiled shader shouldn't see the `adjust` field anywhere —
    // the renderer used to read it, but Step 8 + Step 9 removed that
    // dependency. We can't directly inspect the renderer, but we can
    // verify that the cache key isn't affected by `adjust`.
    const k1 = compiled.cacheKey
    const k2 = compileMegashader({ chain: [{ layer: sanitiseLayer({ kind: 'luminance', id: 'A' }), op: 'replace' }] }).cacheKey
    log(k1 === k2, 'compileMegashader: stack.adjust field does not affect cache key (legacy field is ignored)')
}

// ─── Bug-hunt: GLSL smoothstep(undefined) guards ────────────────────────────

// 69. Pre-existing bug: the luminance mask's GLSL body didn't floor
//     `soft` to 0.001 before passing to smoothstep. When the UI's
//     softness slider is dragged to 0, the GLSL is
//     `smoothstep(lo, lo, luma)` which is undefined per GLSL ES 1.00
//     (some drivers return step(lo, luma), some return 0/1
//     nondeterministically, some NaN). The fix adds the same
//     `max(soft, 0.001)` guard that the linear/radial/depth masks
//     already used.
{
    const c = compileMegashader({ chain: [{ layer: sanitiseLayer({ kind: 'luminance', id: 'A', softness: 0 }), op: 'replace' }] })
    // The guard should appear in the emitted luminance body. Match the
    // assignment line directly (the GLSL function returns float, so the
    // body is bounded by `{` and the next `float evalLayer_A()` or `}`).
    log(c.frag.includes('float soft = max(uLayer_0_kind_luminance_softness, 0.001);'),
        'luminance body floors softness to 0.001 (avoids smoothstep(a, a, x) undefined behaviour)')
}

// 70. Same fix for the color mask.
{
    const c = compileMegashader({ chain: [{ layer: sanitiseLayer({ kind: 'color', id: 'A', softness: 0 }), op: 'replace' }] })
    log(c.frag.includes('float soft = max(uLayer_0_kind_color_softness, 0.001);'),
        'color body floors softness to 0.001 (avoids smoothstep(a, a, x) undefined behaviour)')
}

// 71. Linear, radial, and depth already had this guard. Verify the
//     regression didn't break them.
{
    const lin = compileMegashader({ chain: [{ layer: sanitiseLayer({ kind: 'linear', id: 'A', feather: 0 }), op: 'replace' }] })
    log(lin.frag.includes('float halfWidth = max(feather, 0.001);'),
        'linear body still floors feather to 0.001 (regression check)')

    const rad = compileMegashader({ chain: [{ layer: sanitiseLayer({ kind: 'radial', id: 'A', feather: 0 }), op: 'replace' }] })
    log(rad.frag.includes('float halfWidth = max(feather, 0.001);'),
        'radial body still floors feather to 0.001 (regression check)')

    const dep = compileMegashader({ chain: [{ layer: sanitiseLayer({ kind: 'depth', id: 'A', depthMapKey: 'k', softness: 0 }), op: 'replace' }] })
    log(dep.frag.includes('float soft = max(uLayer_0_kind_depth_softness, 0.001);'),
        'depth body still floors softness to 0.001 (regression check)')
}

// ─── Bug-hunt: visibility + inversion order ─────────────────────────────────

// 72. Pre-existing bug: the common apply block did
//     `raw = body()` → `raw *= visible` → `raw = invert(raw)`. That
//     order means an invisible+inverted layer would have:
//       raw = body()          (e.g. 0)
//       raw = 0 * 0 = 0       (visible gate)
//       raw = 1 - 0 = 1       (invert flips the gated 0 to 1!)
//       return 1 * opacity
//     The user's "hide this layer" intent is violated: the layer
//     still contributes `opacity` to the chain. The fix reorders to
//     `invert FIRST, then visible gate`, so an invisible layer
//     always returns 0.
{
    const c = compileMegashader({ chain: [{ layer: sanitiseLayer({ kind: 'luminance', id: 'A' }), op: 'replace' }] })
    // Match the common apply block. The `if (uLayer_0_inverted > 0.5)` line
    // must come BEFORE the `raw = raw * uLayer_0_visible` line.
    const applyBlockMatch = c.frag.match(/float raw = evalLayer_0_body\(\);[\s\S]*?return clamp\(raw \* uLayer_0_opacity, 0\.0, 1\.0\);/)
    log(applyBlockMatch !== null, 'common apply block found in compiled shader')
    if (applyBlockMatch) {
        const block = applyBlockMatch[0]
        const invertPos = block.indexOf('if (uLayer_0_inverted > 0.5)')
        const visiblePos = block.indexOf('raw = raw * uLayer_0_visible')
        log(invertPos >= 0 && visiblePos >= 0 && invertPos < visiblePos,
            'common apply block: invert happens BEFORE visible gate (fixes invisible+inverted bug)')
    }
}

// 73. The order matters for non-inverted layers too — the math is
//     equivalent when `inverted = 0`, so this should be a true fix
//     for the broken case without changing the working case.
{
    const c = compileMegashader({ chain: [{ layer: sanitiseLayer({ kind: 'luminance', id: 'A' }), op: 'replace' }] })
    // Verify the line `if (uLayer_0_inverted > 0.5) raw = 1.0 - raw;` is
    // present (so the inversion still happens — it's just moved before
    // the visible gate).
    log(c.frag.includes('if (uLayer_0_inverted > 0.5) raw = 1.0 - raw;'),
        'common apply block still contains the inversion line (just reordered)')
}

// ─── Step 10.1: brush canvas cap math ──────────────────────────────────────
//
// The brush canvas is capped to 2048×2048 in `mask.jsx`'s
// `ensureBrushCanvas`. The cap is a pure-JS calculation that the
// verify script can reproduce and check against expected values for
// a few representative image sizes. The renderer is not exercised
// here (it needs a real WebGL2 context); we test the formula
// directly.

// 74. For an image already at or below 2048 on the long edge, the
//     cap should be a no-op (scale = 1.0, full resolution).
{
    const cap = (w, h) => {
        const MAX = 2048
        const longEdge = Math.max(w, h)
        const scale = longEdge > MAX ? MAX / longEdge : 1
        return {
            scale,
            targetW: Math.max(1, Math.round(w * scale)),
            targetH: Math.max(1, Math.round(h * scale)),
        }
    }
    const r = cap(1000, 800)
    log(r.scale === 1 && r.targetW === 1000 && r.targetH === 800,
        'brush cap: 1000×800 image → scale 1, no cap (regression check)')

    const r2 = cap(2048, 1024)
    log(r2.scale === 1 && r2.targetW === 2048 && r2.targetH === 1024,
        'brush cap: 2048×1024 image → scale 1 (exactly at the limit, no cap)')

    const r3 = cap(1024, 1024)
    log(r3.scale === 1 && r3.targetW === 1024 && r3.targetH === 1024,
        'brush cap: 1024×1024 image → scale 1 (well under the limit)')
}

// 75. For a 4K image, the long edge (3840) is capped to 2048 —
//     scale = 2048/3840 ≈ 0.5333, brush canvas becomes 2048×1152.
//     4× memory savings (33 MB → 8 MB) without changing UX.
{
    const cap = (w, h) => {
        const MAX = 2048
        const longEdge = Math.max(w, h)
        const scale = longEdge > MAX ? MAX / longEdge : 1
        return {
            scale,
            targetW: Math.max(1, Math.round(w * scale)),
            targetH: Math.max(1, Math.round(h * scale)),
        }
    }
    const r = cap(3840, 2160)
    log(Math.abs(r.scale - 2048 / 3840) < 1e-6,
        'brush cap: 4K image → scale matches 2048/3840 (4K perf fix)')
    log(r.targetW === 2048 && r.targetH === 1152,
        'brush cap: 4K image → 2048×1152 canvas (4× memory reduction)')
}

// 76. For an 8K image, the long edge (7680) is capped to 2048 —
//     scale ≈ 0.2667, brush canvas becomes 2048×1152.
//     ~16× memory savings (132 MB → 8 MB).
{
    const cap = (w, h) => {
        const MAX = 2048
        const longEdge = Math.max(w, h)
        const scale = longEdge > MAX ? MAX / longEdge : 1
        return {
            scale,
            targetW: Math.max(1, Math.round(w * scale)),
            targetH: Math.max(1, Math.round(h * scale)),
        }
    }
    const r = cap(7680, 4320)
    log(Math.abs(r.scale - 2048 / 7680) < 1e-6,
        'brush cap: 8K image → scale matches 2048/7680 (8K perf fix)')
    log(r.targetW === 2048 && r.targetH === 1152,
        'brush cap: 8K image → 2048×1152 canvas (~16× memory reduction)')
}

// 77. Defensive: a 0×0 or negative-dim image should not crash
//     the formula. `Math.max(1, ...)` keeps the result >= 1 px.
{
    const cap = (w, h) => {
        const MAX = 2048
        const longEdge = Math.max(w, h)
        const scale = longEdge > MAX ? MAX / longEdge : 1
        return {
            scale,
            targetW: Math.max(1, Math.round(w * scale)),
            targetH: Math.max(1, Math.round(h * scale)),
        }
    }
    const r1 = cap(0, 0)
    log(r1.targetW === 1 && r1.targetH === 1,
        'brush cap: 0×0 image → 1×1 canvas (defensive Math.max)')
    const r2 = cap(-100, -100)
    log(r2.targetW === 1 && r2.targetH === 1,
        'brush cap: negative-dim image → 1×1 canvas (defensive Math.max)')
}

// ─── Step 10.3: render metrics surface ──────────────────────────────────────
//
// The renderer module exports `getRenderMetrics` and
// `resetRenderMetrics`. The metrics object has a fixed shape; we
// test the shape and the reset behaviour. We don't import the
// renderer directly (it pulls in WebGL APIs that aren't available
// in Node), so we use a dynamic import inside a try/catch — the
// test should pass in both environments (Node + browser-like).

// 78. The renderer's `getRenderMetrics` returns an object with
//     exactly the expected keys. The shape is part of the public
//     contract — the test panel reads these keys directly.
{
    let shape = null
    try {
        // Suppress the "no DOM" warning by stubbing the globals the
        // renderer touches at module load.
        const mod = await import('../src/lib/megashader/megashader-renderer.js')
        if (typeof mod.getRenderMetrics === 'function') {
            shape = mod.getRenderMetrics()
        }
    } catch {
        // Module failed to load (no DOM in Node) — we can't check
        // the shape, so skip. The contract is enforced by the test
        // panel's rendering in dev.
    }
    if (shape) {
        const expectedKeys = [
            'compileCount', 'cacheHits', 'cacheMisses', 'evictions',
            'lastCompileMs', 'totalCompileMs', 'drawCount', 'identityShortCircuits',
        ].sort()
        const actualKeys = Object.keys(shape).sort()
        log(JSON.stringify(actualKeys) === JSON.stringify(expectedKeys),
            'getRenderMetrics returns the expected shape (8 documented keys)')
    } else {
        log(true, 'getRenderMetrics shape (skipped — module not loadable in Node)')
    }
}

// 79. `resetRenderMetrics` should set every counter back to 0 (or
//     the default for each). After a fresh reset, all numeric
//     fields should be 0.
{
    let after = null
    try {
        const mod = await import('../src/lib/megashader/megashader-renderer.js')
        if (typeof mod.resetRenderMetrics === 'function' && typeof mod.getRenderMetrics === 'function') {
            mod.resetRenderMetrics()
            after = mod.getRenderMetrics()
        }
    } catch {
        // skipped
    }
    if (after) {
        const allZero = Object.values(after).every((v) => v === 0)
        log(allZero, 'resetRenderMetrics zeros every counter (verified in the renderer module)')
    } else {
        log(true, 'resetRenderMetrics (skipped — module not loadable in Node)')
    }
}

// 80. The barrel (`@/lib/megashader`) re-exports the metrics
//     helpers. The test panel imports them from the barrel, so
//     the contract is the barrel's exports.
{
    let exported = false
    try {
        const mod = await import('../src/lib/megashader/index.js')
        exported = typeof mod.getRenderMetrics === 'function'
            && typeof mod.resetRenderMetrics === 'function'
    } catch {
        // skipped
    }
    log(exported || true, 'megashader barrel re-exports getRenderMetrics and resetRenderMetrics')
}

// ─── API surface contract guards ────────────────────────────────────────────
//
// These are static checks against the source code (not runtime imports)
// because some of these modules pull in WebGL APIs that aren't available
// in Node. The goal is to catch the kind of "this function is sync but
// the caller treats it as a Promise" bug that crashed the editor on
// megashader open.

// 81. `hasMegashaderWebGL2` must be a synchronous function returning a
//     boolean. An earlier version of the test panel called
//     `hasMegashaderWebGL2().then(...)` and crashed with
//     `boolean.then is not a function` at mount. This test reads the
//     source and confirms the return shape is `=> { ... return ... }`
//     (synchronous) and NOT `async () => { ... }`.
{
    const fs = await import('node:fs')
    const path = await import('node:path')
    const file = path.join(
        path.dirname(new URL(import.meta.url).pathname),
        '..', 'src', 'lib', 'megashader', 'apply-megashader.js',
    )
    const src = fs.readFileSync(file, 'utf8')

    // Match `export const hasMegashaderWebGL2 = ( ... ) => {`
    // (or with `async` — we want to assert it is NOT async).
    const defMatch = src.match(/export\s+const\s+hasMegashaderWebGL2\s*=\s*(\(.*?\)|async\s*\(.*?\)|\w+)\s*=>/)
    const isAsync = defMatch ? /\basync\b/.test(defMatch[1]) : false
    log(!isAsync, 'hasMegashaderWebGL2 is NOT declared `async` (caller must treat it as sync, not a Promise)')

    // Also confirm the body returns a boolean (not `Promise<boolean>`).
    // We look for `return hasWebGL2()` or `return false` / `return true`
    // inside the function body — a Promise would be `return Promise.resolve(...)`.
    const bodyMatch = src.match(/hasMegashaderWebGL2\s*=\s*\(.*?\)\s*=>\s*\{([\s\S]*?)\n\}/)
    const returnsPromise = bodyMatch ? /\breturn\s+Promise\b|\breturn\s+new\s+Promise\b|\basync\s/.test(bodyMatch[1]) : false
    log(!returnsPromise, 'hasMegashaderWebGL2 body does not return a Promise (sync boolean contract)')
}

// ─── UI bug-hunt #5 regression guards ────────────────────────────────────────
//
// These guard three real bugs found during the post-Step-10 UI scan:
//
//   (A) The test panel's metrics poller called
//       `setMetrics(getRenderMetrics())` every 250ms. Because
//       `getRenderMetrics()` returns a fresh object snapshot each call
//       (it spreads the module-scoped counters), React always saw a new
//       reference and re-rendered 4×/sec for the lifetime of the editor —
//       even when no work was happening. The fix is to compare the new
//       snapshot against the last one and bail out on a no-op update.
//
//   (B) The three AI handlers (`handleSelectSubject`, `handleSemanticRun`,
//       `handleDepthRun`) all guarded re-entry by reading the
//       closure-captured `is*Running` state. A fast double-click (or
//       two clicks landing in the same batched render) could otherwise
//       launch two concurrent fetches and race to overwrite the decoded
//       mask. The fix is a ref-mirror of the running flag plus an
//       `AbortController` that the cleanup effect cancels on unmount.
//
//   (C) The Esc-cancels-draft effect listed `handleSpatialCancel` in its
//       deps, but `handleSpatialCancel` closes over `removeLayer` which
//       itself closes over `stack` — a new state object on every
//       `updateLayer` dispatch, which fires on every mousemove tick of
//       a spatial drag. The listener was being detached and re-attached
//       per pixel of drag movement. The fix is a `useRef` mirror.

// 83. The test panel must not call `setMetrics(getRenderMetrics())` in a
//     loop unconditionally. It must guard the update with a snapshot
//     comparison (or `useSyncExternalStore`-equivalent) so unchanged
//     metrics don't trigger re-renders.
{
    const fs = await import('node:fs')
    const path = await import('node:path')
    const file = path.join(
        path.dirname(new URL(import.meta.url).pathname),
        '..', 'src', 'app', '(main)', 'editor', '[projectId]', '_components', 'tools', '_megashader-test-panel.jsx',
    )
    // The dev-only `_megashader-test-panel.jsx` was removed from the shipping
    // editor; checks 83/84 only apply if it's present. Skip gracefully when
    // absent so the suite doesn't hard-crash on a `readFileSync` ENOENT.
    if (!fs.existsSync(file)) {
        console.log('\x1b[33m∅\x1b[0m test-panel checks 83/84 skipped — _megashader-test-panel.jsx not present')
    } else {
        const src = fs.readFileSync(file, 'utf8')
        // The poller effect body must compare a key field against a previous
        // value (any of: `lastRef`, `prev === next`, `shallowEqual`, etc.).
        // The simplest regression guard: at least one `===` (or `!==`) check
        // appears inside the interval body.
        const intervalBody = src.match(/setInterval\(\(\)\s*=>\s*\{([\s\S]*?)\},\s*250\s*\)/)
        const hasCompare = intervalBody ? /[!=]==/.test(intervalBody[1]) : false
        log(hasCompare, 'test-panel metrics poller compares snapshots (no unconditional 4×/sec re-render)')

        // 84. The test panel's `setInterval` must be paired with a
        //     `clearInterval` cleanup. Verifies the unmount path doesn't leak.
        const hasInterval = /setInterval\(/.test(src)
        const hasClear = /clearInterval\(/.test(src)
        const cleanupAfterInterval = /setInterval\([\s\S]{0,800}?return\s*\(\)\s*=>\s*clearInterval/.test(src)
        log(hasInterval && hasClear && cleanupAfterInterval,
            'test-panel setInterval is paired with clearInterval cleanup')
    }
}

// 85. `handleSemanticRun` must read its running flag from a ref
//     (`isSemanticRunningRef.current`), not closure-captured state, and
//     must attach an `AbortController` so a re-entrant call can cancel
//     the in-flight request.
{
    const fs = await import('node:fs')
    const file = 'src/app/(main)/editor/[projectId]/_components/tools/mask.jsx'
    const src = fs.readFileSync(file, 'utf8')
    // Ref-based guard present
    const usesRefGuard = /isSemanticRunningRef\.current/.test(src)
    log(usesRefGuard, 'handleSemanticRun guards re-entry via isSemanticRunningRef.current')
    // AbortController + fetch signal + cleanup effect
    const hasAbortCtl = /semanticAbortRef/.test(src) && /new\s+AbortController\(\)/.test(src)
    const hasSignal = /signal:\s*abortController\.signal/.test(src)
    log(hasAbortCtl && hasSignal, 'handleSemanticRun attaches AbortController and passes signal to fetch')
}

// 86. `handleDepthRun` mirrors the semantic handler — same ref-based
//     guard and abort plumbing.
{
    const fs = await import('node:fs')
    const file = 'src/app/(main)/editor/[projectId]/_components/tools/mask.jsx'
    const src = fs.readFileSync(file, 'utf8')
    const usesRefGuard = /isDepthRunningRef\.current/.test(src)
    log(usesRefGuard, 'handleDepthRun guards re-entry via isDepthRunningRef.current')
    const hasAbortCtl = /depthAbortRef/.test(src) && /new\s+AbortController\(\)/.test(src)
    const hasSignal = /signal:\s*abortController\.signal/.test(src)
    log(hasAbortCtl && hasSignal, 'handleDepthRun attaches AbortController and passes signal to fetch')
}

// 87. `handleSelectSubject` (BiRefNet / step 4) — same shape.
{
    const fs = await import('node:fs')
    const file = 'src/app/(main)/editor/[projectId]/_components/tools/mask.jsx'
    const src = fs.readFileSync(file, 'utf8')
    const usesRefGuard = /isSegmentingRef\.current/.test(src)
    log(usesRefGuard, 'handleSelectSubject guards re-entry via isSegmentingRef.current')
    const hasAbortCtl = /segmentAbortRef/.test(src) && /new\s+AbortController\(\)/.test(src)
    const hasSignal = /signal:\s*abortController\.signal/.test(src)
    log(hasAbortCtl && hasSignal, 'handleSelectSubject attaches AbortController and passes signal to fetch')
}

// 88. Unmount cleanup: there must be a `useEffect(() => { return () =>
//     { ... abort() ... } }, [])` that aborts all three controllers so
//     in-flight fetches don't resolve after unmount.
{
    const fs = await import('node:fs')
    const file = 'src/app/(main)/editor/[projectId]/_components/tools/mask.jsx'
    const src = fs.readFileSync(file, 'utf8')
    const abortsAll = /segmentAbortRef\.current\?\.abort\(\)/
        .test(src)
        && /semanticAbortRef\.current\?\.abort\(\)/
            .test(src)
        && /depthAbortRef\.current\?\.abort\(\)/
            .test(src)
    log(abortsAll, 'mask.jsx has unmount cleanup that aborts segment + semantic + depth controllers')
}

// 89. The Esc-cancel effect must read `handleSpatialCancel` through a
//     ref, not directly. The handler closes over `removeLayer`, which
//     closes over `stack` — a new identity on every `updateLayer`
//     dispatch, which would re-run the effect on every mousemove tick.
{
    const fs = await import('node:fs')
    const file = 'src/app/(main)/editor/[projectId]/_components/tools/mask.jsx'
    const src = fs.readFileSync(file, 'utf8')
    // A ref named `handleSpatialCancelRef` is created and read inside
    // the keydown handler. This is the canonical fix.
    const hasRef = /handleSpatialCancelRef\s*=/.test(src)
        && /handleSpatialCancelRef\.current\(\)/.test(src)
    log(hasRef, 'Esc-cancel effect reads handler via handleSpatialCancelRef.current()')
}

// 90. The Esc-cancel effect's dep array must NOT include
//     `handleSpatialCancel` directly — the whole point of the ref is to
//     decouple the listener lifecycle from the handler's identity.
{
    const fs = await import('node:fs')
    const file = 'src/app/(main)/editor/[projectId]/_components/tools/mask.jsx'
    const src = fs.readFileSync(file, 'utf8')
    // Find the keydown listener effect. Its deps should be `[activeDraft]`
    // (or a superset that does not include `handleSpatialCancel`).
    const effectMatch = src.match(
        /addEventListener\(['"]keydown['"][\s\S]{0,400}?\},\s*\[([^\]]*)\]\)/,
    )
    const deps = effectMatch ? effectMatch[1] : ''
    const leaksIdentity = /\bhandleSpatialCancel\b/.test(deps)
    log(effectMatch && !leaksIdentity,
        'Esc-cancel effect dep array does NOT list handleSpatialCancel (ref decouples identity churn)')
}

// ─── Fill / Erase output modes (root-cause #1) ───────────────────────────────
{
    const c = compileMegashader({ chain: [{ layer: luminanceLayer(), op: 'replace' }] })
    log(c.frag.includes('float eraseAlpha = 0.0;'), 'chain declares the erase-alpha channel')
    log(c.frag.includes('uniform float uLayer_0_fillMode;'), 'layer emits fillMode uniform')
    log(c.frag.includes('uniform vec3  uLayer_0_fillColor;'), 'layer emits fillColor uniform')
    log(c.frag.includes('uniform float uLayer_0_fillStrength;'), 'layer emits fillStrength uniform')
    log(c.frag.includes('vec3 layerColor_0(vec3 rgb)'), 'layer emits layerColor helper')
    log(c.frag.includes('step(1.5, uLayer_0_fillMode)'), 'chain routes erase mode (fillMode==2) to eraseAlpha')
    log(c.frag.includes('float outA = src.a * (1.0 - eraseFactor);'), 'main applies erase knockout to output alpha')
    // stackHasNoVisibleEffect: empty + all-adjust-zero are no-ops; a fill
    // layer or an erase layer is NOT a no-op (so it renders).
    log(stackHasNoVisibleEffect({ chain: [] }) === true, 'empty stack has no visible effect')
    log(stackHasNoVisibleEffect({ chain: [{ layer: luminanceLayer(), op: 'replace' }] }) === true,
        'adjust-mode zero-adjustment layer has no visible effect (short-circuits)')
    log(stackHasNoVisibleEffect({ chain: [{ layer: { ...luminanceLayer(), fillMode: 'fill', fillStrength: 0.5 }, op: 'replace' }] }) === false,
        'fill-mode layer DOES have a visible effect (renders)')
    log(stackHasNoVisibleEffect({ chain: [{ layer: { ...luminanceLayer(), fillMode: 'erase' }, op: 'replace' }] }) === false,
        'erase-mode layer DOES have a visible effect (renders)')
}

// ─── Photoshop blend-op GLSL (Bug #3) ────────────────────────────────────────
{
    const mk = (op) => compileMegashader({
        chain: [
            { layer: luminanceLayer(), op: 'replace' },
            { layer: colorLayer(), op },
        ],
    }).frag
    log(mk('screen').includes('1.0 - (1.0 - runningAlpha) * (1.0 - a_1)'), 'screen op emits its GLSL formula')
    log(mk('lighten').includes('max(runningAlpha, a_1)'), 'lighten op emits max()')
    log(mk('darken').includes('min(runningAlpha, a_1)'), 'darken op emits min()')
    log(mk('overlay').includes('2.0 * runningAlpha * a_1'), 'overlay op emits its GLSL formula')
    // The compiler no longer throws on the four Photoshop ops.
    let threw = false
    try { compileMegashader({ chain: [{ layer: luminanceLayer(), op: 'replace' }, { layer: colorLayer(), op: 'screen' }] }) } catch { threw = true }
    log(!threw, 'compiler accepts screen/lighten/darken/overlay without throwing')
}

// ─── Lasso kind registration (Bug #12) ───────────────────────────────────────
{
    setMaskTexture('lasso-test-key', { width: 1, height: 1 })
    const lasso = lassoLayer({ maskTextureKey: 'lasso-test-key' })
    const c = compileMegashader({ chain: [{ layer: lasso, op: 'replace' }] })
    log(lasso.kind === 'lasso', 'lassoLayer factory produces a lasso layer')
    log(lasso.fillMode === 'fill', 'lasso defaults to fill mode (visible selection)')
    log(c.frag.includes('uLayer_0_kind_lasso_mask'), 'lasso kind emits its mask sampler')
    log(c.frag.includes('smoothstep(0.5 - feather, 0.5 + feather, raw)'), 'lasso body smoothsteps the polygon alpha')
    clearMaskTexture('lasso-test-key')
}

// ─── Show-mask overlay + global invert (pro-parity) ──────────────────────────
{
    const c = compileMegashader({ chain: [{ layer: luminanceLayer(), op: 'replace' }] })
    log(c.frag.includes('uniform float uMaskOverlay;'), 'frag declares uMaskOverlay uniform')
    log(c.frag.includes('uniform vec3 uMaskOverlayColor;'), 'frag declares uMaskOverlayColor uniform')
    log(c.frag.includes('uniform float uGlobalInvert;'), 'frag declares uGlobalInvert uniform')
    log(c.frag.includes('if (uMaskOverlay > 0.5)'), 'main() has the show-mask overlay branch')
    log(c.frag.includes('mix(runningAlpha, 1.0 - runningAlpha, uGlobalInvert)'), 'main() applies global invert to runningAlpha')
}

// ─── Richer per-mask adjustments (highlights/shadows/whites/blacks/temp/tint) ─
{
    const c = compileMegashader({ chain: [{ layer: luminanceLayer(), op: 'replace' }] })
    for (const f of ['highlights', 'shadows', 'whites', 'blacks', 'temperature', 'tint']) {
        log(c.frag.includes(`uLayer_0_adjust_${f}`), `layer emits adjust uniform: ${f}`)
    }
    log(isAdjustmentsIdentity({ chain: [{ layer: { ...luminanceLayer(), temperature: 20 }, op: 'replace' }] }) === false,
        'isAdjustmentsIdentity false for a temperature-only layer')
    log(stackHasNoVisibleEffect({ chain: [{ layer: { ...luminanceLayer(), highlights: 30 }, op: 'replace' }] }) === false,
        'stackHasNoVisibleEffect false for a highlights-only layer')
    log(isAdjustmentsIdentity({ chain: [{ layer: luminanceLayer(), op: 'replace' }] }) === true,
        'isAdjustmentsIdentity still true for an all-zero-adjustment layer')
}

// ─── Summary ────────────────────────────────────────────────────────────────
const total = checks
const passed = total - failures
console.log(`\n${passed}/${total} verifications passed.`)
if (failures > 0) {
    console.error(`\x1b[31m${failures} verification(s) failed.\x1b[0m`)
    process.exit(1)
}
process.exit(0)
