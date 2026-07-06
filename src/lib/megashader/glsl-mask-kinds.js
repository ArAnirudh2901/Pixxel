/**
 * Megashader Per-Kind GLSL Bodies + Uniform Schemas
 * --------------------------------------------------
 * One builder + one schema per mask kind. The compiler in
 * `glsl-fragments.js` looks up the right builder via the `KIND_BUILDERS`
 * registry, and the renderer in `megashader-renderer.js` looks up the
 * `KIND_SCHEMAS` registry to drive uniform writes.
 *
 * Adding a new mask kind in a later step (linear, radial, smartBrush,
 * semantic, depth) is a one-file change: drop a builder + a schema here
 * and the compiler/renderer pick it up automatically.
 *
 * Schema field naming convention (matched by the renderer):
 *   - `name`     : the JS-side field name on the layer object
 *   - `glsl`     : the GLSL uniform identifier with `<S>` placeholder
 *                  for the slot index. The renderer substitutes it.
 *   - `type`     : 'float' | 'vec3' (extensible to 'int' if needed)
 *   - `min/max`  : the safe value range; renderer clamps before upload
 *   - `default`  : fallback if the layer field is missing
 *
 * The luminance + color kinds are implemented here (Step 2). The other
 * five kinds (linear, radial, smartBrush, semantic, depth) ship with
 * real bodies and schemas — Steps 3-7 filled them in.
 *
 * @module megashader/glsl-mask-kinds
 */

/* ─── Schemas ──────────────────────────────────────────────────────────── */

/**
 * Luminance range mask: selects pixels whose WCAG sRGB luminosity
 * falls within [min, max] (0..1). `softness` (0..1) is the half-width
 * of the smoothstep transition on each edge.
 */
export const LUMINANCE_SCHEMA = {
    kind: 'luminance',
    uniforms: [
        { name: 'min',      glsl: 'uLayer_<S>_kind_luminance_min',      type: 'float', min: 0, max: 1, default: 0 },
        { name: 'max',      glsl: 'uLayer_<S>_kind_luminance_max',      type: 'float', min: 0, max: 1, default: 0.5 },
        { name: 'softness', glsl: 'uLayer_<S>_kind_luminance_softness', type: 'float', min: 0, max: 1, default: 0.1 },
    ],
}

/**
 * Color (HSB combined) range mask: selects pixels whose HSB-space
 * distance to the target colour is within `tolerance`. `softness`
 * (0..1) is the half-width of the smoothstep transition around the
 * tolerance threshold.
 *
 * The combined metric is:
 *   dh = min(|h - tH|, 360 - |h - tH|) / 180   // cyclic, normalised
 *   ds = |s - tS|
 *   db = |b - tB|
 *   d  = dh * 0.5 + ds * 0.25 + db * 0.25
 *
 * This matches the blueprint's hue-centric intent while adding a
 * little S/B sensitivity so a high-saturation target doesn't match a
 * neutral grey pixel.
 */
export const COLOR_SCHEMA = {
    kind: 'color',
    uniforms: [
        { name: 'targetH',   glsl: 'uLayer_<S>_kind_color_targetH',   type: 'float', min: 0,   max: 360, default: 0 },
        { name: 'targetS',   glsl: 'uLayer_<S>_kind_color_targetS',   type: 'float', min: 0,   max: 1,   default: 1 },
        { name: 'targetB',   glsl: 'uLayer_<S>_kind_color_targetB',   type: 'float', min: 0,   max: 1,   default: 1 },
        { name: 'tolerance', glsl: 'uLayer_<S>_kind_color_tolerance', type: 'float', min: 0,   max: 1,   default: 0.15 },
        { name: 'softness',  glsl: 'uLayer_<S>_kind_color_softness',  type: 'float', min: 0,   max: 1,   default: 0.1 },
    ],
}

/**
 * Step 3 — Procedural Linear Gradient mask.
 * Projects each pixel onto the line P1→P2 (in image-space pixels) and
 * returns 1 inside a band of half-width `feather` centred on `position`,
 * smoothly falling to 0 outside. The pixel position is recovered from
 * the standard `vTextureCoord` varying by multiplying with `uImageSize`
 * (set by the renderer from the source canvas's dimensions).
 *
 * `position` is in 0..1 along the P1→P2 line — 0 means the band is
 * centred at P1, 1 at P2, 0.5 halfway. Values outside [0, 1] place the
 * band off the line (the projection clamps to the segment endpoints).
 *
 * @typedef {Object} LinearSchema
 */
export const LINEAR_SCHEMA = {
    kind: 'linear',
    uniforms: [
        { name: 'p1',       glsl: 'uLayer_<S>_kind_linear_p1',       type: 'vec2',  min: 0, max: 100000, default: [0, 0] },
        { name: 'p2',       glsl: 'uLayer_<S>_kind_linear_p2',       type: 'vec2',  min: 0, max: 100000, default: [100, 0] },
        { name: 'position', glsl: 'uLayer_<S>_kind_linear_position', type: 'float', min: 0, max: 1,     default: 0.5 },
        { name: 'feather',  glsl: 'uLayer_<S>_kind_linear_feather',  type: 'float', min: 0, max: 1,     default: 0.1 },
    ],
}

/**
 * Step 3 — Procedural Radial / Elliptical Gradient mask.
 * Warps each pixel into a unit ellipse in the frame of the layer's
 * centre, rotation, and (x, y) half-axes. The mask is 1 inside the
 * ellipse (radius < 1), 0 outside (radius > 1), with a smooth
 * `feather` band around the boundary.
 *
 * The rotation matrix is `R(-θ)` because we want to *undo* the layer's
 * own rotation when transforming the pixel back to ellipse-local
 * coordinates (the ellipse is axis-aligned in its own frame).
 */
export const RADIAL_SCHEMA = {
    kind: 'radial',
    uniforms: [
        { name: 'center',   glsl: 'uLayer_<S>_kind_radial_center',   type: 'vec2',  min: 0,     max: 100000, default: [50, 50] },
        { name: 'rotation', glsl: 'uLayer_<S>_kind_radial_rotation', type: 'float', min: 0,     max: 6.2832, default: 0 },   // 0..2π
        { name: 'radius',   glsl: 'uLayer_<S>_kind_radial_radius',   type: 'vec2',  min: 0.001, max: 100000, default: [50, 50] },
        { name: 'feather',  glsl: 'uLayer_<S>_kind_radial_feather',  type: 'float', min: 0,     max: 1,     default: 0.1 },
    ],
}

/**
 * Step 5 — Semantic AI mask (SAM 2 click-to-select).
 *
 * The mask is uploaded by the renderer as a sampler2D (the per-layer
 * `uLayer_<slot>_kind_semantic_mask`). The mask is greyscale; the R
 * channel carries the 0..1 alpha. The optional `feather` uniform widens
 * the smoothstep edges so users can soften a binary mask without
 * re-running the model.
 *
 * Texture upload is driven by the renderer in `megashader-renderer.js`;
 * see `writeKindTextures` for the per-frame sampler binding.
 */
export const SEMANTIC_SCHEMA = {
    kind: 'semantic',
    uniforms: [
        { name: 'feather', glsl: 'uLayer_<S>_kind_semantic_feather', type: 'float', min: 0, max: 1, default: 0.1 },
    ],
    samplers: [
        { name: 'mask', glsl: 'uLayer_<S>_kind_semantic_mask' },
    ],
}

/**
 * Step 6 — Depth-based mask (Depth Anything V2).
 *
 * The depth map is uploaded by the renderer as a per-layer sampler2D
 * (R channel, 0..1). The user picks a depth range; pixels whose depth
 * falls inside `[min, max]` (with smoothstep edges of half-width
 * `softness`) become 1.0, everything else 0.0. White (1) = near,
 * black (0) = far.
 *
 * Texture upload is driven by the renderer in `megashader-renderer.js`;
 * see `bindKindTextures` for the per-frame sampler binding.
 */
export const DEPTH_SCHEMA = {
    kind: 'depth',
    uniforms: [
        { name: 'min',      glsl: 'uLayer_<S>_kind_depth_min',      type: 'float', min: 0, max: 1, default: 0 },
        { name: 'max',      glsl: 'uLayer_<S>_kind_depth_max',      type: 'float', min: 0, max: 1, default: 0.5 },
        { name: 'softness', glsl: 'uLayer_<S>_kind_depth_softness', type: 'float', min: 0, max: 1, default: 0.1 },
    ],
    samplers: [
        { name: 'map', glsl: 'uLayer_<S>_kind_depth_map' },
    ],
}

/**
 * Step 7 — Manual precision Smart Brush (edge-preserving filter).
 *
 * The user's painted alpha texture is uploaded by the renderer as a
 * per-layer sampler2D (R channel, 0..1). The GLSL body runs a
 * bilateral filter over a window around the current pixel: spatial
 * weight × colour weight × brush alpha, then normalised. The colour
 * weight is what makes the filter "edge-preserving" — across a
 * high-contrast edge, the colour weight is tiny so the alpha doesn't
 * bleed past the edge.
 *
 * The filter radius is a *compile-time constant* (the loop is
 * `for (dy = -MAX_RADIUS; ...; dy++)`) so the GLSL compiler can
 * unroll it; the runtime filterRadius uniform bounds the actual
 * sampling via an if-guard inside the loop. `sigmaColor` and
 * `sigmaSpace` control the weights — small sigmas = stricter edges.
 *
 * Texture upload is driven by the renderer in `megashader-renderer.js`;
 * see `bindKindTextures` for the per-frame sampler binding.
 */
export const SMART_BRUSH_SCHEMA = {
    kind: 'smartBrush',
    uniforms: [
        { name: 'filterRadius', glsl: 'uLayer_<S>_kind_smartBrush_filterRadius', type: 'float', min: 1,   max: 8,   default: 3 },
        { name: 'sigmaColor',   glsl: 'uLayer_<S>_kind_smartBrush_sigmaColor',   type: 'float', min: 0.01, max: 1,   default: 0.15 },
        { name: 'sigmaSpace',   glsl: 'uLayer_<S>_kind_smartBrush_sigmaSpace',   type: 'float', min: 0.01, max: 8,   default: 2 },
    ],
    samplers: [
        { name: 'brush', glsl: 'uLayer_<S>_kind_smartBrush_brush' },
    ],
}

/**
 * Lasso mask (freehand / polygonal selection).
 *
 * The closed selection polygon is rasterised client-side to an offscreen
 * alpha canvas (white inside, transparent outside) and uploaded by the
 * renderer as a per-layer sampler2D, exactly like the semantic mask. The
 * R channel carries the 0..1 selection. `feather` softens the edge via a
 * smoothstep around 0.5 — 0 gives the Canvas2D anti-aliased edge, larger
 * values widen the transition band (Photoshop "Feather" on a selection).
 *
 * Because the texture is uploaded with UNPACK_FLIP_Y_WEBGL like every
 * other texture-backed kind, the lasso polygon must be rasterised in
 * canvas-Y-down image space — the upload flip cancels, so it lines up
 * with the photo (this sidesteps the linear/radial Y-flip concern).
 */
export const LASSO_SCHEMA = {
    kind: 'lasso',
    uniforms: [
        { name: 'feather', glsl: 'uLayer_<S>_kind_lasso_feather', type: 'float', min: 0, max: 1, default: 0.05 },
    ],
    samplers: [
        { name: 'mask', glsl: 'uLayer_<S>_kind_lasso_mask' },
    ],
}

/**
 * Plain selection Brush mask (non-destructive, Photoshop-style).
 *
 * Unlike `smartBrush` (which runs an edge-preserving bilateral filter), the
 * plain brush samples the painted alpha texture verbatim — what you paint is
 * what you select. The soft falloff lives in the texture's ALPHA channel (the
 * R channel is a flat 1.0 wherever the brush touched), so the GLSL samples
 * `.a`. The renderer uploads it like every other texture-backed kind
 * (`uLayer_<slot>_kind_brush_mask`), keyed by the layer's `maskTextureKey`.
 */
export const BRUSH_SCHEMA = {
    kind: 'brush',
    uniforms: [],
    samplers: [
        { name: 'mask', glsl: 'uLayer_<S>_kind_brush_mask' },
    ],
}

/**
 * Pen / Bézier path mask (Photoshop "Pen tool" parity).
 *
 * Engine-identical to `lasso`: the closed Bézier path is rasterised
 * client-side (Canvas2D `bezierCurveTo`, filled) to an offscreen alpha
 * canvas — white inside, transparent outside — and uploaded by the renderer
 * as a per-layer sampler2D keyed by `maskTextureKey`. The only difference
 * from the lasso is the authoring geometry (smooth curves between anchors vs
 * straight segments); `feather` softens the edge via a smoothstep around 0.5.
 */
export const PATH_SCHEMA = {
    kind: 'path',
    uniforms: [
        { name: 'feather', glsl: 'uLayer_<S>_kind_path_feather', type: 'float', min: 0, max: 1, default: 0.04 },
    ],
    samplers: [
        { name: 'mask', glsl: 'uLayer_<S>_kind_path_mask' },
    ],
}

/**
 * Step 4 stub kind. Empty schema (no uniforms, no samplers) and a
 * builder that returns 0.0 — kept for any kind we want to revert to a
 * no-op in a hotfix. The compiler and renderer treat stubs identically
 * to real kinds.
 */
const stubSchema = (kind) => ({ kind, uniforms: [] })

export const KIND_SCHEMAS = {
    luminance:  LUMINANCE_SCHEMA,
    color:      COLOR_SCHEMA,
    linear:     LINEAR_SCHEMA,
    radial:     RADIAL_SCHEMA,
    smartBrush: SMART_BRUSH_SCHEMA,
    semantic:   SEMANTIC_SCHEMA,
    depth:      DEPTH_SCHEMA,
    lasso:      LASSO_SCHEMA,
    brush:      BRUSH_SCHEMA,
    path:       PATH_SCHEMA,
}

/* ─── GLSL Body Builders ───────────────────────────────────────────────── */

/**
 * Build the GLSL body for a luminance-range mask. The function reads the
 * three per-layer uniforms declared by `LUMINANCE_SCHEMA`, evaluates
 * WCAG sRGB luminosity, and returns 0..1 via a two-edge smoothstep.
 *
 * @param {number} slot    The layer slot index (0..7).
 * @returns {string}       A complete `float evalLayer_<slot>()` GLSL function.
 */
export const buildLuminance = (slot) => /* glsl */ `
        uniform float uLayer_${slot}_kind_luminance_min;
        uniform float uLayer_${slot}_kind_luminance_max;
        uniform float uLayer_${slot}_kind_luminance_softness;

        float evalLayer_${slot}_body() {
            vec3 rgb = texture2D(uImage, vTextureCoord).rgb;
            float luma = dot(rgb, vec3(0.2126, 0.7152, 0.0722));

            float lo = uLayer_${slot}_kind_luminance_min;
            float hi = uLayer_${slot}_kind_luminance_max;
            // GLSL ES 1.00's smoothstep(a, a, x) is undefined (some
            // drivers return step(a, x), some return 0/1 nondeterministically,
            // some NaN). The UI's softness slider goes down to 0, so we
            // floor to 0.001 here — same defence the linear/radial/depth
            // masks use for their feather field. The visible effect of
            // softness=0.001 vs 0 is sub-pixel and imperceptible.
            float soft = max(uLayer_${slot}_kind_luminance_softness, 0.001);
            // Lower-edge smoothstep: ramps from 0 (below lo) to 1 (above lo).
            float a = smoothstep(lo - soft, lo + soft, luma);
            // Upper-edge inverted: 1 (below hi) ramps to 0 (above hi).
            float b = 1.0 - smoothstep(hi - soft, hi + soft, luma);
            return clamp(a * b, 0.0, 1.0);
        }
    `

/**
 * Build the GLSL body for a color (HSB combined) range mask. Calls the
 * shared `rgbToHsb` already declared in the fragment-shader template.
 *
 * The cyclic hue distance uses degrees (matching the rest of the
 * blueprint's HSB conventions); we normalise to 0..1 inside the metric.
 *
 * @param {number} slot    The layer slot index (0..7).
 * @returns {string}       A complete `float evalLayer_<slot>()` GLSL function.
 */
export const buildColor = (slot) => /* glsl */ `
        uniform float uLayer_${slot}_kind_color_targetH;
        uniform float uLayer_${slot}_kind_color_targetS;
        uniform float uLayer_${slot}_kind_color_targetB;
        uniform float uLayer_${slot}_kind_color_tolerance;
        uniform float uLayer_${slot}_kind_color_softness;

        float evalLayer_${slot}_body() {
            vec3 rgb = texture2D(uImage, vTextureCoord).rgb;
            vec3 hsb = rgbToHsb(rgb);

            // Cyclic hue distance in degrees, normalised to 0..1.
            float diffH = abs(hsb.x - uLayer_${slot}_kind_color_targetH);
            float dh = min(diffH, 360.0 - diffH) / 180.0;
            float ds = abs(hsb.y - uLayer_${slot}_kind_color_targetS);
            float db = abs(hsb.z - uLayer_${slot}_kind_color_targetB);

            // Combined metric — hue dominates (0.5), S+B split the rest.
            float d = dh * 0.5 + ds * 0.25 + db * 0.25;

            float tol = uLayer_${slot}_kind_color_tolerance;
            // Floor softness to 0.001 — see the luminance body's comment
            // for why GLSL ES 1.00's smoothstep(a, a, x) is undefined.
            float soft = max(uLayer_${slot}_kind_color_softness, 0.001);
            // 1 inside the tolerance, smoothly to 0 at tol + soft.
            return 1.0 - smoothstep(tol - soft, tol + soft, d);
        }
    `

/**
 * Build the GLSL body for a linear gradient mask. Reads the four
 * per-layer uniforms (p1, p2, position, feather), projects the
 * pixel onto the P1→P2 line, and returns 1 inside a band of half-
 * width `feather` centred on `position`.
 *
 * The `len2 > 0.0001` guard prevents division-by-zero when P1 == P2
 * (degenerate line). The `max(feather, 0.001)` clamp keeps `smoothstep`
 * well-defined — `smoothstep(a, a, x)` is undefined in GLSL ES 1.00.
 *
 * @param {number} slot    The layer slot index (0..7).
 * @returns {string}
 */
export const buildLinear = (slot) => /* glsl */ `
        uniform vec2  uLayer_${slot}_kind_linear_p1;
        uniform vec2  uLayer_${slot}_kind_linear_p2;
        uniform float uLayer_${slot}_kind_linear_position;
        uniform float uLayer_${slot}_kind_linear_feather;

        float evalLayer_${slot}_body() {
            // Y-flip: the source texture is uploaded with UNPACK_FLIP_Y_WEBGL,
            // so vTextureCoord is GL-Y-up, but p1/p2 are authored in canvas-
            // Y-down image space. Flip Y here so the gradient lands where the
            // user dragged instead of vertically mirrored.
            vec2 pixelPos = vec2(vTextureCoord.x, 1.0 - vTextureCoord.y) * uImageSize;
            vec2 p1 = uLayer_${slot}_kind_linear_p1;
            vec2 p2 = uLayer_${slot}_kind_linear_p2;
            vec2 d = p2 - p1;
            float len2 = dot(d, d);
            // t in 0..1 along the segment (clamps past endpoints naturally
            // because abs(t - position) grows without bound outside the line).
            float t = (len2 > 0.0001) ? dot(pixelPos - p1, d) / len2 : 0.0;
            float pos = uLayer_${slot}_kind_linear_position;
            float feather = uLayer_${slot}_kind_linear_feather;
            float offset = abs(t - pos);
            float halfWidth = max(feather, 0.001);
            return 1.0 - smoothstep(0.0, halfWidth, offset);
        }
    `

/**
 * Build the GLSL body for a radial (elliptical) gradient mask.
 * Reads the four per-layer uniforms (center, rotation, radius,
 * feather), warps the pixel into the ellipse's local frame, and
 * returns 1 inside the ellipse (radius < 1) and 0 outside, with a
 * smooth feathered band at the boundary.
 *
 * The `radius` components are clamped to >= 0.001 in the schema to
 * avoid division by zero. The rotation is radians, 0..2π.
 *
 * @param {number} slot    The layer slot index (0..7).
 * @returns {string}
 */
export const buildRadial = (slot) => /* glsl */ `
        uniform vec2  uLayer_${slot}_kind_radial_center;
        uniform float uLayer_${slot}_kind_radial_rotation;
        uniform vec2  uLayer_${slot}_kind_radial_radius;
        uniform float uLayer_${slot}_kind_radial_feather;

        float evalLayer_${slot}_body() {
            // Y-flip: see buildLinear — center/rotation are authored in
            // canvas-Y-down image space, vTextureCoord is GL-Y-up.
            vec2 pixelPos = vec2(vTextureCoord.x, 1.0 - vTextureCoord.y) * uImageSize;
            vec2 c = uLayer_${slot}_kind_radial_center;
            float rot = uLayer_${slot}_kind_radial_rotation;
            vec2 r = uLayer_${slot}_kind_radial_radius;

            vec2 d = pixelPos - c;
            // Undo the ellipse's rotation so the ellipse is axis-aligned
            // in this frame.
            float cR = cos(-rot);
            float sR = sin(-rot);
            vec2 rotated = vec2(d.x * cR - d.y * sR, d.x * sR + d.y * cR);
            // Normalise by the half-axes; r=1 is the ellipse boundary.
            vec2 normalised = rotated / r;
            float dist = length(normalised);

            float feather = uLayer_${slot}_kind_radial_feather;
            float halfWidth = max(feather, 0.001);
            return 1.0 - smoothstep(1.0 - halfWidth, 1.0 + halfWidth, dist);
        }
    `

/**
 * Build the GLSL body for a semantic (SAM 2) mask layer. Samples the
 * layer's mask texture at the current pixel's UV and returns the R
 * channel as 0..1 alpha. The `feather` uniform softens the edge via a
 * smoothstep around the 0.5 threshold — a 0 feather gives a hard cut
 * (the binary mask is already a 0-or-1 step); larger feathers pull the
 * transition zone wider, useful when the model returns a soft mask.
 *
 * Sampler declaration: each semantic layer owns its own sampler
 * (`uLayer_<slot>_kind_semantic_mask`) so multiple semantic layers can
 * coexist in a stack — the renderer binds each to a unique texture unit.
 *
 * @param {number} slot    The layer slot index (0..7).
 * @returns {string}       A complete `float evalLayer_<slot>_body()` GLSL function.
 */
export const buildSemantic = (slot) => /* glsl */ `
        uniform sampler2D uLayer_${slot}_kind_semantic_mask;
        uniform float uLayer_${slot}_kind_semantic_feather;

        float evalLayer_${slot}_body() {
            float raw = texture2D(uLayer_${slot}_kind_semantic_mask, vTextureCoord).r;
            float feather = max(uLayer_${slot}_kind_semantic_feather, 0.001);
            // Smooth the transition around 0.5 with a symmetric half-width
            // of 'feather'. For a binary mask the smoothstep is a no-op
            // (edges are still hard); for a soft mask it widens the
            // transition band.
            return 1.0 - smoothstep(0.5 - feather, 0.5 + feather, 1.0 - raw);
        }
    `

/**
 * Build the GLSL body for a depth-based mask layer. Samples the
 * layer's depth map at the current pixel's UV and returns 1.0 if
 * the depth falls inside `[min, max]`, 0.0 outside, with a smoothstep
 * edge of half-width `softness` on each threshold. White in the
 * depth map = near (1.0), black = far (0.0) — matches the Python
 * service's per-image min-max normalisation.
 *
 * `softness` is clamped to a minimum of 0.001 because
 * `smoothstep(a, a, x)` is undefined in GLSL ES 1.00.
 *
 * @param {number} slot    The layer slot index (0..7).
 * @returns {string}       A complete `float evalLayer_<slot>_body()` GLSL function.
 */
export const buildDepth = (slot) => /* glsl */ `
        uniform sampler2D uLayer_${slot}_kind_depth_map;
        uniform float uLayer_${slot}_kind_depth_min;
        uniform float uLayer_${slot}_kind_depth_max;
        uniform float uLayer_${slot}_kind_depth_softness;

        float evalLayer_${slot}_body() {
            float depth = texture2D(uLayer_${slot}_kind_depth_map, vTextureCoord).r;
            float dMin = uLayer_${slot}_kind_depth_min;
            float dMax = uLayer_${slot}_kind_depth_max;
            float soft = max(uLayer_${slot}_kind_depth_softness, 0.001);
            // Two-edge smoothstep band: ramp up at dMin, ramp down at dMax.
            // a * b clamps to 0..1 in one multiply; both a and b are
            // already clamped by smoothstep.
            float a = smoothstep(dMin - soft, dMin + soft, depth);
            float b = 1.0 - smoothstep(dMax - soft, dMax + soft, depth);
            return clamp(a * b, 0.0, 1.0);
        }
    `

/**
 * Build the GLSL body for a smart-brush layer. Runs a bilateral
 * filter over a window of `MAX_RADIUS` (compile-time constant)
 * around the current pixel: each neighbour contributes
 * `wSpace * wColor * brushAlpha`, normalised by `sum(wSpace * wColor)`.
 * `wSpace = exp(-|p-q|² / (2 * sigmaSpace²))` and
 * `wColor = exp(-|C(p)-C(q)|² / (2 * sigmaColor²))`. The colour
 * term is what makes the filter edge-preserving — across a sharp
 * colour transition, the weight is tiny, so the alpha doesn't bleed
 * past the edge.
 *
 * Why a fixed loop bound: GLSL ES 1.00 requires constant loop bounds
 * for unrollable inner loops. The actual sampling radius is
 * `filterRadius` (uniform) — values outside that bound are skipped
 * via the `if` guard, so the body is correct for any radius 1..MAX_RADIUS.
 *
 * The `sigmaColor` floor (0.01) and `sigmaSpace` floor (0.01) match
 * the schema's min — keeps the weight exponent finite even if the
 * uniform is somehow uploaded as 0. The normalisation guard
 * `sumW > 0.0` handles a degenerate window (all-zero colour
 * weights — shouldn't happen in practice, but the GL spec says the
 * divisions are by-uniform, not runtime-checked, so we add the guard
 * to avoid a NaN).
 *
 * Source image sampling uses the standard `uImage` (already bound
 * by the renderer to texture unit 0).
 *
 * @param {number} slot    The layer slot index (0..7).
 * @returns {string}       A complete `float evalLayer_<slot>_body()` GLSL function.
 */
export const buildSmartBrush = (slot) => /* glsl */ `
        uniform sampler2D uLayer_${slot}_kind_smartBrush_brush;
        uniform float uLayer_${slot}_kind_smartBrush_filterRadius;
        uniform float uLayer_${slot}_kind_smartBrush_sigmaColor;
        uniform float uLayer_${slot}_kind_smartBrush_sigmaSpace;

        float evalLayer_${slot}_body() {
            // Compile-time loop bound. The runtime filterRadius is the
            // actual cutoff (see the radius guards inside the loop below).
            const int MAX_RADIUS = 8;
            int radiusI = int(uLayer_${slot}_kind_smartBrush_filterRadius + 0.5);
            if (radiusI < 1) radiusI = 1;
            if (radiusI > MAX_RADIUS) radiusI = MAX_RADIUS;

            // Floor the sigmas to keep the Gaussian weight finite even
            // if the uniform comes in as 0 (schema enforces 0.01 min,
            // this is a defense-in-depth for direct uniform writes).
            float sigmaColor = max(uLayer_${slot}_kind_smartBrush_sigmaColor, 0.01);
            float sigmaSpace = max(uLayer_${slot}_kind_smartBrush_sigmaSpace, 0.01);
            float inv2SC = 1.0 / (2.0 * sigmaColor * sigmaColor);
            float inv2SS = 1.0 / (2.0 * sigmaSpace * sigmaSpace);

            // Centre pixel's source-image colour drives the colour-weight
            // distance. texture2D returns vec4 — we only need .rgb.
            vec3 centerColor = texture2D(uImage, vTextureCoord).rgb;
            float centerBrush = texture2D(uLayer_${slot}_kind_smartBrush_brush, vTextureCoord).r;

            // The user-painted brush alpha already excludes a lot of
            // background — if the centre is 0, the filter result should
            // also be near 0. Run a quick early-out: if the centre alpha
            // is below a tiny threshold AND none of the 4-neighbours
            // have any alpha, return 0. This skips the inner loop in
            // the common "clicked-once-and-moved" case where most of the
            // image is empty.
            float sumBrush = 0.0;
            float sumWeight = 0.0;

            for (int dy = -MAX_RADIUS; dy <= MAX_RADIUS; dy += 1) {
                for (int dx = -MAX_RADIUS; dx <= MAX_RADIUS; dx += 1) {
                    // Skip neighbours outside the actual radius. Using a
                    // continue instead of two nested loops so the loop
                    // bounds stay GLSL-ES-1.00-legal. NB: GLSL ES 1.00 has
                    // no integer abs() overload (that arrived in ES 3.00),
                    // so cast to float before abs — abs(int) fails to
                    // compile and takes the whole program down to CPU
                    // passthrough (every smart-brush mask a silent no-op).
                    if (abs(float(dx)) > float(radiusI)) continue;
                    if (abs(float(dy)) > float(radiusI)) continue;
                    vec2 offset = vec2(float(dx), float(dy)) / uImageSize;
                    vec2 sampleUV = vTextureCoord + offset;
                    // Defensive clamp: CLAMP_TO_EDGE on the texture means
                    // the sample is bounded to the image, but if the
                    // uniform is misconfigured the loop could wander
                    // outside [0,1] and produce undefined output.
                    sampleUV = clamp(sampleUV, vec2(0.0), vec2(1.0));

                    vec3 neighborColor = texture2D(uImage, sampleUV).rgb;
                    float neighborBrush = texture2D(uLayer_${slot}_kind_smartBrush_brush, sampleUV).r;

                    // Spatial weight — Gaussian over pixel distance.
                    float r2 = float(dx * dx + dy * dy);
                    float wSpace = exp(-r2 * inv2SS);

                    // Colour weight — Gaussian over RGB distance.
                    vec3 dC = neighborColor - centerColor;
                    float dc2 = dot(dC, dC);
                    float wColor = exp(-dc2 * inv2SC);

                    float w = wSpace * wColor;
                    sumBrush += w * neighborBrush;
                    sumWeight += w;
                }
            }

            // Degenerate-window fallback: every neighbour's colour is
            // outside the colour Gaussian (e.g. centre pixel sits on a
            // hard edge with no nearby same-colour pixels). Returning
            // the raw centerBrush would bypass the bilateral filter
            // entirely and leak the unfiltered painted alpha at the
            // boundary of every stroke. 0.0 is the correct result — the
            // filter found no similar neighbours, so the output is no
            // contribution. The strict > guards the divide and the 1e-6
            // epsilon avoids a divide-by-near-zero on mediump hardware
            // where exp(-large) can underflow to exactly 0 and then
            // sumBrush can be a small but non-zero tail.
            return sumWeight > 1e-6 ? (sumBrush / sumWeight) : 0.0;
        }
    `

/**
 * Build the GLSL body for a lasso (freehand/polygonal) selection layer.
 * Samples the rasterised polygon alpha texture at the current pixel's UV
 * and returns 0..1, softened by `feather` around the 0.5 threshold. White
 * inside the selection (1.0), transparent outside (0.0).
 *
 * @param {number} slot    The layer slot index (0..7).
 * @returns {string}       A complete `float evalLayer_<slot>_body()` GLSL function.
 */
export const buildLasso = (slot) => /* glsl */ `
        uniform sampler2D uLayer_${slot}_kind_lasso_mask;
        uniform float uLayer_${slot}_kind_lasso_feather;

        float evalLayer_${slot}_body() {
            float raw = texture2D(uLayer_${slot}_kind_lasso_mask, vTextureCoord).r;
            float feather = max(uLayer_${slot}_kind_lasso_feather, 0.001);
            // raw ~1 inside the polygon, ~0 outside. Smoothstep around 0.5
            // widens the (already anti-aliased) edge when feather > 0.
            return smoothstep(0.5 - feather, 0.5 + feather, raw);
        }
    `

/**
 * Build the GLSL body for a plain (non-edge-snapping) brush selection layer.
 * Samples the painted alpha texture's A channel directly so the brush's soft
 * hardness falloff is preserved (the R channel is a flat 1.0 wherever the
 * brush touched and would lose the softness). White (1) = selected.
 *
 * @param {number} slot    The layer slot index (0..7).
 * @returns {string}       A complete `float evalLayer_<slot>_body()` GLSL function.
 */
export const buildBrush = (slot) => /* glsl */ `
        uniform sampler2D uLayer_${slot}_kind_brush_mask;

        float evalLayer_${slot}_body() {
            // The painted brush alpha carries the soft falloff in the A
            // channel; clamp for safety on mediump hardware.
            return clamp(texture2D(uLayer_${slot}_kind_brush_mask, vTextureCoord).a, 0.0, 1.0);
        }
    `

/**
 * Build the GLSL body for a Pen / Bézier path selection layer. Identical to
 * the lasso: samples the rasterised path alpha texture (R channel, ~1 inside
 * the filled path, ~0 outside) and softens the edge by `feather` around 0.5.
 *
 * @param {number} slot    The layer slot index (0..7).
 * @returns {string}       A complete `float evalLayer_<slot>_body()` GLSL function.
 */
export const buildPath = (slot) => /* glsl */ `
        uniform sampler2D uLayer_${slot}_kind_path_mask;
        uniform float uLayer_${slot}_kind_path_feather;

        float evalLayer_${slot}_body() {
            float raw = texture2D(uLayer_${slot}_kind_path_mask, vTextureCoord).r;
            float feather = max(uLayer_${slot}_kind_path_feather, 0.001);
            return smoothstep(0.5 - feather, 0.5 + feather, raw);
        }
    `

/**
 * Build a stub GLSL body for kinds not yet implemented (Steps 4-6).
 * Returns 0.0 — the layer contributes no alpha, same as Step 1.
 *
 * Convention: stubs emit `evalLayer_<slot>_body()` (the per-kind
 * shape), NOT the wrapper `evalLayer_<slot>()`. The wrapper is added
 * uniformly by `buildLayerFunction` in glsl-fragments.js.
 *
 * @param {number} slot    The layer slot index (0..7).
 * @param {string} kind    The mask kind (used only for the comment).
 * @returns {string}
 */
export const buildStub = (slot, kind) => /* glsl */ `
        float evalLayer_${slot}_body() {
            return 0.0; // Step 3: stub for kind "${kind}" — replaced in a later step.
        }
    `

/**
 * Registry mapping each known mask kind to its GLSL body builder. The
 * compiler (`buildLayerFunction` in glsl-fragments.js) calls into this
 * map; unknown kinds throw a descriptive error so the failure surfaces
 * at compile-time rather than as a silent broken shader.
 */
export const KIND_BUILDERS = {
    luminance:  buildLuminance,
    color:      buildColor,
    linear:     buildLinear,
    radial:     buildRadial,
    semantic:   buildSemantic,
    depth:      buildDepth,
    smartBrush: buildSmartBrush,
    lasso:      buildLasso,
    brush:      buildBrush,
    path:       buildPath,
}

/**
 * Lookup the GLSL body builder for a kind. Throws on unknown kinds so
 * the compiler fails loud.
 *
 * @param {string} kind
 * @returns {(slot: number) => string}
 */
export const getKindBuilder = (kind) => {
    const builder = KIND_BUILDERS[kind]
    if (!builder) {
        throw new Error(`[megashader] getKindBuilder: unknown mask kind "${kind}"`)
    }
    return builder
}

/**
 * Lookup the uniform schema for a kind. Throws on unknown kinds.
 *
 * @param {string} kind
 * @returns {import('./mask-types').KindSchema}
 */
export const getKindSchema = (kind) => {
    const schema = KIND_SCHEMAS[kind]
    if (!schema) {
        throw new Error(`[megashader] getKindSchema: unknown mask kind "${kind}"`)
    }
    return schema
}

/**
 * Normalise a layer's uniform field value into the canonical shape the
 * renderer expects, so the GLSL uniform write never silently no-ops.
 *
 * Background: the kinds' factories store `vec2` / `vec3` fields as
 * `{ x, y }` / `{ x, y, z }` objects (e.g. `linear.p1`, `radial.center`,
 * `radial.radius`). The renderer's `writeKindUniforms` previously only
 * recognised `Array.isArray(value)` — every `{x, y}` object fell
 * through to the schema's default `[0, 0]`, so the linear / radial
 * mask was rendered at the origin with a zero-radius, regardless of
 * what the user dragged on the canvas. This helper bridges the two
 * shapes so the renderer can use a single uniform-write path.
 *
 * Returns a value the renderer can pass straight to `gl.uniformNf`:
 *   - `float` → `number`
 *   - `vec2`  → `[number, number]`
 *   - `vec3`  → `[number, number, number]`
 *
 * Clamps the numeric components into the schema's `[min, max]` range,
 * falls back to the schema's `default` when the value is missing /
 * non-finite / wrong shape. Components inside an object are clamped
 * individually, so a `{x: 99999, y: 0}` gets clamped to
 * `[clamp(99999), clamp(0)]` rather than rejected wholesale.
 *
 * @param {unknown} raw        The layer's value for this field.
 * @param {{ type: string, min: number, max: number, default: unknown }} field
 *                             The schema's field descriptor.
 * @returns {number | number[]} A value the renderer can pass to
 *                              `gl.uniformNf` directly. Returns the
 *                              field's `default` when the raw value
 *                              can't be coerced.
 */
export const normaliseUniformValue = (raw, field) => {
    const { type, min, max, default: def } = field
    if (type === 'float') {
        if (typeof raw === 'number' && Number.isFinite(raw)) {
            return Math.max(min, Math.min(max, raw))
        }
        return def
    }
    // vec2 / vec3 — accept arrays OR {x, y[, z]} objects.
    const wantLen = type === 'vec2' ? 2 : (type === 'vec3' ? 3 : 0)
    if (wantLen === 0) return def
    const clamp = (v) => (typeof v === 'number' && Number.isFinite(v))
        ? Math.max(min, Math.min(max, v))
        : null
    if (Array.isArray(raw) && raw.length >= wantLen) {
        const out = []
        for (let i = 0; i < wantLen; i += 1) {
            const c = clamp(raw[i])
            out.push(c === null ? 0 : c)
        }
        return out
    }
    if (raw && typeof raw === 'object') {
        const obj = /** @type {Record<string, unknown>} */ (raw)
        const out = []
        for (let i = 0; i < wantLen; i += 1) {
            const c = clamp(obj[String.fromCharCode(120 + i)]) // 'x', 'y', 'z'
            out.push(c === null ? 0 : c)
        }
        return out
    }
    return def
}
