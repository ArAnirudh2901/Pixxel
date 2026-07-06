/**
 * Browser harness for the phosmith EDITOR megashader render path.
 *
 * Bundled by scripts/verify-mask-render.mjs (`bun build --target=browser`) so
 * the EXACT production path the editor uses — `applyMegashaderFilter(image,
 * stack)` on a real FabricImage with Fabric's Canvas2D filter backend — runs
 * in a real browser with no Next/Clerk/env.
 *
 * Coverage (production regression matrix):
 *   1. smoke: fill / adjust / maskOverlay localisation (the original #14 bug)
 *   2. ALL 10 mask kinds render a localised effect through the Fabric filter
 *   3. boolean blend ops (add / subtract / intersect) compose correctly
 *   4. erase fillMode knocks alpha out; inverted flips coverage; feather
 *      produces a soft boundary
 *   5. chain truncation at MAX_LAYERS doesn't throw
 *   6. persistence round-trip: toObject → fromObject renders pixel-identical
 */

import { StaticCanvas, FabricImage, config as fabricConfig } from 'fabric'
import { applyMegashaderFilter } from '../../src/lib/megashader/apply-megashader.js'
import { MegashaderFilter } from '../../src/lib/megashader/fabric-megashader-filter.js'
import {
    setMaskTexture, getMaskTexture, sanitiseFill,
    semanticLayer, luminanceLayer, colorLayer, linearLayer, radialLayer,
    lassoLayer, brushLayer, pathLayer, smartBrushLayer, depthLayer,
} from '../../src/lib/megashader/mask-types.js'
import { buildPackedLutFromCurves } from '../../src/lib/curve-lut.js'

// Match canvas.jsx exactly: force the Canvas2D filter backend.
if (fabricConfig) fabricConfig.enableGLFiltering = false

// Record every [megashader] warning — failures degrade to silent CPU
// passthrough (#14/#20/#21), so the final check fails the run if ANY fired.
const shaderWarnings = []
{
    const origWarn = console.warn.bind(console)
    console.warn = (...args) => {
        const msg = args.map((a) => (typeof a === 'string' ? a : String(a && a.message || a))).join(' ')
        if (msg.includes('[megashader]')) shaderWarnings.push(msg)
        origWarn(...args)
    }
}

const W = 200, H = 200
const MAGENTA = { r: 1, g: 0, b: 0.6 }

function grayCanvas() {
    const c = document.createElement('canvas'); c.width = W; c.height = H
    const ctx = c.getContext('2d')
    ctx.fillStyle = 'rgb(128,128,128)'; ctx.fillRect(0, 0, W, H)
    return c
}
// Left half white / right black — luma-styled opaque mask (lasso/path/semantic style).
function leftLumaCanvas() {
    const c = document.createElement('canvas'); c.width = W; c.height = H
    const ctx = c.getContext('2d')
    ctx.fillStyle = 'black'; ctx.fillRect(0, 0, W, H)
    ctx.fillStyle = 'white'; ctx.fillRect(0, 0, W / 2, H)
    return c
}
function leftHalfImageData() {
    return leftLumaCanvas().getContext('2d').getImageData(0, 0, W, H)
}
// Top half opaque white / bottom transparent — alpha-styled (brush strokes).
function topAlphaCanvas() {
    const c = document.createElement('canvas'); c.width = W; c.height = H
    const ctx = c.getContext('2d')
    ctx.fillStyle = 'rgba(255,255,255,1)'; ctx.fillRect(0, 0, W, H / 2)
    return c
}
// Horizontal depth ramp: black (far) left → white (near) right.
function depthRampImageData() {
    const c = document.createElement('canvas'); c.width = W; c.height = H
    const ctx = c.getContext('2d')
    const g = ctx.createLinearGradient(0, 0, W, 0)
    g.addColorStop(0, 'black'); g.addColorStop(1, 'white')
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H)
    return ctx.getImageData(0, 0, W, H)
}

function buildFabric() {
    const el = document.createElement('canvas'); el.width = W; el.height = H
    document.body.appendChild(el)
    const fcanvas = new StaticCanvas(el, { width: W, height: H, renderOnAddRemove: false, enableRetinaScaling: false })
    const img = new FabricImage(grayCanvas(), { left: 0, top: 0, objectCaching: false })
    fcanvas.add(img)
    fcanvas.renderAll()
    return { el, fcanvas, img }
}
function samplerOf(img) {
    const el = img._element || img.getElement()
    const ctx = el.getContext('2d', { willReadFrequently: true })
    return (x, y) => { const d = ctx.getImageData(x, y, 1, 1).data; return [d[0], d[1], d[2], d[3]] }
}
const isTinted = (p) => p[0] > 150 && p[1] < 90        // magenta-ish over gray
const isGray = (p) => Math.abs(p[0] - 128) < 25 && Math.abs(p[1] - 128) < 25 && Math.abs(p[2] - 128) < 25

function applyChain(img, chainEntries, opts = {}) {
    applyMegashaderFilter(img, { chain: chainEntries }, { globalMaskAlpha: 1, ...opts })
}
const fillOn = (layer) => ({ ...layer, ...sanitiseFill({ fillMode: 'fill', fillColor: MAGENTA, fillStrength: 1 }) })

async function run() {
    const checks = []
    const ok = (label, cond, detail) => checks.push({ label, ok: !!cond, detail })

    /* ── 1. smoke (original #14 regression) ─────────────────────────── */
    {
        const { img, fcanvas } = buildFabric()
        setMaskTexture('k-smoke', leftHalfImageData())
        applyChain(img, [{ op: 'replace', layer: fillOn(semanticLayer({ maskTextureKey: 'k-smoke', feather: 0 })) }])
        fcanvas.renderAll()
        const s = samplerOf(img)
        ok('smoke fill: masked tinted / unmasked gray', isTinted(s(50, 100)) && isGray(s(150, 100)),
            `L=${s(50, 100)} R=${s(150, 100)}`)

        const { img: i2 } = buildFabric()
        setMaskTexture('k-smoke2', leftHalfImageData())
        applyChain(i2, [{ op: 'replace', layer: { ...semanticLayer({ maskTextureKey: 'k-smoke2', feather: 0 }), ...sanitiseFill({ fillMode: 'adjust' }), exposure: 0.9 } }])
        const s2 = samplerOf(i2)
        ok('smoke adjust: exposure localised', s2(50, 100)[0] - 128 > 20 && Math.abs(s2(150, 100)[0] - 128) < 12,
            `ΔL=${s2(50, 100)[0] - 128} ΔR=${s2(150, 100)[0] - 128}`)

        const { img: i3 } = buildFabric()
        setMaskTexture('k-smoke3', leftHalfImageData())
        applyChain(i3, [{ op: 'replace', layer: { ...semanticLayer({ maskTextureKey: 'k-smoke3', feather: 0 }), ...sanitiseFill({ fillMode: 'adjust' }) } }], { maskOverlay: true })
        const s3 = samplerOf(i3)
        ok('smoke overlay: Show-mask visualises', !isGray(s3(50, 100)) && isGray(s3(150, 100)), `L=${s3(50, 100)}`)
    }

    /* ── 2. every kind renders localised ────────────────────────────── */
    const kindCases = [
        ['semantic', () => { setMaskTexture('kk-sem', leftHalfImageData()); return semanticLayer({ maskTextureKey: 'kk-sem', feather: 0 }) }, [50, 100], [150, 100]],
        ['lasso', () => { setMaskTexture('kk-las', leftLumaCanvas()); return lassoLayer({ maskTextureKey: 'kk-las', feather: 0 }) }, [50, 100], [150, 100]],
        ['path', () => { setMaskTexture('kk-pth', leftLumaCanvas()); return pathLayer({ maskTextureKey: 'kk-pth', feather: 0 }) }, [50, 100], [150, 100]],
        ['brush', () => { setMaskTexture('kk-brs', topAlphaCanvas()); return brushLayer({ maskTextureKey: 'kk-brs' }) }, [100, 40], [100, 160]],
        // smartBrush samples the brush texture's RED channel (luma-styled
        // opaque canvas), not alpha — use the white-on-black mask.
        ['smartBrush', () => { setMaskTexture('kk-sbr', leftLumaCanvas()); return smartBrushLayer({ brushTextureKey: 'kk-sbr', filterRadius: 2 }) }, [50, 100], [150, 100]],
        ['depth', () => { setMaskTexture('kk-dep', depthRampImageData()); return depthLayer({ depthMapKey: 'kk-dep', min: 0.7, max: 1, softness: 0.02 }) }, [190, 100], [10, 100]],
        ['luminance', () => luminanceLayer({ min: 0.3, max: 0.7, softness: 0.02 }), [100, 100], null],
        // color: exercised separately below with a red target region (hue
        // dominates the distance metric; a pure-gray target is degenerate).
        // linear: a BAND of half-width `feather` centred at `position` along
        // p1→p2 — probe inside the band (x≈100) vs far outside (x=10).
        ['linear', () => linearLayer({ imageSize: { width: W, height: H }, p1: { x: 0, y: 100 }, p2: { x: W, y: 100 }, position: 0.5, feather: 0.08 }), [100, 100], [10, 100]],
        ['radial', () => radialLayer({ imageSize: { width: W, height: H }, center: { x: 50, y: 100 }, radius: { x: 45, y: 45 }, feather: 0.02 }), [50, 100], [180, 20]],
    ]
    for (const [kind, make, inPt, outPt] of kindCases) {
        try {
            const { img, fcanvas } = buildFabric()
            applyChain(img, [{ op: 'replace', layer: fillOn(make()) }])
            fcanvas.renderAll()
            const s = samplerOf(img)
            const pin = s(inPt[0], inPt[1])
            const pout = outPt ? s(outPt[0], outPt[1]) : null
            ok(`kind ${kind} renders`, isTinted(pin) || !isGray(pin), `in=${pin}${pout ? ` out=${pout}` : ''}`)
            if (outPt) ok(`kind ${kind} localised`, isGray(pout), `out=${pout}`)
        } catch (e) {
            ok(`kind ${kind} renders`, false, String(e).slice(0, 120))
        }
    }

    // color kind — a red patch on the source + red target (the real UX:
    // the user picks a colour from the image). Gray stays unselected.
    {
        const el = document.createElement('canvas'); el.width = W; el.height = H
        document.body.appendChild(el)
        const fcanvas = new StaticCanvas(el, { width: W, height: H, renderOnAddRemove: false, enableRetinaScaling: false })
        const src = grayCanvas()
        const sctx = src.getContext('2d')
        sctx.fillStyle = 'rgb(220,30,30)'; sctx.fillRect(0, 0, W / 2, H)
        const img = new FabricImage(src, { left: 0, top: 0, objectCaching: false })
        fcanvas.add(img); fcanvas.renderAll()
        applyChain(img, [{ op: 'replace', layer: { ...fillOn(colorLayer({ target: { h: 0, s: 0.86, b: 0.86 }, tolerance: 0.25, softness: 0.05 })), fillColor: { r: 0, g: 1, b: 0.4 } } }])
        fcanvas.renderAll()
        const s = samplerOf(img)
        const inP = s(50, 100), outP = s(150, 100)
        const greenIn = inP[1] > 150 && inP[0] < 110
        ok('kind color renders (red target)', greenIn, `in=${inP}`)
        ok('kind color localised', isGray(outP), `out=${outP}`)
    }

    /* ── 3. boolean blend ops ───────────────────────────────────────── */
    {
        const opCases = [
            ['add', { TL: true, TR: true, BL: true, BR: false }],
            ['subtract', { TL: false, TR: false, BL: true, BR: false }],
            ['intersect', { TL: true, TR: false, BL: false, BR: false }],
        ]
        for (const [op, expect] of opCases) {
            setMaskTexture('op-left', leftHalfImageData())
            setMaskTexture('op-top', topAlphaCanvas())
            const { img, fcanvas } = buildFabric()
            applyChain(img, [
                { op: 'replace', layer: fillOn(semanticLayer({ maskTextureKey: 'op-left', feather: 0 })) },
                { op, layer: fillOn(brushLayer({ maskTextureKey: 'op-top' })) },
            ])
            fcanvas.renderAll()
            const s = samplerOf(img)
            const probes = { TL: s(50, 50), TR: s(150, 50), BL: s(50, 150), BR: s(150, 150) }
            const pass = Object.entries(expect).every(([k, want]) => isTinted(probes[k]) === want)
            ok(`blend op ${op}`, pass, Object.entries(probes).map(([k, p]) => `${k}=${isTinted(p) ? 'T' : 'g'}`).join(' '))
        }
    }

    /* ── 4. erase / inverted / feather semantics ────────────────────── */
    {
        const { img, fcanvas } = buildFabric()
        setMaskTexture('sem-erase', leftHalfImageData())
        applyChain(img, [{ op: 'replace', layer: { ...semanticLayer({ maskTextureKey: 'sem-erase', feather: 0 }), ...sanitiseFill({ fillMode: 'erase' }) } }])
        fcanvas.renderAll()
        const s = samplerOf(img)
        ok('erase: masked alpha knocked out', s(50, 100)[3] < 40 && s(150, 100)[3] > 200,
            `αL=${s(50, 100)[3]} αR=${s(150, 100)[3]}`)

        const { img: i2, fcanvas: f2 } = buildFabric()
        setMaskTexture('sem-inv', leftHalfImageData())
        applyChain(i2, [{ op: 'replace', layer: { ...fillOn(semanticLayer({ maskTextureKey: 'sem-inv', feather: 0 })), inverted: true } }])
        f2.renderAll()
        const s2 = samplerOf(i2)
        ok('inverted: coverage flipped', isGray(s2(50, 100)) && isTinted(s2(150, 100)),
            `L=${s2(50, 100)} R=${s2(150, 100)}`)

        // Feather remaps an existing soft ramp (no-op on binary masks — see
        // buildSemantic), so feed a ramp and assert wider feather ⇒ more
        // partially-covered columns (green 15..115 = partial coverage).
        const ramp = depthRampImageData()  // black(left) → white(right) ramp
        const softColumns = (feather) => {
            setMaskTexture('sem-fth', ramp)
            const { img, fcanvas } = buildFabric()
            applyChain(img, [{ op: 'replace', layer: fillOn(semanticLayer({ maskTextureKey: 'sem-fth', feather })) }])
            fcanvas.renderAll()
            const s = samplerOf(img)
            let n = 0
            for (let x = 20; x < W - 4; x += 8) { const g = s(x, 100)[1]; if (g > 15 && g < 115) n += 1 }
            return n
        }
        const wide = softColumns(0.5)
        const narrow = softColumns(0.02)
        ok('feather: widens soft-mask transition', wide > narrow && wide >= 3,
            `soft columns wide(0.5)=${wide} narrow(0.02)=${narrow}`)
    }

    /* ── 5. chain truncation at MAX_LAYERS doesn't throw ────────────── */
    {
        try {
            const { img, fcanvas } = buildFabric()
            const chain = []
            for (let i = 0; i < 9; i++) {
                chain.push({ op: i === 0 ? 'replace' : 'add', layer: fillOn(luminanceLayer({ min: 0, max: 1, softness: 0 })) })
            }
            applyChain(img, chain)
            fcanvas.renderAll()
            const s = samplerOf(img)
            ok('9-layer chain truncates without throwing', isTinted(s(100, 100)), `px=${s(100, 100)}`)
        } catch (e) {
            ok('9-layer chain truncates without throwing', false, String(e).slice(0, 120))
        }
    }

    /* ── 6. persistence round-trip (toObject → fromObject) ──────────── */
    {
        try {
            setMaskTexture('per-sem', leftHalfImageData())
            setMaskTexture('per-dep', depthRampImageData())
            const chain = [
                { op: 'replace', layer: fillOn(semanticLayer({ maskTextureKey: 'per-sem', feather: 0 })) },
                { op: 'add', layer: fillOn(depthLayer({ depthMapKey: 'per-dep', min: 0.7, max: 1, softness: 0.02 })) },
            ]
            const { img, fcanvas } = buildFabric()
            const filter = applyMegashaderFilter(img, { chain })
            fcanvas.renderAll()
            const s1 = samplerOf(img)
            const ref = [s1(50, 100), s1(190, 100), s1(150, 50)]

            const obj = filter.toObject()
            // simulate reload: wipe both textures so ONLY the persisted copies count
            setMaskTexture('per-sem', null)
            setMaskTexture('per-dep', null)
            const restored = await MegashaderFilter.fromObject(JSON.parse(JSON.stringify(obj)))
            const { img: i2, fcanvas: f2 } = buildFabric()
            i2.filters.push(restored)
            i2.applyFilters()
            f2.renderAll()
            const s2 = samplerOf(i2)
            const got = [s2(50, 100), s2(190, 100), s2(150, 50)]
            const near = ref.every((p, i) => p.every((v, j) => Math.abs(v - got[i][j]) <= 6))
            ok('persistence round-trip pixel-stable', near,
                `ref=${JSON.stringify(ref)} got=${JSON.stringify(got)}`)
        } catch (e) {
            ok('persistence round-trip pixel-stable', false, String(e).slice(0, 140))
        }
    }

    /* ── 6b. grade persistence: curve-LUT rebuild + baseTextureKey ──── */
    // Curves only (gamma 1) so a pixel match after restore can ONLY come from
    // fromObject rebuilding the (never-serialised) LUT texture from the points.
    {
        try {
            setMaskTexture('per-sem2', leftHalfImageData())
            const white = document.createElement('canvas'); white.width = W; white.height = H
            const wctx = white.getContext('2d'); wctx.fillStyle = '#fff'; wctx.fillRect(0, 0, W, H)
            setMaskTexture('per-base2', white)
            const curves = { master: [{ x: 0, y: 0.35 }, { x: 1, y: 1 }] }
            const { packed } = buildPackedLutFromCurves(curves)
            setMaskTexture('curve-rt', new ImageData(new Uint8ClampedArray(packed), 256, 1))
            const layer = {
                ...semanticLayer({ maskTextureKey: 'per-sem2', feather: 0 }),
                fillMode: 'adjust', curves, curveLutKey: 'curve-rt', baseTextureKey: 'per-base2',
            }
            const { img, fcanvas } = buildFabric()
            const filter = applyMegashaderFilter(img, { chain: [{ op: 'replace', layer }] })
            fcanvas.renderAll()
            const s1 = samplerOf(img)
            const ref = [s1(50, 100), s1(190, 100)]
            ok('grade persistence: curve visibly renders pre-save', Math.abs(ref[0][0] - 128) > 10, `in-mask=${ref[0]}`)

            const obj = filter.toObject()
            setMaskTexture('per-sem2', null)
            setMaskTexture('per-base2', null)
            setMaskTexture('curve-rt', null) // LUT is never serialised — must be rebuilt
            const restored = await MegashaderFilter.fromObject(JSON.parse(JSON.stringify(obj)))
            const { img: i2, fcanvas: f2 } = buildFabric()
            i2.filters.push(restored)
            i2.applyFilters()
            f2.renderAll()
            const s2 = samplerOf(i2)
            const got = [s2(50, 100), s2(190, 100)]
            const near = ref.every((p, i) => p.every((v, j) => Math.abs(v - got[i][j]) <= 6))
            ok('grade persistence: curve-LUT round-trip pixel-stable', near,
                `ref=${JSON.stringify(ref)} got=${JSON.stringify(got)}`)
            ok('grade persistence: baseTextureKey restored', !!getMaskTexture('per-base2'),
                `persisted keys=${Object.keys(obj.textures || {}).join(',')}`)
        } catch (e) {
            ok('grade persistence: curve-LUT round-trip pixel-stable', false, String(e).slice(0, 140))
        }
    }

    /* ── 6c. base grade: two-pass pre-grade + persistence ───────────── */
    {
        try {
            // Base curves lift the WHOLE image; a fill layer sits on top. The
            // unmasked region must show the base grade (proves pass 1 ran).
            setMaskTexture('bg-sem', leftHalfImageData())
            const curves = { master: [{ x: 0, y: 0.4 }, { x: 1, y: 1 }] }
            const { packed } = buildPackedLutFromCurves(curves)
            setMaskTexture('curve-base', new ImageData(new Uint8ClampedArray(packed), 256, 1))
            const base = { gamma: 1, curves, curveLutKey: 'curve-base' }
            const chain = [{ op: 'replace', layer: fillOn(semanticLayer({ maskTextureKey: 'bg-sem', feather: 0 })) }]
            const { img, fcanvas } = buildFabric()
            const filter = applyMegashaderFilter(img, { chain, base })
            fcanvas.renderAll()
            const s1 = samplerOf(img)
            const outMask = s1(190, 100) // right half: no layer, base-only
            ok('base grade: pre-pass grades unmasked region', Math.abs(outMask[0] - 128) > 10, `out=${outMask}`)
            ok('base grade: masked region still tinted', isTinted(s1(50, 100)), `in=${s1(50, 100)}`)

            const obj = filter.toObject()
            setMaskTexture('bg-sem', null)
            setMaskTexture('curve-base', null)
            const restored = await MegashaderFilter.fromObject(JSON.parse(JSON.stringify(obj)))
            const { img: i2, fcanvas: f2 } = buildFabric()
            i2.filters.push(restored)
            i2.applyFilters()
            f2.renderAll()
            const s2 = samplerOf(i2)
            const near = [ [50, 100], [190, 100] ].every(([x, y]) =>
                s1(x, y).every((v, j) => Math.abs(v - s2(x, y)[j]) <= 6))
            ok('base grade: round-trip pixel-stable (LUT rebuilt)', near,
                `in=${s2(50, 100)} out=${s2(190, 100)}`)
        } catch (e) {
            ok('base grade: two-pass renders', false, String(e).slice(0, 140))
        }
    }

    /* ── 7. kind-combination compile sweep ──────────────────────────── */
    // Multi-kind programs can fail where single-kind ones pass (helper
    // collisions, slot-name clashes). Probes are loose; gate 8 catches
    // compile failures.
    {
        try {
            setMaskTexture('cmb-sem', leftHalfImageData())
            setMaskTexture('cmb-las', leftLumaCanvas())
            setMaskTexture('cmb-pth', leftLumaCanvas())
            setMaskTexture('cmb-brs', topAlphaCanvas())
            setMaskTexture('cmb-sbr', leftLumaCanvas())
            setMaskTexture('cmb-dep', depthRampImageData())
            const all8 = [
                semanticLayer({ maskTextureKey: 'cmb-sem', feather: 0 }),
                lassoLayer({ maskTextureKey: 'cmb-las', feather: 0 }),
                pathLayer({ maskTextureKey: 'cmb-pth', feather: 0 }),
                brushLayer({ maskTextureKey: 'cmb-brs' }),
                smartBrushLayer({ brushTextureKey: 'cmb-sbr', filterRadius: 2 }),
                depthLayer({ depthMapKey: 'cmb-dep', min: 0.7, max: 1, softness: 0.02 }),
                colorLayer({ target: { h: 0, s: 0.86, b: 0.86 }, tolerance: 0.25, softness: 0.05 }),
                luminanceLayer({ min: 0.3, max: 0.7, softness: 0.02 }),
            ]
            const { img, fcanvas } = buildFabric()
            applyChain(img, all8.map((layer, i) => ({ op: i === 0 ? 'replace' : 'add', layer: fillOn(layer) })))
            fcanvas.renderAll()
            const s = samplerOf(img)
            ok('combo: all-8-kinds program renders', isTinted(s(50, 100)), `px=${s(50, 100)}`)
        } catch (e) {
            ok('combo: all-8-kinds program renders', false, String(e).slice(0, 140))
        }
        try {
            // linear + radial + DUPLICATE heavy kinds (color ×2, smartBrush ×2)
            setMaskTexture('cmb-sb2', leftLumaCanvas())
            const dupes = [
                linearLayer({ imageSize: { width: W, height: H }, p1: { x: 0, y: 100 }, p2: { x: W, y: 100 }, position: 0.5, feather: 0.08 }),
                radialLayer({ imageSize: { width: W, height: H }, center: { x: 50, y: 100 }, radius: { x: 45, y: 45 }, feather: 0.02 }),
                colorLayer({ target: { h: 120, s: 0.5, b: 0.5 }, tolerance: 0.1 }),
                colorLayer({ target: { h: 240, s: 0.5, b: 0.5 }, tolerance: 0.1 }),
                smartBrushLayer({ brushTextureKey: 'cmb-sbr', filterRadius: 2 }),
                smartBrushLayer({ brushTextureKey: 'cmb-sb2', filterRadius: 3 }),
            ]
            const { img, fcanvas } = buildFabric()
            applyChain(img, dupes.map((layer, i) => ({ op: i === 0 ? 'replace' : 'add', layer: fillOn(layer) })))
            fcanvas.renderAll()
            const s = samplerOf(img)
            ok('combo: duplicate-kind slots render', isTinted(s(100, 100)), `px=${s(100, 100)}`)
        } catch (e) {
            ok('combo: duplicate-kind slots render', false, String(e).slice(0, 140))
        }
    }

    /* ── 8. zero-tolerance: no silent megashader degradation (keep LAST) ── */
    ok('no [megashader] warnings during entire run', shaderWarnings.length === 0,
        shaderWarnings.length ? `${shaderWarnings.length} warning(s); first: ${shaderWarnings[0].slice(0, 160)}` : 'clean')

    return { checks }
}

window.__maskRender = { run }
window.__harnessReady = true
console.log('[harness] mask-render ready')
