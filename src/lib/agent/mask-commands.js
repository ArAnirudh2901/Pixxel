/**
 * Mask Commands (agent-facing, UI-decoupled)
 * ------------------------------------------
 * Imperative masking operations that drive the megashader mask chain on the
 * active image WITHOUT any React/UI involvement, so the future in-app agent
 * can perform the full masking workflow headlessly. The Mask tool UI and
 * these commands share the SAME engine primitives (`applyMegashaderFilter`,
 * the layer factories, `setMaskTexture`, the `/api/ai/*` routes) and the SAME
 * source of truth — the `MegashaderFilter` that lives on the Fabric image
 * (which already survives tool switches + save/reload).
 *
 * Reconciliation: after every mutation we dispatch `phosmith:mask-chain-replaced`
 * so a mounted Mask tool can re-sync its panel (see `useMaskLayers`). When the
 * tool is NOT mounted (true headless agent run), the direct
 * `applyMegashaderFilter` call is what renders — no UI needed.
 *
 * NOT wired to any agent yet — see `command-registry.js`.
 *
 * @module agent/mask-commands
 */

import {
    applyMegashaderFilter,
    sanitiseLayer,
    luminanceLayer,
    colorLayer,
    linearLayer,
    radialLayer,
    lassoLayer,
    brushLayer,
    semanticLayer,
    depthLayer,
    setMaskTexture,
    MAX_LAYERS,
} from '@/lib/megashader'
import { rgbToHsb } from '@/lib/color-utils'
import { computeGradientMagnitude, snapToEdgePoint } from '@/lib/mask-edge-snap'
import { expandLayerBoundary, MAX_GROW_PX } from '@/lib/mask-grow'
import { clientSubjectMask } from '@/lib/client-ai'
import { resolveOrder } from '@/lib/ai-routing'
import { createNlMaskRunner } from './nl-mask'

const MEGASHADER_TYPE = 'Megashader'
const UPLOAD_MAX_SIDE = 1024

const getFilter = (image) => (image?.filters || []).find((f) => f && f.type === MEGASHADER_TYPE) || null
const getStack = (image) => {
    const f = getFilter(image)
    return f && f.stack && Array.isArray(f.stack.chain) ? { chain: [...f.stack.chain] } : { chain: [] }
}
const getSourceEl = (image) => image?._element || image?.getElement?.() || image?._originalElement || null
const naturalSize = (image) => {
    const el = getSourceEl(image)
    return { w: el?.naturalWidth || el?.width || image?.width || 0, h: el?.naturalHeight || el?.height || image?.height || 0 }
}

const uniqueKey = (prefix) => `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`

const rid = (v) => Math.round(v)

/**
 * Draw the image's source bitmap to a JPEG blob capped to UPLOAD_MAX_SIDE for
 * the AI routes. Returns the blob + the scale applied (so click coords can be
 * scaled to match).
 */
const imageToUploadBlob = async (image) => {
    const el = getSourceEl(image)
    if (!el) throw new Error('[agent.mask] image element not ready')
    const origW = el.naturalWidth || el.width
    const origH = el.naturalHeight || el.height
    const scale = Math.min(1, UPLOAD_MAX_SIDE / Math.max(origW, origH))
    const w = Math.max(1, rid(origW * scale))
    const h = Math.max(1, rid(origH * scale))
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    const ctx = c.getContext('2d')
    if (!ctx) throw new Error('[agent.mask] could not allocate upload canvas')
    ctx.drawImage(el, 0, 0, w, h)
    const blob = await new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error('toBlob failed'))), 'image/jpeg', 0.85))
    return { blob, scale, origW, origH }
}

/** Decode a PNG blob to an HTMLImageElement. */
const decodePng = (blob) => new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob)
    const img = new Image()
    img.onload = () => { URL.revokeObjectURL(url); resolve(img) }
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('mask decode failed')) }
    img.src = url
})

/** Decode a base64 PNG string to an HTMLImageElement. */
const decodeBase64Png = (b64) => new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('mask decode failed'))
    img.src = `data:image/png;base64,${b64}`
})

/**
 * Fetch per-subject instance masks from /api/ai/segment-instances, cached on
 * the Fabric image (one detection pass per image, however many subjects the
 * agent subsequently selects). bbox/centroid are scaled from the capped
 * upload back to true image-pixel coords so they line up with the coordinate
 * space every other mask command uses.
 */
const fetchSubjectInstances = async (image, { refresh = false } = {}) => {
    if (!refresh && image.__phosmithSubjectInstances) return image.__phosmithSubjectInstances
    const { blob, scale } = await imageToUploadBlob(image)
    const form = new FormData()
    form.append('image', blob, 'image.jpg')
    const resp = await fetch('/api/ai/segment-instances', { method: 'POST', body: form })
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}))
        throw new Error(err.error || `/api/ai/segment-instances failed (${resp.status})`)
    }
    const data = await resp.json()
    const inv = scale > 0 ? 1 / scale : 1
    const instances = (data.instances || []).map((inst) => ({
        ...inst,
        bboxImage: (inst.bbox || []).map((v) => Math.round(v * inv)),
        centroidImage: (inst.centroid || []).map((v) => Math.round(v * inv)),
    }))
    const out = { ...data, instances }
    image.__phosmithSubjectInstances = out
    return out
}

/**
 * Convert a decoded mask image into a coverage canvas where the R channel
 * carries the selection (what the texture-backed kinds sample). Handles both
 * greyscale masks (white=subject, R already correct) and transparent-bg
 * mattes (alpha=subject) by taking max(luma, alpha) into R on opaque black.
 */
const toCoverageCanvas = (img) => {
    const w = img.naturalWidth || img.width
    const h = img.naturalHeight || img.height
    const src = document.createElement('canvas')
    src.width = w
    src.height = h
    const sctx = src.getContext('2d', { willReadFrequently: true })
    sctx.drawImage(img, 0, 0)
    const data = sctx.getImageData(0, 0, w, h)
    const px = data.data
    for (let i = 0; i < px.length; i += 4) {
        const luma = 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]
        const cover = Math.max(luma, px[i + 3]) // matte alpha OR greyscale luma
        px[i] = cover
        px[i + 1] = cover
        px[i + 2] = cover
        px[i + 3] = 255
    }
    sctx.putImageData(data, 0, 0)
    return src
}

/** Rasterise polygon points (image-pixel coords) to a white-on-black canvas. */
const rasterizePolygon = (points, w, h) => {
    const c = document.createElement('canvas')
    c.width = Math.max(1, w)
    c.height = Math.max(1, h)
    const ctx = c.getContext('2d')
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, c.width, c.height)
    ctx.fillStyle = '#fff'
    ctx.beginPath()
    ctx.moveTo(points[0].x, points[0].y)
    for (let i = 1; i < points.length; i += 1) ctx.lineTo(points[i].x, points[i].y)
    ctx.closePath()
    ctx.fill('evenodd')
    return c
}

/**
 * Rasterise polygon points to a transparent canvas with white+alpha INSIDE.
 * The plain `brush` kind samples the ALPHA channel, so the selection must live
 * in alpha (not the R channel like the lasso's opaque-black canvas).
 */
const rasterizePolygonAlpha = (points, w, h) => {
    const c = document.createElement('canvas')
    c.width = Math.max(1, w)
    c.height = Math.max(1, h)
    const ctx = c.getContext('2d')
    ctx.clearRect(0, 0, c.width, c.height)
    ctx.fillStyle = '#fff'
    ctx.beginPath()
    ctx.moveTo(points[0].x, points[0].y)
    for (let i = 1; i < points.length; i += 1) ctx.lineTo(points[i].x, points[i].y)
    ctx.closePath()
    ctx.fill('evenodd')
    return c
}

/**
 * Build a normalised (0..1) Sobel gradient-magnitude map of the image's source
 * bitmap (full natural resolution). Used by `addMagneticLasso` to snap rough
 * points to the nearest edge headlessly — same shared engine the interactive
 * magnetic lasso uses (`@/lib/mask-edge-snap`).
 */
const buildGradientMag = (image) => {
    const el = getSourceEl(image)
    const { w, h } = naturalSize(image)
    if (!el || w < 3 || h < 3) return null
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    const ctx = c.getContext('2d', { willReadFrequently: true })
    if (!ctx) return null
    ctx.drawImage(el, 0, 0, w, h)
    let data
    try { data = ctx.getImageData(0, 0, w, h).data } catch { return null }
    return computeGradientMagnitude(data, w, h)
}

const hexToRgb255 = (hex) => {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex || '')
    if (!m) return null
    const n = parseInt(m[1], 16)
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 }
}

/**
 * Build the mask command set bound to a way of resolving the active primary
 * image. `getPrimaryImage()` must return the live Fabric image to mask.
 *
 * @param {{ getPrimaryImage: () => any }} ctx
 * @returns {Record<string, { description: string, params?: object, run: Function }>}
 */
export const createMaskCommands = ({ getPrimaryImage }) => {
    const requireImage = () => {
        const image = getPrimaryImage?.()
        if (!image) throw new Error('[agent.mask] no image on canvas')
        return image
    }

    const apply = (stack, extraOpts = {}) => {
        const image = requireImage()
        const prev = getFilter(image)
        applyMegashaderFilter(image, stack, {
            globalMaskAlpha: prev?.globalMaskAlpha ?? 1,
            globalInvert: extraOpts.globalInvert ?? prev?.globalInvert ?? false,
            maskOverlay: extraOpts.maskOverlay ?? prev?.maskOverlay ?? false,
            ...extraOpts,
        })
        try { image.canvas?.requestRenderAll?.() } catch { /* no canvas */ }
        try { window.dispatchEvent(new CustomEvent('phosmith:mask-chain-replaced', { detail: { stack } })) } catch { /* SSR */ }
        return stack
    }

    const addLayer = (layer, op) => {
        const image = requireImage()
        const stack = getStack(image)
        if (stack.chain.length >= MAX_LAYERS) throw new Error(`[agent.mask] max ${MAX_LAYERS} layers reached`)
        const safe = sanitiseLayer({ ...layer })
        stack.chain.push({ layer: safe, op: stack.chain.length === 0 ? 'replace' : (op || 'add') })
        apply(stack)
        return safe.id
    }

    const postMask = async (route, formBuilder) => {
        const image = requireImage()
        const { blob, scale } = await imageToUploadBlob(image)
        const form = new FormData()
        form.append('image', blob, 'image.jpg')
        if (formBuilder) formBuilder(form, scale)
        const resp = await fetch(route, { method: 'POST', body: form })
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}))
            throw new Error(err.error || `${route} failed (${resp.status})`)
        }
        return resp.blob()
    }

    // NL → mask pipeline (mask.fromDescription). Shares this module's
    // upload/decode/addLayer helpers via dependency injection.
    const nlRunner = createNlMaskRunner({
        requireImage,
        addLayer,
        applyExtra: (extraOpts) => apply(getStack(requireImage()), extraOpts),
        fetchSubjectInstances,
        imageToUploadBlob,
        decodeBase64Png,
        toCoverageCanvas,
        postMask,
        decodePng,
        naturalSize,
    })

    return {
        list: {
            description: 'List the current mask layers on the active image.',
            params: {},
            run: () => getStack(getPrimaryImage?.()).chain.map((e, i) => ({
                index: i, id: e.layer.id, kind: e.layer.kind, op: e.op,
                fillMode: e.layer.fillMode, label: e.layer.label, visible: e.layer.visible !== false,
            })),
        },
        clear: {
            description: 'Remove all mask layers from the active image.',
            params: {},
            run: () => { apply({ chain: [] }); return { ok: true } },
        },
        removeLayer: {
            description: 'Remove a mask layer by id.',
            params: { id: 'string — layer id from mask.list' },
            run: ({ id }) => {
                const stack = getStack(requireImage())
                stack.chain = stack.chain.filter((e) => e.layer.id !== id)
                if (stack.chain[0]) stack.chain[0] = { ...stack.chain[0], op: 'replace' }
                apply(stack)
                return { ok: true }
            },
        },
        updateLayer: {
            description: 'Patch a layer (adjustments exposure/contrast/highlights/shadows/temperature/tint/…, fillMode, opacity, inverted, op).',
            params: { id: 'string', patch: 'object — fields to merge onto the layer', op: 'optional blend op' },
            run: ({ id, patch = {}, op }) => {
                const stack = getStack(requireImage())
                const idx = stack.chain.findIndex((e) => e.layer.id === id)
                if (idx < 0) throw new Error(`[agent.mask] no layer ${id}`)
                const entry = stack.chain[idx]
                stack.chain[idx] = {
                    op: idx > 0 && op ? op : entry.op,
                    layer: sanitiseLayer({ ...entry.layer, ...patch }),
                }
                apply(stack)
                return { ok: true }
            },
        },
        addLuminance: {
            description: 'Add a luminance-range mask. min/max are 0..1; fillMode fill|adjust|erase.',
            params: { min: 'number 0..1', max: 'number 0..1', softness: 'number 0..1', fillMode: 'string', op: 'blend op' },
            run: (a) => ({ id: addLayer({ ...luminanceLayer(a), fillMode: a.fillMode, ...pickFill(a) }, a.op) }),
        },
        addColor: {
            description: 'Add a colour-range mask. Pass target {h,s,b} or hex "#rrggbb"; tolerance 0..1.',
            params: { hex: 'string "#rrggbb"', target: '{h,s,b}', tolerance: 'number 0..1', softness: 'number 0..1', fillMode: 'string' },
            run: (a) => {
                let target = a.target
                if (!target && a.hex) { const rgb = hexToRgb255(a.hex); if (rgb) target = rgbToHsb(rgb.r, rgb.g, rgb.b) }
                return { id: addLayer({ ...colorLayer({ ...a, target }), fillMode: a.fillMode, ...pickFill(a) }, a.op) }
            },
        },
        addLinear: {
            description: 'Add a linear-gradient mask. p1/p2 in image-pixel coords; position/feather 0..1.',
            params: { p1: '{x,y}', p2: '{x,y}', position: 'number 0..1', feather: 'number 0..1', fillMode: 'string' },
            run: (a) => ({ id: addLayer({ ...linearLayer(a), fillMode: a.fillMode, ...pickFill(a) }, a.op) }),
        },
        addRadial: {
            description: 'Add a radial-gradient mask. center/radius in image-pixel coords; rotation radians; feather 0..1.',
            params: { center: '{x,y}', radius: '{x,y}', rotation: 'number rad', feather: 'number 0..1', fillMode: 'string' },
            run: (a) => ({ id: addLayer({ ...radialLayer(a), fillMode: a.fillMode, ...pickFill(a) }, a.op) }),
        },
        addLasso: {
            description: 'Add a lasso selection from a polygon of image-pixel points. fillMode fill|erase|adjust.',
            params: { points: 'Array<{x,y}> (>=3)', feather: 'number 0..1', fillMode: 'string', op: 'blend op' },
            run: (a) => {
                const pts = Array.isArray(a.points) ? a.points : []
                if (pts.length < 3) throw new Error('[agent.mask] addLasso needs >=3 points')
                const { w, h } = naturalSize(requireImage())
                const key = uniqueKey('lasso')
                setMaskTexture(key, rasterizePolygon(pts, w, h))
                return { id: addLayer(lassoLayer({ maskTextureKey: key, feather: a.feather, fillMode: a.fillMode || 'fill', ...pickFill(a) }), a.op) }
            },
        },
        addBrush: {
            description: 'Add a non-destructive brush selection from a polygon of image-pixel points (samples painted alpha). fillMode fill|erase|adjust.',
            params: { points: 'Array<{x,y}> (>=3)', fillMode: 'string', op: 'blend op' },
            run: (a) => {
                const pts = Array.isArray(a.points) ? a.points : []
                if (pts.length < 3) throw new Error('[agent.mask] addBrush needs >=3 points')
                const { w, h } = naturalSize(requireImage())
                const key = uniqueKey('brush')
                setMaskTexture(key, rasterizePolygonAlpha(pts, w, h))
                return { id: addLayer(brushLayer({ maskTextureKey: key, fillMode: a.fillMode || 'fill', ...pickFill(a) }), a.op) }
            },
        },
        addMagneticLasso: {
            description: 'Add a lasso selection that SNAPS rough points to the nearest image edge (magnetic). width = search radius px, contrast 0..1 edge threshold.',
            params: { points: 'Array<{x,y}> (>=3)', width: 'number px (default 16)', contrast: 'number 0..1 (default 0.12)', feather: 'number 0..1', fillMode: 'string', op: 'blend op' },
            run: (a) => {
                const pts = Array.isArray(a.points) ? a.points : []
                if (pts.length < 3) throw new Error('[agent.mask] addMagneticLasso needs >=3 points')
                const image = requireImage()
                const { w, h } = naturalSize(image)
                const gm = buildGradientMag(image)
                const width = typeof a.width === 'number' ? a.width : 16
                const contrast = typeof a.contrast === 'number' ? a.contrast : 0.12
                const snapped = pts.map((p) => snapToEdgePoint(gm, p.x, p.y, width, contrast))
                const key = uniqueKey('lasso')
                setMaskTexture(key, rasterizePolygon(snapped, w, h))
                return { id: addLayer(lassoLayer({ maskTextureKey: key, feather: a.feather, fillMode: a.fillMode || 'fill', ...pickFill(a) }), a.op) }
            },
        },
        setOverlay: {
            description: 'Toggle the "show mask" red overlay visualisation.',
            params: { on: 'boolean' },
            run: ({ on }) => { apply(getStack(requireImage()), { maskOverlay: !!on }); try { window.dispatchEvent(new CustomEvent('phosmith:mask-overlay', { detail: { value: !!on } })) } catch {} return { ok: true } },
        },
        setInvert: {
            description: 'Toggle global invert of the whole mask.',
            params: { on: 'boolean' },
            run: ({ on }) => { apply(getStack(requireImage()), { globalInvert: !!on }); try { window.dispatchEvent(new CustomEvent('phosmith:mask-invert', { detail: { value: !!on } })) } catch {} return { ok: true } },
        },
        setGlobalAlpha: {
            description: 'Set the overall mask strength (0..1).',
            params: { value: 'number 0..1' },
            run: ({ value }) => { apply(getStack(requireImage()), { globalMaskAlpha: Math.max(0, Math.min(1, Number(value))) }); try { window.dispatchEvent(new CustomEvent('phosmith:mask-global-alpha', { detail: { value } })) } catch {} return { ok: true } },
        },
        selectSubject: {
            description: 'AI: detect the photo subject(s) and add them as a mask layer. Follows the AI routing policy — BiRefNet on the service, or RMBG-1.4 in the browser.',
            params: { fillMode: 'string (default fill)' },
            run: async (a) => {
                const image = requireImage()
                let cover = null
                let lastErr = null
                for (const side of resolveOrder('segment')) {
                    try {
                        if (side === 'client') {
                            const el = getSourceEl(image)
                            const { w, h } = naturalSize(image)
                            if (!el || !w || !h) throw new Error('image element not ready for on-device segmentation')
                            cover = await clientSubjectMask(el, { width: w, height: h })
                        } else {
                            const maskBlob = await postMask('/api/ai/segment')
                            cover = toCoverageCanvas(await decodePng(maskBlob))
                        }
                        break
                    } catch (err) {
                        lastErr = err
                    }
                }
                if (!cover) throw lastErr || new Error('[agent.mask] subject segmentation failed on every configured side')
                const key = uniqueKey('subject')
                setMaskTexture(key, cover)
                return { id: addLayer({ ...semanticLayer({ maskTextureKey: key, feather: 0.05, label: 'AI Subject' }), fillMode: a.fillMode || 'fill', ...pickFill(a) }, a.op) }
            },
        },
        detectSubjects: {
            description: 'AI: detect EVERY subject instance in the photo (multi-subject). Returns [{index,label,confidence,source,bbox,centroid}] in image-pixel coords without changing the mask. Use mask.selectSubjects to mask specific ones.',
            params: { refresh: 'boolean — bypass the per-image cache (default false)' },
            run: async (a) => {
                const data = await fetchSubjectInstances(requireImage(), { refresh: !!a?.refresh })
                return data.instances.map((inst) => ({
                    index: inst.index,
                    label: inst.label,
                    confidence: inst.confidence,
                    source: inst.source,
                    bbox: inst.bboxImage,
                    centroid: inst.centroidImage,
                }))
            },
        },
        selectSubjects: {
            description: 'AI: mask specific subject instances from mask.detectSubjects. Select by indices and/or labels (e.g. ["person"]); omit both to select every subject. separateLayers=true adds one layer per subject so each can be adjusted independently.',
            params: { indices: 'number[] — instance indices from detectSubjects', labels: 'string[] — class labels, case-insensitive', separateLayers: 'boolean (default false — one unioned layer)', fillMode: 'string (default fill)', op: 'blend op' },
            run: async (a) => {
                const image = requireImage()
                const data = await fetchSubjectInstances(image, { refresh: !!a?.refresh })
                if (!data.instances.length) throw new Error('[agent.mask] no subjects detected in this image')

                const wantIdx = Array.isArray(a?.indices) ? new Set(a.indices.map(Number)) : null
                const wantLabels = Array.isArray(a?.labels) ? new Set(a.labels.map((l) => String(l).toLowerCase())) : null
                const picked = data.instances.filter((inst) => {
                    if (!wantIdx && !wantLabels) return true
                    if (wantIdx && wantIdx.has(inst.index)) return true
                    if (wantLabels && wantLabels.has(String(inst.label).toLowerCase())) return true
                    return false
                })
                if (!picked.length) {
                    throw new Error(`[agent.mask] no subjects matched (have: ${data.instances.map((i) => `${i.index}:${i.label}`).join(', ')})`)
                }

                if (a?.separateLayers) {
                    const ids = []
                    for (const inst of picked) {
                        const cover = toCoverageCanvas(await decodeBase64Png(inst.maskPng))
                        const key = uniqueKey(`subject-${inst.index}`)
                        setMaskTexture(key, cover)
                        ids.push({
                            index: inst.index,
                            label: inst.label,
                            id: addLayer({ ...semanticLayer({ maskTextureKey: key, feather: 0.05, label: `AI ${inst.label} #${inst.index + 1}` }), fillMode: a.fillMode || 'fill', ...pickFill(a) }, a.op),
                        })
                    }
                    return { layers: ids }
                }

                // Union the picked instances into ONE coverage canvas (max blend).
                const canvases = []
                for (const inst of picked) canvases.push(toCoverageCanvas(await decodeBase64Png(inst.maskPng)))
                const w = Math.max(...canvases.map((c) => c.width))
                const h = Math.max(...canvases.map((c) => c.height))
                const union = document.createElement('canvas')
                union.width = w
                union.height = h
                const uctx = union.getContext('2d')
                uctx.fillStyle = '#000'
                uctx.fillRect(0, 0, w, h)
                uctx.globalCompositeOperation = 'lighten'
                for (const c of canvases) uctx.drawImage(c, 0, 0, w, h)
                const key = uniqueKey('subjects')
                setMaskTexture(key, union)
                const label = picked.length === 1 ? `AI ${picked[0].label}` : `AI Subjects (${picked.length})`
                return {
                    selected: picked.map((i) => ({ index: i.index, label: i.label })),
                    id: addLayer({ ...semanticLayer({ maskTextureKey: key, feather: 0.05, label }), fillMode: a.fillMode || 'fill', ...pickFill(a) }, a.op),
                }
            },
        },
        addSubjectClicks: {
            description: 'AI (SAM 3.1): click-select a subject. clicks = [[x,y,label=1|0], …] in image-pixel coords.',
            params: { clicks: 'Array<[x,y,label]>', fillMode: 'string' },
            run: async (a) => {
                const clicks = Array.isArray(a.clicks) ? a.clicks : []
                if (!clicks.length) throw new Error('[agent.mask] addSubjectClicks needs >=1 click')
                const maskBlob = await postMask('/api/ai/sam3', (form, scale) => {
                    form.append('clicks', JSON.stringify(clicks.map(([x, y, l]) => [x * scale, y * scale, l ?? 1])))
                })
                const cover = toCoverageCanvas(await decodePng(maskBlob))
                const key = uniqueKey('sam2')
                setMaskTexture(key, cover)
                return { id: addLayer({ ...semanticLayer({ maskTextureKey: key, feather: 0.05, label: 'AI Subject' }), fillMode: a.fillMode || 'fill', ...pickFill(a) }, a.op) }
            },
        },
        addSubjectBox: {
            description: 'AI (SAM 3.1): box-select the object inside a rectangle — SAM 3.1\'s strongest single prompt for whole objects. box = [x, y, w, h] in image-pixel coords; optional clicks refine it.',
            params: { box: '[x, y, w, h] image px', clicks: 'optional Array<[x,y,label]>', fillMode: 'string' },
            run: async (a) => {
                const b = Array.isArray(a.box) && a.box.length === 4 ? a.box.map(Number) : null
                if (!b || b.some((v) => !Number.isFinite(v)) || b[2] <= 0 || b[3] <= 0) {
                    throw new Error('[agent.mask] addSubjectBox needs box = [x, y, w, h]')
                }
                const clicks = Array.isArray(a.clicks) ? a.clicks : []
                const maskBlob = await postMask('/api/ai/sam3', (form, scale) => {
                    form.append('box', JSON.stringify([
                        b[0] * scale, b[1] * scale,
                        (b[0] + b[2]) * scale, (b[1] + b[3]) * scale,
                    ]))
                    if (clicks.length) {
                        form.append('clicks', JSON.stringify(clicks.map(([x, y, l]) => [x * scale, y * scale, l ?? 1])))
                    }
                })
                const cover = toCoverageCanvas(await decodePng(maskBlob))
                const key = uniqueKey('sam2-box')
                setMaskTexture(key, cover)
                return { id: addLayer({ ...semanticLayer({ maskTextureKey: key, feather: 0.05, label: 'AI Box Subject' }), fillMode: a.fillMode || 'fill', ...pickFill(a) }, a.op) }
            },
        },
        addDepthRange: {
            description: 'AI: generate a depth map and add a depth-range mask. min/max/softness 0..1.',
            params: { min: 'number 0..1', max: 'number 0..1', softness: 'number 0..1', fillMode: 'string' },
            run: async (a) => {
                const depthBlob = await postMask('/api/ai/depth')
                const cover = toCoverageCanvas(await decodePng(depthBlob))
                const key = uniqueKey('depth')
                setMaskTexture(key, cover)
                return { id: addLayer({ ...depthLayer({ depthMapKey: key, min: a.min, max: a.max, softness: a.softness, label: 'Depth range' }), fillMode: a.fillMode, ...pickFill(a) }, a.op) }
            },
        },
        fromDescription: {
            description: 'AI: mask a region described in natural language (e.g. "the dog on the left", "everything except the sky", "the shadows but not the person", "the red jacket, extend by 12px"). Plans with Gemini (heuristic fallback), then resolves via instance detection, text grounding (CLIPSeg+SAM 3.1), depth, luminance, colour or geometry, composing multiple parts with add/subtract/intersect. Returns {plan, layers, notes}.',
            params: {
                description: 'string — the region in plain language',
                fillMode: 'string fill|adjust|erase (overrides the language)',
                feather: 'number 0..0.5 (overrides the language)',
                grow: 'number ±px boundary extension (overrides the language)',
                dryRun: 'boolean — return the plan without adding layers',
            },
            run: async (a) => {
                const hadChainBefore = getStack(requireImage()).chain.length > 0
                return nlRunner.fromDescription({ ...a, hadChainBefore })
            },
        },
        expandLayer: {
            description: `Extend (or shrink) the BOUNDARY of a texture-backed mask layer by N pixels — works on AI-detected subject masks, grounded masks, lassos and brushes. pixels is ABSOLUTE (12 then 20 yields 20; 0 restores the original edge), clamped to ±${MAX_GROW_PX}.`,
            params: { id: 'string — layer id from mask.list', pixels: `number — signed px, ±${MAX_GROW_PX} max` },
            run: ({ id, pixels }) => expandLayerBoundary(requireImage(), id, pixels),
        },
    }
}

/** Pull fill colour/strength out of args if the caller specified them. */
function pickFill(a) {
    const out = {}
    if (a.fillColor) out.fillColor = a.fillColor
    if (typeof a.fillStrength === 'number') out.fillStrength = a.fillStrength
    return out
}
