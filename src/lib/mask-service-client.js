/**
 * mask-service-client — browser client for the AI masking endpoints.
 *
 * Calls the Next.js proxy routes (/api/ai/*), NOT the Python service directly,
 * so every request keeps Clerk auth + rate limiting. Encapsulates the upload
 * downscale, mask decode, and PNG→coverage-canvas plumbing that was duplicated
 * across the editor (mask.jsx) and the agent (mask-commands.js).
 *
 * Coordinate contract: click/box points are passed in ORIGINAL image-pixel
 * coords; this module scales them to the (downscaled) upload so the proxy route
 * sees consistent coords, and returns mask canvases scaled to the natural image
 * resolution so they drop onto the editor canvas 1:1.
 */

// SAM resizes to ~1024 internally, so uploading the full canvas wastes encode +
// network + decode. Send a downscaled copy; the returned mask is scaled back up.
const SAM_INPUT = 1024
const SUBJECT_INPUT = 2048 // BiRefNet/YOLO benefit from more resolution
const SUBJECT_CONCEPT = 'main subject'

const naturalSize = (el) => ({
    w: el?.naturalWidth || el?.width || 0,
    h: el?.naturalHeight || el?.height || 0,
})

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))

export const canvasToBlob = (canvas, type = 'image/jpeg', quality = 0.85) =>
    new Promise((res, rej) =>
        canvas.toBlob((b) => (b ? res(b) : rej(new Error('toBlob failed'))), type, quality),
    )

/**
 * Downscale a source image/canvas to `maxSide` and JPEG-encode it for upload.
 * Returns the blob, the scale factor (uploaded/original — for scaling prompts),
 * the uploaded dims, and the natural dims (for scaling masks back up).
 */
export const imageToUploadBlob = async (sourceEl, { maxSide = SAM_INPUT, quality = 0.85 } = {}) => {
    const { w: origW, h: origH } = naturalSize(sourceEl)
    if (origW < 1 || origH < 1) throw new Error('image is still loading — try again in a moment')
    const scale = Math.min(1, maxSide / Math.max(origW, origH))
    const w = Math.max(1, Math.round(origW * scale))
    const h = Math.max(1, Math.round(origH * scale))
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    c.getContext('2d').drawImage(sourceEl, 0, 0, w, h)
    const blob = await canvasToBlob(c, 'image/jpeg', quality)
    return { blob, scale, width: w, height: h, origWidth: origW, origHeight: origH }
}

/** Decode a mask PNG blob into ImageData + a data URL (the editor's existing shape). */
export const decodeMaskBlob = async (blob) => {
    const url = URL.createObjectURL(blob)
    try {
        const img = await new Promise((resolve, reject) => {
            const image = new Image()
            image.crossOrigin = 'anonymous'
            image.onload = () => resolve(image)
            image.onerror = () => reject(new Error('Failed to decode mask PNG'))
            image.src = url
        })
        const c = document.createElement('canvas')
        c.width = img.naturalWidth || img.width
        c.height = img.naturalHeight || img.height
        const ctx = c.getContext('2d', { willReadFrequently: true })
        ctx.drawImage(img, 0, 0)
        return {
            imageData: ctx.getImageData(0, 0, c.width, c.height),
            width: c.width,
            height: c.height,
            dataUrl: c.toDataURL('image/png'),
        }
    } finally {
        URL.revokeObjectURL(url)
    }
}

export const base64PngToBlob = (b64) => {
    const bin = atob(b64)
    const arr = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i += 1) arr[i] = bin.charCodeAt(i)
    return new Blob([arr], { type: 'image/png' })
}

/**
 * Decode a service mask (greyscale PNG white=subject, or RGBA cutout carrying
 * the mask in alpha) into the opaque R=G=B=coverage / A=255 canvas the semantic
 * shader samples. Optionally rescales to (w, h).
 */
export const pngToMaskCanvas = (src, w, h) =>
    new Promise((resolve, reject) => {
        const img = new Image()
        img.onload = () => {
            const cv = document.createElement('canvas')
            cv.width = w || img.naturalWidth
            cv.height = h || img.naturalHeight
            const ctx = cv.getContext('2d', { willReadFrequently: true })
            ctx.drawImage(img, 0, 0, cv.width, cv.height)
            const id = ctx.getImageData(0, 0, cv.width, cv.height)
            const d = id.data
            for (let i = 0; i < d.length; i += 4) {
                const v = d[i + 3] < 250 ? d[i + 3] : d[i] // alpha or luma
                d[i] = v
                d[i + 1] = v
                d[i + 2] = v
                d[i + 3] = 255
            }
            ctx.putImageData(id, 0, 0)
            resolve(cv)
        }
        img.onerror = () => reject(new Error('mask decode failed'))
        img.src = src
    })

/** Tight [x0, y0, x1, y1] bounding box of mask coverage (red channel > thresh). */
export const bboxOfMaskCanvas = (canvas, thresh = 24) => {
    const w = canvas.width
    const h = canvas.height
    if (!w || !h) return null
    const d = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h).data
    let x0 = w
    let y0 = h
    let x1 = -1
    let y1 = -1
    for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
            if (d[(y * w + x) * 4] > thresh) {
                if (x < x0) x0 = x
                if (x > x1) x1 = x
                if (y < y0) y0 = y
                if (y > y1) y1 = y
            }
        }
    }
    return x1 < 0 ? null : [x0, y0, x1, y1]
}

const errorFromResponse = async (resp, fallback) => {
    const j = await resp.json().catch(() => null)
    const err = new Error(j?.error || `${fallback} (${resp.status})`)
    // Carry the HTTP status so callers can tell permanent conditions (501 =
    // model not installed on the service, 404 = route missing) from transient
    // ones and stop retrying the server per interaction.
    err.status = resp.status
    return err
}

/** GET /api/ai/health — is the masking service up, and is SAM 3.1 loaded? */
export const checkMaskService = async (timeoutMs = 4000) => {
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), timeoutMs)
    try {
        const r = await fetch('/api/ai/health', { signal: ctrl.signal })
        if (!r.ok) return { available: false }
        return await r.json()
    } catch {
        return { available: false }
    } finally {
        clearTimeout(timer)
    }
}

/**
 * POST /api/ai/sam3 — point-click selection. `points` are [[x, y], ...] and
 * `labels` [1|0, ...] in ORIGINAL image coords. Returns a coverage canvas at
 * (width, height) (defaults to the natural image size).
 */
export const serviceSamClick = async (sourceEl, points, labels, { width, height, signal } = {}) => {
    const up = await imageToUploadBlob(sourceEl, { maxSide: SAM_INPUT })
    const clicks = points.map(([x, y], i) => [
        clamp(Math.round(x * up.scale), 0, up.width - 1),
        clamp(Math.round(y * up.scale), 0, up.height - 1),
        labels[i] ? 1 : 0,
    ])
    const form = new FormData()
    form.append('image', up.blob, 'image.jpg')
    form.append('clicks', JSON.stringify(clicks))
    const r = await fetch('/api/ai/sam3', { method: 'POST', body: form, signal })
    if (!r.ok) throw await errorFromResponse(r, 'SAM click failed')
    const blob = await r.blob()
    const url = URL.createObjectURL(blob)
    try {
        return await pngToMaskCanvas(url, width || up.origWidth, height || up.origHeight)
    } finally {
        URL.revokeObjectURL(url)
    }
}

/**
 * POST /api/ai/sam3 — box-prompted selection. `box` is [x0, y0, x1, y1] in
 * ORIGINAL image coords. Returns a coverage canvas at (width, height).
 */
export const serviceSamBox = async (sourceEl, box, { width, height, signal } = {}) => {
    const up = await imageToUploadBlob(sourceEl, { maxSide: SAM_INPUT })
    const scaled = [
        clamp(Math.round(box[0] * up.scale), 0, up.width - 1),
        clamp(Math.round(box[1] * up.scale), 0, up.height - 1),
        clamp(Math.round(box[2] * up.scale), 1, up.width),
        clamp(Math.round(box[3] * up.scale), 1, up.height),
    ]
    if (scaled[2] <= scaled[0] || scaled[3] <= scaled[1]) throw new Error('box too small')
    const form = new FormData()
    form.append('image', up.blob, 'image.jpg')
    form.append('box', JSON.stringify(scaled))
    const r = await fetch('/api/ai/sam3', { method: 'POST', body: form, signal })
    if (!r.ok) throw await errorFromResponse(r, 'SAM box failed')
    const blob = await r.blob()
    const url = URL.createObjectURL(blob)
    try {
        return await pngToMaskCanvas(url, width || up.origWidth, height || up.origHeight)
    } finally {
        URL.revokeObjectURL(url)
    }
}

/**
 * POST /api/ai/segment-instances — subject/concept mask. With subjectBox the
 * service takes the fast saliency-bbox → SAM 3 box-seed path. Returns the union
 * coverage canvas plus mode/model/instances so callers can decide whether to
 * upgrade a saliency matte with a SAM 3 box prompt (`mode !== 'sam3'`).
 */
export const serviceSubjectMask = async (
    sourceEl,
    { concept = SUBJECT_CONCEPT, subjectBox = true, width, height, signal } = {},
) => {
    const up = await imageToUploadBlob(sourceEl, { maxSide: SUBJECT_INPUT, quality: 0.92 })
    const form = new FormData()
    form.append('image', up.blob, 'image.jpg')
    if (concept) form.append('prompt', concept)
    if (subjectBox) form.append('subject_box', 'true')
    const r = await fetch('/api/ai/segment-instances', { method: 'POST', body: form, signal })
    if (!r.ok) throw await errorFromResponse(r, 'subject detection failed')
    const j = await r.json()
    if (!j.unionPng) throw new Error(j.count === 0 ? 'no subject found' : 'no mask returned')
    const canvas = await pngToMaskCanvas(
        'data:image/png;base64,' + j.unionPng,
        width || j.width || up.origWidth,
        height || j.height || up.origHeight,
    )
    return { canvas, count: j.count || 0, mode: j.mode || 'sam3', model: j.model || '', instances: j.instances || [] }
}

/** POST /api/ai/ground — text-grounded mask. Returns the coverage canvas + score. */
export const serviceGroundText = async (sourceEl, phrase, { width, height, signal } = {}) => {
    const up = await imageToUploadBlob(sourceEl, { maxSide: SUBJECT_INPUT, quality: 0.9 })
    const form = new FormData()
    form.append('image', up.blob, 'image.jpg')
    form.append('phrases', JSON.stringify([phrase]))
    const r = await fetch('/api/ai/ground', { method: 'POST', body: form, signal })
    if (!r.ok) throw await errorFromResponse(r, 'text grounding failed')
    const j = await r.json()
    const res = Array.isArray(j.results) ? j.results[0] : null
    if (!res || !res.found || !res.maskPng) throw new Error(`no region matched "${phrase}"`)
    const canvas = await pngToMaskCanvas(
        'data:image/png;base64,' + res.maskPng,
        width || j.width || up.origWidth,
        height || j.height || up.origHeight,
    )
    return { canvas, engine: j.model || 'clipseg', score: res.score, coverage: res.coverage }
}

/** POST /api/ai/depth — depth map. Returns the decoded ImageData (white=near). */
export const serviceDepthMap = async (sourceEl, { signal } = {}) => {
    const up = await imageToUploadBlob(sourceEl, { maxSide: SAM_INPUT })
    const form = new FormData()
    form.append('image', up.blob, 'image.jpg')
    const r = await fetch('/api/ai/depth', { method: 'POST', body: form, signal })
    if (!r.ok) throw await errorFromResponse(r, 'depth failed')
    return decodeMaskBlob(await r.blob())
}
