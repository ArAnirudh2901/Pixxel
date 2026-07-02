'use client'

/**
 * Client for the masking microservice, reached through the same-origin Next
 * proxy routes (/api/ai/*) that read MASKING_SERVICE_URL and add auth + rate
 * limiting. Centralises the upload-prep + mask-decode helpers the Mask tool
 * used inline, so there's one implementation to test. Every call fails soft.
 */

/** GET /api/ai/health — masking-service availability (SAM 3.1 etc). Never throws. */
export async function checkMaskService(timeoutMs = 4000) {
  try {
    const signal = typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(timeoutMs) : undefined
    const r = await fetch('/api/ai/health', { signal })
    if (!r.ok) return { available: false }
    return await r.json()
  } catch {
    return { available: false }
  }
}

/**
 * Draw a source image element/canvas to a downscaled JPEG blob for upload.
 * Caps the longest side (the routes re-scale server-side, so this only trims
 * bandwidth). Returns { blob, scale, width, height }. Throws a clear message on
 * a still-loading image or a CORS-tainted canvas.
 */
export async function prepareImageBlob(sourceEl, { maxSide = 2048, quality = 0.92 } = {}) {
  const ow = sourceEl?.naturalWidth || sourceEl?.width || 0
  const oh = sourceEl?.naturalHeight || sourceEl?.height || 0
  if (ow < 1 || oh < 1) throw new Error('Image is still loading — try again in a moment')
  const scale = Math.min(1, maxSide / Math.max(ow, oh))
  const w = Math.max(1, Math.round(ow * scale))
  const h = Math.max(1, Math.round(oh * scale))
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const ctx = c.getContext('2d')
  if (!ctx) throw new Error('Could not allocate upload canvas')
  ctx.drawImage(sourceEl, 0, 0, w, h)
  let blob
  try {
    blob = await new Promise((resolve, reject) => {
      c.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/jpeg', quality)
    })
  } catch (e) {
    if (e?.name === 'SecurityError') {
      throw new Error('This image is from another site without CORS, so it can’t be read for AI selection')
    }
    throw e
  }
  return { blob, scale, width: w, height: h }
}

/** PNG Blob → { imageData, width, height, dataUrl }. The imageData is what the
 * megashader mask cache stores; dataUrl feeds a small preview. */
export async function decodeMaskBlob(blob) {
  const objectUrl = URL.createObjectURL(blob)
  try {
    const img = await new Promise((resolve, reject) => {
      const image = new Image()
      image.crossOrigin = 'anonymous'
      image.onload = () => resolve(image)
      image.onerror = () => reject(new Error('Failed to decode mask PNG'))
      image.src = objectUrl
    })
    const c = document.createElement('canvas')
    c.width = img.naturalWidth || img.width
    c.height = img.naturalHeight || img.height
    const ctx = c.getContext('2d', { willReadFrequently: true })
    if (!ctx) throw new Error('Could not get 2D context for mask decode')
    ctx.drawImage(img, 0, 0)
    const imageData = ctx.getImageData(0, 0, c.width, c.height)
    return { imageData, width: c.width, height: c.height, dataUrl: c.toDataURL('image/png') }
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
}

/** base64 PNG → Blob (for the base64 masks that /segment-instances + /ground return). */
export function base64PngToBlob(b64) {
  const bin = atob(b64)
  const len = bin.length
  const arr = new Uint8Array(len)
  for (let i = 0; i < len; i++) arr[i] = bin.charCodeAt(i)
  return new Blob([arr], { type: 'image/png' })
}
