import { auth } from '@clerk/nextjs/server'
import { NextResponse } from 'next/server'
import sharp from 'sharp'
import { enforceRateLimit, rateLimitResponse } from '@/lib/rate-limit'

export const maxDuration = 180
export const runtime = 'nodejs'

/* ═══════════════════════════════════════════════════════════════════════════
 * TEXT-GROUNDED MASKING — /api/ai/ground
 *
 * Thin proxy in front of the local mask service's `/ground/text` endpoint
 * (services/segment/main.py). Turns free-text phrases ("the red jacket",
 * "the waterfall") into soft greyscale masks: CLIPSeg heatmap → component
 * analysis → SAM 2 box refinement → matte cleanup. The NL mask pipeline
 * (src/lib/agent/nl-mask.js) calls this for open-vocabulary targets that
 * instance detection (/api/ai/segment-instances) can't name.
 *
 * Returns 501 when MASKING_SERVICE_URL is unset — exactly like sam2/depth/
 * segment-instances. No HuggingFace fallback.
 *
 * Request (multipart/form-data):
 *   - image:     JPEG/PNG/WebP, max 24 MB
 *   - phrases:   JSON array of 1..4 strings
 *   - threshold: optional float 0..1
 *   - refine:    optional "0" to skip SAM 2 refinement
 *
 * Response: the service payload with every mask upscaled — and every bbox
 * scaled — back to the ORIGINAL image resolution.
 * ═══════════════════════════════════════════════════════════════════════════ */

// Masking microservice (own HF Space); falls back to the legacy shared
// MASK_SERVICE_URL so existing single-service deploys keep working.
const MASKING_SERVICE_URL = process.env.MASKING_SERVICE_URL?.trim().replace(/\/+$/, '')
  || process.env.MASK_SERVICE_URL?.trim().replace(/\/+$/, '') || ''
const MAX_INPUT_BYTES = 24 * 1024 * 1024
const MAX_MODEL_SIDE = 2048

const fileToBuffer = async (file, label) => {
  if (!file || typeof file.arrayBuffer !== 'function') {
    throw new Error(`${label} is required`)
  }
  if (file.size > MAX_INPUT_BYTES) {
    throw new Error(`${label} is too large (max ${MAX_INPUT_BYTES / 1024 / 1024}MB)`)
  }
  return Buffer.from(await file.arrayBuffer())
}

const prepareImage = async (inputBuffer) => {
  const meta = await sharp(inputBuffer, { failOn: 'none' }).metadata()
  const origW = meta.width || 512
  const origH = meta.height || 512
  const scale = Math.min(1, MAX_MODEL_SIDE / Math.max(origW, origH))
  const w = Math.round(origW * scale)
  const h = Math.round(origH * scale)

  const buffer = await sharp(inputBuffer, { failOn: 'none' })
    .resize(w, h, { fit: 'fill' })
    .removeAlpha()
    .jpeg({ quality: 85 })
    .toBuffer()

  return { buffer, width: w, height: h, origWidth: origW, origHeight: origH }
}

/** Same soft upscale recipe as /api/ai/segment-instances. */
const upscaleMask = async (maskPngBuffer, origW, origH) =>
  sharp(maskPngBuffer, { failOn: 'none' })
    .greyscale()
    .median(3)
    .resize(origW, origH, { fit: 'fill', kernel: 'lanczos3' })
    .blur(1.2)
    .png()
    .toBuffer()

export async function POST(request) {
  try {
    const { userId } = await auth()
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    if (!MASKING_SERVICE_URL) {
      return NextResponse.json(
        { error: 'MASKING_SERVICE_URL is not configured — text grounding requires the masking service (bun run masking:dev)' },
        { status: 501 },
      )
    }

    const limited = rateLimitResponse(await enforceRateLimit('ai-ground', userId))
    if (limited) return limited

    const contentLength = request.headers.get('content-length')
    if (contentLength) {
      const cl = Number.parseInt(contentLength, 10)
      if (Number.isFinite(cl) && cl > MAX_INPUT_BYTES) {
        return NextResponse.json(
          { error: `request body too large (${(cl / 1024 / 1024).toFixed(1)}MB > ${MAX_INPUT_BYTES / 1024 / 1024}MB)` },
          { status: 413 },
        )
      }
    }

    const formData = await request.formData()
    const imageBuffer = await fileToBuffer(formData.get('image'), 'image')

    const rawPhrases = formData.get('phrases')
    let phrases
    try {
      phrases = JSON.parse(typeof rawPhrases === 'string' ? rawPhrases : '[]')
    } catch {
      return NextResponse.json({ error: 'phrases must be a JSON array of strings' }, { status: 400 })
    }
    phrases = (Array.isArray(phrases) ? phrases : [])
      .map((p) => String(p).trim())
      .filter(Boolean)
      .slice(0, 4)
    if (!phrases.length) {
      return NextResponse.json({ error: 'phrases must contain at least one phrase' }, { status: 400 })
    }

    const threshold = formData.get('threshold')
    const refine = formData.get('refine')

    const prepared = await prepareImage(imageBuffer)
    console.info('[ai-ground] image:', prepared.width, 'x', prepared.height,
      '→ orig:', prepared.origWidth, 'x', prepared.origHeight, 'phrases:', phrases)

    const serviceForm = new FormData()
    serviceForm.append('image', new Blob([prepared.buffer], { type: 'image/jpeg' }), 'image.jpg')
    serviceForm.append('phrases', JSON.stringify(phrases))
    if (typeof threshold === 'string' && threshold.trim() !== '') {
      serviceForm.append('threshold', threshold.trim())
    }
    if (typeof refine === 'string' && refine.trim() !== '') {
      serviceForm.append('refine', refine.trim())
    }

    const response = await fetch(`${MASKING_SERVICE_URL}/ground/text`, {
      method: 'POST',
      body: serviceForm,
      // First call lazy-loads CLIPSeg (+ SAM 2 for refinement) on the service.
      signal: AbortSignal.timeout(170_000),
    })

    if (!response.ok) {
      const text = await response.text().catch(() => '')
      console.warn('[ai-ground] service HTTP', response.status, text.slice(0, 200))
      return NextResponse.json(
        { error: `mask service error (HTTP ${response.status})`, detail: text.slice(0, 200) },
        { status: response.status >= 500 ? 502 : response.status },
      )
    }

    const payload = await response.json()

    const sx = prepared.origWidth / (payload.width || prepared.width)
    const sy = prepared.origHeight / (payload.height || prepared.height)
    const needsScale = Math.abs(sx - 1) > 1e-3 || Math.abs(sy - 1) > 1e-3

    const results = await Promise.all((payload.results || []).map(async (r) => {
      let maskPng = r.maskPng || null
      if (needsScale && maskPng) {
        const upscaled = await upscaleMask(
          Buffer.from(maskPng, 'base64'),
          prepared.origWidth,
          prepared.origHeight,
        )
        maskPng = upscaled.toString('base64')
      }
      const bbox = Array.isArray(r.bbox) && r.bbox.length === 4
        ? [
            Math.round(r.bbox[0] * sx),
            Math.round(r.bbox[1] * sy),
            Math.round(r.bbox[2] * sx),
            Math.round(r.bbox[3] * sy),
          ]
        : null
      return { ...r, bbox, maskPng }
    }))

    console.info('[ai-ground] ✓', results.map((r) => `${r.phrase}:${r.found ? r.score : 'miss'}`).join(', '),
      `(${payload.elapsed_ms}ms)`)

    return NextResponse.json({
      width: prepared.origWidth,
      height: prepared.origHeight,
      model: payload.model || 'clipseg',
      refine: !!payload.refine,
      elapsedMs: payload.elapsed_ms,
      results,
    }, { headers: { 'Cache-Control': 'no-store' } })
  } catch (error) {
    console.error('[ai-ground] ✗', error?.message)
    const timeout = /abort|timeout/i.test(error?.message || '')
    return NextResponse.json(
      { error: error?.message || 'Text grounding failed' },
      { status: timeout ? 504 : 500 },
    )
  }
}
