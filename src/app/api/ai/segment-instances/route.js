import { auth } from '@clerk/nextjs/server'
import { NextResponse } from 'next/server'
import sharp from 'sharp'
import { enforceRateLimit, rateLimitResponse } from '@/lib/rate-limit'

export const maxDuration = 120
export const runtime = 'nodejs'

/* ═══════════════════════════════════════════════════════════════════════════
 * MULTI-SUBJECT SELECTION — /api/ai/segment-instances
 *
 * Thin proxy in front of the local mask service's `/segment/instances`
 * endpoint (services/segment/main.py). Where /api/ai/segment returns ONE
 * union mask covering every subject, this returns one refined soft mask PER
 * detected subject instance (each person, each animal, each salient object),
 * with label, confidence, bounding box, area and centroid — so the editor
 * (or the agent via the `mask.selectInstances` command) can target
 * "person 2" or "the dog" individually.
 *
 * No HuggingFace fallback: per-instance segmentation requires the local
 * YOLO-seg + BiRefNet stack. Without MASKING_SERVICE_URL this returns 501,
 * matching /api/ai/sam2 and /api/ai/depth.
 * ═══════════════════════════════════════════════════════════════════════════ */

// Masking microservice (own HF Space); falls back to the legacy shared
// MASK_SERVICE_URL so existing single-service deploys keep working.
const MASKING_SERVICE_URL = process.env.MASKING_SERVICE_URL?.trim().replace(/\/+$/, '')
  || process.env.MASK_SERVICE_URL?.trim().replace(/\/+$/, '') || ''

const MAX_INPUT_BYTES = 24 * 1024 * 1024
// Mirrors /api/ai/segment: BiRefNet runs at a fixed 1024² internally, but a
// 2048 longest side feeds YOLO a higher-resolution image so small/distant
// people in group photos are detected, and gives a crisper Lanczos upscale of
// each instance mask back to the true source resolution.
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

/**
 * Upscale a model-resolution greyscale mask PNG to the original image
 * resolution. Same recipe as /api/ai/segment's refineMaskEdges soft path:
 * median(3) denoise → Lanczos3 upscale → blur(1.2) anti-alias. No threshold —
 * the service's masks are already soft-edged and the canvas maps mask
 * luminance directly to clip alpha.
 */
const upscaleMask = async (maskPngBuffer, origW, origH) => {
  return sharp(maskPngBuffer, { failOn: 'none' })
    .greyscale()
    .median(3)
    .resize(origW, origH, { fit: 'fill', kernel: 'lanczos3' })
    .blur(1.2)
    .png()
    .toBuffer()
}

export async function POST(request) {
  try {
    const { userId } = await auth()
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    if (!MASKING_SERVICE_URL) {
      return NextResponse.json(
        { error: 'MASKING_SERVICE_URL is not configured — multi-subject selection requires the masking service (bun run masking:dev)' },
        { status: 501 },
      )
    }

    const limited = rateLimitResponse(await enforceRateLimit('ai-segment', userId))
    if (limited) return limited

    // Pre-check Content-Length BEFORE parsing the multipart body (mirrors
    // /api/ai/segment and /api/ai/depth).
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

    const prepared = await prepareImage(imageBuffer)
    console.info('[ai-segment-instances] image:', prepared.width, 'x', prepared.height,
      '→ orig:', prepared.origWidth, 'x', prepared.origHeight)

    const serviceForm = new FormData()
    serviceForm.append('image', new Blob([prepared.buffer], { type: 'image/jpeg' }), 'image.jpg')

    const response = await fetch(`${MASKING_SERVICE_URL}/segment/instances`, {
      method: 'POST',
      body: serviceForm,
      // Same budget as /api/ai/segment: covers a cold CPU BiRefNet run plus
      // YOLO + per-instance compositing.
      signal: AbortSignal.timeout(180_000),
    })

    if (!response.ok) {
      const text = await response.text().catch(() => '')
      console.warn('[ai-segment-instances] service HTTP', response.status, text.slice(0, 200))
      return NextResponse.json(
        { error: `mask service error (HTTP ${response.status})` },
        { status: response.status >= 500 ? 502 : response.status },
      )
    }

    const payload = await response.json()

    // The service worked at model resolution; rescale every mask + geometry
    // back to the ORIGINAL image resolution so the editor canvas can use the
    // result 1:1, exactly like /api/ai/segment does for the union mask.
    const sx = prepared.origWidth / (payload.width || prepared.width)
    const sy = prepared.origHeight / (payload.height || prepared.height)
    const needsScale = Math.abs(sx - 1) > 1e-3 || Math.abs(sy - 1) > 1e-3

    const instances = await Promise.all((payload.instances || []).map(async (inst) => {
      let maskPng = inst.mask_png
      if (needsScale && maskPng) {
        const upscaled = await upscaleMask(
          Buffer.from(maskPng, 'base64'),
          prepared.origWidth,
          prepared.origHeight,
        )
        maskPng = upscaled.toString('base64')
      }
      const [bx, by, bw, bh] = inst.bbox || [0, 0, 0, 0]
      const [cx, cy] = inst.centroid || [0, 0]
      return {
        index: inst.index,
        label: inst.label,
        classId: inst.class_id,
        confidence: inst.confidence,
        source: inst.source,
        bbox: [Math.round(bx * sx), Math.round(by * sy), Math.round(bw * sx), Math.round(bh * sy)],
        areaFrac: inst.area_frac,
        centroid: [Math.round(cx * sx), Math.round(cy * sy)],
        maskPng,
      }
    }))

    let unionPng = payload.union_png || null
    if (needsScale && unionPng) {
      const upscaled = await upscaleMask(
        Buffer.from(unionPng, 'base64'),
        prepared.origWidth,
        prepared.origHeight,
      )
      unionPng = upscaled.toString('base64')
    }

    console.info('[ai-segment-instances] ✓', instances.length, 'instances (mode:', payload.mode + ')')

    return NextResponse.json({
      width: prepared.origWidth,
      height: prepared.origHeight,
      model: payload.model || 'local-rembg',
      subjectModel: payload.subject_model || null,
      mode: payload.mode,
      count: instances.length,
      truncated: !!payload.truncated,
      instances,
      unionPng,
    }, {
      headers: { 'Cache-Control': 'no-store' },
    })
  } catch (error) {
    console.error('[ai-segment-instances] ✗', error?.message)
    const timeout = /abort|timeout/i.test(error?.message || '')
    return NextResponse.json(
      { error: error?.message || 'Instance segmentation failed' },
      { status: timeout ? 504 : 500 },
    )
  }
}
