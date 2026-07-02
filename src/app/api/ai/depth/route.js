import { auth } from '@clerk/nextjs/server'
import { NextResponse } from 'next/server'
import sharp from 'sharp'
import { enforceRateLimit, rateLimitResponse } from '@/lib/rate-limit'

export const maxDuration = 120
export const runtime = 'nodejs'

/* ═══════════════════════════════════════════════════════════════════════════
 * Depth Anything V2 — whole-image depth estimation
 *
 * The Next.js route here is a thin proxy in front of the local Python
 * service (`services/segment/`), which runs `depth-anything/Depth-
 * Anything-V2-Small-hf` via Hugging Face `transformers`. The Python
 * service returns a per-image min-max-normalised greyscale PNG depth
 * map, resized to the input's natural dimensions.
 *
 * Why a separate route instead of a direct call from the editor?
 *   - The editor and the Python service run in different processes; the
 *     service can be scaled independently (GPU host, free-tier CPU, etc.).
 *   - Server-side keeps the HF model out of the browser bundle.
 *   - Server-side can attach auth + rate limiting + a per-user quota.
 *
 * Request format (multipart/form-data):
 *   - image:  JPEG/PNG/WebP, max 24 MB
 *
 * Response:
 *   - image/png  (greyscale depth map at *original* image resolution,
 *                 0 = far, 255 = near, per-image min-max-normalised)
 *   - X-Model    (HF checkpoint id)
 *   - X-Width    (depth-map width in px, equals input width)
 *   - X-Height   (depth-map height in px, equals input height)
 *   - X-Elapsed-Ms
 * ═══════════════════════════════════════════════════════════════════════════ */

// Masking microservice (own HF Space); falls back to the legacy shared
// MASK_SERVICE_URL so existing single-service deploys keep working.
const MASKING_SERVICE_URL = process.env.MASKING_SERVICE_URL?.trim().replace(/\/+$/, '')
  || process.env.MASK_SERVICE_URL?.trim().replace(/\/+$/, '') || ''
const MAX_INPUT_BYTES = 24 * 1024 * 1024
// Cap the image's longest side before forwarding to the Python service.
// Depth Anything V2 runs internally at ~518×518 and the route resizes
// the returned depth map back to the *input* dimensions. A user uploading
// a 12K image would (a) saturate the 24MB body limit, (b) trigger a
// 144M-op Lanczos resize in Python (~30s on CPU, OOM risk), and (c)
// produce a 144MB depth array in the LRU cache. 2048 gives the user
// four times the model's native resolution for headroom while bounding
// peak memory at 4MB per cache entry and a sub-second resize.
const MAX_DEPTH_SIDE = 2048

const fileToBuffer = async (file, label) => {
  if (!file || typeof file.arrayBuffer !== 'function') {
    throw new Error(`${label} is required`)
  }
  if (typeof file.size !== 'number' || file.size > MAX_INPUT_BYTES) {
    throw new Error(`${label} is too large (max ${MAX_INPUT_BYTES / 1024 / 1024}MB)`)
  }
  return Buffer.from(await file.arrayBuffer())
}

const readImageMeta = async (inputBuffer) => {
  const meta = await sharp(inputBuffer, { failOn: 'none' }).metadata()
  const origW = meta.width || 0
  const origH = meta.height || 0
  if (!origW || !origH) {
    throw new Error('image has no usable dimensions')
  }
  return { origWidth: origW, origHeight: origH }
}

const callDepthService = async (imageBuffer) => {
  if (!MASKING_SERVICE_URL) {
    return { ok: false, reason: 'MASKING_SERVICE_URL not configured' }
  }
  const endpoint = `${MASKING_SERVICE_URL}/depth`
  try {
    const formData = new FormData()
    formData.append('image', new Blob([imageBuffer], { type: 'image/jpeg' }), 'image.jpg')

    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(120_000),
    })
    if (!response.ok) {
      const text = await response.text().catch(() => '')
      return { ok: false, reason: `HTTP ${response.status}: ${text.slice(0, 200)}` }
    }
    const ct = response.headers.get('content-type') || ''
    if (!ct.includes('image')) {
      return { ok: false, reason: `non-image response: ${ct}` }
    }
    const buf = Buffer.from(await response.arrayBuffer())
    return {
      ok: true,
      buffer: buf,
      model: response.headers.get('x-model') || 'depth-anything',
      width: response.headers.get('x-width') || '',
      height: response.headers.get('x-height') || '',
      elapsedMs: response.headers.get('x-elapsed-ms') || '',
      coldLoad: response.headers.get('x-cold-load') === 'true',
    }
  } catch (e) {
    const reason = e?.message || String(e)
    const isTimeout = /timeout|abort/i.test(reason)
    return {
      ok: false,
      reason: isTimeout
        ? 'AI model is loading for the first time (downloading weights). This takes 30–90 seconds — please try again in a moment.'
        : reason,
      isModelLoading: isTimeout,
    }
  }
}

export async function POST(request) {
  try {
    const { userId } = await auth()
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    // Shares the ai-segment bucket — depth is comparable in cost to a
    // full-image semantic inference (single forward pass on the full
    // image, no per-prompt work like SAM 2 clicks).
    const limited = rateLimitResponse(await enforceRateLimit('ai-segment', userId))
    if (limited) return limited

    if (!MASKING_SERVICE_URL) {
      return NextResponse.json(
        { error: 'MASKING_SERVICE_URL is not configured. Start services/masking/main.py (bun run masking:dev) and set MASKING_SERVICE_URL in .env.local.' },
        { status: 501 },
      )
    }

    // Pre-check the Content-Length header BEFORE parsing the multipart
    // body. Next.js's request.formData() is streaming but it still
    // allocates a Buffer for the whole body in some cases; a 1GB upload
    // from a malicious client would otherwise be parsed before our
    // per-file `fileToBuffer` size check runs.
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

    // Read original dimensions for the log line and X-Width/X-Height headers.
    // The Python service resizes its internal model output back to these
    // dimensions, so the response PNG should be origWidth × origHeight.
    const meta = await readImageMeta(imageBuffer)

    // Reject images whose longest side exceeds MAX_DEPTH_SIDE. The
    // Python service would happily forward them, but its Lanczos
    // resize-from-518²-to-N² is O(n²) and the resulting depth array
    // can saturate the LRU cache (each entry is H×W uint8).
    const longestSide = Math.max(meta.origWidth, meta.origHeight)
    if (longestSide > MAX_DEPTH_SIDE) {
      return NextResponse.json(
        {
          error:
            `image too large (${meta.origWidth}x${meta.origHeight}); ` +
            `max longest side is ${MAX_DEPTH_SIDE}px. ` +
            `Pre-resize the upload client-side.`,
        },
        { status: 413 },
      )
    }

    console.info('[ai-depth] image:', meta.origWidth, 'x', meta.origHeight)

    const result = await callDepthService(imageBuffer)
    if (!result.ok) {
      console.warn('[ai-depth] service failed:', result.reason)
      const status = /not configured|not available/i.test(result.reason) ? 501 : 502
      return NextResponse.json({ error: result.reason }, { status })
    }

    console.info(
      '[ai-depth] ✓ depth map:', result.buffer.length, 'bytes',
      `(${result.width}x${result.height}, model: ${result.model}, elapsed: ${result.elapsedMs}ms)`,
    )

    return new NextResponse(result.buffer, {
      status: 200,
      headers: {
        'Content-Type': 'image/png',
        'Cache-Control': 'no-store',
        'X-Model': result.model,
        'X-Width': result.width,
        'X-Height': result.height,
        'X-Elapsed-Ms': result.elapsedMs,
        ...(result.coldLoad ? { 'X-Cold-Load': 'true' } : {}),
      },
    })
  } catch (error) {
    console.error('[ai-depth] ✗', error?.message)
    const msg = error?.message || 'Depth estimation failed'
    const status = /not configured/i.test(msg) ? 501 : 400
    return NextResponse.json({ error: msg }, { status })
  }
}
