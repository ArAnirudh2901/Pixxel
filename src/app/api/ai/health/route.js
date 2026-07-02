import { NextResponse } from 'next/server'

export const runtime = 'nodejs'

/* Health proxy for the masking microservice — lets the Mask tool probe SAM 3.1 /
 * model availability same-origin (no CORS, no exposed service URL). Fails soft:
 * always 200 with { available:false } when the service is unset/down. */

const MASKING_SERVICE_URL = process.env.MASKING_SERVICE_URL?.trim().replace(/\/+$/, '')
  || process.env.MASK_SERVICE_URL?.trim().replace(/\/+$/, '') || ''

export async function GET() {
  if (!MASKING_SERVICE_URL) {
    return NextResponse.json({ available: false, configured: false })
  }
  try {
    const r = await fetch(`${MASKING_SERVICE_URL}/health`, { signal: AbortSignal.timeout(4000) })
    if (!r.ok) return NextResponse.json({ available: false, configured: true })
    const h = await r.json()
    return NextResponse.json({
      available: true,
      configured: true,
      sam3: !!h.sam3_available,
      sam3Loaded: !!h.sam3_loaded,
      sam2: !!h.sam2_available,
      depth: !!h.depth_available,
      ground: !!h.ground_available,
      subjectEngine: h.subject_engine || (h.sam3_available ? 'sam3' : 'saliency'),
      model: h.sam3_model || h.model || '',
    }, { headers: { 'Cache-Control': 'no-store' } })
  } catch {
    return NextResponse.json({ available: false, configured: true })
  }
}
