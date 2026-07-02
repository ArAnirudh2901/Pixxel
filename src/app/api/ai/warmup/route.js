import { NextResponse } from 'next/server'

export const maxDuration = 300 // warmup can take a while on cold starts
export const runtime = 'nodejs'

/* ═══════════════════════════════════════════════════════════════════════════
 * /api/ai/warmup — Proactively warm up all lazy-loaded AI models
 *
 * The Python mask service lazy-loads heavy models (SAM2, Depth, CLIPSeg,
 * LaMa) on first use. This can take 30–120 seconds on a free-tier CPU
 * host (downloading weights from HuggingFace Hub + loading into memory).
 *
 * This route lets the frontend trigger warmup proactively (e.g. when the
 * editor opens) so the first real AI request doesn't surprise the user
 * with a long wait. It's fire-and-forget — the UI can show "AI models
 * warming up" but doesn't need to block on the result.
 *
 * No auth required — warmup is idempotent and free (just loads models).
 * ═══════════════════════════════════════════════════════════════════════════ */

const MASKING_SERVICE_URL = process.env.MASKING_SERVICE_URL?.trim().replace(/\/+$/, '')
  || process.env.MASK_SERVICE_URL?.trim().replace(/\/+$/, '') || ''

export async function POST() {
  if (!MASKING_SERVICE_URL) {
    return NextResponse.json(
      { status: 'skipped', reason: 'MASKING_SERVICE_URL not configured' },
      { status: 200 },
    )
  }

  try {
    // First check if the service is up at all
    const healthResp = await fetch(`${MASKING_SERVICE_URL}/health`, {
      signal: AbortSignal.timeout(5000),
    })
    if (!healthResp.ok) {
      return NextResponse.json(
        { status: 'error', reason: `health check failed: HTTP ${healthResp.status}` },
        { status: 502 },
      )
    }
    const health = await healthResp.json()

    // If all models are already loaded, skip warmup
    const allLoaded =
      health.sam2_loaded &&
      health.depth_loaded &&
      health.ground_loaded
    if (allLoaded) {
      return NextResponse.json({ status: 'already_warm', models: health })
    }

    // Trigger warmup — this can take 1–3 minutes on cold start
    console.info('[ai-warmup] triggering warmup at', MASKING_SERVICE_URL)
    const warmupResp = await fetch(`${MASKING_SERVICE_URL}/warmup`, {
      method: 'POST',
      signal: AbortSignal.timeout(290_000), // just under maxDuration
    })

    if (!warmupResp.ok) {
      const text = await warmupResp.text().catch(() => '')
      return NextResponse.json(
        { status: 'error', reason: `warmup failed: HTTP ${warmupResp.status}: ${text.slice(0, 200)}` },
        { status: 502 },
      )
    }

    const result = await warmupResp.json()
    console.info('[ai-warmup] ✓ warmup complete:', result)
    return NextResponse.json({ status: 'warmed', ...result })
  } catch (e) {
    console.warn('[ai-warmup] warmup failed:', e?.message)
    return NextResponse.json(
      { status: 'error', reason: e?.message || 'warmup failed' },
      { status: 502 },
    )
  }
}

// GET for simple health probes
export async function GET() {
  if (!MASKING_SERVICE_URL) {
    return NextResponse.json({
      configured: false,
      status: 'not_configured',
    })
  }

  try {
    const resp = await fetch(`${MASKING_SERVICE_URL}/health`, {
      signal: AbortSignal.timeout(5000),
    })
    if (!resp.ok) {
      return NextResponse.json({ configured: true, status: 'unreachable' }, { status: 502 })
    }
    const health = await resp.json()
    return NextResponse.json({
      configured: true,
      status: 'ok',
      modelsLoaded: {
        sam3: health.sam3_loaded || false,
        sam2: health.sam2_loaded || false,
        depth: health.depth_loaded || false,
        ground: health.ground_loaded || false,
      },
      allWarm: Boolean(
        health.sam2_loaded &&
        health.depth_loaded &&
        health.ground_loaded,
      ),
    })
  } catch {
    return NextResponse.json({ configured: true, status: 'unreachable' }, { status: 502 })
  }
}
