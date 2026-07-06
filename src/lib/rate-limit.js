// Per-user rate limiters for the AI endpoints. Backed by the Redis adapter
// when configured, with the local in-memory cache as a development fallback.
//
// Implementation: fixed-window counter. Each request derives a window key
// from floor(now/windowMs), so all requests within the same window share a
// counter. When the window rolls, the next request gets a fresh counter.
// Simpler than sliding-window and good enough for spam protection.
//
// The limits aren't meant to slow real users — they exist to:
//   - Protect the shared Gemini free quota (1,500 RPD across the user base)
//     from a single noisy client.
//   - Cap ImageKit AI credit burn per user per minute.
//   - Surface a clean 429 instead of an opaque failure when quotas hit.

import { NextResponse } from "next/server"
import { getRedis } from "./redis"

const LIMITERS = {
    "edit-plan":         { count: 30,  windowSec: 60 },  // 30 / minute
    "edit-judge":        { count: 20,  windowSec: 60 },  // 20 / minute — vision judge (2 images per call)
    "ai-extend":         { count: 5,   windowSec: 60 },  // 5  / minute — expensive AI
    "ai-segment":        { count: 5,   windowSec: 60 },  // 5  / minute — HuggingFace segmentation
    "ai-sam3":           { count: 15,  windowSec: 60 },  // 15 / minute — interactive click/box prompts (the AI Object Eraser fires one per click, so this must absorb a multi-subject click burst)
    "ai-inpaint":        { count: 5,   windowSec: 60 },  // 5  / minute — GPU-heavy inpaint (LaMa / HF SD)
    "ai-auto-crop":      { count: 10,  windowSec: 60 },  // 10 / minute — local subject + depth + saliency pipeline
    "ai-ground":         { count: 10,  windowSec: 60 },  // 10 / minute — CLIPSeg + SAM2 text grounding
    "ai-mask-plan":      { count: 30,  windowSec: 60 },  // 30 / minute — NL mask planning (Gemini text-only)
    "ai-collage-plan":   { count: 12,  windowSec: 60 },  // 12 / minute — vision collage planner (up to ~6 thumbnails per call)
    "ai-stretch-plan":   { count: 10,  windowSec: 60 },  // 10 / minute — AI pixel stretch planner (one-shot per image)
    "shape-mask":        { count: 120, windowSec: 60 },  // 120/minute — deterministic local shape rasterization
    "imagekit-resolve":  { count: 60,  windowSec: 60 },  // 60 / minute
    "imagekit-upload":   { count: 30,  windowSec: 60 },  // 30 / minute
    "canvas-snapshot":   { count: 240, windowSec: 60 },  // 240/ minute — write-behind cache
    "canvas-presence":   { count: 60,  windowSec: 60 },  // 60 / minute — concurrent-device heartbeat (~6/min/tab steady state)
}

export const enforceRateLimit = async (kind, identifier) => {
    const cfg = LIMITERS[kind]
    if (!cfg || !identifier) return { ok: true }

    const windowMs = cfg.windowSec * 1000
    const windowStart = Math.floor(Date.now() / windowMs) * windowMs
    const windowEnd = windowStart + windowMs
    const key = `rl:${kind}:${identifier}:${windowStart}`

    const count = await getRedis().incr(key, cfg.windowSec + 1)
    const remaining = Math.max(0, cfg.count - count)

    if (count > cfg.count) {
        return {
            ok: false,
            limit: cfg.count,
            remaining,
            reset: windowEnd,
            reason: `Rate limit exceeded for ${kind} (${count}/${cfg.count} in current ${cfg.windowSec}s window)`,
        }
    }
    return { ok: true, limit: cfg.count, remaining, reset: windowEnd }
}

export const rateLimitResponse = (result) => {
    if (result.ok) return null
    const retryAfterSeconds = Math.max(1, Math.ceil((result.reset - Date.now()) / 1000))
    return NextResponse.json(
        {
            error: result.reason || "Too many requests",
            limit: result.limit,
            remaining: result.remaining,
            retryAfter: retryAfterSeconds,
        },
        {
            status: 429,
            headers: {
                "Retry-After": String(retryAfterSeconds),
                "X-RateLimit-Limit": String(result.limit ?? ""),
                "X-RateLimit-Remaining": String(result.remaining ?? 0),
            },
        },
    )
}
