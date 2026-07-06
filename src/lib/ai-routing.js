/**
 * ai-routing — per-capability client/server execution policy
 * ------------------------------------------------------------
 * The editor's AI features decompose into CAPABILITIES, each of which may be
 * servable on the CLIENT (in-browser models via transformers.js — see
 * client-ai.js), on the SERVER (Next routes → local Python mask service /
 * Gemini), or both. This module is the single source of truth for:
 *
 *   1. what each capability supports (the registry below),
 *   2. the user's per-capability preference: 'auto' | 'client' | 'server',
 *   3. the ORDERED attempt list executors should follow (`resolveOrder`).
 *
 * Routing semantics (deliberately preference-with-fallback, not strict):
 *   - 'client' / 'server': try the preferred side first; if it fails at
 *     runtime, the executor falls through to the other side (when supported)
 *     rather than failing the user's request. Masking should degrade, not die.
 *   - 'auto': server first (higher quality — SAM 2 refinement, YOLO
 *     instances, Gemini planning), client as the fallback. This is also what
 *     makes the editor work out-of-the-box without the Python service.
 *
 * Persistence: localStorage `phosmith:ai-routing` (JSON map). The legacy
 * boolean `phosmith:client-ai` (the old all-or-nothing "On-device AI" toggle)
 * migrates on first read: when set, every client-capable capability starts
 * as 'client'.
 *
 * Headless-safe: without `window` everything works in memory with 'auto'
 * defaults, so the verify script can pin the resolver's semantics.
 */

export const ROUTING_MODES = ['auto', 'client', 'server']

/**
 * The capability registry. `client`/`server` say which sides CAN serve the
 * capability; labels surface in the routing UI so users know what each side
 * actually runs.
 */
export const AI_CAPABILITIES = {
    maskPlan: {
        label: 'Language planning',
        hint: 'Turns your description into a mask plan',
        client: true,
        server: true,
        clientImpl: 'On-device rule parser',
        serverImpl: 'Gemini planner',
    },
    ground: {
        label: 'Text grounding',
        hint: 'Finds "the red jacket" in the image',
        client: true,
        server: true,
        clientImpl: 'CLIPSeg in browser',
        serverImpl: 'CLIPSeg + SAM 2 refine',
    },
    depth: {
        label: 'Depth estimation',
        hint: 'Foreground / background planes',
        client: true,
        server: true,
        clientImpl: 'Depth Anything V2 in browser',
        serverImpl: 'Depth Anything V2 (service)',
    },
    subjects: {
        label: 'Subject detection',
        hint: 'Per-instance people/animals/objects',
        // "client" here is the degraded path: the executor resolves subject
        // phrases via on-device text grounding (no instance separation).
        client: true,
        server: true,
        clientImpl: 'Text grounding (no instance split)',
        serverImpl: 'YOLO + BiRefNet (service)',
    },
    segment: {
        label: 'Select subject',
        hint: 'One-click subject matte',
        client: true,
        server: true,
        clientImpl: 'RMBG-1.4 in browser',
        serverImpl: 'BiRefNet (service)',
    },
    sam: {
        label: 'Click / box select',
        hint: 'Point or drag to select an object',
        client: true,
        server: true,
        clientImpl: 'SAM 3 Tracker in browser',
        serverImpl: 'SAM 3.1 (service)',
    },
    inpaint: {
        label: 'Object fill',
        hint: 'Fills removed objects with background texture',
        // "client" here means the LOCAL mask service's LaMa (on this machine),
        // not in-browser — the closest thing to on-device inpainting we have.
        // The /api/ai/inpaint route maps the preference to its backend param:
        // client → lama, server → hf, auto → lama-first with HF fallback.
        client: true,
        server: true,
        clientImpl: 'LaMa (local mask service)',
        serverImpl: 'Stable Diffusion (Hugging Face)',
    },
}

const STORAGE_KEY = 'phosmith:ai-routing'
const LEGACY_CLIENT_KEY = 'phosmith:client-ai'
const CHANGED_EVENT = 'phosmith:ai-routing-changed'

const hasWindow = () => typeof window !== 'undefined'

/** In-memory policy for headless use; browser reads merge localStorage. */
let memoryPolicy = {}

const sanitizeMode = (mode) => (ROUTING_MODES.includes(mode) ? mode : 'auto')

const readStored = () => {
    if (!hasWindow()) return { ...memoryPolicy }
    try {
        const raw = window.localStorage.getItem(STORAGE_KEY)
        if (raw) {
            const parsed = JSON.parse(raw)
            return parsed && typeof parsed === 'object' ? parsed : {}
        }
        // First read with no stored policy: migrate the legacy boolean.
        if (window.localStorage.getItem(LEGACY_CLIENT_KEY) === '1') {
            const migrated = {}
            for (const [cap, def] of Object.entries(AI_CAPABILITIES)) {
                if (def.client) migrated[cap] = 'client'
            }
            window.localStorage.setItem(STORAGE_KEY, JSON.stringify(migrated))
            return migrated
        }
        return {}
    } catch {
        return { ...memoryPolicy }
    }
}

const writeStored = (policy) => {
    memoryPolicy = { ...policy }
    if (!hasWindow()) return
    try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(policy))
    } catch { /* storage blocked — in-memory policy still applies */ }
}

const emit = () => {
    if (!hasWindow()) return
    try {
        window.dispatchEvent(new CustomEvent(CHANGED_EVENT, { detail: getRoutingPolicy() }))
    } catch { /* SSR */ }
}

/** Full policy map: every known capability → sanitized mode. */
export const getRoutingPolicy = () => {
    const stored = readStored()
    const out = {}
    for (const cap of Object.keys(AI_CAPABILITIES)) {
        out[cap] = sanitizeMode(stored[cap])
    }
    return out
}

/** One capability's mode ('auto' for unknown capabilities). */
export const getRoutingMode = (capability) => getRoutingPolicy()[capability] || 'auto'

export const setRoutingMode = (capability, mode) => {
    if (!AI_CAPABILITIES[capability]) return
    const policy = { ...readStored(), [capability]: sanitizeMode(mode) }
    writeStored(policy)
    emit()
}

export const resetRoutingPolicy = () => {
    writeStored({})
    if (hasWindow()) {
        try { window.localStorage.removeItem(LEGACY_CLIENT_KEY) } catch { /* ignore */ }
    }
    emit()
}

/**
 * The ordered attempt list for a capability: which side to try first, and
 * whether a fallback side exists. Always non-empty for known capabilities.
 *
 * @param {string} capability
 * @returns {Array<'client'|'server'>}
 */
export const resolveOrder = (capability) => {
    const def = AI_CAPABILITIES[capability]
    if (!def) return ['server']
    const sides = []
    const mode = getRoutingMode(capability)
    const preferClient = mode === 'client'
    const first = preferClient ? 'client' : 'server'
    const second = preferClient ? 'server' : 'client'
    if (def[first]) sides.push(first)
    if (def[second]) sides.push(second)
    return sides
}

/** Convenience: is the preferred side for `capability` the client? */
export const prefersClient = (capability) => resolveOrder(capability)[0] === 'client'

/**
 * Capability ids the user has explicitly routed to the CLIENT (Device). These
 * are the ones whose in-browser models are worth downloading ahead of time —
 * 'auto' (server-first) and 'server' need no client model up front, so they're
 * excluded to avoid pulling hundreds of MB the user didn't ask for. Drives the
 * background model prefetch (see prefetchClientModels in client-ai.js).
 */
export const getClientPreferredCapabilities = () =>
    Object.keys(AI_CAPABILITIES).filter((cap) => getRoutingMode(cap) === 'client')

/** Subscribe to policy changes (returns unsubscribe). */
export const subscribeRouting = (cb) => {
    if (!hasWindow() || typeof cb !== 'function') return () => {}
    const handler = () => cb(getRoutingPolicy())
    window.addEventListener(CHANGED_EVENT, handler)
    return () => window.removeEventListener(CHANGED_EVENT, handler)
}
