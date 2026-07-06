"use client"

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react"
import { luminanceLayer, colorLayer, linearLayer, radialLayer, semanticLayer, depthLayer, smartBrushLayer, lassoLayer, brushLayer, BLEND_OPS, FILL_MODES, MAX_LAYERS, sanitiseFill } from "@/lib/megashader"
import { isAgentActing, recordChange } from "@/lib/change-journal"

// Journal a mask-panel edit. Agent-driven chain mutations go through
// mask-commands (which the command registry already records), so only
// direct user edits in the panel log here.
const journalMaskEdit = (label) => {
    if (!isAgentActing()) recordChange({ label, domain: 'mask' })
}

/**
 * useMaskLayers
 * --------------
 * React-side state container for the megashader mask stack. The hook
 * mirrors the lifecycle of `usePixelMaskTool` (the codebase's existing
 * brush-driven mask hook) but is structured around a *chain* of layers
 * rather than a single editable canvas — the megashader is non-destructive
 * and stacks.
 *
 * Returned API:
 *   - `stack`        : the current MaskStack
 *   - `addLayer`     : (kind, params?) => adds a new layer at the end
 *   - `removeLayer`  : (id) => removes the layer
 *   - `updateLayer`  : (id, patch) => merges a partial layer update
 *   - `setLayerOp`   : (id, op) => sets the compositing op
 *   - `moveLayer`    : (id, direction) => 'up' | 'down'
 *   - `clearAll`     : () => empties the chain
 *   - `setGlobalAlpha`: (0..1) => sets the global mask strength fader
 *   - `isReady`      : false while the initial empty stack is being
 *                      established; otherwise true (kept for symmetry with
 *                      the rest of the editor's hooks).
 *
 * The hook deliberately does NOT compile GLSL or touch the canvas itself.
 * It only owns the data + emits change events via the returned `stack`.
 * A separate `useEffect` in the canvas component subscribes to `stack`
 * and calls `applyMegashaderFilter` whenever it changes.
 *
 * Persistence:
 *   Steps 1-7 keep the layer metadata + per-layer textures in module
 *   scope. Persistence to Neon/Redis is Step 8.
 *
 * @returns {object}
 */
const MAX_STACK_HISTORY = 50

export const useMaskLayers = () => {
    const initial = useMemo(() => ({ chain: [], base: null }), [])
    const [stack, dispatch] = useReducer(reducer, initial)
    const isReady = true
    const lastSignatureRef = useRef(null)

    // Currently-selected layer (the "active" layer a brush/lasso edits and
    // the UI highlights). Separate from the reducer state so selection
    // changes don't churn the megashader filter.
    const [selectedLayerId, setSelectedLayerId] = useState(null)

    // Stack-level undo/redo. We snapshot the chain on STRUCTURAL mutations
    // (add / remove / reorder / op / fillMode / clear) into a past ring;
    // param slider edits (updateLayer) are intentionally NOT snapshotted so
    // dragging a slider doesn't flood the history. Textures are NOT freed on
    // remove (so undo can restore a removed texture-backed layer); the
    // module-scoped texture cache lives for the editor session and is freed
    // on full page reload, while persistence round-trips textures through the
    // on-image MegashaderFilter (see fabric-megashader-filter.js).
    const stackRef = useRef(stack)
    const pastRef = useRef([])
    const futureRef = useRef([])
    useEffect(() => { stackRef.current = stack }, [stack])

    const snapshot = useCallback(() => {
        const chain = stackRef.current?.chain || []
        const base = stackRef.current?.base || null
        // structuredClone keeps the snapshot independent of later mutations;
        // textures live in the module cache keyed by string, so cloning the
        // chain (which only holds the key strings) is sufficient.
        try {
            pastRef.current.push(structuredClone({ chain, base }))
        } catch {
            pastRef.current.push({
                chain: chain.map((e) => ({ op: e.op, layer: { ...e.layer } })),
                base: base ? { ...base } : null,
            })
        }
        if (pastRef.current.length > MAX_STACK_HISTORY) pastRef.current.shift()
        futureRef.current = []
    }, [])

    // One history entry per param-edit burst (grade sliders, gamma, wheels,
    // boundary): snapshot before the FIRST change, then coalesce while edits
    // keep arriving within 350 ms of each other.
    const paramSnapAtRef = useRef(0)
    const paramSnapshot = useCallback(() => {
        const now = Date.now()
        if (now - paramSnapAtRef.current > 350) snapshot()
        paramSnapAtRef.current = now
    }, [snapshot])

    // Notify subscribers on EVERY stack change. We use a custom event on
    // `window` (matching the codebase's phosmith:* convention — see page.jsx's
    // "phosmith:tool-sub").
    //
    // We deliberately do NOT gate on `computeSignature` here. The signature
    // only tracks structural fields (kind/op/inverted/visible/lock), so a
    // uniform-only change — a per-layer adjustment slider, or the new
    // fillMode / fillColor / fillStrength — would never reach the canvas and
    // the edit would silently not render. The canvas effect debounces these
    // events (150 ms) and the compiled program is LRU-cached, so firing on
    // every change is cheap; the signature is still used elsewhere for
    // coarse change detection.
    useEffect(() => {
        lastSignatureRef.current = computeSignature(stack)
        try {
            window.dispatchEvent(new CustomEvent('phosmith:mask-layers-changed', { detail: { stack } }))
        } catch { /* SSR safe */ }
    }, [stack])

    const addLayer = useCallback((kind, params = {}) => {
        if (!kind) return null
        // Hard cap (matches the compiler's MAX_LAYERS). Adding beyond it would
        // be silently truncated by the shader compiler — wiping the user's new
        // selection with no feedback — so refuse and signal the UI to toast.
        if ((stackRef.current?.chain?.length || 0) >= MAX_LAYERS) {
            try { window.dispatchEvent(new CustomEvent('phosmith:mask-layer-limit', { detail: { max: MAX_LAYERS } })) } catch { /* SSR */ }
            return null
        }
        // For kinds with real per-kind parameters (luminance, color in Step 2;
        // linear, radial, smartBrush, semantic, depth in later steps), defer
        // to the factory in mask-types.js so the layer comes out with the
        // correct defaults and clamping. Unknown / unsupported kinds fall
        // through to the legacy generic shape (the compiler's stub bodies
        // don't care about field shape).
        let layer = null
        if (kind === 'luminance') {
            layer = luminanceLayer(params)
        } else if (kind === 'color') {
            layer = colorLayer(params)
        } else if (kind === 'linear') {
            layer = linearLayer(params)
        } else if (kind === 'radial') {
            layer = radialLayer(params)
        } else if (kind === 'smartBrush') {
            layer = smartBrushLayer(params)
        } else if (kind === 'semantic') {
            layer = semanticLayer(params)
        } else if (kind === 'depth') {
            layer = depthLayer(params)
        } else if (kind === 'lasso') {
            layer = lassoLayer(params)
        } else if (kind === 'brush') {
            layer = brushLayer(params)
        } else {
            // Unknown / future kind: build a minimal layer from `params`
            // (the compiler's stub bodies don't care about field shape).
            // Re-assert id/label/visible/inverted AFTER the `...params`
            // spread so a caller that supplies `id: ''` or `label: ''`
            // (empty string is truthy-falsy in the wrong direction) still
            // ends up with valid values. Same fix as `sanitiseLayer` —
            // see the comment there for the full explanation.
            const id = `layer-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`
            layer = {
                id,
                kind,
                label: params.label || `Mask ${kind}`,
                opacity: typeof params.opacity === 'number' ? params.opacity : 1,
                visible: params.visible !== false,
                inverted: params.inverted === true,
                ...params,
                id: params.id || id,
                label: params.label || `Mask ${kind}`,
                visible: params.visible !== false,
                inverted: params.inverted === true,
            }
            // Sub-task 1 — `lock` is on every layer (MaskLayerBase), so the
            // generic-kinds fallback has to set it too. The factory-built
            // kinds (luminance/color/linear/radial/smartBrush/semantic/
            // depth) all default it in their factory.
            if (typeof layer.lock !== 'boolean') {
                layer.lock = false
            }
        }
        // Boundary-grow fields ride through the kind factories (which whitelist
        // their params): the pristine base + absolute grow offset.
        if (layer && typeof params.baseTextureKey === 'string' && params.baseTextureKey) {
            layer.baseTextureKey = params.baseTextureKey
        }
        if (layer && typeof params.growPx === 'number') {
            layer.growPx = params.growPx
        }
        // Root-cause #1: new layers default to 'fill' mode so the selection
        // is visible the instant it's added (no slider drag required). The
        // factory default is 'adjust'; we override here unless the caller
        // explicitly asked for a mode via params. The lasso factory already
        // defaults to 'fill', so this is a no-op for it.
        const fillSeed = {
            fillMode: params.fillMode || layer.fillMode || 'fill',
            fillColor: params.fillColor || layer.fillColor,
            fillStrength: typeof params.fillStrength === 'number' ? params.fillStrength : layer.fillStrength,
        }
        layer = { ...layer, ...sanitiseFill(fillSeed) }
        snapshot()
        dispatch({ type: 'add', layer })
        setSelectedLayerId(layer.id)
        journalMaskEdit(`Mask: add ${kind} layer`)
        return layer.id
    }, [snapshot])

    const removeLayer = useCallback((id) => {
        // NOTE: we intentionally do NOT free the texture cache entry here.
        // Undo can restore a removed texture-backed layer, and the cache is
        // keyed by an opaque string the restored layer still references. The
        // cache lives for the editor session and is freed on full reload; the
        // on-image filter also keeps the chain (and its textures) alive across
        // tool switches.
        snapshot()
        dispatch({ type: 'remove', id })
        setSelectedLayerId((cur) => (cur === id ? null : cur))
        journalMaskEdit('Mask: remove layer')
    }, [snapshot])

    const updateLayer = useCallback((id, patch) => {
        // Burst-debounced snapshot so grade/param edits are undoable without
        // flooding the ring (one entry per drag, not per tick).
        paramSnapshot()
        dispatch({ type: 'update', id, patch })
    }, [paramSnapshot])

    // Root-cause #1: change a layer's output mode (adjust/fill/erase). This
    // IS snapshotted (it's a meaningful, undoable change, unlike a slider).
    const setFillMode = useCallback((id, mode) => {
        if (!FILL_MODES.includes(mode)) return
        snapshot()
        dispatch({ type: 'update', id, patch: { fillMode: mode } })
    }, [snapshot])

    const setLayerOp = useCallback((id, op) => {
        // Sub-task 1 — `BLEND_OPS` now includes the four Photoshop-parity
        // modes (screen/lighten/darken/overlay) in addition to the original
        // four. Validate before dispatch so a stray string from a stale
        // picker can't poison the GLSL chain (the compiler has a default
        // branch but it renders the layer as `add` — silent corruption is
        // worse than a thrown error here).
        if (typeof op !== 'string' || !BLEND_OPS.includes(/** @type {any} */ (op))) {
            if (typeof console !== 'undefined') {
                console.warn(`[useMaskLayers] setLayerOp: ignoring unknown op "${op}"`)
            }
            return
        }
        snapshot()
        dispatch({ type: 'setOp', id, op })
    }, [snapshot])

    const moveLayer = useCallback((id, direction) => {
        // Sub-task 1 — accept 'up' | 'down' (legacy) and 'top' | 'bottom'
        // (Photoshop's "Bring to front" / "Send to back"). Drag-to-reorder
        // uses the same dispatcher; a drag that lands at index 0 dispatches
        // 'top' and a drag from the top dispatches 'bottom'.
        if (!['up', 'down', 'top', 'bottom'].includes(direction)) {
            if (typeof console !== 'undefined') {
                console.warn(`[useMaskLayers] moveLayer: ignoring unknown direction "${direction}"`)
            }
            return
        }
        snapshot()
        dispatch({ type: 'move', id, direction })
    }, [snapshot])

    const clearAll = useCallback(() => {
        snapshot()
        dispatch({ type: 'clear' })
        setSelectedLayerId(null)
        journalMaskEdit('Mask: clear all layers')
    }, [snapshot])

    const selectLayer = useCallback((id) => setSelectedLayerId(id), [])

    // Undo/redo the STACK structure. Dispatches a wholesale `set` of the
    // chain; the canvas effect re-applies the megashader on the resulting
    // change. Bound to the same phosmith:mask-undo/redo events below so the
    // global Ctrl+Z path can drive them when the Mask tool owns undo.
    const currentEntry = () => {
        const chain = stackRef.current?.chain || []
        const base = stackRef.current?.base || null
        try { return structuredClone({ chain, base }) }
        catch { return { chain: chain.map((e) => ({ op: e.op, layer: { ...e.layer } })), base: base ? { ...base } : null } }
    }
    // Entries are { chain, base }; tolerate bare chain arrays (older shape).
    const restoreEntry = (entry) => {
        const chain = Array.isArray(entry) ? entry : entry?.chain || []
        const base = Array.isArray(entry) ? stackRef.current?.base || null : entry?.base || null
        dispatch({ type: 'restore', chain, base })
    }

    const undo = useCallback(() => {
        if (pastRef.current.length === 0) return false
        const prev = pastRef.current.pop()
        futureRef.current.push(currentEntry())
        restoreEntry(prev)
        return true
    }, [])

    const redo = useCallback(() => {
        if (futureRef.current.length === 0) return false
        const next = futureRef.current.pop()
        pastRef.current.push(currentEntry())
        restoreEntry(next)
        return true
    }, [])

    const setBase = useCallback((patch) => {
        paramSnapshot()
        dispatch({ type: 'setBase', base: patch })
    }, [paramSnapshot])

    const canUndo = pastRef.current.length > 0
    const canRedo = futureRef.current.length > 0

    // Hydrate the whole chain from persisted project state (textures must be
    // re-registered via setMaskTexture by the caller BEFORE calling this so
    // the renderer doesn't sample a missing texture). Resets history.
    const setChain = useCallback((chain) => {
        const safe = Array.isArray(chain) ? chain : []
        pastRef.current = []
        futureRef.current = []
        dispatch({ type: 'set', chain: safe })
    }, [])

    // Reconcile with the agent command layer: when an agent (or any non-UI
    // caller) mutates the chain on the image via src/lib/agent/mask-commands,
    // it dispatches `phosmith:mask-chain-replaced` so this panel re-syncs. The
    // dispatch is distinct from `phosmith:mask-layers-changed` (which WE emit),
    // so there's no feedback loop.
    useEffect(() => {
        const onReplaced = (e) => {
            const chain = e?.detail?.stack?.chain
            if (Array.isArray(chain)) setChain(chain)
        }
        try { window.addEventListener('phosmith:mask-chain-replaced', onReplaced) } catch { /* SSR */ }
        return () => { try { window.removeEventListener('phosmith:mask-chain-replaced', onReplaced) } catch { /* SSR */ } }
    }, [setChain])

    const setGlobalAlpha = useCallback((value) => {
        // The global alpha is NOT part of MaskStack — it's a separate UI
        // fader that lives in the hook consumer. Exposed as a callback
        // so the test panel can write to it without owning a second
        // piece of state. Step 1 stores it in the test panel's local
        // state; the canvas reads it through a window event.
        try {
            window.dispatchEvent(new CustomEvent('phosmith:mask-global-alpha', { detail: { value } }))
        } catch { /* SSR safe */ }
    }, [])

    // "Show mask" overlay (view mode) + global invert. Like globalAlpha,
    // these are chain-wide render options the canvas reads via window events
    // (the canvas re-applies the megashader immediately). Local state mirrors
    // them so the UI toggles reflect the current value.
    const [showMaskOverlay, setShowMaskOverlayState] = useState(false)
    const [globalInvert, setGlobalInvertState] = useState(false)
    const setShowMaskOverlay = useCallback((value) => {
        const v = !!value
        setShowMaskOverlayState(v)
        try { window.dispatchEvent(new CustomEvent('phosmith:mask-overlay', { detail: { value: v } })) } catch { /* SSR safe */ }
    }, [])
    const setGlobalInvert = useCallback((value) => {
        const v = !!value
        setGlobalInvertState(v)
        try { window.dispatchEvent(new CustomEvent('phosmith:mask-invert', { detail: { value: v } })) } catch { /* SSR safe */ }
    }, [])

    return {
        stack,
        addLayer,
        removeLayer,
        updateLayer,
        setLayerOp,
        setFillMode,
        moveLayer,
        clearAll,
        setGlobalAlpha,
        showMaskOverlay,
        setShowMaskOverlay,
        globalInvert,
        setGlobalInvert,
        selectedLayerId,
        selectLayer,
        setBase,
        undo,
        redo,
        canUndo,
        canRedo,
        setChain,
        isReady,
    }
}

/**
 * Compute a coarse signature for change detection. Used by the canvas
 * effect to decide whether to reinstall the filter or just update its
 * stack reference. Two stacks with the same signature can share a
 * WebGLProgram (the compiler's cache key is more granular than this
 * signature — this is for *filter re-installation* deduping, not shader
 * recompilation).
 */
const computeSignature = (stack) => {
    if (!stack || !Array.isArray(stack.chain)) return 'empty'
    return stack.chain
        .map((e) => `${e.layer.id}:${e.layer.kind}:${e.op}:${e.layer.inverted ? 1 : 0}:${e.layer.visible === false ? 0 : 1}:${e.layer.lock === true ? 1 : 0}`)
        .join('|')
}

const reducer = (state, action) => {
    switch (action.type) {
        case 'add': {
            const chain = state.chain.slice()
            const layer = action.layer
            // First layer always uses 'replace' (compile enforces this too).
            const op = chain.length === 0 ? 'replace' : 'add'
            chain.push({ layer, op })
            return { ...state, chain }
        }
        case 'restore':
            // Undo/redo: replace chain AND base wholesale (no merge).
            return reducer({ ...state, base: action.base ?? null }, { type: 'set', chain: action.chain })
        case 'setBase': {
            // Whole-image pre-grade (gamma/curves/wheels). null clears; a patch
            // merges. Lives on the stack so filter sync + persistence ride free.
            const base = action.base == null ? null : { ...(state.base || {}), ...action.base }
            return { ...state, base }
        }
        case 'set': {
            // Wholesale replacement of the chain (undo/redo, hydrate from
            // persisted state). Re-pin slot 0 to 'replace' and demote any
            // stray 'replace' elsewhere (see the 'move' case for why).
            const chain = (Array.isArray(action.chain) ? action.chain : []).map((e, i) => {
                if (i === 0) return e.op === 'replace' ? e : { ...e, op: 'replace' }
                return e.op === 'replace' ? { ...e, op: 'add' } : e
            })
            return { ...state, chain }
        }
        case 'remove': {
            const chain = state.chain.filter((e) => e.layer.id !== action.id)
            // The first entry's op must stay 'replace' after removal; and a
            // formerly-first entry that carried 'replace' must be demoted to
            // 'add' if it is no longer at slot 0.
            for (let i = 0; i < chain.length; i += 1) {
                if (i === 0 && chain[0].op !== 'replace') chain[0] = { ...chain[0], op: 'replace' }
                if (i > 0 && chain[i].op === 'replace') chain[i] = { ...chain[i], op: 'add' }
            }
            return { ...state, chain }
        }
        case 'clear': {
            // Textures are NOT freed here (undo can restore a cleared chain);
            // the cache is released on full page reload.
            return { ...state, chain: [] }
        }
        case 'update': {
            const chain = state.chain.map((e) => {
                if (e.layer.id !== action.id) return e
                return { ...e, layer: { ...e.layer, ...action.patch } }
            })
            return { ...state, chain }
        }
        case 'setOp': {
            const idx = state.chain.findIndex((e) => e.layer.id === action.id)
            if (idx <= 0) return state // cannot change op of first layer
            const chain = state.chain.map((e, i) => {
                if (i !== idx) return e
                return { ...e, op: action.op }
            })
            return { ...state, chain }
        }
        case 'move': {
            const idx = state.chain.findIndex((e) => e.layer.id === action.id)
            if (idx < 0) return state
            let swap = idx
            if (action.direction === 'up') swap = idx - 1
            else if (action.direction === 'down') swap = idx + 1
            else if (action.direction === 'top') swap = 0
            else if (action.direction === 'bottom') swap = state.chain.length - 1
            if (swap === idx) return state
            if (swap < 0 || swap >= state.chain.length) return state
            const chain = state.chain.slice()
            const [moved] = chain.splice(idx, 1)
            chain.splice(swap, 0, moved)
            // After reordering, slot 0 must be 'replace' and NO other slot
            // may be 'replace' — otherwise the compiler treats a stray
            // mid-chain 'replace' as a wholesale overwrite, wiping every
            // layer below it. (This happens when the original slot-0 entry,
            // which carries 'replace', is moved down.)
            for (let i = 0; i < chain.length; i += 1) {
                if (i === 0 && chain[0].op !== 'replace') chain[0] = { ...chain[0], op: 'replace' }
                if (i > 0 && chain[i].op === 'replace') chain[i] = { ...chain[i], op: 'add' }
            }
            return { ...state, chain }
        }
        default:
            return state
    }
}

export default useMaskLayers
