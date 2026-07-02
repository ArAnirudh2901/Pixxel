"use client"

import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
    ArrowDown, ArrowDownLeft, ArrowDownRight, ArrowLeft, ArrowRight,
    ArrowUp, ArrowUpLeft, ArrowUpRight,
    Blend, ChevronDown, ChevronRight, Circle, Contrast, Cpu, Crosshair, Eye, Layers,
    Lasso, Loader2, Minus, Mountain, MousePointer, Palette, Paintbrush, Plus, RotateCcw, Scissors, Spline, Sparkles, Square, Sun, Wand2, X,
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { Circle as FabricCircle, Ellipse, FabricImage, Line, Polyline, Rect as FabricRect } from 'fabric'
import { toast } from 'sonner'
import { useCanvas } from '../../../../../../../context/context'
import usePixelMaskTool, { MIN_BRUSH, MAX_BRUSH } from '../../../../../../../hooks/usePixelMaskTool'
import useMaskLayers from '../../../../../../../hooks/useMaskLayers'
import { computeImageHistogram, getHistogramSourceElement } from '@/lib/image-histogram'
import { rgbToHsb } from '@/lib/color-utils'
import { setMaskTexture, rasterisePath, smoothToBezier } from '@/lib/megashader'
import { buildPackedLutFromCurves } from '@/lib/curve-lut'
import { expandLayerBoundary } from '@/lib/mask-grow'
import { AI_CAPABILITIES, getRoutingPolicy, resetRoutingPolicy, setRoutingMode, subscribeRouting } from '@/lib/ai-routing'
import { getClientAIState, runClientAISelfTest, subscribeClientAI } from '@/lib/client-ai'
import { checkMaskService, decodeMaskBlob, base64PngToBlob, prepareImageBlob } from '@/lib/mask-service-client'
import { computeGradientMagnitude, snapToEdgePoint } from '@/lib/mask-edge-snap'
import {
    BrushSizeControl,
    LabeledSlider,
    LuminanceHistogram,
    MaskActionButtons,
    MaskChainCard,
    ModeToggle,
    TipCard,
    ToolEmptyState,
} from './_pixel-tool-ui'

/* ─── collapsible section ─── */
const Section = ({ title, icon: Icon, defaultOpen = false, children, badge }) => {
    const [open, setOpen] = useState(defaultOpen)
    return (
        <div className={`mask-section ${open ? 'mask-section--open' : ''}`}>
            <button
                type="button"
                onClick={() => setOpen(v => !v)}
                className="mask-section__header"
            >
                {Icon && <Icon className="mask-section__icon" />}
                <span className="mask-section__title">{title}</span>
                {badge && (
                    <span className="mask-section__badge">{badge}</span>
                )}
                <ChevronRight className="mask-section__chevron" />
            </button>
            <AnimatePresence initial={false}>
                {open && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
                        className="overflow-hidden"
                    >
                        <div className="mask-section__body space-y-3">
                            {children}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

/* ─── category divider between section groups ─── */
const CategoryHeader = ({ label }) => (
    <div className="mask-category-header">
        <span>{label}</span>
    </div>
)

/* ─── gradient direction icons ─── */
const DIRECTIONS = [
    { id: 'top', icon: ArrowUp, label: 'Top → Bottom' },
    { id: 'bottom', icon: ArrowDown, label: 'Bottom → Top' },
    { id: 'left', icon: ArrowLeft, label: 'Left → Right' },
    { id: 'right', icon: ArrowRight, label: 'Right → Left' },
    { id: 'top-left', icon: ArrowUpLeft, label: 'Top-Left' },
    { id: 'top-right', icon: ArrowUpRight, label: 'Top-Right' },
    { id: 'bottom-left', icon: ArrowDownLeft, label: 'Bottom-Left' },
    { id: 'bottom-right', icon: ArrowDownRight, label: 'Bottom-Right' },
]

/* ─── color swatch component ─── */
const ColorSwatch = ({ color, size = 24 }) => {
    if (!color) return null
    return (
        <div
            className="rounded border shrink-0"
            style={{
                width: size, height: size,
                background: `rgb(${color.r}, ${color.g}, ${color.b})`,
                borderColor: 'var(--border-subtle)',
            }}
        />
    )
}

const CLOSED_BRUSH_MIN_POINTS = 8
const CLOSED_BRUSH_MIN_AREA = 64

const pathLength = (points) => {
    let length = 0
    for (let i = 1; i < points.length; i += 1) {
        const dx = points[i].x - points[i - 1].x
        const dy = points[i].y - points[i - 1].y
        length += Math.sqrt(dx * dx + dy * dy)
    }
    return length
}

const polygonArea = (points) => {
    let area = 0
    for (let i = 0; i < points.length; i += 1) {
        const a = points[i]
        const b = points[(i + 1) % points.length]
        area += a.x * b.y - b.x * a.y
    }
    return Math.abs(area) / 2
}

const isClosedBrushPath = (points, brushSize) => {
    if (!Array.isArray(points) || points.length < CLOSED_BRUSH_MIN_POINTS) return false
    const first = points[0]
    const last = points[points.length - 1]
    const dx = last.x - first.x
    const dy = last.y - first.y
    const closeDistance = Math.sqrt(dx * dx + dy * dy)
    const radius = Math.max(0.5, brushSize / 2)
    const closeThreshold = Math.max(10, Math.min(96, radius * 1.5))
    const area = polygonArea(points)
    return (
        closeDistance <= closeThreshold &&
        pathLength(points) >= closeThreshold * 3 &&
        area >= Math.max(CLOSED_BRUSH_MIN_AREA, radius * radius * 3)
    )
}

const fillClosedBrushPath = (ctx, points, scale, brushSize) => {
    if (!ctx || !isClosedBrushPath(points, brushSize)) return false
    const s = Math.max(0.0001, scale || 1)
    ctx.save()
    ctx.globalCompositeOperation = 'source-over'
    ctx.fillStyle = 'rgba(255, 255, 255, 1)'
    ctx.beginPath()
    ctx.moveTo(points[0].x * s, points[0].y * s)
    for (let i = 1; i < points.length; i += 1) {
        ctx.lineTo(points[i].x * s, points[i].y * s)
    }
    ctx.closePath()
    ctx.fill('evenodd')
    ctx.restore()
    return true
}

const MaskControls = ({ dominantColor }) => {
    const { canvasEditor } = useCanvas()
    // The pixel brush is the default manual eraser, but it must hand
    // off to the smart brush (Step 7) when the user enters that mode
    // — both write to different masks (clipPath vs texture cache) and
    // a single click would otherwise fire both. The pixel tool's undo
    // stack + brush-size UI stay live; only its pointer handlers are
    // suppressed via the `disabled` prop. We mirror `brushActive` into
    // `pixelToolDisabled` via an effect (declared below, after the
    // `brushActive` useState) so the hook body never needs to read
    // `brushActive` directly — avoiding a temporal-dead-zone error.
    const [pixelToolDisabled, setPixelToolDisabled] = useState(false)
    // Quick Erase (the destructive clipPath brush) is OFF by default. Opening
    // the Mask tool and painting must SELECT a region (non-destructive), not
    // erase the image — so the pixel tool only takes the pointer when the user
    // explicitly enables Quick Erase. See the `pixelToolDisabled` effect below.
    const [quickEraseActive, setQuickEraseActive] = useState(false)
    const tool = usePixelMaskTool({ canvasEditor, defaultMode: 'erase', supportsMagic: false, disabled: pixelToolDisabled })

    // AI Subject Selection state
    const [isSegmenting, setIsSegmenting] = useState(false)
    // Multi-subject (instance) detection state. Populated by the "Detect All
    // Subjects" button; persists until the image changes so the user can
    // re-select an individual subject without re-running BiRefNet + YOLO.
    const [isDetectingInstances, setIsDetectingInstances] = useState(false)
    const isDetectingInstancesRef = useRef(false)
    const instancesAbortRef = useRef(/** @type {AbortController | null} */ (null))
    const [subjectInstances, setSubjectInstances] = useState(/** @type {Array<any> | null} */ (null))
    const [activeInstanceIndex, setActiveInstanceIndex] = useState(/** @type {number | null} */ (null))
    const lastInstancesImageRef = useRef(/** @type {any} */ (null))
    // Id of the non-destructive megashader layer that currently holds the AI
    // subject selection. The "Select Subject" / "Detect All Subjects" / chip
    // picker all SELECT a region (they no longer erase the background); picking
    // a different subject SWAPS this single layer instead of stacking duplicates.
    const subjectLayerIdRef = useRef(/** @type {string | null} */ (null))
    // Ref-mirrors of the "running" flags. The `useCallback` closures for
    // `handleSelectSubject` / `handleSemanticRun` / `handleDepthRun` capture
    // the React state at render time, so a fast double-click (or two clicks
    // landing in the same batched render) could otherwise launch two
    // concurrent fetches and race to write the same state. The refs are
    // always current and are also the only place we read the in-flight
    // state from.
    const isSegmentingRef = useRef(false)
    // AbortController for the most recent in-flight request per tool.
    // When a new request is fired, the previous one is cancelled so the
    // older fetch can't resolve later and clobber fresher state.
    const segmentAbortRef = useRef(/** @type {AbortController | null} */ (null))

    // Color Range state
    const [colorPickerActive, setColorPickerActive] = useState(false)
    const [pickedColor, setPickedColor] = useState(null)
    const [colorTolerance, setColorTolerance] = useState(35)
    const colorPickerRef = useRef(null)

    // Luminance Range state
    const [lumaMin, setLumaMin] = useState(0)
    const [lumaMax, setLumaMax] = useState(128)

    // Gradient state
    const [gradDirection, setGradDirection] = useState('bottom')
    const [gradPosition, setGradPosition] = useState(50)
    const [gradFeather, setGradFeather] = useState(30)

    // Megashader chain state (independent from brush+clipPath). Step 2 keeps
    // this per-component; module-scope persistence is Step 7. See AGENTS.md
    // notes in hooks/useMaskLayers.js.
    const chain = useMaskLayers()
    const {
        stack, addLayer: addChainLayer, removeLayer, updateLayer,
        setLayerOp, setFillMode, moveLayer, clearAll,
        showMaskOverlay, setShowMaskOverlay, globalInvert, setGlobalInvert,
        selectedLayerId, selectLayer,
        undo: undoChain, redo: redoChain, canUndo, canRedo, setChain,
    } = chain

    // Always-current mirror of the chain so async callbacks (which capture the
    // `stack` from the render where they were created) can check the LIVE layer
    // list — used by the subject-selection swap to see whether its tracked
    // layer still exists without re-binding on every chain edit.
    const chainStackRef = useRef(stack)
    useEffect(() => { chainStackRef.current = stack }, [stack])

    // Per-mask tone curves: build the packed 256×1 RGBA LUT from the curve
    // points (master/R/G/B), register it as a texture, and point the layer at
    // it via `curveLutKey`; identity curves clear the LUT so the engine skips
    // the lookup. Drives the `LayerGradeEditor`'s curve graph in each mask card
    // (gamma + colour wheels are plain layer fields and flow through onUpdate).
    const applyCurve = useCallback((id, curves) => {
        const { packed, identity } = buildPackedLutFromCurves(curves || {})
        if (identity) {
            updateLayer(id, { curveLutKey: undefined, curves: undefined })
            return
        }
        const key = `curve-${id}`
        setMaskTexture(key, new ImageData(new Uint8ClampedArray(packed), 256, 1))
        updateLayer(id, { curveLutKey: key, curves })
    }, [updateLayer])

    // Persistence rehydrate: if the primary image already carries a persisted
    // MegashaderFilter (restored from project state by loadFromJSON, with its
    // textures re-registered by MegashaderFilter.fromObject), seed the Mask
    // Layers UI from it once so the panel reflects the saved chain.
    const chainHydratedRef = useRef(false)
    useEffect(() => {
        if (chainHydratedRef.current) return
        const img = tool.mainImage
        if (!img || !Array.isArray(img.filters)) return
        const mega = img.filters.find((f) => f && f.type === 'Megashader')
        const persisted = mega?.stack?.chain
        if (Array.isArray(persisted) && persisted.length > 0 && stack.chain.length === 0) {
            setChain(persisted)
            chainHydratedRef.current = true
        }
    }, [tool.mainImage, stack.chain.length, setChain])

    // Per-capability AI routing policy (Auto / Device / Server). Initialised
    // to all-auto and synced from localStorage AFTER mount — reading the
    // persisted policy during the first render would make SSR and client
    // markup disagree (hydration mismatch) whenever a custom policy is saved.
    const [routingPolicy, setRoutingPolicyState] = useState(() =>
        Object.fromEntries(Object.keys(AI_CAPABILITIES).map((cap) => [cap, 'auto'])))
    useEffect(() => {
        setRoutingPolicyState(getRoutingPolicy())
        return subscribeRouting(setRoutingPolicyState)
    }, [])
    const routingBadge = Object.values(routingPolicy).some((m) => m !== 'auto') ? 'Custom' : 'Auto'

    // On-device model download/readiness, surfaced so the background prefetch is
    // observable (per-capability "Ready" / "Downloading…" hints below).
    const [clientAI, setClientAI] = useState(getClientAIState)
    useEffect(() => {
        setClientAI(getClientAIState())
        return subscribeClientAI(setClientAI)
    }, [])
    // Map a routing capability to its in-browser readiness flag. Capabilities
    // with no browser model (maskPlan rule-parser, inpaint LaMa-on-service) are
    // absent — they need no download.
    const CLIENT_READY = {
        ground: clientAI.groundReady,
        subjects: clientAI.groundReady,
        depth: clientAI.depthReady,
        segment: clientAI.segmentReady,
    }

    // On-device AI self-test: runs the REAL in-browser models on a synthetic
    // scene with a known answer (see runClientAISelfTest). First run also
    // downloads + caches the models, so it doubles as a warm-up.
    const [selfTest, setSelfTest] = useState({ running: false, progress: null, report: null })
    const handleSelfTest = useCallback(async () => {
        setSelfTest({ running: true, progress: 'Starting…', report: null })
        try {
            const report = await runClientAISelfTest({
                onProgress: (msg) => setSelfTest((s) => ({ ...s, progress: msg })),
            })
            setSelfTest({ running: false, progress: null, report })
            if (report.ok) {
                toast.success(`Device AI works — ${report.device.toUpperCase()}, ${(report.totalMs / 1000).toFixed(1)}s`)
            } else {
                toast.error('Device AI self-test found problems — see the AI Processing section')
            }
        } catch (err) {
            setSelfTest({
                running: false,
                progress: null,
                report: { ok: false, device: 'unknown', totalMs: 0, checks: [{ label: 'Self-test crashed', ok: false, detail: String(err?.message || err) }] },
            })
            toast.error(err?.message || 'Device AI self-test failed')
        }
    }, [])

    // Selected layer — drives the re-editable gradient-handle gizmo (linear/
    // radial) and the brush pointer-arbitration (handles must suppress the
    // pixel brush so clicking the image doesn't paint while editing a gradient).
    const selectedLayer = stack.chain.find((e) => e.layer.id === selectedLayerId)?.layer || null
    const selKind = selectedLayer?.kind
    const isGradientSelected = selKind === 'linear' || selKind === 'radial'

    // Lightroom-style histogram for the luminance panel. Computed lazily on
    // the next tick after mainImage changes (the source <img> must be
    // decoded before drawImage is safe). Null while loading.
    const [histogram, setHistogram] = useState(null)
    useEffect(() => {
        const el = getHistogramSourceElement(tool.mainImage)
        if (!el) { setHistogram(null); return undefined }
        let cancelled = false
        // Yield to the next frame so React's commit is done before we touch
        // the DOM. The histogram walk is ~10ms on a 1MP image; on a 5MP+ image
        // it can take 50-200ms. The await keeps the UI responsive.
        Promise.resolve().then(() => {
            if (cancelled) return
            try {
                const h = computeImageHistogram(el)
                if (!cancelled) setHistogram(h)
            } catch {
                if (!cancelled) setHistogram(null)
            }
        })
        return () => { cancelled = true }
    }, [tool.mainImage])

    // Cancel any in-flight AI requests on unmount (or when the mask tool
    // itself is torn down). Without this, a slow request that resolves
    // after unmount would call setState on an unmounted component, and
    // we'd leak the AbortController plus its underlying fetch socket.
    useEffect(() => {
        return () => {
            try { segmentAbortRef.current?.abort() } catch { /* ignore */ }
            try { semanticAbortRef.current?.abort() } catch { /* ignore */ }
            try { depthAbortRef.current?.abort() } catch { /* ignore */ }
            segmentAbortRef.current = null
            semanticAbortRef.current = null
            depthAbortRef.current = null
        }
    }, [])

    /* ─── Spatial draft mode (Step 3) ─── */

    // Image dimensions of the source image (not the displayed scaled size).
    // Used to default new linear/radial layers to a full-image line/ellipse.
    // MUST come from the underlying HTMLImageElement's naturalWidth/Height —
    // falling back to fabric's `width/height` would be the *scaled* size,
    // which would put the layer's p1/p2 in scaled pixels while the GLSL's
    // uImageSize (uploaded from sourceCanvas.width/height) is in natural
    // pixels. The mismatch would be invisible until the user drags.
    const imageSize = (() => {
        if (!tool.mainImage) return null
        const sourceEl = tool.mainImage._element || tool.mainImage.getElement?.()
        const w = sourceEl?.naturalWidth || 0
        const h = sourceEl?.naturalHeight || 0
        return w > 0 && h > 0 ? { width: w, height: h } : null
    })()

    // activeDraft is the layer that's currently being placed on the canvas.
    // Set by the "Add Linear/Radial to Mask Layers" buttons; cleared on
    // mouse-up (commit) or Esc (cancel — also removes the layer).
    const [activeDraft, setActiveDraft] = useState(null)
    // dragStateRef holds the in-progress drag's start point (image-pixel
    // coords) and the active flag. We use a ref because the drag fires
    // faster than React's render cycle (window-level mousemove).
    const dragStateRef = useRef(null)
    // overlayRef holds the Fabric overlay object so we can remove it
    // when the draft ends (or when the component re-creates the overlay
    // on layer updates).
    const overlayRef = useRef(null)

    // Convert a fabric event's pointer to image-pixel coordinates. The
    // fabric canvas uses display-space; the user wants to set the layer
    // geometry in image-space pixels (so the GLSL's `vTextureCoord *
    // uImageSize` matches what the user pointed at).
    const pointerToImage = useCallback((fabricCanvas, e) => {
        if (!tool.mainImage || !fabricCanvas) return null
        const pointer = fabricCanvas.getPointer(e.e || e)
        const fabricObj = tool.mainImage
        const objLeft = fabricObj.left || 0
        const objTop = fabricObj.top || 0
        const objScaleX = fabricObj.scaleX || 1
        const objScaleY = fabricObj.scaleY || 1
        return {
            x: (pointer.x - objLeft) / objScaleX,
            y: (pointer.y - objTop) / objScaleY,
        }
    }, [tool.mainImage])

    // Convert image-pixel coordinates to display-space (where Fabric
    // objects live). Used for drawing the live preview overlay so it
    // sits exactly on top of the corresponding image pixel.
    const imageToDisplay = useCallback((imageX, imageY) => {
        if (!tool.mainImage) return null
        const fabricObj = tool.mainImage
        return {
            x: imageX * (fabricObj.scaleX || 1) + (fabricObj.left || 0),
            y: imageY * (fabricObj.scaleY || 1) + (fabricObj.top || 0),
        }
    }, [tool.mainImage])

    // Begin a spatial drag. Snaps the layer's anchor to the click point
    // and prepares the dragStateRef for the move handler.
    const handleSpatialDragStart = useCallback((e) => {
        if (!activeDraft) return
        const fabricCanvas = canvasEditor
        if (!fabricCanvas) return
        const pos = pointerToImage(fabricCanvas, e)
        if (!pos) return
        dragStateRef.current = { startX: pos.x, startY: pos.y }
        if (activeDraft.kind === 'linear') {
            updateLayer(activeDraft.layerId, {
                p1: { x: pos.x, y: pos.y },
                p2: { x: pos.x, y: pos.y },
            })
        } else if (activeDraft.kind === 'radial') {
            updateLayer(activeDraft.layerId, {
                center: { x: pos.x, y: pos.y },
                radius: { x: 0.001, y: 0.001 },
            })
        }
    }, [activeDraft, canvasEditor, pointerToImage, updateLayer])

    // Update the layer as the user drags. Linear uses start→current as
    // the line endpoints; radial derives center (midpoint) + radius
    // (half the drag distance per axis) + rotation (atan2 of the drag).
    const handleSpatialDragMove = useCallback((e) => {
        if (!activeDraft || !dragStateRef.current) return
        const fabricCanvas = canvasEditor
        if (!fabricCanvas) return
        const pos = pointerToImage(fabricCanvas, e)
        if (!pos) return
        if (activeDraft.kind === 'linear') {
            updateLayer(activeDraft.layerId, {
                p1: { x: dragStateRef.current.startX, y: dragStateRef.current.startY },
                p2: { x: pos.x, y: pos.y },
            })
        } else if (activeDraft.kind === 'radial') {
            const cx = (dragStateRef.current.startX + pos.x) / 2
            const cy = (dragStateRef.current.startY + pos.y) / 2
            const dx = pos.x - dragStateRef.current.startX
            const dy = pos.y - dragStateRef.current.startY
            const rx = Math.max(0.001, Math.abs(dx) / 2)
            const ry = Math.max(0.001, Math.abs(dy) / 2)
            const rot = Math.atan2(dy, dx)
            updateLayer(activeDraft.layerId, {
                center: { x: cx, y: cy },
                radius: { x: rx, y: ry },
                rotation: rot,
            })
        }
    }, [activeDraft, canvasEditor, pointerToImage, updateLayer])

    // Commit the draft. The layer keeps whatever geometry it has.
    // Only clears `activeDraft` if a drag actually started — clicking
    // outside the canvas (e.g. on a UI button) before clicking the canvas
    // should NOT silently commit a layer with default p1/p2.
    const handleSpatialDragEnd = useCallback(() => {
        if (!activeDraft) return
        if (dragStateRef.current) {
            dragStateRef.current = null
            setActiveDraft(null)
        }
    }, [activeDraft])

    // Cancel the draft (Esc key). The just-added layer is removed so the
    // chain stays clean — the user opted out.
    const handleSpatialCancel = useCallback(() => {
        if (!activeDraft) return
        removeLayer(activeDraft.layerId)
        setActiveDraft(null)
        dragStateRef.current = null
    }, [activeDraft, removeLayer])

    // Wire fabric's mouse:down when a draft is active. We use
    // window-level mousemove/mouseup (added in a separate effect) so
    // drags that escape the canvas still finalise.
    useEffect(() => {
        const fabricCanvas = canvasEditor
        if (!fabricCanvas || !activeDraft) return
        fabricCanvas.defaultCursor = 'crosshair'
        fabricCanvas.selection = false
        const onDown = (e) => handleSpatialDragStart(e)
        fabricCanvas.on('mouse:down', onDown)
        return () => {
            fabricCanvas.defaultCursor = 'default'
            fabricCanvas.selection = true
            fabricCanvas.off('mouse:down', onDown)
        }
    }, [activeDraft, canvasEditor, handleSpatialDragStart])

    // Window-level move + up so off-canvas drags still update + commit.
    useEffect(() => {
        if (!activeDraft) return
        const onMove = (e) => {
            // Build a fabric-event-shaped object so pointerToImage can
            // use the same getPointer() API. We synthesise `e.e` with the
            // raw DOM event so the conversion is uniform.
            const fakeEvent = { e }
            handleSpatialDragMove(fakeEvent)
        }
        const onUp = () => handleSpatialDragEnd()
        window.addEventListener('mousemove', onMove)
        window.addEventListener('mouseup', onUp)
        return () => {
            window.removeEventListener('mousemove', onMove)
            window.removeEventListener('mouseup', onUp)
        }
    }, [activeDraft, handleSpatialDragMove, handleSpatialDragEnd])

    // Esc cancels the active draft. We use a ref for the cancel handler
    // so the keydown listener is only attached/detached when the draft
    // itself appears or disappears. If we listed `handleSpatialCancel`
    // in the deps directly, its identity changes on every stack mutation
    // (because `removeLayer` closes over `stack`), and that fires on
    // every mousemove tick of the drag — the listener would be detached
    // and re-added per pixel of movement.
    const handleSpatialCancelRef = useRef(handleSpatialCancel)
    handleSpatialCancelRef.current = handleSpatialCancel
    useEffect(() => {
        if (!activeDraft) return
        const onKey = (e) => { if (e.key === 'Escape') handleSpatialCancelRef.current() }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
    }, [activeDraft])

    /* ─── Semantic (SAM 2) click-to-select (Step 5) ─── */

    // Active click-mode flag — when true, mouse clicks on the canvas are
    // captured as SAM 2 click points (positive by default, negative with
    // the Alt key held). Disabled while another tool (color picker,
    // spatial draft) is active so handlers don't fight over mouse:down.
    const [semanticActive, setSemanticActive] = useState(false)
    // List of `[x, y, label]` clicks in *original* (natural) image-pixel
    // coordinates. We accumulate here, then send the whole array to
    // /api/ai/sam2 in one request when the user hits "Run".
    const [semanticClicks, setSemanticClicks] = useState(/** @type {Array<[number, number, 0 | 1]>} */ ([]))
    const [isSemanticRunning, setIsSemanticRunning] = useState(false)
    const isSemanticRunningRef = useRef(false)
    const semanticAbortRef = useRef(/** @type {AbortController | null} */ (null))
    // Decoded mask ImageData for the most recent successful run, kept
    // around so the user can re-add it as a megashader layer without
    // re-hitting the API. Cleared on reset.
    const [lastSemanticMask, setLastSemanticMask] = useState(null)
    // Live preview of the decoded mask, rendered as an HTMLCanvasElement
    // so the user sees what they got before committing to a layer.
    const [lastSemanticPreview, setLastSemanticPreview] = useState(/** @type {string | null} */ (null))
    // Refs to the live preview marker dots (one Fabric Circle per click)
    // so we can draw/erase them in lockstep with `semanticClicks`.
    const semanticMarkerRefs = useRef(/** @type {Array<any>} */ ([]))

    // SAM 2 BOX prompt — the strongest single prompt for whole-object
    // selection. When armed, the next drag on the canvas defines the box
    // (in natural image-pixel coords, [x0, y0, x1, y1]); it can be combined
    // with clicks to refine. One box at a time — a new drag replaces it.
    const [semanticBox, setSemanticBox] = useState(/** @type {[number,number,number,number] | null} */ (null))
    const [boxArmed, setBoxArmed] = useState(false)
    const boxArmedRef = useRef(false)
    boxArmedRef.current = boxArmed
    const boxDraftRef = useRef(/** @type {{x:number,y:number} | null} */ (null))
    const boxRectRef = useRef(/** @type {any} */ (null))

    // Masking-service (SAM 3.1) availability — probed once via the same-origin
    // /api/ai/health proxy. Drives the badge + graceful "on-device" messaging;
    // fails soft to 'offline' (the tool still works via server/on-device AI).
    const [maskService, setMaskService] = useState(/** @type {{status:string,sam3?:boolean,model?:string}} */({ status: 'unchecked' }))
    useEffect(() => {
        let alive = true
        ;(async () => {
            const s = await checkMaskService()
            if (alive) setMaskService(s?.available
                ? { status: 'available', sam3: !!s.sam3, model: s.model || '' }
                : { status: 'offline' })
        })()
        return () => { alive = false }
    }, [])

    // AI Background selection state (independent from Select Subject).
    const [isSelectingBg, setIsSelectingBg] = useState(false)
    const isSelectingBgRef = useRef(false)
    const bgAbortRef = useRef(/** @type {AbortController | null} */(null))

    const handleSemanticClick = useCallback((e) => {
        if (!semanticActive || !tool.mainImage) return
        if (boxArmedRef.current) return // a box drag owns the pointer right now
        const fabricCanvas = canvasEditor
        if (!fabricCanvas) return
        const pos = pointerToImage(fabricCanvas, e)
        if (!pos) return
        // The route validates bounds server-side, but rejecting obvious
        // out-of-bounds clicks here saves a roundtrip + 400 response.
        const sourceEl = tool.mainImage._element || tool.mainImage.getElement?.()
        const w = sourceEl?.naturalWidth || 0
        const h = sourceEl?.naturalHeight || 0
        if (w > 0 && h > 0 && (pos.x < 0 || pos.y < 0 || pos.x >= w || pos.y >= h)) {
            toast('Click is outside the image bounds', { icon: '⚠️' })
            return
        }
        // Alt-click = negative (exclude) click. e.e.altKey is the DOM
        // event alt modifier; fall back to false for synthetic events.
        const isNegative = Boolean(e?.e?.altKey)
        const x = Math.round(pos.x * 100) / 100
        const y = Math.round(pos.y * 100) / 100
        setSemanticClicks((prev) => [...prev, [x, y, isNegative ? 0 : 1]])
    }, [semanticActive, tool.mainImage, canvasEditor, pointerToImage])

    // Wire the canvas click handler whenever semantic mode is active.
    // The cleanup is critical — without it, a leaked mouse:down handler
    // would keep capturing clicks after the user leaves the tool.
    useEffect(() => {
        const fabricCanvas = canvasEditor
        if (!fabricCanvas) return
        if (!semanticActive) return
        fabricCanvas.defaultCursor = 'crosshair'
        const onDown = (e) => handleSemanticClick(e)
        fabricCanvas.on('mouse:down', onDown)
        return () => {
            fabricCanvas.defaultCursor = 'default'
            fabricCanvas.off('mouse:down', onDown)
        }
    }, [semanticActive, canvasEditor, handleSemanticClick])

    // Box-drag capture. While armed, mouse:down anchors the box, mouse:move
    // resizes a dashed live rect (display coords), mouse:up commits the box
    // in natural image coords and disarms. The rect marker itself persists
    // (mirroring semanticBox) until reset/stop.
    useEffect(() => {
        const fabricCanvas = canvasEditor
        if (!fabricCanvas || !semanticActive || !boxArmed) return undefined
        fabricCanvas.defaultCursor = 'crosshair'
        const onDown = (e) => {
            const pos = pointerToImage(fabricCanvas, e)
            if (!pos) return
            boxDraftRef.current = { x: pos.x, y: pos.y }
        }
        const onMove = (e) => {
            if (!boxDraftRef.current) return
            const pos = pointerToImage(fabricCanvas, e)
            if (!pos) return
            const x0 = Math.min(boxDraftRef.current.x, pos.x)
            const y0 = Math.min(boxDraftRef.current.y, pos.y)
            const x1 = Math.max(boxDraftRef.current.x, pos.x)
            const y1 = Math.max(boxDraftRef.current.y, pos.y)
            setSemanticBox([x0, y0, x1, y1])
        }
        const onUp = () => {
            if (!boxDraftRef.current) return
            boxDraftRef.current = null
            setBoxArmed(false)
            setSemanticBox((b) => {
                // Discard degenerate boxes (a stray click instead of a drag).
                if (!b || b[2] - b[0] < 4 || b[3] - b[1] < 4) return null
                return b
            })
        }
        fabricCanvas.on('mouse:down', onDown)
        fabricCanvas.on('mouse:move', onMove)
        fabricCanvas.on('mouse:up', onUp)
        return () => {
            fabricCanvas.defaultCursor = 'default'
            fabricCanvas.off('mouse:down', onDown)
            fabricCanvas.off('mouse:move', onMove)
            fabricCanvas.off('mouse:up', onUp)
            boxDraftRef.current = null
        }
    }, [semanticActive, boxArmed, canvasEditor, pointerToImage])

    // Dashed live rect mirroring `semanticBox` (display coords).
    useEffect(() => {
        const fabricCanvas = canvasEditor
        if (!fabricCanvas || !tool.mainImage) return undefined
        if (boxRectRef.current) {
            try { fabricCanvas.remove(boxRectRef.current) } catch { /* gone */ }
            boxRectRef.current = null
        }
        if (!semanticActive || !semanticBox) {
            fabricCanvas.requestRenderAll()
            return undefined
        }
        const tl = imageToDisplay(semanticBox[0], semanticBox[1])
        const br = imageToDisplay(semanticBox[2], semanticBox[3])
        if (!tl || !br) return undefined
        const rect = new FabricRect({
            left: Math.min(tl.x, br.x),
            top: Math.min(tl.y, br.y),
            width: Math.abs(br.x - tl.x),
            height: Math.abs(br.y - tl.y),
            fill: 'rgba(6, 184, 212, 0.08)',
            stroke: 'rgba(6, 184, 212, 0.95)',
            strokeWidth: 1.5,
            strokeDashArray: [5, 4],
            strokeUniform: true,
            selectable: false,
            evented: false,
            excludeFromExport: true,
        })
        fabricCanvas.add(rect)
        boxRectRef.current = rect
        fabricCanvas.requestRenderAll()
        return () => {
            if (boxRectRef.current) {
                try { fabricCanvas.remove(boxRectRef.current) } catch { /* gone */ }
                boxRectRef.current = null
                fabricCanvas.requestRenderAll()
            }
        }
    }, [semanticActive, semanticBox, canvasEditor, tool.mainImage, imageToDisplay])

    // Live markers — one small Fabric circle per click. Cyan = positive
    // (include), red = negative (exclude). Mirrors the click list so
    // removing a click from the list removes its marker too.
    useEffect(() => {
        const fabricCanvas = canvasEditor
        if (!fabricCanvas || !tool.mainImage) return
        for (const m of semanticMarkerRefs.current) {
            try { fabricCanvas.remove(m) } catch { /* canvas gone */ }
        }
        semanticMarkerRefs.current = []
        if (!semanticActive) return
        for (const [x, y, label] of semanticClicks) {
            const d = imageToDisplay(x, y)
            if (!d) continue
            const marker = new FabricCircle({
                left: d.x,
                top: d.y,
                radius: 6,
                fill: label === 1 ? 'rgba(6, 184, 212, 0.85)' : 'rgba(239, 68, 68, 0.85)',
                stroke: '#ffffff',
                strokeWidth: 1.5,
                originX: 'center',
                originY: 'center',
                selectable: false,
                evented: false,
                excludeFromExport: true,
            })
            fabricCanvas.add(marker)
            semanticMarkerRefs.current.push(marker)
        }
        fabricCanvas.requestRenderAll()
        return () => {
            for (const m of semanticMarkerRefs.current) {
                try { fabricCanvas.remove(m) } catch { /* canvas gone */ }
            }
            semanticMarkerRefs.current = []
            fabricCanvas.requestRenderAll()
        }
    }, [semanticActive, semanticClicks, canvasEditor, tool.mainImage, imageToDisplay])

    const handleSemanticReset = useCallback(() => {
        setSemanticClicks([])
        setSemanticBox(null)
        setBoxArmed(false)
        setLastSemanticMask(null)
        setLastSemanticPreview(null)
    }, [])

    const handleSemanticStop = useCallback(() => {
        setSemanticActive(false)
        setSemanticClicks([])
        setSemanticBox(null)
        setBoxArmed(false)
        setLastSemanticMask(null)
        setLastSemanticPreview(null)
    }, [])

    // Decode a Blob (PNG) to an HTMLImageElement + ImageData. The
    // decodeMaskBlob / base64PngToBlob / prepareImageBlob are imported from
    // @/lib/mask-service-client (single implementation, testable).

    const handleSemanticRun = useCallback(async () => {
        if (!tool.mainImage) return
        // Read the in-flight flag from a ref, not from closure-captured
        // state — a fast double-click can otherwise launch two concurrent
        // fetches and race to overwrite the decoded mask.
        if (isSemanticRunningRef.current) return
        if (semanticClicks.length === 0 && !semanticBox) {
            toast('Add a click or draw a box first')
            return
        }
        const sourceEl = tool.mainImage._element || tool.mainImage.getElement?.()
        if (!sourceEl) {
            toast.error('Image not ready')
            return
        }
        // Cancel any earlier in-flight request so it can't resolve and
        // clobber our state if it was slower than us.
        try { semanticAbortRef.current?.abort() } catch { /* ignore */ }
        const abortController = new AbortController()
        semanticAbortRef.current = abortController
        isSemanticRunningRef.current = true
        setIsSemanticRunning(true)
        try {
            // Resize source for the upload (matches the route's MAX_MODEL_SIDE
            // so we don't burn bandwidth shipping 12MP images over the wire).
            const origW = sourceEl.naturalWidth || sourceEl.width || 0
            const origH = sourceEl.naturalHeight || sourceEl.height || 0
            // An <img> mid-load reports 0 — sending a 1px upload would 400 or
            // mis-select. Fail with a clear message instead.
            if (origW < 1 || origH < 1) {
                throw new Error('Image is still loading — try again in a moment')
            }
            const MAX_UPLOAD_SIDE = 1024
            const scale = Math.min(1, MAX_UPLOAD_SIDE / Math.max(origW, origH))
            const uploadW = Math.max(1, Math.round(origW * scale))
            const uploadH = Math.max(1, Math.round(origH * scale))

            const c = document.createElement('canvas')
            c.width = uploadW
            c.height = uploadH
            const ctx = c.getContext('2d')
            if (!ctx) throw new Error('Could not allocate upload canvas')
            ctx.drawImage(sourceEl, 0, 0, uploadW, uploadH)
            let blob
            try {
                blob = await new Promise((resolve, reject) => {
                    c.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/jpeg', 0.85)
                })
            } catch (e) {
                // Cross-origin image without CORS taints the canvas → toBlob
                // throws SecurityError. Say so plainly.
                if (e?.name === 'SecurityError') {
                    throw new Error('This image is from another site without CORS, so it can’t be read for AI selection')
                }
                throw e
            }

            // Bug #10: clicks are captured in TRUE-natural image space, but
            // the route derives "original" dims from the (already downscaled
            // ≤1024) upload and validates/scales clicks against THAT. Sending
            // raw natural coords for any >1024px source either 400s
            // (out-of-bounds) or selects the wrong subject (off by
            // ~natural/1024). Scale the clicks into the uploaded blob's space
            // here so both ends agree on one reference frame.
            const scaledClicks = semanticClicks.map(([x, y, l]) => [
                Math.round(x * scale * 100) / 100,
                Math.round(y * scale * 100) / 100,
                l,
            ])
            const form = new FormData()
            form.append('image', blob, 'image.jpg')
            if (scaledClicks.length) form.append('clicks', JSON.stringify(scaledClicks))
            if (semanticBox) {
                // Same reference-frame fix as clicks: scale the box into the
                // uploaded blob's space, clamped inside it.
                const scaledBox = [
                    Math.max(0, Math.round(semanticBox[0] * scale * 100) / 100),
                    Math.max(0, Math.round(semanticBox[1] * scale * 100) / 100),
                    Math.min(uploadW, Math.round(semanticBox[2] * scale * 100) / 100),
                    Math.min(uploadH, Math.round(semanticBox[3] * scale * 100) / 100),
                ]
                form.append('box', JSON.stringify(scaledBox))
            }

            const resp = await fetch('/api/ai/sam2', { method: 'POST', body: form, signal: abortController.signal })
            if (!resp.ok) {
                const errJson = await resp.json().catch(() => ({}))
                throw new Error(errJson.error || `SAM 2 request failed (${resp.status})`)
            }
            const maskBlob = await resp.blob()
            const decoded = await decodeMaskBlob(maskBlob)

            // If a newer request was kicked off while we were decoding,
            // its results supersede ours — bail without writing state.
            if (semanticAbortRef.current !== abortController) return
            setLastSemanticMask(decoded.imageData)
            setLastSemanticPreview(decoded.dataUrl)
            toast.success(`Subject selected (${decoded.width}×${decoded.height})`)
        } catch (err) {
            // AbortError is the expected outcome of a user-initiated
            // cancel; don't surface it as a failure toast.
            if (err?.name === 'AbortError') return
            console.error('[mask] SAM 2 selection failed:', err)
            toast.error(err?.message || 'AI selection failed')
        } finally {
            // Only clear the running flag if we're still the latest
            // request — a newer one will manage its own lifecycle.
            if (semanticAbortRef.current === abortController) {
                isSemanticRunningRef.current = false
                setIsSemanticRunning(false)
            }
        }
    }, [tool, semanticClicks, semanticBox])

    // Live click/box selection: run SAM the moment a click lands or a box is
    // drawn, so the mask refines per interaction (the "Run" button stays as a
    // manual re-run). A short debounce coalesces rapid clicks into one call with
    // ALL accumulated points; re-firing on isSemanticRunning picks up clicks
    // added mid-run; the signature ref stops an idempotent re-run loop.
    const liveRunSigRef = useRef(/** @type {string | null} */(null))
    useEffect(() => {
        if (!semanticActive) { liveRunSigRef.current = null; return }
        if (semanticClicks.length === 0 && !semanticBox) { liveRunSigRef.current = null; return }
        if (isSemanticRunning) return
        const sig = JSON.stringify([semanticClicks, semanticBox])
        if (sig === liveRunSigRef.current) return
        const t = setTimeout(() => { liveRunSigRef.current = sig; handleSemanticRun() }, 110)
        return () => clearTimeout(t)
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [semanticClicks, semanticBox, semanticActive, isSemanticRunning])

    // Add the most recent decoded mask as a megashader layer. The
    // texture is stored in the module-level mask cache under a freshly
    // minted key, then the layer is pushed through the normal
    // `addLayer` path so all the existing dispatch / state machinery
    // works unchanged.
    const handleAddSemanticLayer = useCallback(() => {
        if (!lastSemanticMask) {
            toast('Run the click selection first')
            return
        }
        const key = `semantic-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`
        setMaskTexture(key, lastSemanticMask)
        const id = addChainLayer('semantic', { maskTextureKey: key, feather: 0.1, label: 'AI Subject' })
        if (id) toast.success('Semantic layer added to chain')
        // Clear the working state so the user knows the layer is now
        // managed by the chain (and so they don't accidentally add
        // the same mask twice).
        setLastSemanticMask(null)
        setLastSemanticPreview(null)
        setSemanticClicks([])
        setSemanticActive(false)
    }, [lastSemanticMask, addChainLayer])

    /* ─── Depth (Depth Anything V2) (Step 6) ─── */

    // Single-shot depth state. Unlike SAM 2 there's no click accumulation
    // — the user hits "Generate" once, gets a whole-image depth map, and
    // can then add it to the chain with custom min/max/softness.
    const [isDepthRunning, setIsDepthRunning] = useState(false)
    const isDepthRunningRef = useRef(false)
    const depthAbortRef = useRef(/** @type {AbortController | null} */ (null))
    const [lastDepthMap, setLastDepthMap] = useState(null)
    const [lastDepthPreview, setLastDepthPreview] = useState(/** @type {string | null} */ (null))
    // Per-add depth range settings. Re-used for the next "Add to Mask
    // Layers" — reset on add so the next generation starts fresh.
    const [depthMin, setDepthMin] = useState(0)
    const [depthMax, setDepthMax] = useState(0.5)
    const [depthSoftness, setDepthSoftness] = useState(0.1)

    // Keep the min slider below the max slider in real time. Without this,
    // a user dragging min past max would see a layer with an empty range
    // (the factory silently swaps them, but the sliders stay crossed,
    // and the visual result won't match what the UI is showing). The 0.01
    // floor on the gap matches the slider's `step` so the result is a
    // legal [min, max] pair for the factory.
    const setDepthMinBounded = useCallback((v) => {
        setDepthMin((cur) => {
            const upper = depthMax
            if (v >= upper) return Math.max(0, upper - 0.01)
            return v
        })
    }, [depthMax])
    const setDepthMaxBounded = useCallback((v) => {
        setDepthMax((cur) => {
            const lower = depthMin
            if (v <= lower) return Math.min(1, lower + 0.01)
            return v
        })
    }, [depthMin])

    const handleDepthRun = useCallback(async () => {
        if (!tool.mainImage) return
        if (isDepthRunningRef.current) return
        const sourceEl = tool.mainImage._element || tool.mainImage.getElement?.()
        if (!sourceEl) {
            toast.error('Image not ready')
            return
        }
        try { depthAbortRef.current?.abort() } catch { /* ignore */ }
        const abortController = new AbortController()
        depthAbortRef.current = abortController
        isDepthRunningRef.current = true
        setIsDepthRunning(true)
        try {
            // Cap to 1024 for upload size — depth model runs at 518×518
            // internally so anything bigger is wasted bandwidth.
            const origW = sourceEl.naturalWidth || sourceEl.width
            const origH = sourceEl.naturalHeight || sourceEl.height
            const MAX_UPLOAD_SIDE = 1024
            const scale = Math.min(1, MAX_UPLOAD_SIDE / Math.max(origW, origH))
            const uploadW = Math.max(1, Math.round(origW * scale))
            const uploadH = Math.max(1, Math.round(origH * scale))

            const c = document.createElement('canvas')
            c.width = uploadW
            c.height = uploadH
            const ctx = c.getContext('2d')
            if (!ctx) throw new Error('Could not allocate upload canvas')
            ctx.drawImage(sourceEl, 0, 0, uploadW, uploadH)
            const blob = await new Promise((resolve, reject) => {
                c.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/jpeg', 0.85)
            })

            const form = new FormData()
            form.append('image', blob, 'image.jpg')

            const resp = await fetch('/api/ai/depth', { method: 'POST', body: form, signal: abortController.signal })
            if (!resp.ok) {
                const errJson = await resp.json().catch(() => ({}))
                throw new Error(errJson.error || `Depth request failed (${resp.status})`)
            }
            const depthBlob = await resp.blob()
            const decoded = await decodeMaskBlob(depthBlob)

            if (depthAbortRef.current !== abortController) return
            setLastDepthMap(decoded.imageData)
            setLastDepthPreview(decoded.dataUrl)
            const elapsed = resp.headers.get('x-elapsed-ms') || '?'
            toast.success(`Depth map generated (${decoded.width}×${decoded.height}, ${elapsed}ms)`)
        } catch (err) {
            if (err?.name === 'AbortError') return
            console.error('[mask] Depth generation failed:', err)
            toast.error(err?.message || 'Depth generation failed')
        } finally {
            if (depthAbortRef.current === abortController) {
                isDepthRunningRef.current = false
                setIsDepthRunning(false)
            }
        }
    }, [tool])

    // Add the most recent depth map as a megashader layer with the user's
    // current min/max/softness. The texture lives in the same module-level
    // cache as semantic masks, so the renderer can pick it up by key.
    const handleAddDepthLayer = useCallback(() => {
        if (!lastDepthMap) {
            toast('Generate a depth map first')
            return
        }
        const key = `depth-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`
        setMaskTexture(key, lastDepthMap)
        const id = addChainLayer('depth', {
            depthMapKey: key,
            min: depthMin,
            max: depthMax,
            softness: depthSoftness,
            label: 'Depth range',
        })
        if (id) toast.success('Depth layer added to chain')
        // Clear the working state so the user knows the layer is now
        // managed by the chain.
        setLastDepthMap(null)
        setLastDepthPreview(null)
    }, [lastDepthMap, addChainLayer, depthMin, depthMax, depthSoftness])

    const handleDepthReset = useCallback(() => {
        setLastDepthMap(null)
        setLastDepthPreview(null)
    }, [])

    /* ─── Smart Brush (Step 7) ─── */
    //
    // The user paints an alpha stroke on an offscreen HTMLCanvasElement
    // that lives in the source image's natural pixel space. The canvas
    // is uploaded to the megashader as a sampler2D, where the GLSL body
    // runs a bilateral filter so the stroke "snaps" to colour edges in
    // the source image rather than bleeding across them.
    //
    // Lifecycle:
    //   1. User clicks "Start Painting" — we allocate the brush canvas
    //      (matching the image's natural size), add a FabricImage live
    //      preview, and lock the canvas cursor to a crosshair.
    //   2. User drags on the canvas — pointer events stamp soft radial
    //      alpha circles onto the brush canvas. The live overlay's
    //      element is the same brush canvas, so updates are instant.
    //   3. User clicks "Add to Mask Layers" — we put the brush canvas
    //      into the mask texture cache, push a smartBrushLayer onto
    //      the chain with the current filter parameters, then reset
    //      the brush canvas and exit paint mode.
    //
    // The brush size / hardness / filter settings are local state
    // (not chain state) — they only become "real" when the user adds
    // the layer. After adding, the user can still tweak filter params
    // via the chain card's `KindParamEditor` (already supports per-layer
    // params via the smartBrushLayer factory's clamps).

    /* ─── Lasso selection (freehand + polygonal) ─── */
    // The lasso captures a closed polygon and turns it into a megashader
    // 'lasso' layer (Bug #12 kind). `lassoSink` picks the layer's output:
    // 'select' → fill mode (visible selection), 'erase' → erase mode (cut).
    // `lassoModifier` maps Shift/Alt + the modifier buttons to the chain
    // blend op so a second lasso can add/subtract/intersect with the first.
    const [lassoActive, setLassoActive] = useState(false)
    const [lassoMode, setLassoMode] = useState('freehand') // 'freehand' | 'polygonal' | 'magnetic'
    const [lassoSink, setLassoSink] = useState('select')   // 'select' | 'erase'
    const [lassoModifier, setLassoModifier] = useState('new') // new|add|subtract|intersect
    const [lassoFeather, setLassoFeather] = useState(0.04)
    // Pen mode: smooth the captured anchors into a Bézier 'path' layer (vs the
    // straight-edged 'lasso' layer). Reuses the whole lasso capture pipeline.
    const [lassoSmooth, setLassoSmooth] = useState(false)
    const [lassoVertexCount, setLassoVertexCount] = useState(0)
    // ─── Magnetic lasso (edge-snapping) options — Photoshop parity ───
    // Width = search radius (image px), Contrast = edge threshold (0..100),
    // Frequency = anchor spacing (image px). Refs mirror them so the
    // window-level pointer handlers read fresh values without re-binding.
    const [magneticWidth, setMagneticWidth] = useState(16)
    const [magneticContrast, setMagneticContrast] = useState(12)
    const [magneticFrequency, setMagneticFrequency] = useState(14)
    const magneticWidthRef = useRef(magneticWidth)
    const magneticContrastRef = useRef(magneticContrast)
    const magneticFrequencyRef = useRef(magneticFrequency)
    useEffect(() => { magneticWidthRef.current = magneticWidth }, [magneticWidth])
    useEffect(() => { magneticContrastRef.current = magneticContrast }, [magneticContrast])
    useEffect(() => { magneticFrequencyRef.current = magneticFrequency }, [magneticFrequency])
    // Cached Sobel gradient-magnitude map of the source image (0..1), used to
    // snap the magnetic lasso to edges. Rebuilt when the source/size changes.
    const gradientMapRef = useRef(/** @type {{mag:Float32Array,W:number,H:number,scale:number,token:string}|null} */ (null))
    const lassoPointsRef = useRef(/** @type {Array<{x:number,y:number}>} */ ([]))
    const lassoDrawingRef = useRef(false)
    const lassoCursorRef = useRef(/** @type {{x:number,y:number}|null} */ (null))
    const lassoOverlayRef = useRef(/** @type {any} */ (null))
    const lassoMarkerRefs = useRef(/** @type {Array<any>} */ ([]))
    const lassoModeRef = useRef(lassoMode)
    const lassoModifierRef = useRef(lassoModifier)
    useEffect(() => { lassoModeRef.current = lassoMode }, [lassoMode])
    useEffect(() => { lassoModifierRef.current = lassoModifier }, [lassoModifier])

    const [brushActive, setBrushActive] = useState(false)
    const [brushSize, setBrushSize] = useState(40)
    const [brushHardness, setBrushHardness] = useState(0.8)
    const [filterRadius, setFilterRadius] = useState(3)
    const [sigmaColor, setSigmaColor] = useState(0.15)
    const [sigmaSpace, setSigmaSpace] = useState(2)
    const [brushHasContent, setBrushHasContent] = useState(false)
    const [isShapeFilling, setIsShapeFilling] = useState(false)
    // Selection-brush output options (mirror the lasso). `brushSink` picks the
    // layer's fillMode (select → visible 'fill', erase → 'erase' knockout);
    // `brushModifier` maps to the chain blend op; `brushEdgeSnap` toggles the
    // edge-preserving bilateral filter (smartBrush kind) vs a plain brush kind.
    // `brushFeather` is baked PER-LAYER at add time, so each region keeps its
    // own soft edge regardless of the slider's later position.
    const [brushSink, setBrushSink] = useState('select')      // 'select' | 'erase'
    const [brushModifier, setBrushModifier] = useState('new') // new|add|subtract|intersect
    const [brushEdgeSnap, setBrushEdgeSnap] = useState(false)
    const [brushFeather, setBrushFeather] = useState(0)       // image px, baked per region

    // Refs (so pointer handlers and effects read consistent values
    // without re-binding effects on every slider tweak).
    const brushCanvasRef = useRef(/** @type {HTMLCanvasElement | null} */ (null))
    const brushOverlayRef = useRef(/** @type {any} */ (null))
    const isPaintingRef = useRef(false)
    const lastBrushPointRef = useRef(null)
    const brushPointsRef = useRef(/** @type {Array<{x:number,y:number}>} */ ([]))
    const shapeFillTokenRef = useRef(0)
    const brushSizeRef = useRef(brushSize)
    const brushHardnessRef = useRef(brushHardness)
    const filterRadiusRef = useRef(filterRadius)
    const sigmaColorRef = useRef(sigmaColor)
    const sigmaSpaceRef = useRef(sigmaSpace)

    // Step 10.1: cap the brush canvas to 2048×2048 max. A 4K image
    // would otherwise allocate a 33 MB RGBA brush canvas (and an 8K
    // image, 132 MB). The cap preserves the aspect ratio of the
    // source image — `brushScale` is the factor (1.0 for ≤2048 long
    // edge, <1.0 for larger). Pointer events and stamp radii are
    // pre-multiplied by `brushScale` when handed to the brush canvas
    // 2D context, so the brush state is in brush-canvas space while
    // the UI's slider value remains in image-pixel space.
    //
    // The GLSL doesn't need to know about `brushScale` — it samples
    // the brush texture using normalized UV (`vTextureCoord`), so a
    // 2048×2048 brush canvas is correctly mapped onto a 4K render
    // output via the GPU's default linear filtering. The visible
    // difference is a slightly softer brush at high zoom, which is
    // the natural trade-off for ~16× less memory.
    const BRUSH_CANVAS_MAX_DIM = 2048
    const brushScaleRef = useRef(1)

    useEffect(() => { brushSizeRef.current = brushSize }, [brushSize])
    useEffect(() => { brushHardnessRef.current = brushHardness }, [brushHardness])
    useEffect(() => { filterRadiusRef.current = filterRadius }, [filterRadius])
    useEffect(() => { sigmaColorRef.current = sigmaColor }, [sigmaColor])
    useEffect(() => { sigmaSpaceRef.current = sigmaSpace }, [sigmaSpace])

    // Push the live `brushActive` flag into `pixelToolDisabled` so the
    // pixel tool's pointer handlers short-circuit. Without this, a
    // single click in smart-brush mode would fire BOTH brushes — the
    // pixel tool would write a clipPath dab AND the smart brush would
    // write a texture dab, leaving the user with two unrelated masks
    // for one click. Declaring this effect after the `brushActive`
    // `useState` is intentional: it satisfies the temporal-dead-zone
    // rule (the variable is in scope on this line of source) and the
    // effect itself just runs once on mount with the current value.
    // Bug #9 — central pointer arbitration. The pixel brush binds its own
    // mouse:down and paints a clipPath dab on every in-image click; it is
    // suppressed ONLY by this flag. Any other click-capturing mode (smart
    // brush, colour eyedropper, SAM2 click, spatial gradient draft, lasso)
    // MUST drive it true, or a single click would fire BOTH that mode's
    // handler AND the brush — placing a layer while painting a stray dab.
    useEffect(() => {
        // The destructive pixel brush only takes the pointer when the user has
        // explicitly turned on Quick Erase AND no other capture mode owns the
        // canvas. Anything else (incl. the default "nothing engaged" state)
        // suppresses it, so a stray click never erases image pixels.
        setPixelToolDisabled(!quickEraseActive || brushActive || colorPickerActive || semanticActive || lassoActive || !!activeDraft || isGradientSelected)
    }, [quickEraseActive, brushActive, colorPickerActive, semanticActive, lassoActive, activeDraft, isGradientSelected])

    // Allocate a fresh brush canvas, capped to BRUSH_CANVAS_MAX_DIM on
    // the long edge. Called both on entering brush mode (if the image
    // is ready) and on adding a layer (to clear the slate for the
    // next stroke).
    //
    // The cap preserves aspect ratio: for a 3840×2160 (4K) image, the
    // long edge is 3840; scale = 2048/3840 ≈ 0.5333; the brush canvas
    // becomes 2048×1152. For an 8K (7680×4320) image, scale ≈ 0.2667;
    // the brush canvas becomes 2048×1152 — a 16× memory reduction
    // (33 MB → 2 MB). For images already at or below 2048 on the
    // long edge, scale = 1 and behaviour is unchanged.
    const ensureBrushCanvas = useCallback(() => {
        if (!imageSize) return null
        const longEdge = Math.max(imageSize.width, imageSize.height)
        const scale = longEdge > BRUSH_CANVAS_MAX_DIM
            ? BRUSH_CANVAS_MAX_DIM / longEdge
            : 1
        const targetW = Math.max(1, Math.round(imageSize.width * scale))
        const targetH = Math.max(1, Math.round(imageSize.height * scale))
        if (brushCanvasRef.current
            && brushCanvasRef.current.width === targetW
            && brushCanvasRef.current.height === targetH) {
            brushScaleRef.current = scale
            return brushCanvasRef.current
        }
        const c = document.createElement('canvas')
        c.width = targetW
        c.height = targetH
        brushCanvasRef.current = c
        brushScaleRef.current = scale
        return c
    }, [imageSize])

    // Paint a single soft alpha stamp at (x, y) with the given radius
    // and hardness (0..1). Hardness controls the size of the opaque
    // inner disc relative to the full stamp radius — a soft brush
    // (hardness=0) is a pure radial gradient; a hard brush
    // (hardness=1) is a full opaque disc.
    const stampBrush = useCallback((ctx, x, y, radius, hardness) => {
        if (!ctx || radius <= 0) return
        const h = Math.max(0, Math.min(1, hardness))
        if (h >= 0.999) {
            // Hard brush: single fill (faster, no gradient).
            ctx.fillStyle = 'rgba(255, 255, 255, 1)'
            ctx.beginPath()
            ctx.arc(x, y, radius, 0, Math.PI * 2)
            ctx.fill()
            return
        }
        const grad = ctx.createRadialGradient(x, y, 0, x, y, radius)
        grad.addColorStop(0, 'rgba(255, 255, 255, 1)')
        grad.addColorStop(h, 'rgba(255, 255, 255, 1)')
        grad.addColorStop(1, 'rgba(255, 255, 255, 0)')
        ctx.fillStyle = grad
        ctx.beginPath()
        ctx.arc(x, y, radius, 0, Math.PI * 2)
        ctx.fill()
    }, [])

    // Interpolate stamps along a line from (x1, y1) → (x2, y2) so fast
    // drags don't show gaps. The step is `radius / 3` — three stamps
    // per diameter gives smooth coverage without overdraw.
    const strokeBrush = useCallback((ctx, x1, y1, x2, y2, radius, hardness) => {
        if (!ctx) return
        const dx = x2 - x1
        const dy = y2 - y1
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < 0.5) {
            // Sub-pixel drag — single stamp.
            stampBrush(ctx, x2, y2, radius, hardness)
            return
        }
        const step = Math.max(1, radius / 3)
        const n = Math.max(1, Math.ceil(dist / step))
        for (let i = 1; i <= n; i += 1) {
            const t = i / n
            stampBrush(ctx, x1 + dx * t, y1 + dy * t, radius, hardness)
        }
    }, [stampBrush])

    // Mark the brush canvas as "has content" (drives the Add button
    // enable state). Called from a `useEffect` that observes the
    // overlay's `dirty` flag via a polling tick — too lazy but cheap.
    // A `requestAnimationFrame` after each stamp is more idiomatic; see
    // the pointer move handler below.
    const markBrushChanged = useCallback(() => {
        setBrushHasContent(true)
    }, [])

    /* ─── Brush pointer handlers (active only while `brushActive`) ─── */

    const handleBrushDown = useCallback((e) => {
        if (!brushActive || !imageSize) return
        const fabricCanvas = canvasEditor
        if (!fabricCanvas) return
        const pos = pointerToImage(fabricCanvas, e)
        if (!pos) return
        // Reject clicks outside the image bounds (the brush canvas is
        // sized to the image, so out-of-bounds clicks would silently
        // no-op the stamp, which is confusing).
        if (pos.x < 0 || pos.y < 0 || pos.x >= imageSize.width || pos.y >= imageSize.height) return
        const canvas = ensureBrushCanvas()
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        if (!ctx) return
        isPaintingRef.current = true
        lastBrushPointRef.current = { x: pos.x, y: pos.y }
        brushPointsRef.current = [{ x: pos.x, y: pos.y }]
        // Step 10.1: scale the stamp position and radius into
        // brush-canvas space. `brushScale` is 1.0 for images already
        // at or below 2048 on the long edge, <1.0 for higher-res
        // sources. The UI's `brushSize` slider stays in image-pixel
        // units, so this is the only place the scaling happens.
        const s = brushScaleRef.current
        const r = Math.max(0.5, brushSizeRef.current / 2) * s
        stampBrush(ctx, pos.x * s, pos.y * s, r, brushHardnessRef.current)
        markBrushChanged()
        if (brushOverlayRef.current) {
            brushOverlayRef.current.set('dirty', true)
            fabricCanvas.requestRenderAll()
        }
    }, [brushActive, imageSize, canvasEditor, pointerToImage, ensureBrushCanvas, stampBrush, markBrushChanged])

    const handleBrushMove = useCallback((e) => {
        if (!isPaintingRef.current || !brushActive) return
        const fabricCanvas = canvasEditor
        if (!fabricCanvas) return
        const pos = pointerToImage(fabricCanvas, e)
        if (!pos) return
        // Clamp to image bounds so a drag past the edge doesn't paint
        // outside the brush canvas (which would be clipped to transparent
        // by `destination-out` in the stamp).
        const w = imageSize?.width || 0
        const h = imageSize?.height || 0
        const cx = w > 0 ? Math.max(0, Math.min(w, pos.x)) : pos.x
        const cy = h > 0 ? Math.max(0, Math.min(h, pos.y)) : pos.y
        const canvas = ensureBrushCanvas()
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        if (!ctx) return
        const last = lastBrushPointRef.current || { x: cx, y: cy }
        // Step 10.1: scale into brush-canvas space. `last` is stored
        // in image-pixel space (so the deltas across moves are
        // consistent), but `strokeBrush` is called with scaled
        // coordinates so the 2D context stamps the right pixels.
        const s = brushScaleRef.current
        const r = Math.max(0.5, brushSizeRef.current / 2) * s
        strokeBrush(ctx, last.x * s, last.y * s, cx * s, cy * s, r, brushHardnessRef.current)
        lastBrushPointRef.current = { x: cx, y: cy }
        const pts = brushPointsRef.current
        const prev = pts[pts.length - 1]
        if (!prev || (cx - prev.x) * (cx - prev.x) + (cy - prev.y) * (cy - prev.y) >= 4) {
            pts.push({ x: cx, y: cy })
        }
        markBrushChanged()
        if (brushOverlayRef.current) {
            brushOverlayRef.current.set('dirty', true)
            // Throttle the render to one per frame — without this, fast
            // drags fire dozens of mousemoves per frame and the GL
            // context sputters.
            fabricCanvas.requestRenderAll()
        }
    }, [brushActive, imageSize, canvasEditor, pointerToImage, ensureBrushCanvas, strokeBrush, markBrushChanged])

    const drawShapeMaskBlob = useCallback(async (blob, token) => {
        const objectUrl = URL.createObjectURL(blob)
        try {
            const img = await new Promise((resolve, reject) => {
                const image = new Image()
                image.onload = () => resolve(image)
                image.onerror = () => reject(new Error('Failed to decode shape mask PNG'))
                image.src = objectUrl
            })
            if (shapeFillTokenRef.current !== token) return false
            const brushCanvas = brushCanvasRef.current
            if (!brushCanvas) return false
            const ctx = brushCanvas.getContext('2d')
            if (!ctx) return false
            ctx.save()
            ctx.globalCompositeOperation = 'source-over'
            ctx.drawImage(img, 0, 0, brushCanvas.width, brushCanvas.height)
            ctx.restore()
            markBrushChanged()
            if (brushOverlayRef.current) {
                brushOverlayRef.current.set('dirty', true)
                canvasEditor?.requestRenderAll?.()
            }
            return true
        } finally {
            URL.revokeObjectURL(objectUrl)
        }
    }, [canvasEditor, markBrushChanged])

    const requestPythonClosedShapeFill = useCallback(async (points, token) => {
        const brushCanvas = brushCanvasRef.current
        if (!brushCanvas || !Array.isArray(points) || points.length < 3) return false
        const scale = brushScaleRef.current || 1
        const scaledPoints = points.map((p) => [
            Math.round(p.x * scale * 100) / 100,
            Math.round(p.y * scale * 100) / 100,
        ])

        setIsShapeFilling(true)
        try {
            const response = await fetch('/api/ai/shape-mask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    width: brushCanvas.width,
                    height: brushCanvas.height,
                    points: scaledPoints,
                }),
                signal: AbortSignal.timeout(8_000),
            })
            if (!response.ok) {
                const err = await response.json().catch(() => ({}))
                throw new Error(err.error || `shape mask failed (${response.status})`)
            }
            return drawShapeMaskBlob(await response.blob(), token)
        } catch (error) {
            console.warn('[mask] Python closed-shape fill unavailable; using canvas fallback:', error?.message)
            return false
        } finally {
            if (shapeFillTokenRef.current === token) setIsShapeFilling(false)
        }
    }, [drawShapeMaskBlob])

    const fillClosedBrushStroke = useCallback(() => {
        const brushCanvas = brushCanvasRef.current
        if (!brushCanvas) return false
        const ctx = brushCanvas.getContext('2d')
        if (!ctx) return false
        const points = brushPointsRef.current.slice()
        const filled = fillClosedBrushPath(
            ctx,
            points,
            brushScaleRef.current,
            brushSizeRef.current,
        )
        if (!filled) return false
        const token = shapeFillTokenRef.current + 1
        shapeFillTokenRef.current = token
        void requestPythonClosedShapeFill(points, token)
        markBrushChanged()
        if (brushOverlayRef.current) {
            brushOverlayRef.current.set('dirty', true)
            canvasEditor?.requestRenderAll?.()
        }
        return true
    }, [canvasEditor, markBrushChanged, requestPythonClosedShapeFill])

    const handleBrushUp = useCallback((e) => {
        if (!isPaintingRef.current) return
        if (e && canvasEditor) {
            const pos = pointerToImage(canvasEditor, e)
            const w = imageSize?.width || 0
            const h = imageSize?.height || 0
            if (pos && w > 0 && h > 0) {
                const cx = Math.max(0, Math.min(w, pos.x))
                const cy = Math.max(0, Math.min(h, pos.y))
                const pts = brushPointsRef.current
                const prev = pts[pts.length - 1]
                if (!prev || (cx - prev.x) * (cx - prev.x) + (cy - prev.y) * (cy - prev.y) >= 1) {
                    pts.push({ x: cx, y: cy })
                }
            }
        }
        fillClosedBrushStroke()
        isPaintingRef.current = false
        lastBrushPointRef.current = null
        brushPointsRef.current = []
    }, [canvasEditor, imageSize, pointerToImage, fillClosedBrushStroke])

    // Wire the pointer events when brush mode is active. Window-level
    // move + up so a drag that escapes the canvas still finalises.
    useEffect(() => {
        const fabricCanvas = canvasEditor
        if (!fabricCanvas || !brushActive) return
        fabricCanvas.defaultCursor = 'crosshair'
        fabricCanvas.hoverCursor = 'crosshair'
        fabricCanvas.selection = false
        const onDown = (e) => handleBrushDown(e)
        const onMove = (e) => {
            const fake = { e }
            handleBrushMove(fake)
        }
        const onUp = (ev) => handleBrushUp({ e: ev })
        fabricCanvas.on('mouse:down', onDown)
        window.addEventListener('mousemove', onMove)
        window.addEventListener('mouseup', onUp)
        return () => {
            fabricCanvas.defaultCursor = 'default'
            fabricCanvas.hoverCursor = 'move'
            fabricCanvas.selection = true
            fabricCanvas.off('mouse:down', onDown)
            window.removeEventListener('mousemove', onMove)
            window.removeEventListener('mouseup', onUp)
        }
    }, [brushActive, canvasEditor, handleBrushDown, handleBrushMove, handleBrushUp])

    // Add / remove the live-preview FabricImage when brush mode toggles.
    // The overlay shares the brush canvas as its `_element` — any draw
    // to the canvas is visible on the next render with no extra sync.
    useEffect(() => {
        const fabricCanvas = canvasEditor
        if (!fabricCanvas || !tool.mainImage) return
        if (!brushActive) {
            if (brushOverlayRef.current) {
                try { fabricCanvas.remove(brushOverlayRef.current) } catch { /* canvas gone */ }
                brushOverlayRef.current = null
                fabricCanvas.requestRenderAll()
            }
            return
        }
        const brushCanvas = ensureBrushCanvas()
        if (!brushCanvas) return
        // The brush canvas is in image-pixel space; mirror the main
        // image's transform so the overlay sits exactly on top of the
        // corresponding image pixels.
        const img = tool.mainImage
        const overlay = new FabricImage(brushCanvas, {
            left: img.left,
            top: img.top,
            scaleX: img.scaleX,
            scaleY: img.scaleY,
            angle: img.angle,
            originX: img.originX,
            originY: img.originY,
            flipX: img.flipX,
            flipY: img.flipY,
            selectable: false,
            evented: false,
            hasControls: false,
            hasBorders: false,
            objectCaching: false,
            excludeFromExport: true,
            opacity: 0.6,
        })
        fabricCanvas.add(overlay)
        // Move the overlay above the image but below the markers.
        // We just add it at the end — Fabric's z-order is the add order,
        // so the overlay sits on top of the image. The user's selection
        // handles still take priority (Fabric draws those last).
        brushOverlayRef.current = overlay
        fabricCanvas.requestRenderAll()
        return () => {
            if (brushOverlayRef.current) {
                try { fabricCanvas.remove(brushOverlayRef.current) } catch { /* canvas gone */ }
                brushOverlayRef.current = null
            }
        }
    }, [brushActive, canvasEditor, tool.mainImage, ensureBrushCanvas])

    // Re-position the overlay when the image's transform changes (pan,
    // zoom, rotate, flip). The overlay's `_element` doesn't change, so
    // we just patch the geometry fields. This effect is independent of
    // the add/remove effect above so pan-during-paint works.
    //
    // Why we depend on the transform fields explicitly (not just the
    // image reference): Fabric mutates the same FabricImage object on
    // zoom/rotate — the *reference* doesn't change, but `scaleX`/
    // `scaleY`/`angle`/`flipX`/`flipY`/`left`/`top` do. Subscribing to
    // the reference alone would miss those edits. We also depend on
    // the canvas viewport transform (indices 4 and 5) for pure pan
    // (where the image itself doesn't move, only the viewport does).
    const img = tool.mainImage
    const imgLeft = img?.left ?? 0
    const imgTop = img?.top ?? 0
    const imgScaleX = img?.scaleX ?? 1
    const imgScaleY = img?.scaleY ?? 1
    const imgAngle = img?.angle ?? 0
    const imgFlipX = !!img?.flipX
    const imgFlipY = !!img?.flipY
    const fabricViewport = canvasEditor?.viewportTransform
    const panX = fabricViewport?.[4] ?? 0
    const panY = fabricViewport?.[5] ?? 0
    useEffect(() => {
        const overlay = brushOverlayRef.current
        const liveImg = tool.mainImage
        const fabricCanvas = canvasEditor
        if (!overlay || !liveImg || !fabricCanvas) return
        overlay.set({
            left: liveImg.left,
            top: liveImg.top,
            scaleX: liveImg.scaleX,
            scaleY: liveImg.scaleY,
            angle: liveImg.angle,
            originX: liveImg.originX,
            originY: liveImg.originY,
            flipX: liveImg.flipX,
            flipY: liveImg.flipY,
        })
        overlay.setCoords?.()
        fabricCanvas.requestRenderAll()
    }, [
        tool.mainImage,
        brushActive,
        canvasEditor,
        // Image transform fields — cover zoom, rotate, flip, move
        imgLeft, imgTop, imgScaleX, imgScaleY, imgAngle, imgFlipX, imgFlipY,
        // Viewport translate — covers pure pan (image not moving)
        panX, panY,
    ])

    const handleStartBrush = useCallback(() => {
        if (!imageSize) {
            toast.error('Image not ready yet')
            return
        }
        // Cancel any other click-mode (incl. Quick Erase) so its handler
        // doesn't fire and the destructive brush can't resume afterwards.
        setColorPickerActive(false)
        setSemanticActive(false)
        handleSemanticStop()
        setQuickEraseActive(false)
        if (activeDraft) {
            toast('Finish or cancel the current draft first', { icon: 'ℹ️' })
            return
        }
        ensureBrushCanvas()
        // Reset stale "hasContent" state from a prior session — without
        // this, the Add button would be enabled before the user painted
        // anything (the flag was left true from the previous stroke).
        setBrushHasContent(false)
        isPaintingRef.current = false
        lastBrushPointRef.current = null
        brushPointsRef.current = []
        shapeFillTokenRef.current += 1
        setIsShapeFilling(false)
        setBrushActive(true)
        toast(brushSink === 'erase'
            ? 'Paint to mark the cut region, then "Add cut to layers". Nothing is erased until you add it.'
            : 'Paint a selection, then "Add selection to layers". Non-destructive.')
    }, [imageSize, ensureBrushCanvas, activeDraft, handleSemanticStop, brushSink])

    const handleStopBrush = useCallback(() => {
        setBrushActive(false)
        isPaintingRef.current = false
        lastBrushPointRef.current = null
        brushPointsRef.current = []
        shapeFillTokenRef.current += 1
        setIsShapeFilling(false)
    }, [])

    const handleClearBrush = useCallback(() => {
        const c = brushCanvasRef.current
        if (!c) return
        const ctx = c.getContext('2d')
        if (!ctx) return
        ctx.clearRect(0, 0, c.width, c.height)
        setBrushHasContent(false)
        brushPointsRef.current = []
        shapeFillTokenRef.current += 1
        setIsShapeFilling(false)
        if (brushOverlayRef.current) {
            brushOverlayRef.current.set('dirty', true)
            canvasEditor?.requestRenderAll?.()
        }
    }, [canvasEditor])

    // Commit the painted brush canvas as a NON-DESTRUCTIVE megashader
    // selection layer. The painted alpha is stored in the mask texture cache
    // and referenced by the new layer (plain `brush` kind, or edge-preserving
    // `smartBrush` when "Snap to edges" is on). `brushSink` picks fill vs
    // erase; `brushModifier` maps to the chain blend op; `brushFeather` is
    // baked into the texture so each region keeps its own soft edge.
    const handleAddBrushLayer = useCallback(() => {
        if (isShapeFilling) {
            toast('Finishing the closed shape fill...')
            return
        }
        const brushCanvas = brushCanvasRef.current
        if (!brushCanvas) {
            toast('Start painting first')
            return
        }
        if (stack.chain.length >= 8) {
            toast.error('Mask layer limit reached (8). Remove a layer to add more.')
            return
        }
        const ctx = brushCanvas.getContext('2d', { willReadFrequently: true })
        if (!ctx) return
        // Empty check: a single full-canvas read, scanning only the alpha
        // channel (one getImageData, not one per pixel).
        let anyPainted = false
        try {
            const data = ctx.getImageData(0, 0, brushCanvas.width, brushCanvas.height).data
            for (let i = 3; i < data.length; i += 4) { if (data[i] > 0) { anyPainted = true; break } }
        } catch { anyPainted = true /* tainted → assume painted */ }
        if (!anyPainted) {
            toast('Paint something first')
            return
        }

        // Bake the feather PER REGION (plain brush only — the smart brush's
        // bilateral filter is its own edge control). Blurring the painted
        // alpha into a fresh canvas means moving the slider later never
        // re-feathers already-committed regions.
        let textureCanvas = brushCanvas
        const featherPx = Math.max(0, Math.round(brushFeather * brushScaleRef.current))
        if (!brushEdgeSnap && featherPx > 0 && typeof document !== 'undefined') {
            try {
                const fc = document.createElement('canvas')
                fc.width = brushCanvas.width
                fc.height = brushCanvas.height
                const fctx = fc.getContext('2d')
                if (fctx) {
                    fctx.filter = `blur(${featherPx}px)`
                    fctx.drawImage(brushCanvas, 0, 0)
                    fctx.filter = 'none'
                    textureCanvas = fc
                }
            } catch { textureCanvas = brushCanvas }
        }

        const key = `brush-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`
        setMaskTexture(key, textureCanvas)
        const fillMode = brushSink === 'erase' ? 'erase' : 'fill'
        const id = brushEdgeSnap
            ? addChainLayer('smartBrush', {
                brushTextureKey: key,
                filterRadius,
                sigmaColor,
                sigmaSpace,
                fillMode,
                label: brushSink === 'erase' ? 'Brush cut (smart)' : 'Smart brush',
            })
            : addChainLayer('brush', {
                maskTextureKey: key,
                fillMode,
                label: brushSink === 'erase' ? 'Brush cut' : 'Brush selection',
            })
        if (id) {
            if (brushModifier === 'add' || brushModifier === 'subtract' || brushModifier === 'intersect') {
                setLayerOp(id, brushModifier)
            }
            toast.success(brushSink === 'erase' ? 'Brush cut added to layers' : 'Brush selection added to layers')
            // Reset the brush canvas for the next stroke. We allocate
            // a fresh canvas rather than clearing in place — the old
            // canvas is still referenced by the just-added layer, and
            // clearing it would mutate a cached texture.
            brushCanvasRef.current = null
            setBrushHasContent(false)
            setBrushActive(false)
            isPaintingRef.current = false
            lastBrushPointRef.current = null
            brushPointsRef.current = []
            shapeFillTokenRef.current += 1
            setIsShapeFilling(false)
        }
    }, [addChainLayer, setLayerOp, stack.chain.length, brushSink, brushModifier, brushEdgeSnap, brushFeather, filterRadius, sigmaColor, sigmaSpace, isShapeFilling])

    /* ─── Lasso handlers ─── */

    // Remove the live lasso overlay (dashed polyline + vertex markers).
    const clearLassoOverlay = useCallback(() => {
        const fabricCanvas = canvasEditor
        if (!fabricCanvas) return
        if (lassoOverlayRef.current) {
            try { fabricCanvas.remove(lassoOverlayRef.current) } catch { /* canvas gone */ }
            lassoOverlayRef.current = null
        }
        for (const m of lassoMarkerRefs.current) {
            try { fabricCanvas.remove(m) } catch { /* canvas gone */ }
        }
        lassoMarkerRefs.current = []
    }, [canvasEditor])

    // Rebuild the dashed-polyline preview from the current points (plus a
    // rubber-band segment to the cursor in polygonal mode), and the vertex
    // markers. Recreated each redraw so Fabric positions the polyline at the
    // absolute point coords (mutating .points drifts the offset).
    const redrawLassoOverlay = useCallback(() => {
        const fabricCanvas = canvasEditor
        if (!fabricCanvas) return
        clearLassoOverlay()
        const pts = lassoPointsRef.current
        const disp = (pts || []).map((p) => imageToDisplay(p.x, p.y)).filter(Boolean)
        const linePts = disp.slice()
        if ((lassoModeRef.current === 'polygonal' || lassoModeRef.current === 'magnetic') && lassoCursorRef.current && disp.length > 0) {
            const c = imageToDisplay(lassoCursorRef.current.x, lassoCursorRef.current.y)
            if (c) linePts.push(c)
        }
        if (linePts.length >= 2) {
            const poly = new Polyline(linePts, {
                stroke: '#06b8d4',
                strokeWidth: 1.5,
                strokeDashArray: [4, 4],
                fill: 'rgba(6,184,212,0.10)',
                selectable: false,
                evented: false,
                excludeFromExport: true,
                objectCaching: false,
            })
            fabricCanvas.add(poly)
            lassoOverlayRef.current = poly
        }
        if (lassoModeRef.current === 'polygonal') {
            for (const d of disp) {
                const marker = new FabricCircle({
                    left: d.x, top: d.y, radius: 4,
                    fill: '#06b8d4', stroke: '#ffffff', strokeWidth: 1,
                    originX: 'center', originY: 'center',
                    selectable: false, evented: false, excludeFromExport: true,
                })
                fabricCanvas.add(marker)
                lassoMarkerRefs.current.push(marker)
            }
        }
        fabricCanvas.requestRenderAll()
    }, [canvasEditor, imageToDisplay, clearLassoOverlay])

    // Clear the in-progress path (points + overlay + cursor) but keep the
    // tool active so the user can draw another selection.
    const resetLassoPath = useCallback(() => {
        lassoPointsRef.current = []
        lassoCursorRef.current = null
        lassoDrawingRef.current = false
        setLassoVertexCount(0)
        clearLassoOverlay()
        canvasEditor?.requestRenderAll?.()
    }, [clearLassoOverlay, canvasEditor])

    // Rasterise the closed polygon to an offscreen alpha canvas (white
    // inside on opaque black, so Canvas2D anti-aliasing lands in the R
    // channel the lasso GLSL samples). Capped to BRUSH_CANVAS_MAX_DIM like
    // the smart brush; the GLSL samples by normalised UV so the cap is free.
    const rasterizeLasso = useCallback((points) => {
        if (!imageSize || !points || points.length < 3) return null
        const longEdge = Math.max(imageSize.width, imageSize.height)
        const scale = longEdge > BRUSH_CANVAS_MAX_DIM ? BRUSH_CANVAS_MAX_DIM / longEdge : 1
        const W = Math.max(1, Math.round(imageSize.width * scale))
        const H = Math.max(1, Math.round(imageSize.height * scale))
        const c = document.createElement('canvas')
        c.width = W
        c.height = H
        const ctx = c.getContext('2d')
        if (!ctx) return null
        ctx.fillStyle = '#000000'
        ctx.fillRect(0, 0, W, H)
        ctx.fillStyle = '#ffffff'
        ctx.beginPath()
        ctx.moveTo(points[0].x * scale, points[0].y * scale)
        for (let i = 1; i < points.length; i += 1) ctx.lineTo(points[i].x * scale, points[i].y * scale)
        ctx.closePath()
        ctx.fill('evenodd')
        return c
    }, [imageSize])

    // Pen mode — smooth the captured anchors into a Bézier path and rasterise
    // it to the same capped-scale alpha canvas the lasso uses. Reuses the exact
    // capture pipeline; only the rasterisation (curves) and layer kind differ.
    const rasterizePenPath = useCallback((points) => {
        if (!imageSize || !points || points.length < 3) return null
        const longEdge = Math.max(imageSize.width, imageSize.height)
        const scale = longEdge > BRUSH_CANVAS_MAX_DIM ? BRUSH_CANVAS_MAX_DIM / longEdge : 1
        const W = Math.max(1, Math.round(imageSize.width * scale))
        const H = Math.max(1, Math.round(imageSize.height * scale))
        const anchors = smoothToBezier(points, { closed: true })
        const sc = (p) => (p ? { x: p.x * scale, y: p.y * scale } : undefined)
        const scaled = anchors.map((a) => ({ x: a.x * scale, y: a.y * scale, cOut: sc(a.cOut), cIn: sc(a.cIn) }))
        return rasterisePath(scaled, W, H, { closed: true })
    }, [imageSize])

    // ─── Magnetic lasso edge engine ───
    // Build (and cache) a Sobel gradient-magnitude map of the source image,
    // normalised to 0..1, at the SAME capped scale rasterizeLasso uses so the
    // snapped points line up exactly with the rasterised selection texture.
    const ensureGradientMap = useCallback(() => {
        if (!imageSize || !tool.mainImage) return null
        const sourceEl = tool.mainImage._element || tool.mainImage.getElement?.()
        if (!sourceEl) return null
        const longEdge = Math.max(imageSize.width, imageSize.height)
        const scale = longEdge > BRUSH_CANVAS_MAX_DIM ? BRUSH_CANVAS_MAX_DIM / longEdge : 1
        const W = Math.max(1, Math.round(imageSize.width * scale))
        const H = Math.max(1, Math.round(imageSize.height * scale))
        const token = `${sourceEl.currentSrc || sourceEl.src || ''}|${W}x${H}`
        const cached = gradientMapRef.current
        if (cached && cached.token === token) return cached
        let data
        try {
            const c = document.createElement('canvas')
            c.width = W
            c.height = H
            const cx = c.getContext('2d', { willReadFrequently: true })
            if (!cx) return null
            cx.drawImage(sourceEl, 0, 0, W, H)
            data = cx.getImageData(0, 0, W, H).data
        } catch {
            return null // tainted canvas — magnetic snapping unavailable
        }
        const { mag } = computeGradientMagnitude(data, W, H)
        const result = { mag, W, H, scale, token }
        gradientMapRef.current = result
        return result
    }, [imageSize, tool.mainImage])

    // Snap an image-space point to the strongest edge within `magneticWidth`,
    // biased toward the cursor (proximity-dominant). Returns the input unchanged
    // when no edge clears the contrast threshold or the map is unavailable. The
    // search math is the shared, unit-tested engine in `@/lib/mask-edge-snap`.
    const snapToEdge = useCallback((imgX, imgY) => {
        const gm = ensureGradientMap()
        if (!gm) return { x: imgX, y: imgY }
        const radius = Math.max(1, magneticWidthRef.current * gm.scale)
        const threshold = Math.max(0, Math.min(1, magneticContrastRef.current / 100))
        const snapped = snapToEdgePoint(
            { mag: gm.mag, w: gm.W, h: gm.H },
            imgX * gm.scale,
            imgY * gm.scale,
            radius,
            threshold,
        )
        return { x: snapped.x / gm.scale, y: snapped.y / gm.scale }
    }, [ensureGradientMap])

    // Resolve the chain blend op from Shift/Alt on the closing event, or
    // (when no modifier is held) the modifier-button state.
    const resolveLassoModifier = useCallback((rawEvent) => {
        const shift = rawEvent?.shiftKey || rawEvent?.e?.shiftKey
        const alt = rawEvent?.altKey || rawEvent?.e?.altKey
        if (shift && alt) return 'intersect'
        if (shift) return 'add'
        if (alt) return 'subtract'
        return lassoModifierRef.current
    }, [])

    // Commit the current polygon → a megashader 'lasso' layer. `select`
    // sink uses fill mode (visible selection); `erase` uses erase mode
    // (cuts the region). The modifier maps to the chain blend op so a
    // second lasso composes with the first.
    const finishLassoSelection = useCallback((rawEvent) => {
        const pts = lassoPointsRef.current
        if (!pts || pts.length < 3) { resetLassoPath(); return }
        const usePen = lassoSmooth
        const canvas = usePen ? rasterizePenPath(pts) : rasterizeLasso(pts)
        if (!canvas) { resetLassoPath(); return }
        const prefix = usePen ? 'path' : 'lasso'
        const key = `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`
        setMaskTexture(key, canvas)
        const id = addChainLayer(usePen ? 'path' : 'lasso', {
            maskTextureKey: key,
            feather: lassoFeather,
            fillMode: lassoSink === 'erase' ? 'erase' : 'fill',
            label: usePen
                ? (lassoSink === 'erase' ? 'Pen cut' : 'Pen path')
                : (lassoSink === 'erase' ? 'Lasso cut' : 'Lasso selection'),
        })
        if (id) {
            const mod = resolveLassoModifier(rawEvent)
            // 'new' leaves the default op (replace for the first layer, add
            // otherwise). add/subtract/intersect only apply to non-first
            // layers (setLayerOp is a no-op on slot 0).
            if (mod === 'add' || mod === 'subtract' || mod === 'intersect') {
                setLayerOp(id, mod)
            }
            toast.success(lassoSink === 'erase' ? 'Lasso cut added to layers' : 'Lasso selection added to layers')
        }
        resetLassoPath()
    }, [rasterizeLasso, rasterizePenPath, lassoSmooth, addChainLayer, setLayerOp, lassoFeather, lassoSink, resolveLassoModifier, resetLassoPath])

    const removeLastLassoVertex = useCallback(() => {
        if (lassoPointsRef.current.length === 0) return
        lassoPointsRef.current = lassoPointsRef.current.slice(0, -1)
        setLassoVertexCount(lassoPointsRef.current.length)
        redrawLassoOverlay()
    }, [redrawLassoOverlay])

    // Distance (image px) under which a polygonal click snaps to the first
    // vertex to close the loop — ~8 display px regardless of zoom.
    const lassoCloseThreshold = useCallback(() => {
        const sx = tool.mainImage?.scaleX || 1
        return 8 / (sx || 1)
    }, [tool.mainImage])

    const handleLassoDown = useCallback((e) => {
        if (!lassoActive || !imageSize) return
        const pos = pointerToImage(canvasEditor, e)
        if (!pos) return
        if (lassoModeRef.current === 'freehand') {
            lassoDrawingRef.current = true
            lassoPointsRef.current = [{ x: pos.x, y: pos.y }]
            setLassoVertexCount(1)
            redrawLassoOverlay()
            return
        }
        if (lassoModeRef.current === 'magnetic') {
            // First click starts the path; subsequent moves auto-lay snapped
            // points (no button held, Photoshop-style); clicks drop a manual
            // anchor; clicking near the first point (≥3 points) closes.
            const pts = lassoPointsRef.current
            if (pts.length >= 3) {
                const first = pts[0]
                const dx = pos.x - first.x
                const dy = pos.y - first.y
                if (Math.sqrt(dx * dx + dy * dy) <= lassoCloseThreshold()) {
                    finishLassoSelection(e)
                    return
                }
            }
            const snapped = snapToEdge(pos.x, pos.y)
            if (!lassoDrawingRef.current) {
                ensureGradientMap()
                lassoDrawingRef.current = true
                lassoPointsRef.current = [snapped]
            } else {
                lassoPointsRef.current = [...pts, snapped]
            }
            lassoCursorRef.current = snapped
            setLassoVertexCount(lassoPointsRef.current.length)
            redrawLassoOverlay()
            return
        }
        // polygonal: clicking near the first vertex (with ≥3 points) closes.
        const pts = lassoPointsRef.current
        if (pts.length >= 3) {
            const first = pts[0]
            const dx = pos.x - first.x
            const dy = pos.y - first.y
            if (Math.sqrt(dx * dx + dy * dy) <= lassoCloseThreshold()) {
                finishLassoSelection(e)
                return
            }
        }
        lassoPointsRef.current = [...pts, { x: pos.x, y: pos.y }]
        setLassoVertexCount(lassoPointsRef.current.length)
        redrawLassoOverlay()
    }, [lassoActive, imageSize, pointerToImage, canvasEditor, redrawLassoOverlay, lassoCloseThreshold, finishLassoSelection, snapToEdge, ensureGradientMap])

    const handleLassoMove = useCallback((e) => {
        if (!lassoActive) return
        const pos = pointerToImage(canvasEditor, e)
        if (!pos) return
        if (lassoModeRef.current === 'magnetic') {
            // Lay snapped points as the cursor moves (no button needed once
            // started). Append only every `frequency` px so the path stays
            // light; always update the rubber-band to the live snapped cursor.
            if (!lassoDrawingRef.current) return
            const snapped = snapToEdge(pos.x, pos.y)
            lassoCursorRef.current = snapped
            const pts = lassoPointsRef.current
            const last = pts[pts.length - 1]
            if (last) {
                const dx = snapped.x - last.x
                const dy = snapped.y - last.y
                const spacing = Math.max(3, magneticFrequencyRef.current)
                if (dx * dx + dy * dy >= spacing * spacing) {
                    lassoPointsRef.current = [...pts, snapped]
                    setLassoVertexCount(lassoPointsRef.current.length)
                }
            }
            redrawLassoOverlay()
            return
        }
        if (lassoModeRef.current === 'freehand') {
            if (!lassoDrawingRef.current) return
            const pts = lassoPointsRef.current
            const last = pts[pts.length - 1]
            if (last) {
                const dx = pos.x - last.x
                const dy = pos.y - last.y
                if (dx * dx + dy * dy < 4) return // min 2px image-space spacing
            }
            lassoPointsRef.current = [...pts, { x: pos.x, y: pos.y }]
            // Avoid a setState per sampled point on long strokes — only the
            // overlay needs updating live; the count is cosmetic.
            redrawLassoOverlay()
            return
        }
        // polygonal: rubber-band the segment to the cursor.
        lassoCursorRef.current = { x: pos.x, y: pos.y }
        if (lassoPointsRef.current.length > 0) redrawLassoOverlay()
    }, [lassoActive, pointerToImage, canvasEditor, redrawLassoOverlay, snapToEdge])

    const handleLassoUp = useCallback((rawEvent) => {
        if (!lassoActive) return
        if (lassoModeRef.current !== 'freehand') return
        if (!lassoDrawingRef.current) return
        lassoDrawingRef.current = false
        finishLassoSelection(rawEvent)
    }, [lassoActive, finishLassoSelection])

    // Wire fabric mouse:down + dblclick (canvas) and window move/up + keys
    // while the lasso is active.
    useEffect(() => {
        const fabricCanvas = canvasEditor
        if (!fabricCanvas || !lassoActive) return undefined
        fabricCanvas.defaultCursor = 'crosshair'
        fabricCanvas.hoverCursor = 'crosshair'
        fabricCanvas.selection = false
        const onDown = (opt) => handleLassoDown(opt)
        const onDbl = () => finishLassoSelection(null) // polygonal close
        const onMove = (ev) => handleLassoMove({ e: ev })
        const onUp = (ev) => handleLassoUp(ev)
        const onKey = (ev) => {
            if (ev.key === 'Escape') { resetLassoPath() }
            else if (ev.key === 'Enter') { finishLassoSelection(ev) }
            else if (ev.key === 'Backspace') { ev.preventDefault(); removeLastLassoVertex() }
        }
        fabricCanvas.on('mouse:down', onDown)
        fabricCanvas.on('mouse:dblclick', onDbl)
        window.addEventListener('mousemove', onMove)
        window.addEventListener('mouseup', onUp)
        window.addEventListener('keydown', onKey)
        return () => {
            fabricCanvas.defaultCursor = 'default'
            fabricCanvas.hoverCursor = 'move'
            fabricCanvas.selection = true
            fabricCanvas.off('mouse:down', onDown)
            fabricCanvas.off('mouse:dblclick', onDbl)
            window.removeEventListener('mousemove', onMove)
            window.removeEventListener('mouseup', onUp)
            window.removeEventListener('keydown', onKey)
        }
    }, [lassoActive, canvasEditor, handleLassoDown, handleLassoMove, handleLassoUp, finishLassoSelection, resetLassoPath, removeLastLassoVertex])

    // Re-draw the lasso overlay when the image transform changes (pan/zoom)
    // so the in-progress selection stays glued to the photo.
    useEffect(() => {
        if (!lassoActive) return
        if (lassoPointsRef.current.length > 0) redrawLassoOverlay()
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [imgLeft, imgTop, imgScaleX, imgScaleY, imgAngle, imgFlipX, imgFlipY, panX, panY])

    const handleStartLasso = useCallback(() => {
        if (!imageSize) { toast.error('Image not ready yet'); return }
        if (activeDraft) { toast('Finish or cancel the current draft first', { icon: 'ℹ️' }); return }
        // Cancel any other click-mode so handlers don't fight over the click.
        setColorPickerActive(false)
        setSemanticActive(false)
        handleSemanticStop()
        setBrushActive(false)
        setQuickEraseActive(false)
        resetLassoPath()
        if (lassoMode === 'magnetic') {
            // Build the edge map up front; warn if it can't be made (tainted /
            // cross-origin image) so the user knows snapping is off rather than
            // silently getting an un-snapped click-path.
            const gm = ensureGradientMap()
            if (!gm) toast('Magnetic snapping unavailable for this image — points won’t snap. Try Freehand or Polygonal.', { icon: '⚠️' })
        }
        setLassoActive(true)
        toast(lassoMode === 'freehand'
            ? 'Drag to draw a freehand selection'
            : lassoMode === 'magnetic'
                ? 'Click to start, move along an edge — the path snaps to it. Click to anchor, double-click or Enter to close.'
                : 'Click to add points, double-click or Enter to close')
    }, [imageSize, activeDraft, handleSemanticStop, resetLassoPath, lassoMode, ensureGradientMap])

    const handleStopLasso = useCallback(() => {
        setLassoActive(false)
        resetLassoPath()
    }, [resetLassoPath])

    // Quick Erase (destructive) — explicit opt-in. Cancels any selection mode
    // so the pixel-clipPath brush and the megashader selections never fight
    // over the same click.
    const handleStartQuickErase = useCallback(() => {
        setBrushActive(false)
        setColorPickerActive(false)
        setSemanticActive(false)
        handleSemanticStop()
        setLassoActive(false)
        resetLassoPath()
        setQuickEraseActive(true)
    }, [handleSemanticStop, resetLassoPath])
    const handleStopQuickErase = useCallback(() => setQuickEraseActive(false), [])

    // Surface the centralized layer-cap rejection (useMaskLayers.addLayer
    // refuses past MAX_LAYERS and dispatches this) so every "Add ..." path
    // gives feedback instead of silently failing.
    useEffect(() => {
        const onLimit = (e) => {
            const max = e?.detail?.max || 8
            toast.error(`Mask layer limit reached (${max}). Remove a layer to add more.`)
        }
        try { window.addEventListener('phosmith:mask-layer-limit', onLimit) } catch { /* SSR */ }
        return () => { try { window.removeEventListener('phosmith:mask-layer-limit', onLimit) } catch { /* SSR */ } }
    }, [])

    // Remove any stray lasso overlay objects when the canvas changes or the
    // tool unmounts (the wiring effect only restores the cursor + listeners).
    useEffect(() => () => { clearLassoOverlay() }, [clearLassoOverlay])

    /* ─── Re-editable gradient handles (linear / radial) ─── */

    // Point-based inverse of imageToDisplay — converts a scene/display point
    // back to image-pixel space for handle drags.
    const displayToImage = useCallback((x, y) => {
        const img = tool.mainImage
        if (!img) return null
        return {
            x: (x - (img.left || 0)) / (img.scaleX || 1),
            y: (y - (img.top || 0)) / (img.scaleY || 1),
        }
    }, [tool.mainImage])

    // Latest selected layer, read inside drag handlers via a ref so the
    // handle-build effect does NOT depend on geometry (which would tear down
    // and rebuild the handles mid-drag). `selectedLayer`/`selKind`/
    // `isGradientSelected` are computed once near the top of the component.
    const selectedLayerRef = useRef(selectedLayer)
    selectedLayerRef.current = selectedLayer
    const gradientHandlesRef = useRef(/** @type {Array<any>} */ ([]))
    const draggingHandleRef = useRef(false)
    const [handleTick, setHandleTick] = useState(0)

    const clearGradientHandles = useCallback(() => {
        const c = canvasEditor
        for (const o of gradientHandlesRef.current) {
            try { c?.remove(o) } catch { /* canvas gone */ }
        }
        gradientHandlesRef.current = []
    }, [canvasEditor])

    const captureActive = brushActive || colorPickerActive || semanticActive || lassoActive || !!activeDraft

    useEffect(() => {
        const fabricCanvas = canvasEditor
        clearGradientHandles()
        if (!fabricCanvas || !tool.mainImage) return undefined
        if (!isGradientSelected || captureActive) return undefined
        const layer = selectedLayerRef.current
        if (!layer) return undefined

        const sx = tool.mainImage.scaleX || 1
        const sy = tool.mainImage.scaleY || 1

        const makeHandle = (imgPt, onDrag) => {
            const d = imageToDisplay(imgPt.x, imgPt.y)
            if (!d) return null
            const h = new FabricCircle({
                left: d.x, top: d.y, radius: 7,
                fill: 'rgba(6,184,212,0.95)', stroke: '#ffffff', strokeWidth: 2,
                originX: 'center', originY: 'center',
                hasControls: false, hasBorders: false, selectable: true, evented: true,
                hoverCursor: 'grab', moveCursor: 'grabbing',
                excludeFromExport: true, objectCaching: false,
            })
            h.on('mousedown', () => { draggingHandleRef.current = true })
            h.on('moving', () => {
                const p = displayToImage(h.left, h.top)
                if (p) onDrag(p)
            })
            fabricCanvas.add(h)
            gradientHandlesRef.current.push(h)
            return h
        }

        if (layer.kind === 'linear' && layer.p1 && layer.p2) {
            const p1d = imageToDisplay(layer.p1.x, layer.p1.y)
            const p2d = imageToDisplay(layer.p2.x, layer.p2.y)
            let line = null
            let h1 = null
            let h2 = null
            const reflowLine = () => {
                if (line && h1 && h2) {
                    line.set({ x1: h1.left, y1: h1.top, x2: h2.left, y2: h2.top })
                    line.setCoords?.()
                }
            }
            if (p1d && p2d) {
                line = new Line([p1d.x, p1d.y, p2d.x, p2d.y], {
                    stroke: '#06b8d4', strokeWidth: 1.5, strokeDashArray: [5, 5],
                    selectable: false, evented: false, excludeFromExport: true, objectCaching: false,
                })
                fabricCanvas.add(line)
                gradientHandlesRef.current.push(line)
            }
            h1 = makeHandle(layer.p1, (p) => { updateLayer(layer.id, { p1: { x: p.x, y: p.y } }); reflowLine() })
            h2 = makeHandle(layer.p2, (p) => { updateLayer(layer.id, { p2: { x: p.x, y: p.y } }); reflowLine() })
        } else if (layer.kind === 'radial' && layer.center && layer.radius) {
            let ell = null
            let cH = null
            let rxH = null
            let ryH = null
            // Reposition the outline + all handles from the latest layer.
            // For the handle being dragged this is a round-trip identity
            // (its position defines the geometry), so it never fights the drag.
            const reflow = () => {
                const l = selectedLayerRef.current
                if (!l || !l.center || !l.radius) return
                const rot = l.rotation || 0
                const co = Math.cos(rot)
                const si = Math.sin(rot)
                const cc = imageToDisplay(l.center.x, l.center.y)
                if (ell && cc) { ell.set({ left: cc.x, top: cc.y, rx: Math.max(1, l.radius.x * sx), ry: Math.max(1, l.radius.y * sy), angle: rot * 180 / Math.PI }); ell.setCoords?.() }
                if (cH && cc) { cH.set({ left: cc.x, top: cc.y }); cH.setCoords?.() }
                const rp = imageToDisplay(l.center.x + co * l.radius.x, l.center.y + si * l.radius.x)
                if (rxH && rp) { rxH.set({ left: rp.x, top: rp.y }); rxH.setCoords?.() }
                const yp = imageToDisplay(l.center.x - si * l.radius.y, l.center.y + co * l.radius.y)
                if (ryH && yp) { ryH.set({ left: yp.x, top: yp.y }); ryH.setCoords?.() }
            }
            const center = layer.center
            const radius = layer.radius
            const rotation = layer.rotation || 0
            const cd = imageToDisplay(center.x, center.y)
            if (cd) {
                ell = new Ellipse({
                    left: cd.x, top: cd.y,
                    rx: Math.max(1, radius.x * sx), ry: Math.max(1, radius.y * sy),
                    angle: rotation * 180 / Math.PI,
                    fill: 'rgba(6,184,212,0.08)', stroke: '#06b8d4', strokeWidth: 1.5, strokeDashArray: [5, 5],
                    originX: 'center', originY: 'center',
                    selectable: false, evented: false, excludeFromExport: true, objectCaching: false,
                })
                fabricCanvas.add(ell)
                gradientHandlesRef.current.push(ell)
            }
            const co0 = Math.cos(rotation)
            const si0 = Math.sin(rotation)
            cH = makeHandle(center, (p) => {
                updateLayer(layer.id, { center: { x: p.x, y: p.y } })
                reflow()
            })
            rxH = makeHandle({ x: center.x + co0 * radius.x, y: center.y + si0 * radius.x }, (p) => {
                const l = selectedLayerRef.current
                const dx = p.x - l.center.x
                const dy = p.y - l.center.y
                updateLayer(layer.id, { radius: { x: Math.max(2, Math.hypot(dx, dy)), y: l.radius.y }, rotation: Math.atan2(dy, dx) })
                reflow()
            })
            ryH = makeHandle({ x: center.x - si0 * radius.y, y: center.y + co0 * radius.y }, (p) => {
                const l = selectedLayerRef.current
                const rot = l.rotation || 0
                const dx = p.x - l.center.x
                const dy = p.y - l.center.y
                const newRy = Math.max(2, Math.abs(-Math.sin(rot) * dx + Math.cos(rot) * dy))
                updateLayer(layer.id, { radius: { x: l.radius.x, y: newRy } })
                reflow()
            })
        }

        // On drag end, snap the gizmo to the final geometry with a clean
        // rebuild (and re-enable normal interaction).
        const onUp = () => {
            if (draggingHandleRef.current) {
                draggingHandleRef.current = false
                setHandleTick((t) => t + 1)
            }
        }
        fabricCanvas.on('mouse:up', onUp)
        fabricCanvas.requestRenderAll()
        return () => {
            fabricCanvas.off('mouse:up', onUp)
            clearGradientHandles()
        }
        // Geometry is intentionally NOT a dependency (drag handlers reposition
        // imperatively); the image transform + handleTick drive rebuilds.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedLayerId, selKind, isGradientSelected, captureActive, canvasEditor, tool.mainImage,
        imgLeft, imgTop, imgScaleX, imgScaleY, imgAngle, handleTick])

    // Live preview overlay: a thin dashed line/ellipse on the canvas
    // that tracks the layer's current geometry. Re-rendered on every
    // stack change so the user sees the preview update mid-drag.
    useEffect(() => {
        const fabricCanvas = canvasEditor
        if (!fabricCanvas) return
        if (overlayRef.current) {
            fabricCanvas.remove(overlayRef.current)
            overlayRef.current = null
        }
        if (!activeDraft) return
        const layer = stack.chain.find((e) => e.layer.id === activeDraft.layerId)?.layer
        if (!layer) return

        let overlay = null
        if (activeDraft.kind === 'linear' && layer.p1 && layer.p2) {
            const p1 = imageToDisplay(layer.p1.x, layer.p1.y)
            const p2 = imageToDisplay(layer.p2.x, layer.p2.y)
            if (p1 && p2) {
                overlay = new Line([p1.x, p1.y, p2.x, p2.y], {
                    stroke: '#06b8d4',
                    strokeWidth: 2,
                    strokeDashArray: [5, 5],
                    selectable: false,
                    evented: false,
                    excludeFromExport: true,
                })
            }
        } else if (activeDraft.kind === 'radial' && layer.center && layer.radius) {
            const c = imageToDisplay(layer.center.x, layer.center.y)
            if (c) {
                const scaleX = tool.mainImage?.scaleX || 1
                const scaleY = tool.mainImage?.scaleY || 1
                overlay = new Ellipse({
                    left: c.x,
                    top: c.y,
                    rx: layer.radius.x * scaleX,
                    ry: layer.radius.y * scaleY,
                    fill: 'rgba(6, 184, 212, 0.12)',
                    stroke: '#06b8d4',
                    strokeWidth: 2,
                    strokeDashArray: [5, 5],
                    originX: 'center',
                    originY: 'center',
                    angle: ((layer.rotation || 0) * 180) / Math.PI,
                    selectable: false,
                    evented: false,
                    excludeFromExport: true,
                })
            }
        }

        if (overlay) {
            fabricCanvas.add(overlay)
            overlayRef.current = overlay
            fabricCanvas.requestRenderAll()
        }
        return () => {
            if (overlayRef.current) {
                try { fabricCanvas.remove(overlayRef.current) } catch { /* canvas gone */ }
                overlayRef.current = null
                fabricCanvas.requestRenderAll()
            }
        }
    }, [activeDraft, stack, canvasEditor, imageToDisplay, tool.mainImage])

    /* ─── AI Subject Selection ─── */

    // Turn a decoded AI subject mask (white = subject, black = background) into
    // a NON-DESTRUCTIVE selection: it's stored in the megashader texture cache
    // and added as a `semantic` chain layer (fillMode 'fill' → the subject is
    // tinted/visible while the background is left fully intact). This is the
    // same path SAM 2 / Lasso / Brush use, so the selection shows up in the
    // MASK LAYERS panel and stays editable (boundary, blend op, feather, and
    // add/subtract brush or lasso layers) "even after subject detection".
    //
    // Picking a different subject swaps the single tracked layer rather than
    // piling up duplicates — matching the radio-style chip picker.
    const applySubjectSelection = useCallback((imageData, label) => {
        if (!imageData) return null
        const key = `subject-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`
        setMaskTexture(key, imageData)
        // If our tracked selection layer still exists, swap its texture in
        // place — this keeps the layer's slot / blend op and sidesteps the
        // MAX_LAYERS cap, while resetting the boundary (growPx / baseTextureKey)
        // so the "Boundary" slider tracks the NEW subject's edge from 0.
        const existingId = subjectLayerIdRef.current
        const present = existingId && (chainStackRef.current?.chain || []).some((e) => e.layer.id === existingId)
        if (present) {
            updateLayer(existingId, { maskTextureKey: key, baseTextureKey: key, growPx: 0, label: label || 'Subject' })
            return existingId
        }
        const id = addChainLayer('semantic', { maskTextureKey: key, feather: 0.1, label: label || 'Subject' })
        subjectLayerIdRef.current = id
        return id
    }, [addChainLayer, updateLayer])

    const handleSelectSubject = useCallback(async () => {
        if (!tool.mainImage) return
        if (isSegmentingRef.current) return
        try { segmentAbortRef.current?.abort() } catch { /* ignore */ }
        const abortController = new AbortController()
        segmentAbortRef.current = abortController
        isSegmentingRef.current = true
        setIsSegmenting(true)

        try {
            // Get the source image as a blob
            const fabricObj = tool.mainImage
            const sourceEl = fabricObj?._element || fabricObj?.getElement?.()
            if (!sourceEl) throw new Error('Cannot access image element')

            // Draw source to canvas and export as blob
            const c = document.createElement('canvas')
            const origW = sourceEl.naturalWidth || sourceEl.width || fabricObj.width
            const origH = sourceEl.naturalHeight || sourceEl.height || fabricObj.height
            // Cap the longest side to 2048 before upload. BiRefNet runs at a
            // fixed 1024² internally so this doesn't change matte detail, but it
            // gives the service's YOLO pass enough resolution to find small /
            // distant subjects in group photos (and a crisper mask upscale).
            // JPEG q=0.92 keeps a 2048-side frame well under the 24 MB cap.
            const scale = Math.min(1, 2048 / Math.max(origW, origH))
            c.width = Math.round(origW * scale)
            c.height = Math.round(origH * scale)
            const ctx = c.getContext('2d')
            ctx.drawImage(sourceEl, 0, 0, c.width, c.height)

            const blob = await new Promise((resolve, reject) => {
                c.toBlob(b => b ? resolve(b) : reject(new Error('toBlob failed')), 'image/jpeg', 0.92)
            })

            const form = new FormData()
            form.append('image', blob, 'image.jpg')

            const resp = await fetch('/api/ai/segment', { method: 'POST', body: form, signal: abortController.signal })
            if (!resp.ok) {
                const errJson = await resp.json().catch(() => ({}))
                throw new Error(errJson.error || `Segmentation failed (${resp.status})`)
            }

            const maskBlob = await resp.blob()
            if (segmentAbortRef.current !== abortController) return
            const decoded = await decodeMaskBlob(maskBlob)
            // A newer request may have superseded us while decoding.
            if (segmentAbortRef.current !== abortController) return
            const id = applySubjectSelection(decoded.imageData, 'Subject')
            if (id) {
                setActiveInstanceIndex(null)
                toast.success('Subject selected — adjust it in Mask Layers, or add a brush/lasso to refine.')
            } else {
                toast.error('Could not create subject selection')
            }
        } catch (err) {
            if (err?.name === 'AbortError') return
            console.error('[mask] AI subject selection failed:', err)
            toast.error(err?.message || 'AI subject selection failed')
        } finally {
            if (segmentAbortRef.current === abortController) {
                isSegmentingRef.current = false
                setIsSegmenting(false)
            }
        }
    }, [tool, applySubjectSelection])

    // AI Background — detect the subject, then INVERT it. Foreground/background
    // separation (not text grounding) is the only reliable "whole sky/background"
    // select. Adds its own layer, kept separate from the Subject layer.
    const handleSelectBackground = useCallback(async () => {
        if (!tool.mainImage) return
        if (isSelectingBgRef.current) return
        try { bgAbortRef.current?.abort() } catch { /* ignore */ }
        const abortController = new AbortController()
        bgAbortRef.current = abortController
        isSelectingBgRef.current = true
        setIsSelectingBg(true)
        try {
            const fabricObj = tool.mainImage
            const sourceEl = fabricObj?._element || fabricObj?.getElement?.()
            if (!sourceEl) throw new Error('Cannot access image element')
            const { blob } = await prepareImageBlob(sourceEl, { maxSide: 2048, quality: 0.92 })
            const form = new FormData()
            form.append('image', blob, 'image.jpg')
            const resp = await fetch('/api/ai/segment', { method: 'POST', body: form, signal: abortController.signal })
            if (!resp.ok) {
                const errJson = await resp.json().catch(() => ({}))
                throw new Error(errJson.error || `Segmentation failed (${resp.status})`)
            }
            const maskBlob = await resp.blob()
            if (bgAbortRef.current !== abortController) return
            const decoded = await decodeMaskBlob(maskBlob)
            if (bgAbortRef.current !== abortController) return
            // invert subject coverage → everything that ISN'T the subject
            const d = decoded.imageData.data
            for (let i = 0; i < d.length; i += 4) {
                const v = 255 - d[i]
                d[i] = v; d[i + 1] = v; d[i + 2] = v; d[i + 3] = 255
            }
            const key = `background-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`
            setMaskTexture(key, decoded.imageData)
            const id = addChainLayer('semantic', { maskTextureKey: key, feather: 0.1, label: 'AI Background' })
            if (id) toast.success('Background selected — everything except the subject.')
            else toast.error('Could not create background selection')
        } catch (err) {
            if (err?.name === 'AbortError') return
            console.error('[mask] AI background selection failed:', err)
            toast.error(err?.message || 'AI background selection failed')
        } finally {
            if (bgAbortRef.current === abortController) {
                isSelectingBgRef.current = false
                setIsSelectingBg(false)
            }
        }
    }, [tool, addChainLayer])

    /* ─── Multi-Subject Detection (Detect All Subjects) ──────────────────────
     * Calls /api/ai/segment-instances to enumerate every subject in the image
     * and lets the user mask either the union or a specific instance with one
     * click. Results are cached on the Fabric image (one detection pass per
     * image) so re-clicking individual subjects is free.
     */
    const handleDetectAllSubjects = useCallback(async () => {
        if (!tool.mainImage) return
        if (isDetectingInstancesRef.current) return
        try { instancesAbortRef.current?.abort() } catch { /* ignore */ }
        const abortController = new AbortController()
        instancesAbortRef.current = abortController
        isDetectingInstancesRef.current = true
        setIsDetectingInstances(true)

        try {
            const fabricObj = tool.mainImage
            const sourceEl = fabricObj?._element || fabricObj?.getElement?.()
            if (!sourceEl) throw new Error('Cannot access image element')

            const origW = sourceEl.naturalWidth || sourceEl.width || fabricObj.width
            const origH = sourceEl.naturalHeight || sourceEl.height || fabricObj.height
            const scale = Math.min(1, 2048 / Math.max(origW, origH))
            const c = document.createElement('canvas')
            c.width = Math.round(origW * scale)
            c.height = Math.round(origH * scale)
            c.getContext('2d').drawImage(sourceEl, 0, 0, c.width, c.height)
            const blob = await new Promise((resolve, reject) =>
                c.toBlob(b => b ? resolve(b) : reject(new Error('toBlob failed')), 'image/jpeg', 0.9),
            )

            const form = new FormData()
            form.append('image', blob, 'image.jpg')

            const resp = await fetch('/api/ai/segment-instances', {
                method: 'POST', body: form, signal: abortController.signal,
            })
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}))
                throw new Error(err.error || `Detection failed (${resp.status})`)
            }
            const data = await resp.json()
            if (instancesAbortRef.current !== abortController) return

            if (!data.instances?.length) {
                toast.info('No distinct subjects detected — try Select Subject for the unified selection.')
                setSubjectInstances([])
                return
            }
            setSubjectInstances(data.instances)
            setActiveInstanceIndex(null)
            lastInstancesImageRef.current = fabricObj
            // Cache on the Fabric image so other tools (and the agent's
            // mask-commands cache) can reuse the same payload.
            try { fabricObj.__phosmithSubjectInstances = { ...data, instances: data.instances } } catch { /* ignore */ }

            // Auto-select the union (non-destructively) so the panel shows
            // immediate feedback — the user can then click a chip to swap to a
            // single subject. The background is never removed.
            if (data.unionPng) {
                const decoded = await decodeMaskBlob(base64PngToBlob(data.unionPng))
                if (instancesAbortRef.current !== abortController) return
                applySubjectSelection(decoded.imageData, `All subjects (${data.instances.length})`)
                setActiveInstanceIndex(-1)
            }
            toast.success(
                `Detected ${data.instances.length} subject${data.instances.length === 1 ? '' : 's'}` +
                (data.truncated ? ' (capped, refine to see more)' : ''),
            )
        } catch (err) {
            if (err?.name === 'AbortError') return
            console.error('[mask] detect-all-subjects failed:', err)
            toast.error(err?.message || 'Subject detection failed')
        } finally {
            if (instancesAbortRef.current === abortController) {
                isDetectingInstancesRef.current = false
                setIsDetectingInstances(false)
            }
        }
    }, [tool, applySubjectSelection])

    const handleApplyInstance = useCallback(async (instance) => {
        if (!tool.mainImage || !instance?.maskPng) return
        try {
            const blob = base64PngToBlob(instance.maskPng)
            const decoded = await decodeMaskBlob(blob)
            const label = `${instance.label || 'Subject'} #${(instance.index ?? 0) + 1}`
            const id = applySubjectSelection(decoded.imageData, label)
            if (id) {
                setActiveInstanceIndex(instance.index ?? null)
                toast.success(`Selected ${instance.label || `subject ${instance.index ?? ''}`}`)
            } else {
                toast.error('Could not create subject selection')
            }
        } catch (err) {
            console.error('[mask] apply-instance failed:', err)
            toast.error('Could not create subject selection')
        }
    }, [tool, applySubjectSelection])

    const handleApplyAllSubjectsUnion = useCallback(async () => {
        if (!tool.mainImage) return
        const cached = tool.mainImage.__phosmithSubjectInstances || subjectInstances && { unionPng: null, instances: subjectInstances }
        if (cached?.unionPng) {
            try {
                const blob = base64PngToBlob(cached.unionPng)
                const decoded = await decodeMaskBlob(blob)
                const count = cached.instances?.length || subjectInstances?.length || ''
                const id = applySubjectSelection(decoded.imageData, `All subjects (${count})`)
                if (id) {
                    setActiveInstanceIndex(-1)
                    toast.success(`Selected all ${count} subjects`)
                } else {
                    toast.error('Could not create union selection')
                }
            } catch (err) {
                console.error('[mask] apply-union failed:', err)
                toast.error('Could not create union selection')
            }
        }
    }, [tool, subjectInstances, applySubjectSelection])

    // Invalidate the multi-subject cache when the user switches to a
    // different image so the chips don't show stale data.
    useEffect(() => {
        if (lastInstancesImageRef.current && tool.mainImage !== lastInstancesImageRef.current) {
            setSubjectInstances(null)
            setActiveInstanceIndex(null)
            // The subject selection layer belonged to the previous image's chain;
            // drop the handle so the next selection adds a fresh layer.
            subjectLayerIdRef.current = null
            lastInstancesImageRef.current = tool.mainImage
        }
    }, [tool.mainImage])

    /* ─── Color Range ─── */
    const handleColorPick = useCallback((e) => {
        if (!colorPickerActive || !tool.mainImage) return

        const fabricCanvas = canvasEditor
        if (!fabricCanvas) return

        // Get the pixel color at click position from the canvas
        const pointer = fabricCanvas.getPointer(e.e || e)
        const fabricObj = tool.mainImage
        const sourceEl = fabricObj?._element || fabricObj?.getElement?.()
        if (!sourceEl) return

        // Transform pointer to image coordinates
        const imgW = sourceEl.naturalWidth || sourceEl.width || fabricObj.width
        const imgH = sourceEl.naturalHeight || sourceEl.height || fabricObj.height
        const objLeft = fabricObj.left || 0
        const objTop = fabricObj.top || 0
        const objScaleX = fabricObj.scaleX || 1
        const objScaleY = fabricObj.scaleY || 1

        const localX = (pointer.x - objLeft) / objScaleX
        const localY = (pointer.y - objTop) / objScaleY

        if (localX < 0 || localY < 0 || localX >= imgW || localY >= imgH) return

        // Read the pixel color
        const c = document.createElement('canvas')
        c.width = imgW
        c.height = imgH
        const ctx = c.getContext('2d', { willReadFrequently: true })
        try { ctx.drawImage(sourceEl, 0, 0) } catch { return }

        const pixel = ctx.getImageData(Math.floor(localX), Math.floor(localY), 1, 1).data
        const color = { r: pixel[0], g: pixel[1], b: pixel[2] }
        setPickedColor(color)
        setColorPickerActive(false)

        // Apply the color range mask
        tool.applyColorRangeMask({ ...color, tolerance: colorTolerance })
    }, [colorPickerActive, tool, canvasEditor, colorTolerance])

    // Attach/detach canvas click listener for color picker
    React.useEffect(() => {
        const fabricCanvas = canvasEditor
        if (!fabricCanvas) return

        if (colorPickerActive) {
            fabricCanvas.defaultCursor = 'crosshair'
            fabricCanvas.on('mouse:down', handleColorPick)
            return () => {
                fabricCanvas.defaultCursor = 'default'
                fabricCanvas.off('mouse:down', handleColorPick)
            }
        }
    }, [colorPickerActive, canvasEditor, handleColorPick])

    const handleApplyColorRange = useCallback(() => {
        if (!pickedColor) return
        tool.applyColorRangeMask({ ...pickedColor, tolerance: colorTolerance })
    }, [tool, pickedColor, colorTolerance])

    /* ─── Luminance Range ─── */
    const handleApplyLuminance = useCallback(() => {
        tool.applyLuminanceRangeMask({ minLuma: lumaMin, maxLuma: lumaMax })
    }, [tool, lumaMin, lumaMax])

    /* ─── Add to Mask Layers (megashader, non-destructive) ─── */

    // Push the current luminance settings onto the megashader chain as a
    // new layer, rather than baking them into the brush clipPath. The
    // megashader renders through the WebGL2 filter and composes
    // independently of the brush — both stay live until cleared.
    const handleAddLuminanceLayer = useCallback(() => {
        const min = Math.min(lumaMin, lumaMax - 1) / 255
        const max = Math.max(lumaMin + 1, lumaMax) / 255
        addChainLayer('luminance', { min, max, softness: 0.1 })
        toast.success('Luminance layer added')
    }, [addChainLayer, lumaMin, lumaMax])

    // Convert picked {r,g,b} (0..255) to HSB (0..360, 0..1, 0..1) so the
    // megashader can match against the same colour space the GLSL uses.
    // Tolerance maps from the 5..100 slider to the megashader's 0..1 range.
    const handleAddColorLayer = useCallback(() => {
        if (!pickedColor) return
        const target = rgbToHsb(pickedColor.r, pickedColor.g, pickedColor.b)
        const tolerance = Math.max(0.01, Math.min(0.5, colorTolerance / 100))
        addChainLayer('color', {
            target,
            tolerance,
            softness: 0.1,
        })
        toast.success('Color layer added')
    }, [addChainLayer, pickedColor, colorTolerance])

    const handleAddGradientLayer = useCallback(() => {
        // Linear gradient: adds a real linear layer to the megashader chain
        // and enters draft mode. The user then drags on the canvas to set
        // p1 (start) → p2 (end) of the line. Esc cancels, mouse-up commits.
        if (activeDraft) {
            toast('Finish or cancel the current draft first', { icon: 'ℹ️' })
            return
        }
        if (!imageSize) {
            toast.error('Image not ready yet')
            return
        }
        // Cancel color-picker mode so its mouse:down handler doesn't fire
        // alongside the spatial drag's mouse:down on the same canvas click.
        setColorPickerActive(false)
        const id = addChainLayer('linear', { imageSize, label: 'Linear gradient' })
        if (id) {
            setActiveDraft({ kind: 'linear', layerId: id })
            toast('Drag on the canvas to set the line (Esc to cancel)')
        }
    }, [activeDraft, addChainLayer, imageSize])

    const handleAddRadialLayer = useCallback(() => {
        // Radial gradient: adds a real radial layer and enters draft mode.
        // The user drags a bounding box; center = midpoint, rx/ry = half the
        // drag distance per axis, rotation = atan2 of the drag vector.
        if (activeDraft) {
            toast('Finish or cancel the current draft first', { icon: 'ℹ️' })
            return
        }
        if (!imageSize) {
            toast.error('Image not ready yet')
            return
        }
        setColorPickerActive(false)
        const id = addChainLayer('radial', { imageSize, label: 'Radial gradient' })
        if (id) {
            setActiveDraft({ kind: 'radial', layerId: id })
            toast('Drag on the canvas to set the ellipse (Esc to cancel)')
        }
    }, [activeDraft, addChainLayer, imageSize])

    /* ─── Gradient ─── */
    const handleApplyGradient = useCallback(() => {
        tool.applyLinearGradientMask({
            direction: gradDirection,
            position: gradPosition,
            featherPct: gradFeather,
        })
    }, [tool, gradDirection, gradPosition, gradFeather])

    if (!canvasEditor) {
        return (
            <div className="p-4">
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Canvas not ready</p>
            </div>
        )
    }

    if (!tool.mainImage) {
        return (
            <ToolEmptyState
                icon={Scissors}
                title="No image on canvas"
                subtitle="Add an image first, then use the mask tool"
            />
        )
    }

    return (
        <div className="space-y-0 overflow-y-auto pr-1 panel-scroll">
            {/* ────────── Mask Layers (megashader chain) — pinned to top ────────── */}
            <Section
                title="Mask Layers"
                icon={Layers}
                defaultOpen={true}
                badge={stack.chain.length > 0 ? `${stack.chain.length}` : null}
            >
                <div className="space-y-1.5">
                    {stack.chain.length > 0 && (
                        <div className="flex items-center gap-2 pb-1">
                            <button
                                type="button"
                                onClick={() => setShowMaskOverlay(!showMaskOverlay)}
                                aria-pressed={showMaskOverlay}
                                title="Show the selected area as a red overlay"
                                className={`mask-btn flex-1 text-[10px] py-1.5 ${showMaskOverlay ? 'mask-btn--danger' : ''}`}
                            >
                                <Eye className="h-3 w-3" />
                                Show mask
                            </button>
                            <button
                                type="button"
                                onClick={() => setGlobalInvert(!globalInvert)}
                                aria-pressed={globalInvert}
                                title="Invert the whole mask"
                                className={`mask-btn flex-1 text-[10px] py-1.5 ${globalInvert ? 'mask-btn--primary' : ''}`}
                            >
                                <Contrast className="h-3 w-3" />
                                Invert
                            </button>
                        </div>
                    )}
                    {stack.chain.length > 0 && (
                        <div className="flex items-center justify-end gap-1.5 pb-1">
                            <button
                                type="button"
                                onClick={undoChain}
                                disabled={!canUndo}
                                title="Undo layer change"
                                className="mask-icon-btn"
                            >
                                <RotateCcw className="h-3 w-3" />
                            </button>
                            <button
                                type="button"
                                onClick={redoChain}
                                disabled={!canRedo}
                                title="Redo layer change"
                                className="mask-icon-btn"
                                style={{ transform: 'scaleX(-1)' }}
                            >
                                <RotateCcw className="h-3 w-3" />
                            </button>
                        </div>
                    )}
                    <AnimatePresence>
                        {stack.chain.map((entry, i) => (
                            <MaskChainCard
                                key={entry.layer.id}
                                entry={entry}
                                index={i}
                                total={stack.chain.length}
                                isFirst={i === 0}
                                imageSize={imageSize}
                                selected={selectedLayerId === entry.layer.id}
                                onSelect={selectLayer}
                                onUpdate={(patch) => updateLayer(entry.layer.id, patch)}
                                onRemove={removeLayer}
                                onMove={moveLayer}
                                onSetOp={setLayerOp}
                                onSetFillMode={setFillMode}
                                onApplyCurve={applyCurve}
                                histogram={histogram}
                                dominantColor={dominantColor}
                                onExpandBoundary={(layerId, px) => {
                                    // Regenerates the layer's texture from its
                                    // pristine base and re-syncs the panel via
                                    // the chain-replaced event — so the edge of
                                    // an AI-detected subject stays extendable.
                                    try {
                                        expandLayerBoundary(tool.mainImage, layerId, px)
                                    } catch (err) {
                                        toast.error(err?.message || 'Could not adjust the mask boundary')
                                    }
                                }}
                            />
                        ))}
                    </AnimatePresence>

                    {stack.chain.length === 0 && (
                        <p
                            className="text-[10px] text-center py-3 rounded-md"
                            style={{ color: 'var(--text-muted)', border: '1px dashed var(--border-subtle)' }}
                        >
                            No layers yet — use any selection tool below to add one.
                        </p>
                    )}

                    {stack.chain.length > 0 && (
                        <button
                            type="button"
                            onClick={clearAll}
                            className="mask-btn mask-btn--danger w-full text-[10px] py-1.5 mt-1"
                        >
                            Clear all layers
                        </button>
                    )}
                </div>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    Each selection you create becomes a non-destructive mask layer.
                    Layers are composited by the megashader filter.
                </p>
            </Section>

            <CategoryHeader label="AI Tools" />

            <div className="px-1 -mt-1 mb-1 flex items-center gap-1.5 text-[10px]" style={{ color: 'var(--text-muted)' }} title="Masking microservice (SAM 3.1) status">
                <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: maskService.status === 'available' ? '#22c55e' : maskService.status === 'offline' ? '#f59e0b' : '#6b7280' }} />
                {maskService.status === 'available'
                    ? (maskService.sam3 ? 'Masking service · SAM 3.1 ready' : 'Masking service online')
                    : maskService.status === 'offline'
                        ? 'Masking service offline — using on-device AI'
                        : 'Checking masking service…'}
            </div>

            {/* AI processing routing: per-capability choice of where each AI
                function runs — Auto (server first, device fallback), Device
                (in-browser models via transformers.js, downloaded once and
                cached), or Server (local mask service / Gemini). The NL-mask
                executor follows this policy with runtime fallback to the
                other side, so a misconfigured side degrades instead of
                failing (see src/lib/ai-routing.js). */}
            <Section title="AI Processing" icon={Cpu} defaultOpen={false} badge={routingBadge}>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    Choose where each AI function runs. <strong>Device</strong> keeps
                    everything in this browser (models download once);{' '}
                    <strong>Server</strong> uses the local AI service / Gemini;{' '}
                    <strong>Auto</strong> prefers the server and falls back to the device.
                </p>
                {Object.entries(AI_CAPABILITIES).map(([cap, def]) => (
                    <div key={cap} className="space-y-1">
                        {/* flex-wrap: in a narrow sidebar the three-mode pill
                            row drops below the label instead of overlapping it */}
                        <div className="flex items-center justify-between gap-2 flex-wrap">
                            <span
                                className="text-[10px] font-semibold"
                                style={{ color: 'var(--text-secondary)' }}
                                title={def.hint}
                            >
                                {def.label}
                            </span>
                            <div className="mask-fill-modes" style={{ marginTop: 0 }}>
                                {['auto', ...(def.client ? ['client'] : []), ...(def.server ? ['server'] : [])].map((mode) => (
                                    <button
                                        key={mode}
                                        type="button"
                                        onClick={() => setRoutingMode(cap, mode)}
                                        className={`mask-fill-mode-btn ${routingPolicy[cap] === mode ? 'mask-fill-mode-btn--active' : ''}`}
                                        title={mode === 'client' ? (def.clientImpl || 'In this browser')
                                            : mode === 'server' ? (def.serverImpl || 'On the server')
                                                : 'Server first, device fallback'}
                                    >
                                        {mode === 'auto' ? 'Auto' : mode === 'client' ? 'Device' : 'Server'}
                                    </button>
                                ))}
                            </div>
                        </div>
                        {/* Device-model readiness: this capability is set to run
                            on-device and has a downloadable browser model. The
                            background prefetch downloads it; show its progress. */}
                        {routingPolicy[cap] === 'client' && cap in CLIENT_READY && (
                            <span
                                className="text-[9px]"
                                style={{ color: CLIENT_READY[cap] ? '#4ade80' : 'var(--text-muted)' }}
                            >
                                {CLIENT_READY[cap]
                                    ? '✓ Model ready on device'
                                    : clientAI.loading
                                        ? `Downloading ${clientAI.loading}…`
                                        : 'Model downloads in the background'}
                            </span>
                        )}
                    </div>
                ))}
                <div className="flex items-center justify-between gap-2">
                    <button
                        type="button"
                        onClick={resetRoutingPolicy}
                        className="text-[10px]"
                        style={{ color: 'var(--text-muted)' }}
                    >
                        Reset to Auto
                    </button>
                    <button
                        type="button"
                        onClick={handleSelfTest}
                        disabled={selfTest.running}
                        className="mask-btn px-2 py-1 text-[10px] font-semibold"
                        title="Runs the in-browser models on a test image with a known answer. First run downloads the models (one-time)."
                    >
                        {selfTest.running ? (
                            <>
                                <Loader2 className="h-3 w-3 animate-spin" />
                                Testing…
                            </>
                        ) : (
                            <>
                                <Cpu className="h-3 w-3" />
                                Test device AI
                            </>
                        )}
                    </button>
                </div>
                {selfTest.running && selfTest.progress && (
                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{selfTest.progress}</p>
                )}
                {selfTest.report && (
                    <div className="space-y-1">
                        {selfTest.report.checks.map((c) => (
                            <div key={c.label} className="flex items-center gap-1.5 text-[10px]">
                                <span style={{ color: c.ok ? '#4ade80' : '#ef4444' }}>{c.ok ? '✓' : '✗'}</span>
                                <span style={{ color: 'var(--text-secondary)' }}>{c.label}</span>
                                <span className="ml-auto font-mono" style={{ color: 'var(--text-muted)' }}>{c.detail}</span>
                            </div>
                        ))}
                        <p className="text-[9px]" style={{ color: 'var(--text-muted)' }}>
                            {selfTest.report.device?.toUpperCase()} · {(selfTest.report.totalMs / 1000).toFixed(1)}s
                        </p>
                    </div>
                )}
            </Section>

            {/* ────────── AI Masking ────────── */}
            <Section title="Select Subject" icon={Sparkles} defaultOpen={true} badge="AI">
                <motion.button
                    type="button"
                    onClick={handleSelectSubject}
                    disabled={isSegmenting}
                    whileTap={{ scale: 0.97 }}
                    className="mask-btn mask-btn--primary w-full py-2.5 text-xs font-semibold"
                >
                    {isSegmenting ? (
                        <>
                            <Loader2 className="h-4 w-4 animate-spin" />
                            Detecting Subject…
                        </>
                    ) : (
                        <>
                            <Sparkles className="h-4 w-4" />
                            Select Subject
                        </>
                    )}
                </motion.button>
                <p className="text-[10px] text-center" style={{ color: 'var(--text-muted)' }}>
                    AI selects the main subject as a layer — background stays intact
                </p>

                <motion.button
                    type="button"
                    onClick={handleSelectBackground}
                    disabled={isSelectingBg}
                    whileTap={{ scale: 0.97 }}
                    className="mask-btn w-full py-2 text-[11px] font-semibold"
                    style={{
                        background: 'rgba(6,184,212,0.10)',
                        border: '1px solid rgba(6,184,212,0.30)',
                        color: 'var(--accent-primary)',
                    }}
                >
                    {isSelectingBg ? (
                        <>
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            Selecting Background…
                        </>
                    ) : (
                        <>
                            <Layers className="h-3.5 w-3.5" />
                            Select Background / Sky
                        </>
                    )}
                </motion.button>

                {/* ── Multi-subject: per-instance picker ────────────────── */}
                <motion.button
                    type="button"
                    onClick={handleDetectAllSubjects}
                    disabled={isDetectingInstances}
                    whileTap={{ scale: 0.97 }}
                    className="mask-btn w-full py-2 text-[11px] font-semibold"
                    style={{
                        background: 'rgba(124,58,237,0.10)',
                        border: '1px solid rgba(124,58,237,0.30)',
                        color: '#C4B5FD',
                    }}
                >
                    {isDetectingInstances ? (
                        <>
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            Detecting all subjects…
                        </>
                    ) : (
                        <>
                            <Sparkles className="h-3.5 w-3.5" />
                            Detect All Subjects
                        </>
                    )}
                </motion.button>

                {subjectInstances && subjectInstances.length > 0 && (
                    <div className="space-y-1.5">
                        <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                            Detected {subjectInstances.length} subject{subjectInstances.length === 1 ? '' : 's'}. Pick one or All:
                        </p>
                        <div className="flex flex-wrap gap-1">
                            <button
                                type="button"
                                onClick={handleApplyAllSubjectsUnion}
                                className="rounded-md px-2 py-1 text-[10px] font-semibold editor-interactive"
                                style={{
                                    background: activeInstanceIndex === -1 ? 'rgba(124,58,237,0.25)' : 'var(--bg-elevated)',
                                    border: `1px solid ${activeInstanceIndex === -1 ? 'rgba(124,58,237,0.55)' : 'var(--border-subtle)'}`,
                                    color: activeInstanceIndex === -1 ? '#C4B5FD' : 'var(--text-primary)',
                                }}
                            >
                                All ({subjectInstances.length})
                            </button>
                            {subjectInstances.map((inst) => {
                                const isActive = activeInstanceIndex === inst.index
                                return (
                                    <button
                                        key={inst.index}
                                        type="button"
                                        onClick={() => handleApplyInstance(inst)}
                                        title={`${inst.label} · conf ${(inst.confidence * 100).toFixed(0)}%`}
                                        className="rounded-md px-2 py-1 text-[10px] font-medium editor-interactive"
                                        style={{
                                            background: isActive ? 'rgba(124,58,237,0.25)' : 'var(--bg-elevated)',
                                            border: `1px solid ${isActive ? 'rgba(124,58,237,0.55)' : 'var(--border-subtle)'}`,
                                            color: isActive ? '#C4B5FD' : 'var(--text-primary)',
                                        }}
                                    >
                                        {inst.label} #{inst.index + 1}
                                    </button>
                                )
                            })}
                        </div>
                    </div>
                )}
            </Section>

            {/* ────────── Click-to-Select (SAM 2) ────────── */}
            <Section title="Click to Select" icon={MousePointer} badge="AI">
                <div className="space-y-2">
                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        Click to mark the subject — or draw a box around it — then run SAM 2. Hold{' '}
                        <kbd className="px-1 rounded text-[9px]" style={{ background: 'var(--bg-elevated)' }}>Alt</kbd>{' '}
                        to mark background (negative click).
                    </p>

                    <div className="flex items-center gap-1.5">
                        {!semanticActive ? (
                            <motion.button
                                type="button"
                                onClick={() => {
                                    setColorPickerActive(false)
                                    if (activeDraft) {
                                        toast('Finish or cancel the current draft first', { icon: 'ℹ️' })
                                        return
                                    }
                                    setSemanticActive(true)
                                    toast('Click the subject on the canvas')
                                }}
                                whileTap={{ scale: 0.97 }}
                                className="flex-1 flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                                style={{
                                    background: 'rgba(124,58,237,0.08)',
                                    border: '1px solid rgba(124,58,237,0.25)',
                                    color: '#A78BFA',
                                }}
                            >
                                <Crosshair className="h-3.5 w-3.5" />
                                Start Clicking
                            </motion.button>
                        ) : (
                            <motion.button
                                type="button"
                                onClick={handleSemanticStop}
                                whileTap={{ scale: 0.97 }}
                                className="flex-1 flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                                style={{
                                    background: 'rgba(239,68,68,0.10)',
                                    border: '1px solid rgba(239,68,68,0.30)',
                                    color: '#FCA5A5',
                                }}
                            >
                                <X className="h-3.5 w-3.5" />
                                Stop
                            </motion.button>
                        )}
                        <motion.button
                            type="button"
                            onClick={handleSemanticReset}
                            disabled={!semanticActive || (semanticClicks.length === 0 && !lastSemanticMask)}
                            whileTap={{ scale: 0.97 }}
                            className="flex items-center justify-center gap-1 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive disabled:opacity-40"
                            style={{
                                background: 'var(--bg-elevated)',
                                border: '1px solid var(--border-subtle)',
                                color: 'var(--text-secondary)',
                            }}
                            title="Clear clicks and last result"
                        >
                            <RotateCcw className="h-3.5 w-3.5" />
                        </motion.button>
                    </div>

                    {/* Box prompt — SAM 2's strongest single prompt for whole
                        objects. One box at a time; a new drag replaces it. */}
                    {semanticActive && (
                        <div className="flex items-center gap-1.5">
                            <motion.button
                                type="button"
                                onClick={() => setBoxArmed((v) => !v)}
                                whileTap={{ scale: 0.97 }}
                                className="flex-1 flex items-center justify-center gap-1.5 rounded-lg px-2 py-1.5 text-[10px] font-medium editor-interactive"
                                style={{
                                    background: boxArmed ? 'rgba(6,184,212,0.18)' : 'var(--bg-elevated)',
                                    border: `1px solid ${boxArmed ? 'rgba(6,184,212,0.45)' : 'var(--border-subtle)'}`,
                                    color: boxArmed ? 'var(--accent-primary)' : 'var(--text-secondary)',
                                }}
                                title="Drag a rectangle around the object — SAM 2 selects what's inside"
                            >
                                <Square className="h-3 w-3" />
                                {boxArmed ? 'Drag on the image…' : semanticBox ? 'Redraw box' : 'Draw box'}
                            </motion.button>
                            {semanticBox && (
                                <button
                                    type="button"
                                    onClick={() => setSemanticBox(null)}
                                    className="flex items-center gap-1 text-[9px] px-1.5 py-1.5 rounded"
                                    title="Remove the box prompt"
                                    style={{
                                        background: 'rgba(6,184,212,0.15)',
                                        color: 'var(--accent-primary)',
                                        border: '1px solid rgba(6,184,212,0.35)',
                                    }}
                                >
                                    {Math.round(semanticBox[2] - semanticBox[0])}×{Math.round(semanticBox[3] - semanticBox[1])}
                                    <X className="h-2.5 w-2.5" />
                                </button>
                            )}
                        </div>
                    )}

                    {/* Click list — cyan dot = positive, red = negative */}
                    {semanticActive && semanticClicks.length > 0 && (
                        <div
                            className="rounded-md p-1.5 flex flex-wrap gap-1"
                            style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)' }}
                        >
                            {semanticClicks.map((c, i) => (
                                <button
                                    key={i}
                                    type="button"
                                    onClick={() => setSemanticClicks((prev) => prev.filter((_, j) => j !== i))}
                                    className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded"
                                    title={`${c[0].toFixed(0)}, ${c[1].toFixed(0)} — click to remove`}
                                    style={{
                                        background: c[2] === 1 ? 'rgba(6,184,212,0.15)' : 'rgba(239,68,68,0.15)',
                                        color: c[2] === 1 ? 'var(--accent-primary)' : '#FCA5A5',
                                        border: `1px solid ${c[2] === 1 ? 'rgba(6,184,212,0.35)' : 'rgba(239,68,68,0.35)'}`,
                                    }}
                                >
                                    <span>{c[2] === 1 ? '+' : '−'}</span>
                                    <span>({c[0].toFixed(0)}, {c[1].toFixed(0)})</span>
                                </button>
                            ))}
                        </div>
                    )}

                    <motion.button
                        type="button"
                        onClick={handleSemanticRun}
                        disabled={isSemanticRunning || (semanticClicks.length === 0 && !semanticBox)}
                        whileTap={{ scale: 0.97 }}
                        className="flex w-full items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-xs font-semibold editor-interactive disabled:opacity-40"
                        style={{
                            background: 'linear-gradient(135deg, rgba(6,184,212,0.20) 0%, rgba(124,58,237,0.18) 100%)',
                            border: '1px solid rgba(6,184,212,0.35)',
                            color: 'var(--accent-primary)',
                        }}
                    >
                        {isSemanticRunning ? (
                            <>
                                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                Running SAM 2…
                            </>
                        ) : (
                            <>
                                <Wand2 className="h-3.5 w-3.5" />
                                Run ({semanticClicks.length}{semanticBox ? ' + box' : ''})
                            </>
                        )}
                    </motion.button>

                    {/* Mask preview + add-to-chain */}
                    {lastSemanticPreview && (
                        <div
                            className="rounded-md p-2 space-y-1.5"
                            style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)' }}
                        >
                            <div className="flex items-center gap-2">
                                <img
                                    src={lastSemanticPreview}
                                    alt="SAM 2 mask preview"
                                    className="rounded"
                                    style={{ width: 64, height: 64, objectFit: 'contain', background: '#000' }}
                                />
                                <div className="flex-1 text-[10px] leading-tight" style={{ color: 'var(--text-muted)' }}>
                                    <div className="font-semibold mb-0.5" style={{ color: 'var(--text-secondary)' }}>
                                        Mask ready
                                    </div>
                                    White = keep, black = remove. The mask is at the
                                    original image&apos;s resolution.
                                </div>
                            </div>
                            <motion.button
                                type="button"
                                onClick={handleAddSemanticLayer}
                                whileTap={{ scale: 0.97 }}
                                className="flex w-full items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                                style={{
                                    background: 'rgba(6,184,212,0.10)',
                                    border: '1px solid rgba(6,184,212,0.30)',
                                    color: 'var(--accent-primary)',
                                }}
                            >
                                <Plus className="h-3.5 w-3.5" />
                                Add to Mask Layers
                            </motion.button>
                        </div>
                    )}
                </div>
            </Section>

            {/* ────────── Smart Brush (Step 7) ────────── */}
            <CategoryHeader label="Draw Selection" />

            <Section title="Selection Brush" icon={Paintbrush}>
                <div className="space-y-2">
                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        Paint a <strong>selection</strong> — it shows as a live
                        overlay and becomes an editable, non-destructive layer.
                        Closed outlines fill their entire inside automatically.
                        Nothing is erased. Use <strong>Erase / Cut</strong> below
                        to knock the painted region out instead.
                    </p>

                    {/* Output: select (fill) vs erase (cut) — same as the lasso */}
                    <div className="grid grid-cols-2 gap-1.5">
                        {[
                            { id: 'select', label: 'Select', icon: Layers, hint: 'visible selection layer' },
                            { id: 'erase', label: 'Erase / Cut', icon: Scissors, hint: 'cut the painted region out' },
                        ].map((s) => {
                            const SIcon = s.icon
                            const active = brushSink === s.id
                            return (
                                <button
                                    key={s.id}
                                    type="button"
                                    onClick={() => setBrushSink(s.id)}
                                    title={s.hint}
                                    className="flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                                    style={{
                                        background: active ? 'rgba(124,58,237,0.10)' : 'var(--bg-elevated)',
                                        border: `1px solid ${active ? 'rgba(124,58,237,0.35)' : 'var(--border-subtle)'}`,
                                        color: active ? '#A78BFA' : 'var(--text-secondary)',
                                    }}
                                >
                                    <SIcon className="h-3.5 w-3.5" />
                                    {s.label}
                                </button>
                            )
                        })}
                    </div>

                    <BrushSizeControl
                        value={brushSize}
                        setValue={setBrushSize}
                        min={2}
                        max={200}
                        dominantColor={dominantColor}
                    />
                    <LabeledSlider
                        label="Hardness"
                        value={Math.round(brushHardness * 100)}
                        min={0}
                        max={100}
                        suffix="%"
                        onChange={(v) => setBrushHardness(Math.max(0, Math.min(1, v / 100)))}
                        dominantColor={dominantColor}
                    />
                    {!brushEdgeSnap && (
                        <LabeledSlider
                            label="Edge Feather (this region)"
                            value={brushFeather}
                            min={0}
                            max={50}
                            suffix="px"
                            onChange={setBrushFeather}
                            dominantColor={dominantColor}
                        />
                    )}

                    {/* Snap-to-edges toggle → smartBrush (bilateral) vs plain brush */}
                    <button
                        type="button"
                        onClick={() => setBrushEdgeSnap((v) => !v)}
                        aria-pressed={brushEdgeSnap}
                        title="Snap the stroke to underlying edges (bilateral filter)"
                        className="flex w-full items-center justify-center gap-1.5 rounded-lg px-2 py-1.5 text-[11px] font-medium editor-interactive"
                        style={{
                            background: brushEdgeSnap ? 'rgba(6,184,212,0.12)' : 'var(--bg-elevated)',
                            border: `1px solid ${brushEdgeSnap ? 'var(--accent-primary)' : 'var(--border-subtle)'}`,
                            color: brushEdgeSnap ? 'var(--accent-primary)' : 'var(--text-secondary)',
                        }}
                    >
                        <Sparkles className="h-3.5 w-3.5" />
                        Snap to edges {brushEdgeSnap ? 'ON' : 'OFF'}
                    </button>

                    {/* Boolean modifier — how this selection combines with the chain */}
                    <div>
                        <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>
                            Combine (Shift = add, Alt = subtract)
                        </label>
                        <div className="grid grid-cols-4 gap-1">
                            {[
                                { id: 'new', label: 'New', icon: Circle },
                                { id: 'add', label: 'Add', icon: Plus },
                                { id: 'subtract', label: 'Sub', icon: Minus },
                                { id: 'intersect', label: 'Int', icon: Crosshair },
                            ].map((m) => {
                                const MIcon = m.icon
                                const active = brushModifier === m.id
                                return (
                                    <button
                                        key={m.id}
                                        type="button"
                                        onClick={() => setBrushModifier(m.id)}
                                        title={m.label}
                                        className="flex items-center justify-center gap-1 rounded-md px-1 py-1.5 text-[10px] font-medium editor-interactive"
                                        style={{
                                            background: active ? 'rgba(6,184,212,0.12)' : 'var(--bg-elevated)',
                                            border: `1px solid ${active ? 'var(--accent-primary)' : 'var(--border-subtle)'}`,
                                            color: active ? 'var(--accent-primary)' : 'var(--text-muted)',
                                        }}
                                    >
                                        <MIcon className="h-3 w-3" />
                                        {m.label}
                                    </button>
                                )
                            })}
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-1.5 pt-1">
                        {!brushActive ? (
                            <motion.button
                                type="button"
                                onClick={handleStartBrush}
                                whileTap={{ scale: 0.97 }}
                                className="flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                                style={{
                                    background: 'rgba(124,58,237,0.10)',
                                    border: '1px solid rgba(124,58,237,0.30)',
                                    color: '#A78BFA',
                                }}
                            >
                                <Paintbrush className="h-3.5 w-3.5" />
                                Start Painting
                            </motion.button>
                        ) : (
                            <motion.button
                                type="button"
                                onClick={handleStopBrush}
                                whileTap={{ scale: 0.97 }}
                                className="flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                                style={{
                                    background: 'rgba(239,68,68,0.10)',
                                    border: '1px solid rgba(239,68,68,0.30)',
                                    color: '#FCA5A5',
                                }}
                            >
                                <X className="h-3.5 w-3.5" />
                                Stop
                            </motion.button>
                        )}
                        <motion.button
                            type="button"
                            onClick={handleClearBrush}
                            disabled={!brushHasContent && !brushActive}
                            whileTap={{ scale: 0.97 }}
                            className="flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive disabled:opacity-40"
                            style={{
                                background: 'var(--bg-elevated)',
                                border: '1px solid var(--border-subtle)',
                                color: 'var(--text-secondary)',
                            }}
                            title="Clear the painted stroke"
                        >
                            <RotateCcw className="h-3.5 w-3.5" />
                            Clear
                        </motion.button>
                    </div>

                    {/* Edge-snap (bilateral) filter settings — only relevant
                        when "Snap to edges" is on; become the layer's params. */}
                    {brushEdgeSnap && (
                        <div className="space-y-1.5 pt-1">
                            <LabeledSlider
                                label="Filter Radius"
                                value={filterRadius}
                                min={1}
                                max={8}
                                step={1}
                                onChange={setFilterRadius}
                                format={(v) => `${v} px`}
                            />
                            <LabeledSlider
                                label="Color Sigma (edge strictness)"
                                value={sigmaColor}
                                min={0.01}
                                max={1}
                                step={0.01}
                                onChange={setSigmaColor}
                                format={(v) => v.toFixed(2)}
                            />
                            <LabeledSlider
                                label="Space Sigma (spatial spread)"
                                value={sigmaSpace}
                                min={0.5}
                                max={8}
                                step={0.1}
                                onChange={setSigmaSpace}
                                format={(v) => v.toFixed(1)}
                            />
                        </div>
                    )}

                    <motion.button
                        type="button"
                        onClick={handleAddBrushLayer}
                        disabled={!brushHasContent || isShapeFilling}
                        whileTap={{ scale: 0.97 }}
                        className="flex w-full items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-xs font-semibold editor-interactive disabled:opacity-40"
                        style={{
                            background: 'linear-gradient(135deg, rgba(6,184,212,0.20) 0%, rgba(124,58,237,0.18) 100%)',
                            border: '1px solid rgba(6,184,212,0.35)',
                            color: 'var(--accent-primary)',
                        }}
                    >
                        {isShapeFilling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
                        {isShapeFilling
                            ? 'Filling shape...'
                            : brushSink === 'erase'
                                ? 'Add cut to layers'
                                : 'Add selection to layers'}
                    </motion.button>
                </div>
            </Section>

            {/* ────────── Lasso (freehand + polygonal + magnetic) ────────── */}
            <Section title="Lasso Select" icon={Lasso}>
                <div className="space-y-2">
                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        Draw a selection. <strong>Freehand</strong> = drag;{' '}
                        <strong>Polygonal</strong> = click points;{' '}
                        <strong>Magnetic</strong> = click to start, glide along an
                        edge and the path snaps to it. Double-click or{' '}
                        <kbd className="px-1 rounded text-[9px]" style={{ background: 'var(--bg-elevated)' }}>Enter</kbd>{' '}
                        to close (<kbd className="px-1 rounded text-[9px]" style={{ background: 'var(--bg-elevated)' }}>Backspace</kbd> undoes a point).
                    </p>

                    {/* Mode: freehand / polygonal / magnetic */}
                    <div className="grid grid-cols-3 gap-1.5">
                        {[
                            { id: 'freehand', label: 'Freehand', icon: Lasso },
                            { id: 'polygonal', label: 'Polygonal', icon: Spline },
                            { id: 'magnetic', label: 'Magnetic', icon: Wand2 },
                        ].map((m) => {
                            const MIcon = m.icon
                            const active = lassoMode === m.id
                            return (
                                <button
                                    key={m.id}
                                    type="button"
                                    onClick={() => setLassoMode(m.id)}
                                    className="flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                                    style={{
                                        background: active ? 'rgba(6,184,212,0.12)' : 'var(--bg-elevated)',
                                        border: `1px solid ${active ? 'var(--accent-primary)' : 'var(--border-subtle)'}`,
                                        color: active ? 'var(--accent-primary)' : 'var(--text-secondary)',
                                    }}
                                >
                                    <MIcon className="h-3.5 w-3.5" />
                                    {m.label}
                                </button>
                            )
                        })}
                    </div>

                    {/* Magnetic options — Width / Contrast / Frequency (Photoshop parity) */}
                    {lassoMode === 'magnetic' && (
                        <div className="space-y-1.5 rounded-lg px-2 py-2" style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)' }}>
                            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                                The path snaps to the strongest nearby edge. Tune how
                                far it looks (Width), how strong an edge must be
                                (Contrast), and how often it drops anchors (Frequency).
                            </p>
                            <LabeledSlider
                                label="Width (search)"
                                value={magneticWidth}
                                min={4}
                                max={60}
                                suffix="px"
                                onChange={setMagneticWidth}
                                dominantColor={dominantColor}
                            />
                            <LabeledSlider
                                label="Contrast (edge threshold)"
                                value={magneticContrast}
                                min={1}
                                max={60}
                                suffix="%"
                                onChange={setMagneticContrast}
                                dominantColor={dominantColor}
                            />
                            <LabeledSlider
                                label="Frequency (anchor spacing)"
                                value={magneticFrequency}
                                min={4}
                                max={48}
                                suffix="px"
                                onChange={setMagneticFrequency}
                                dominantColor={dominantColor}
                            />
                        </div>
                    )}

                    {/* Pen mode — smooth the captured anchors into a Bézier 'path'
                        layer (Photoshop Pen-tool parity) instead of a straight lasso. */}
                    <button
                        type="button"
                        onClick={() => setLassoSmooth((v) => !v)}
                        title="Smooth the captured points into a Bézier pen path before committing"
                        className="flex w-full items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                        style={{
                            background: lassoSmooth ? 'rgba(83,216,255,0.12)' : 'var(--bg-elevated)',
                            border: `1px solid ${lassoSmooth ? 'var(--accent-primary)' : 'var(--border-subtle)'}`,
                            color: lassoSmooth ? 'var(--accent-primary)' : 'var(--text-secondary)',
                        }}
                    >
                        <Spline className="h-3.5 w-3.5" />
                        {lassoSmooth ? 'Pen: smooth curves ON' : 'Pen: smooth curves'}
                    </button>

                    {/* Output: select (fill) vs erase (cut) */}
                    <div className="grid grid-cols-2 gap-1.5">
                        {[
                            { id: 'select', label: 'Select', icon: Layers, hint: 'visible selection layer' },
                            { id: 'erase', label: 'Erase / Cut', icon: Scissors, hint: 'cut the region out' },
                        ].map((s) => {
                            const SIcon = s.icon
                            const active = lassoSink === s.id
                            return (
                                <button
                                    key={s.id}
                                    type="button"
                                    onClick={() => setLassoSink(s.id)}
                                    title={s.hint}
                                    className="flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                                    style={{
                                        background: active ? 'rgba(124,58,237,0.10)' : 'var(--bg-elevated)',
                                        border: `1px solid ${active ? 'rgba(124,58,237,0.35)' : 'var(--border-subtle)'}`,
                                        color: active ? '#A78BFA' : 'var(--text-secondary)',
                                    }}
                                >
                                    <SIcon className="h-3.5 w-3.5" />
                                    {s.label}
                                </button>
                            )
                        })}
                    </div>

                    {/* Boolean modifier — how this selection combines with the chain */}
                    <div>
                        <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>
                            Combine (Shift = add, Alt = subtract)
                        </label>
                        <div className="grid grid-cols-4 gap-1">
                            {[
                                { id: 'new', label: 'New', icon: Circle },
                                { id: 'add', label: 'Add', icon: Plus },
                                { id: 'subtract', label: 'Sub', icon: Minus },
                                { id: 'intersect', label: 'Int', icon: Crosshair },
                            ].map((m) => {
                                const MIcon = m.icon
                                const active = lassoModifier === m.id
                                return (
                                    <button
                                        key={m.id}
                                        type="button"
                                        onClick={() => setLassoModifier(m.id)}
                                        title={m.label}
                                        className="flex items-center justify-center gap-1 rounded-md px-1 py-1.5 text-[10px] font-medium editor-interactive"
                                        style={{
                                            background: active ? 'rgba(6,184,212,0.12)' : 'var(--bg-elevated)',
                                            border: `1px solid ${active ? 'var(--accent-primary)' : 'var(--border-subtle)'}`,
                                            color: active ? 'var(--accent-primary)' : 'var(--text-muted)',
                                        }}
                                    >
                                        <MIcon className="h-3 w-3" />
                                        {m.label}
                                    </button>
                                )
                            })}
                        </div>
                    </div>

                    <LabeledSlider
                        label="Feather"
                        value={Math.round(lassoFeather * 100)}
                        min={0}
                        max={40}
                        suffix="%"
                        onChange={(v) => setLassoFeather(Math.max(0, Math.min(0.4, v / 100)))}
                        dominantColor={dominantColor}
                    />

                    <div className="flex items-center gap-1.5">
                        {!lassoActive ? (
                            <motion.button
                                type="button"
                                onClick={handleStartLasso}
                                whileTap={{ scale: 0.97 }}
                                className="flex-1 flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-semibold editor-interactive"
                                style={{
                                    background: 'linear-gradient(135deg, rgba(6,184,212,0.20) 0%, rgba(124,58,237,0.18) 100%)',
                                    border: '1px solid rgba(6,184,212,0.35)',
                                    color: 'var(--accent-primary)',
                                }}
                            >
                                <Lasso className="h-3.5 w-3.5" />
                                Start Lasso
                            </motion.button>
                        ) : (
                            <>
                                {(lassoMode === 'polygonal' || lassoMode === 'magnetic') && lassoVertexCount >= 3 && (
                                    <motion.button
                                        type="button"
                                        onClick={() => finishLassoSelection(null)}
                                        whileTap={{ scale: 0.97 }}
                                        className="flex items-center justify-center gap-1 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                                        style={{ background: 'rgba(6,184,212,0.12)', border: '1px solid var(--accent-primary)', color: 'var(--accent-primary)' }}
                                        title="Close the selection"
                                    >
                                        <Plus className="h-3.5 w-3.5" /> Close
                                    </motion.button>
                                )}
                                <motion.button
                                    type="button"
                                    onClick={handleStopLasso}
                                    whileTap={{ scale: 0.97 }}
                                    className="flex-1 flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                                    style={{ background: 'rgba(239,68,68,0.10)', border: '1px solid rgba(239,68,68,0.30)', color: '#FCA5A5' }}
                                >
                                    <X className="h-3.5 w-3.5" />
                                    Stop{lassoVertexCount > 0 ? ` (${lassoVertexCount})` : ''}
                                </motion.button>
                            </>
                        )}
                    </div>
                </div>
            </Section>

            {/* ────────── Depth Range (Depth Anything V2) ────────── */}
            <Section title="Depth Range" icon={Mountain} badge="AI">
                <div className="space-y-2">
                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        Generate a per-pixel depth map, then add it as a
                        non-destructive layer with a custom depth range.
                    </p>

                    <div className="flex items-center gap-1.5">
                        <motion.button
                            type="button"
                            onClick={handleDepthRun}
                            disabled={isDepthRunning}
                            whileTap={{ scale: 0.97 }}
                            className="flex-1 flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive disabled:opacity-40"
                            style={{
                                background: 'linear-gradient(135deg, rgba(6,184,212,0.20) 0%, rgba(20,184,166,0.18) 100%)',
                                border: '1px solid rgba(6,184,212,0.35)',
                                color: 'var(--accent-primary)',
                            }}
                        >
                            {isDepthRunning ? (
                                <>
                                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                    Generating…
                                </>
                            ) : (
                                <>
                                    <Mountain className="h-3.5 w-3.5" />
                                    Generate Depth Map
                                </>
                            )}
                        </motion.button>
                        <motion.button
                            type="button"
                            onClick={handleDepthReset}
                            disabled={!lastDepthMap}
                            whileTap={{ scale: 0.97 }}
                            className="flex items-center justify-center gap-1 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive disabled:opacity-40"
                            style={{
                                background: 'var(--bg-elevated)',
                                border: '1px solid var(--border-subtle)',
                                color: 'var(--text-secondary)',
                            }}
                            title="Clear last result"
                        >
                            <RotateCcw className="h-3.5 w-3.5" />
                        </motion.button>
                    </div>

                    {/* Depth preview + range controls + add-to-chain */}
                    {lastDepthPreview && (
                        <div
                            className="rounded-md p-2 space-y-1.5"
                            style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)' }}
                        >
                            <div className="flex items-center gap-2">
                                <img
                                    src={lastDepthPreview}
                                    alt="Depth map preview"
                                    className="rounded"
                                    style={{ width: 64, height: 64, objectFit: 'contain', background: '#000' }}
                                />
                                <div className="flex-1 text-[10px] leading-tight" style={{ color: 'var(--text-muted)' }}>
                                    <div className="font-semibold mb-0.5" style={{ color: 'var(--text-secondary)' }}>
                                        Depth map ready
                                    </div>
                                    White = near, black = far. Use the sliders
                                    below to select a range.
                                </div>
                            </div>

                            <div className="space-y-1.5 pt-1">
                                <LabeledSlider
                                    label="Min (near floor)"
                                    value={depthMin}
                                    onChange={setDepthMinBounded}
                                    min={0}
                                    max={1}
                                    step={0.01}
                                    format={(v) => v.toFixed(2)}
                                />
                                <LabeledSlider
                                    label="Max (far ceiling)"
                                    value={depthMax}
                                    onChange={setDepthMaxBounded}
                                    min={0}
                                    max={1}
                                    step={0.01}
                                    format={(v) => v.toFixed(2)}
                                />
                                <LabeledSlider
                                    label="Softness (edge width)"
                                    value={depthSoftness}
                                    onChange={setDepthSoftness}
                                    min={0}
                                    max={0.5}
                                    step={0.01}
                                    format={(v) => v.toFixed(2)}
                                />
                            </div>

                            <motion.button
                                type="button"
                                onClick={handleAddDepthLayer}
                                whileTap={{ scale: 0.97 }}
                                className="flex w-full items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                                style={{
                                    background: 'rgba(6,184,212,0.10)',
                                    border: '1px solid rgba(6,184,212,0.30)',
                                    color: 'var(--accent-primary)',
                                }}
                            >
                                <Plus className="h-3.5 w-3.5" />
                                Add to Mask Layers
                            </motion.button>
                        </div>
                    )}
                </div>
            </Section>

            <CategoryHeader label="Range Selection" />

            {/* ────────── Color Range ────────── */}
            <Section title="Color Range" icon={Palette}>
                <div className="flex items-center gap-2">
                    <motion.button
                        type="button"
                        onClick={() => setColorPickerActive(v => !v)}
                        whileTap={{ scale: 0.95 }}
                        className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium editor-interactive flex-1"
                        style={{
                            background: colorPickerActive
                                ? 'rgba(6,184,212,0.12)'
                                : 'var(--bg-elevated)',
                            border: `1px solid ${colorPickerActive ? 'var(--accent-primary)' : 'var(--border-subtle)'}`,
                            color: colorPickerActive ? 'var(--accent-primary)' : 'var(--text-secondary)',
                        }}
                    >
                        <Crosshair className="h-3.5 w-3.5" />
                        {colorPickerActive ? 'Click image to pick…' : 'Pick Color'}
                    </motion.button>
                    {pickedColor && <ColorSwatch color={pickedColor} />}
                </div>

                {pickedColor && (
                    <div className="space-y-2">
                        <LabeledSlider
                            label="Tolerance"
                            value={colorTolerance}
                            min={5}
                            max={100}
                            suffix=""
                            onChange={setColorTolerance}
                            dominantColor={dominantColor}
                        />
                        <div className="grid grid-cols-2 gap-1.5">
                            <motion.button
                                type="button"
                                onClick={handleApplyColorRange}
                                whileTap={{ scale: 0.97 }}
                                className="flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                                style={{
                                    background: 'var(--bg-elevated)',
                                    border: '1px solid var(--border-subtle)',
                                    color: 'var(--text-secondary)',
                                }}
                            >
                                <Palette className="h-3.5 w-3.5" />
                                Apply (bake)
                            </motion.button>
                            <motion.button
                                type="button"
                                onClick={handleAddColorLayer}
                                whileTap={{ scale: 0.97 }}
                                className="flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                                style={{
                                    background: 'rgba(6,184,212,0.10)',
                                    border: '1px solid rgba(6,184,212,0.3)',
                                    color: 'var(--accent-primary)',
                                }}
                                title="Add as a live megashader layer"
                            >
                                <Layers className="h-3.5 w-3.5" />
                                Add to Mask Layers
                            </motion.button>
                        </div>
                    </div>
                )}
            </Section>

            {/* ────────── Luminance Range ────────── */}
            <Section title="Luminance Range" icon={Sun}>
                <div className="space-y-2">
                    <LuminanceHistogram
                        histogram={histogram}
                        min={lumaMin / 255}
                        max={lumaMax / 255}
                        softness={0.1}
                    />
                    <LabeledSlider
                        label="Min Brightness"
                        value={lumaMin}
                        min={0}
                        max={254}
                        suffix=""
                        onChange={(v) => { setLumaMin(Math.min(v, lumaMax - 1)) }}
                        dominantColor={dominantColor}
                    />
                    <LabeledSlider
                        label="Max Brightness"
                        value={lumaMax}
                        min={1}
                        max={255}
                        suffix=""
                        onChange={(v) => { setLumaMax(Math.max(v, lumaMin + 1)) }}
                        dominantColor={dominantColor}
                    />
                    <div className="grid grid-cols-2 gap-1.5">
                        <motion.button
                            type="button"
                            onClick={handleApplyLuminance}
                            whileTap={{ scale: 0.97 }}
                            className="flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                            style={{
                                background: 'var(--bg-elevated)',
                                border: '1px solid var(--border-subtle)',
                                color: 'var(--text-secondary)',
                            }}
                        >
                            <Sun className="h-3.5 w-3.5" />
                            Apply (bake)
                        </motion.button>
                        <motion.button
                            type="button"
                            onClick={handleAddLuminanceLayer}
                            whileTap={{ scale: 0.97 }}
                            className="flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-medium editor-interactive"
                            style={{
                                background: 'rgba(6,184,212,0.10)',
                                border: '1px solid rgba(6,184,212,0.3)',
                                color: 'var(--accent-primary)',
                            }}
                            title="Add as a live megashader layer"
                        >
                            <Layers className="h-3.5 w-3.5" />
                            Add to Mask Layers
                        </motion.button>
                    </div>
                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        Apply (bake) writes to the brush mask. Add to Mask Layers
                        keeps it as a live, editable megashader layer.
                    </p>
                </div>
            </Section>

            {/* ────────── Linear Gradient ────────── */}
            <Section title="Linear Gradient" icon={Blend}>
                <div className="space-y-3">
                    {/* Direction grid */}
                    <div>
                        <label className="text-[10px] block mb-1.5" style={{ color: 'var(--text-muted)' }}>Direction</label>
                        <div className="grid grid-cols-4 gap-1">
                            {DIRECTIONS.map(d => {
                                const DirIcon = d.icon
                                const active = gradDirection === d.id
                                return (
                                    <motion.button
                                        key={d.id}
                                        type="button"
                                        onClick={() => setGradDirection(d.id)}
                                        whileTap={{ scale: 0.9 }}
                                        className="flex items-center justify-center rounded-md p-1.5 editor-interactive"
                                        style={{
                                            background: active ? 'rgba(6,184,212,0.12)' : 'var(--bg-elevated)',
                                            border: `1px solid ${active ? 'var(--accent-primary)' : 'var(--border-subtle)'}`,
                                            color: active ? 'var(--accent-primary)' : 'var(--text-muted)',
                                        }}
                                        title={d.label}
                                    >
                                        <DirIcon className="h-3.5 w-3.5" />
                                    </motion.button>
                                )
                            })}
                        </div>
                    </div>

                    <LabeledSlider
                        label="Position"
                        value={gradPosition}
                        min={10}
                        max={90}
                        suffix="%"
                        onChange={setGradPosition}
                        dominantColor={dominantColor}
                    />
                    <LabeledSlider
                        label="Feather"
                        value={gradFeather}
                        min={5}
                        max={80}
                        suffix="%"
                        onChange={setGradFeather}
                        dominantColor={dominantColor}
                    />
                    <motion.button
                        type="button"
                        onClick={handleApplyGradient}
                        whileTap={{ scale: 0.97 }}
                        className="flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-medium editor-interactive"
                        style={{
                            background: 'var(--bg-elevated)',
                            border: '1px solid var(--border-subtle)',
                            color: 'var(--text-secondary)',
                        }}
                    >
                        <Blend className="h-3.5 w-3.5" />
                        Apply Gradient Mask
                    </motion.button>
                    <motion.button
                        type="button"
                        onClick={handleAddGradientLayer}
                        whileTap={{ scale: 0.97 }}
                        className="flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-medium editor-interactive"
                        style={{
                            background: 'rgba(124,58,237,0.08)',
                            border: '1px solid rgba(124,58,237,0.25)',
                            color: '#A78BFA',
                        }}
                        title="Adds a linear megashader layer — drag on canvas to set p1/p2"
                    >
                        <Layers className="h-3.5 w-3.5" />
                        Add Linear to Mask Layers

                    </motion.button>
                </div>
            </Section>

            {/* ────────── Radial Gradient ────────── */}
            <Section title="Radial Gradient" icon={Circle}>
                <div className="space-y-3">
                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        Drag a bounding box on the canvas to define the ellipse.
                        Center, radii, and rotation are derived from the drag.
                    </p>
                    <motion.button
                        type="button"
                        onClick={handleAddRadialLayer}
                        whileTap={{ scale: 0.97 }}
                        className="flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-medium editor-interactive"
                        style={{
                            background: 'rgba(124,58,237,0.08)',
                            border: '1px solid rgba(124,58,237,0.25)',
                            color: '#A78BFA',
                        }}
                        title="Adds a radial megashader layer — drag on canvas to set the bounding box"
                    >
                        <Layers className="h-3.5 w-3.5" />
                        Add Radial to Mask Layers

                    </motion.button>
                </div>
            </Section>

            <CategoryHeader label="Destructive" />

            {/* ────────── Brush (manual) ────────── */}
            <Section title="Quick Erase" icon={Scissors} defaultOpen={false}>
                <div className="space-y-2">
                    <p className="text-[10px]" style={{ color: '#FCA5A5' }}>
                        ⚠ This paints directly onto the image and hides pixels
                        (destructive). For a reversible result, use the
                        <strong> Selection Brush</strong> with Erase / Cut instead.
                        Turn this on to paint; turn it off to stop.
                    </p>
                    <button
                        type="button"
                        onClick={quickEraseActive ? handleStopQuickErase : handleStartQuickErase}
                        aria-pressed={quickEraseActive}
                        className="flex w-full items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[11px] font-semibold editor-interactive"
                        style={{
                            background: quickEraseActive ? 'rgba(239,68,68,0.14)' : 'var(--bg-elevated)',
                            border: `1px solid ${quickEraseActive ? 'rgba(239,68,68,0.45)' : 'var(--border-subtle)'}`,
                            color: quickEraseActive ? '#FCA5A5' : 'var(--text-secondary)',
                        }}
                    >
                        {quickEraseActive ? <X className="h-3.5 w-3.5" /> : <Scissors className="h-3.5 w-3.5" />}
                        {quickEraseActive ? 'Stop erasing' : 'Enable Quick Erase'}
                    </button>
                    {quickEraseActive && (
                        <>
                            <ModeToggle mode={tool.mode} setMode={tool.setMode} altActive={tool.altActive} />
                            <BrushSizeControl
                                value={tool.brushSize}
                                setValue={tool.setBrushSize}
                                min={MIN_BRUSH}
                                max={MAX_BRUSH}
                                dominantColor={dominantColor}
                            />
                            <LabeledSlider label="Hardness" value={tool.hardness} min={1} max={100} suffix="%" onChange={tool.setHardness} dominantColor={dominantColor} />
                            <LabeledSlider label="Flow" value={tool.flow} min={5} max={100} suffix="%" onChange={tool.setFlow} dominantColor={dominantColor} />
                            <LabeledSlider label="Edge Feather" value={tool.feather} min={0} max={50} suffix="px" onChange={tool.setFeather} dominantColor={dominantColor} />
                        </>
                    )}
                </div>
            </Section>

            {/* ────────── Actions ────────── */}
            <div style={{ paddingTop: '4px' }}>
                <MaskActionButtons
                    hasMask={tool.hasMask}
                    undoDepth={tool.undoDepth}
                    redoDepth={tool.redoDepth}
                    onUndo={tool.undo}
                    onRedo={tool.redo}
                    onInvert={tool.invert}
                    onClear={tool.clear}
                />
            </div>

            <TipCard>
                <p><strong>AI Tools</strong> — Select Subject (one-click), Click to Select (SAM 2), and Depth Range use AI models to generate masks automatically.</p>
                <p><strong>Draw Selection</strong> — Selection Brush paints a region; Lasso draws freehand, polygonal, or edge-snapping (magnetic) outlines.</p>
                <p><strong>Range Selection</strong> — Color, Luminance, and Gradient masks select by pixel properties. Combine multiple methods into one mask.</p>
                <p>Each selection becomes its own <strong>Mask Layer</strong> with per-layer feather, blend mode, and fill / adjust / erase output.</p>
                <p>Shift = add, Alt = subtract while drawing to combine selections.</p>
            </TipCard>
        </div>
    )
}

export default MaskControls
