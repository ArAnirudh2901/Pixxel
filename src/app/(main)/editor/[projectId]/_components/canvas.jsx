"use client"

import React, { useCallback, useEffect, useRef, useState } from "react"
import { useCanvas } from "../../../../../../context/context"
import { useDatabaseMutation } from "../../../../../../hooks/useDatabaseQuery"
import { api } from "@/lib/neon-api";
import { Hand, Maximize2, ZoomIn, ZoomOut, ArrowLeftRight } from "lucide-react"
import {
    Canvas,
    FabricImage,
    InteractiveFabricObject,
    Point,
    config as fabricConfig,
} from "fabric"
import { bindDoodlesToImage, followDoodles } from "@/lib/canvas-doodle-bind"
// Side-effect import: registers PhosmithCurves filter in Fabric's classRegistry so
// loadFromJSON can rehydrate saved canvas state that contains it.
import "../../../../../lib/curves-filter"
// Eagerly register the MegashaderFilter class with Fabric's classRegistry at
// module load. Without this, a project saved with a persisted megashader mask
// chain would have its `type: "Megashader"` filter silently dropped by
// loadFromJSON (the class wouldn't be registered yet — it was previously only
// imported lazily when the mask tool fired its first change event).
import "@/lib/megashader/fabric-megashader-filter"

// Force the Canvas2D filter backend instead of WebGL. The custom curves LUT filter
// has a WebGL fragment-shader path that worked in isolation but had subtle issues
// in real filter chains (corrupted source texture state on some images, leading to
// black canvases and other downstream filters not visibly applying). 2D is slower
// for large images but correct in every chain shape.
if (typeof window !== "undefined" && fabricConfig) {
    fabricConfig.enableGLFiltering = false
}

// Neo-brutalist defaults for selected-object controls (corners, border, padding,
// rotation handle). Fabric reads InteractiveFabricObject.ownDefaults at object
// construction, so this needs to run before any FabricImage / Rect / etc. is
// created — at module init, before Canvas mounts.
if (typeof window !== "undefined" && InteractiveFabricObject?.ownDefaults) {
    InteractiveFabricObject.ownDefaults = {
        ...InteractiveFabricObject.ownDefaults,
        // Square cyan corners with a hard cream stroke — same palette as the
        // editor's Projects header, preview controls, and resolution HUD.
        cornerStyle: "rect",
        cornerColor: "#06B8D4",
        cornerStrokeColor: "#F4F4F5",
        cornerSize: 11,
        touchCornerSize: 22,
        transparentCorners: false,
        cornerDashArray: null,
        // Solid cream marquee border with a slight float-off-the-image padding.
        // Thicker than default so it reads at any zoom.
        borderColor: "#F4F4F5",
        borderScaleFactor: 1.6,
        borderDashArray: null,
        borderOpacityWhenMoving: 0.9,
        padding: 6,
    }
}
import { normalizeCanvasState, serializeCanvasState } from "../../../../../lib/canvas-state"
import { hydrateCanvasImages, restoreCanvasFromHistory } from "../../../../../lib/canvas-history"
import { registerExtendHost } from "../../../../../lib/extend-poller"
import { isAgentActing, recordChange, setJournalProject } from "../../../../../lib/change-journal"
import { describeCanvasChange } from "../../../../../lib/canvas-change-describe"
import { isExpansionFrameLike, removeExpansionFramesFromCanvas } from "../../../../../lib/expansion-pipeline"
import { addImageFilesToCanvas } from "../../../../../lib/canvas-images"
import {
    fetchCachedSnapshot,
    flushToNeon,
    snapshotToCache,
} from "../../../../../lib/canvas-cache"
import { createCanvasSync, loadLocalState, clearLocalState } from "../../../../../lib/canvas-sync"
import { createPresenceChannel } from "../../../../../lib/canvas-presence"
import { toast } from "sonner"
import { isPhosmithMaskOverlay } from "../../../../../lib/canvas-mask"
import { syncBackgroundGrade } from "../../../../../lib/canvas-background"
import AuroraLoader from "./AuroraLoader"

const MIN_ZOOM = 0.05
const MAX_ZOOM = 64
const MIN_PREVIEW_ZOOM_PERCENT = 5
const MAX_PREVIEW_ZOOM_PERCENT = 300
const PREVIEW_ZOOM_STEP_PERCENT = 1
const VIEWPORT_PADDING = 32
const MAX_PERSISTED_HISTORY = 30
const MIN_PERSISTED_HISTORY_ENTRIES = 3
const MAX_NEON_STATE_CHARS = 900_000
// Tiered autosave: MAJOR changes (real edits) save on the fast debounce as
// before; MINOR ones (sub-threshold nudges — see canvas-change-describe) ride
// a slow trickle so fidgeting can't generate a stream of snapshot/flush API
// calls. Minor edits are still durable immediately via the IndexedDB mirror
// and the unload beacon, and any pending major save carries them for free.
const MAJOR_SAVE_DEBOUNCE_MS = 2000
const MINOR_SAVE_TRICKLE_MS = 30_000
// 'text:changed' fires per keystroke. Without a debounce every keystroke
// became its own undo state and evicted the whole 30-entry history while
// typing a sentence — push ONE history state per typing pause instead.
const TEXT_HISTORY_DEBOUNCE_MS = 900
// How long the network snapshot (Redis write-behind) is allowed to lag the
// freshest state — see minSnapshotIntervalMs in canvas-sync.
const MIN_SNAPSHOT_INTERVAL_MS = 4000
// Neon flush debounce. Was 8s; 15s halves database writes in active sessions
// with no durability cost (Redis snapshot + IndexedDB hold the latest state,
// and tab-hide/unload force a flush anyway).
const FLUSH_DEBOUNCE_MS = 15_000
const clamp = (value, min, max) => Math.min(Math.max(value, min), max)
const readPreviewZoomPercent = (canvas) => Math.round((canvas?.getZoom?.() || 1) * 100)
const getPrimaryRemoteImageUrl = (canvas) => {
    const image = canvas
        ?.getObjects?.()
        ?.find((object) => object?.type?.toLowerCase() === 'image')
    const src =
        image?.getSrc?.() ||
        image?._originalElement?.src ||
        image?._element?.src ||
        image?.src ||
        ''

    if (!src || src.startsWith('data:') || src.startsWith('blob:')) return null
    return src.startsWith('http') ? src : null
}

const fitImageInsideProject = (image, projectSize) => {
    const projectW = Math.max(1, projectSize?.width || image?.width || 1)
    const projectH = Math.max(1, projectSize?.height || image?.height || 1)
    const imageW = Math.max(1, image?.width || projectW)
    const imageH = Math.max(1, image?.height || projectH)
    const scale = Math.min(projectW / imageW, projectH / imageH)

    image.set({
        left: projectW / 2,
        top: projectH / 2,
        originX: "center",
        originY: "center",
        scaleX: scale,
        scaleY: scale,
        selectable: true,
        evented: true,
    })
    image.setCoords()
}

const CanvasEditor = ({ project }) => {
    const [isLoading, setIsLoading] = useState(true)
    const canvasRef = useRef()
    const containerRef = useRef()
    const canvasInstanceRef = useRef(null)
    const isPanningRef = useRef(false)
    const ctrlPressedRef = useRef(false)
    const spacePressedRef = useRef(false)
    const handToolActiveRef = useRef(false)
    const [isHandToolActive, setIsHandToolActive] = useState(false)
    const [isProjectFrameVisible, setIsProjectFrameVisible] = useState(false)
    const [previewZoomPercent, setPreviewZoomPercent] = useState(100)
    const projectFrameStyleRef = useRef({ left: 0, top: 0, width: 0, height: 0 })
    const [projectFrameStyle, setProjectFrameStyle] = useState({ left: 0, top: 0, width: 0, height: 0 })
    // Before/After compare: a per-tool "session baseline" — the flattened image
    // captured when the current tool was opened. Held-down Compare overlays it on
    // the live canvas so the user sees before (baseline) vs after (current edits).
    const [compareBaselineUrl, setCompareBaselineUrl] = useState(null)
    const [isComparing, setIsComparing] = useState(false)
    const lastPointerRef = useRef(null)
    const historyRef = useRef([])
    const historyIndexRef = useRef(-1)
    const isRestoringRef = useRef(false)
    const resizeFrameRef = useRef(null)
    const initGenerationRef = useRef(0)
    const previewZoomPercentRef = useRef(100)
    const projectRef = useRef(project)
    projectRef.current = project

    const { canvasEditor, setCanvasEditor, activeTool, expansionPreview, processingMessage } = useCanvas()
    // Hide the floating canvas chrome (zoom bar, hand tool, resolution HUD) any
    // time we're showing a full-screen processing overlay or the initial canvas
    // loader. Otherwise those controls bleed through the blurred background.
    const isBusy = Boolean(processingMessage) || isLoading
    const activeToolRef = useRef(activeTool)
    activeToolRef.current = activeTool
    const { mutate: updateProject } = useDatabaseMutation(api.projects.updateProject)
    const { mutate: createProjectRevisionMut } = useDatabaseMutation(api.projects.createProjectRevision)
    const { mutate: createProjectMut } = useDatabaseMutation(api.projects.create)

    const disposeCanvasInstance = useCallback(() => {
        const existing = canvasInstanceRef.current
        if (existing) {
            existing.__cleanupInfiniteWorkspace?.()
            try {
                existing.dispose()
            } catch {
                /* already disposed */
            }
            // After dispose(), Fabric's internal canvas context is null.
            // Stale references held by in-flight RAF callbacks, debounced saves,
            // or React state that hasn't caught up yet will still call
            // requestRenderAll / renderAll and crash inside Fabric with
            // "t.clearRect" (t = null context). Replace them with no-ops so any
            // caller that fires after dispose silently does nothing.
            try {
                const noop = () => {}
                existing.requestRenderAll = noop
                existing.renderAll = noop
                existing.renderAndReset = noop
            } catch { /* ignore — already dead */ }
            canvasInstanceRef.current = null
        }
        setCanvasEditor(null)
    }, [setCanvasEditor])

    const getContainerSize = () => {
        if (typeof window === 'undefined' || !containerRef.current) return { width: 0, height: 0 }
        return { width: containerRef.current.clientWidth, height: containerRef.current.clientHeight }
    }

    const getViewportState = useCallback((canvas) => {
        const viewportTransform = canvas.viewportTransform || [1, 0, 0, 1, 0, 0]
        const zoom = viewportTransform[0] || 1
        return {
            zoom,
            center: { x: (canvas.getWidth() / 2 - viewportTransform[4]) / zoom, y: (canvas.getHeight() / 2 - viewportTransform[5]) / zoom },
        }
    }, [])

    const setViewportState = useCallback((canvas, viewportState, fallbackCenter) => {
        const zoom = clamp(viewportState?.zoom || 1, MIN_ZOOM, MAX_ZOOM)
        const center = viewportState?.center || fallbackCenter
        if (!center) return
        canvas.setViewportTransform([zoom, 0, 0, zoom, canvas.getWidth() / 2 - center.x * zoom, canvas.getHeight() / 2 - center.y * zoom])
    }, [])

    const syncPreviewZoomState = useCallback((canvas) => {
        const nextPercent = readPreviewZoomPercent(canvas)
        if (previewZoomPercentRef.current === nextPercent) return
        previewZoomPercentRef.current = nextPercent
        setPreviewZoomPercent(nextPercent)
    }, [])

    const setCanvasPreviewZoom = useCallback((canvas, percent) => {
        if (!canvas) return
        const viewportState = getViewportState(canvas)
        setViewportState(canvas, {
            ...viewportState,
            zoom: clamp(Number(percent) / 100, MIN_ZOOM, MAX_ZOOM),
        })
        canvas.calcOffset()
        canvas.requestRenderAll()
        syncPreviewZoomState(canvas)
    }, [getViewportState, setViewportState, syncPreviewZoomState])

    const fitProjectToViewport = useCallback((canvas, size) => {
        const canvasW = canvas.getWidth()
        const canvasH = canvas.getHeight()
        const proj = size || projectRef.current
        const projectW = Math.max(1, proj?.width || 1)
        const projectH = Math.max(1, proj?.height || 1)
        if (!canvasW || !canvasH || !projectW || !projectH) return

        // Use 85% of canvas area as the safe zone, ensuring visible breathing
        // room on all four sides. Math.min ensures BOTH the percentage margin
        // AND the fixed padding are respected (the tighter constraint wins).
        const safeW = Math.min(canvasW * 0.85, canvasW - VIEWPORT_PADDING * 2)
        const safeH = Math.min(canvasH * 0.85, canvasH - VIEWPORT_PADDING * 2)
        if (safeW <= 0 || safeH <= 0) return
        const fitZoom = Math.min(safeW / projectW, safeH / projectH)
        setViewportState(canvas, {
            zoom: clamp(fitZoom || 1, MIN_ZOOM, MAX_ZOOM),
            center: { x: projectW / 2, y: projectH / 2 },
        })
    }, [setViewportState])

    const createInitialViewport = useCallback((canvas) => fitProjectToViewport(canvas), [fitProjectToViewport])

    const emitHistoryChange = (canvas) => {
        if (!canvas) return
        canvas.fire('history:changed', {
            canUndo: historyIndexRef.current > 0,
            canRedo: historyIndexRef.current < historyRef.current.length - 1,
            index: historyIndexRef.current,
            length: historyRef.current.length,
        })
    }

    const pushHistoryState = useCallback((canvas, meta) => {
        if (!canvas || isRestoringRef.current) return
        const nextState = serializeCanvasState(canvas)
        if (!nextState?.canvas) return
        const nextSignature = JSON.stringify(nextState)
        const currentState = historyRef.current[historyIndexRef.current]
        const currentSignature = currentState ? JSON.stringify(currentState) : null
        if (nextSignature === currentSignature) return
        historyRef.current = historyRef.current.slice(0, historyIndexRef.current + 1)
        historyRef.current.push(nextState)
        while (historyRef.current.length > MAX_PERSISTED_HISTORY) {
            historyRef.current.shift()
            historyIndexRef.current = Math.max(0, historyIndexRef.current - 1)
        }
        historyIndexRef.current = historyRef.current.length - 1
        emitHistoryChange(canvas)
        // Journal AFTER the dedup check, so no-op pushes never log. While an
        // agent command runs, runCommand already records the SPECIFIC command
        // — skip the generic entry to avoid double-logging the same change.
        // meta.silent pushes the undo state WITHOUT a journal entry (used by
        // the initial rehydration push, which isn't a user change).
        if (!isAgentActing() && !meta?.silent) {
            recordChange({
                label: meta?.label || 'Canvas edit',
                detail: meta?.detail,
                source: 'user',
                domain: meta?.domain || 'canvas',
                coalesceKey: meta?.coalesceKey,
            })
        }
    }, [])

    const restoreCanvasState = useCallback(async (canvas, state) => {
        if (!canvas || !state) return
        const proj = projectRef.current
        isRestoringRef.current = true
        try {
            const imageUrl = proj?.currentImageUrl || proj?.originalImageUrl
            await restoreCanvasFromHistory(canvas, state, {
                imageUrl,
                setViewportState,
                fallbackCenter: { x: proj.width / 2, y: proj.height / 2 },
            })
        } finally {
            isRestoringRef.current = false
            emitHistoryChange(canvas)
        }
    }, [setViewportState])

    const undoCanvasState = useCallback(async () => {
        const canvas = canvasInstanceRef.current
        if (!canvas || historyIndexRef.current <= 0) return false
        historyIndexRef.current -= 1
        await restoreCanvasState(canvas, historyRef.current[historyIndexRef.current])
        recordChange({ label: 'Undo', domain: 'history' })
        return true
    }, [restoreCanvasState])

    const redoCanvasState = useCallback(async () => {
        const canvas = canvasInstanceRef.current
        if (!canvas || historyIndexRef.current >= historyRef.current.length - 1) return false
        historyIndexRef.current += 1
        await restoreCanvasState(canvas, historyRef.current[historyIndexRef.current])
        recordChange({ label: 'Redo', domain: 'history' })
        return true
    }, [restoreCanvasState])

    // Canvas sync manager, lazily created per project (see the useEffect below
    // that recreates it when projectId changes). It owns the write-behind path:
    // dedup, single-flight, the durable IndexedDB mirror, reconnect replay, and
    // the unload beacon. We never let two managers exist for the same project.
    const syncRef = useRef(null)
    // Stable indirection to the latest direct-Neon writer so the sync manager
    // (created once per project) always calls the current updateProject mutation.
    const directWriteRef = useRef(async () => {})
    const wasOfflineRef = useRef(false)
    // 'idle' | 'saving' | 'saved' | 'offline' | 'error' — drives the status pill.
    const [syncStatus, setSyncStatus] = useState("idle")

    // Surface sync-manager status: update the pill and toast on offline/online
    // transitions (deduped via wasOfflineRef so we don't spam).
    const handleSyncStatus = useCallback((status) => {
        setSyncStatus(status)
        if (status === "offline") {
            if (!wasOfflineRef.current) {
                wasOfflineRef.current = true
                toast.warning(
                    "You're offline — changes are saved on this device and will sync when you reconnect.",
                    { id: "canvas-sync", duration: Infinity },
                )
            }
        } else if (status === "saved" && wasOfflineRef.current) {
            wasOfflineRef.current = false
            toast.success("Back online — your changes are synced.", { id: "canvas-sync", duration: 3000 })
        }
    }, [])

    // Save the CURRENT live canvas (with its history) as a brand-new project and
    // return the new project's id (null on failure, e.g. the free-plan limit).
    // Shared by the proactive concurrent-device path (presence) and the reactive
    // flush-conflict path, so "your work becomes its own copy" behaves identically.
    const forkLiveCanvasToProject = useCallback(async (titleSuffix) => {
        const canvas = canvasInstanceRef.current
        const proj = projectRef.current
        if (!proj) return null
        let localCanvasState
        let localCurrentImageUrl
        if (canvas) {
            const localState = serializeCanvasState(canvas)
            localCurrentImageUrl = getPrimaryRemoteImageUrl(canvas) || undefined
            localCanvasState = {
                ...localState,
                history: historyRef.current.slice(-MAX_PERSISTED_HISTORY),
                historyIndex: historyIndexRef.current,
            }
        } else {
            // Canvas not initialised yet (concurrency detected mid-load) — fork
            // from the loaded project state, which is exactly what this device sees.
            localCanvasState = proj.canvasState || null
            localCurrentImageUrl = proj.currentImageUrl || undefined
        }
        try {
            return await createProjectMut({
                title: `${proj.title || "Project"} — ${titleSuffix}`,
                canvasState: localCanvasState,
                currentImageUrl: localCurrentImageUrl || proj.currentImageUrl,
                width: proj.width,
                height: proj.height,
            })
        } catch (forkErr) {
            console.warn("[canvas] fork to new project failed:", forkErr?.message || forkErr)
            return null
        }
    }, [createProjectMut])

    // Conflict handler: another session (device/tab) advanced the project's
    // revision while we were editing, so the flush was rejected (no overwrite).
    // We resolve it NON-DESTRUCTIVELY — both versions land in version history —
    // then let the user pick which becomes current.
    const handleConflict = useCallback(async (serverProject) => {
        setSyncStatus("conflict")
        const canvas = canvasInstanceRef.current
        const proj = projectRef.current
        if (!canvas || !proj) return

        const localState = serializeCanvasState(canvas)
        const localCurrentImageUrl = getPrimaryRemoteImageUrl(canvas) || undefined
        const localCanvasState = {
            ...localState,
            history: historyRef.current.slice(-MAX_PERSISTED_HISTORY),
            historyIndex: historyIndexRef.current,
        }

        // Always preserve OUR working copy in version history first, so it's
        // recoverable no matter which way the user resolves.
        try {
            await createProjectRevisionMut({
                projectId: proj._id,
                canvasState: localCanvasState,
                width: proj.width,
                height: proj.height,
                currentImageUrl: localCurrentImageUrl,
                title: "Your version (edit conflict)",
                summary: "Auto-saved because this project was edited on another device.",
            })
        } catch (error) {
            console.warn("[canvas] failed to preserve local conflict copy:", error?.message || error)
        }

        // Auto-save the local diverged state as a NEW PROJECT so both versions
        // coexist independently — each device can continue on its own copy
        // without either overwriting the other.
        const forkProjectId = await forkLiveCanvasToProject("your device's copy")

        if (forkProjectId) {
            // Fork succeeded: auto-reload this tab to the server's version so the
            // two devices no longer compete for the same project. The user can open
            // the forked project separately at any time.
            toast.success("Conflict detected — your version saved as a new project.", {
                id: "canvas-conflict",
                duration: Infinity,
                dismissible: true,
                description: `"${proj.title || "Project"} — your device's copy" has been created. Reload this tab to see the other version, or open your copy.`,
                action: {
                    label: "Open my copy",
                    onClick: () => {
                        if (typeof window !== "undefined") {
                            window.open(`/editor/${forkProjectId}`, "_blank")
                        }
                    },
                },
                cancel: {
                    label: "Reload this tab",
                    onClick: () => {
                        Promise.resolve()
                            .then(() => clearLocalState(proj._id).catch(() => {}))
                            .then(() => {
                                if (serverProject?.canvasState) {
                                    return snapshotToCache(proj._id, serverProject.canvasState, serverProject.currentImageUrl || null, serverProject.revision).catch(() => {})
                                }
                            })
                            .finally(() => {
                                if (typeof window !== "undefined") window.location.reload()
                            })
                    },
                },
            })
        } else {
            // Fallback: fork failed (plan limit etc.) — show the original dialog
            // so the user can manually choose which version to keep.
            toast.warning("This project was edited on another device.", {
                id: "canvas-conflict",
                duration: Infinity,
                dismissible: true,
                description: "Your version is saved in version history. Reload the latest, or keep yours (overwrites the other copy).",
                // "Keep mine" must be the `action` — Sonner v2's action.onClick fires
                // reliably, but cancel.onClick is broken (auto-dismiss kills it).
                action: {
                    label: "Keep mine",
                    onClick: () => {
                        syncRef.current?.overwriteRemote()
                        toast.dismiss("canvas-conflict")

                        if (serverProject?.canvasState) {
                            createProjectRevisionMut({
                                projectId: proj._id,
                                canvasState: serverProject.canvasState,
                                width: serverProject.width || proj.width,
                                height: serverProject.height || proj.height,
                                currentImageUrl: serverProject.currentImageUrl || undefined,
                                title: "Other device's version (edit conflict)",
                                summary: "Preserved before overwriting with your version.",
                            }).catch((error) => {
                                console.warn("[canvas] failed to preserve remote conflict copy:", error?.message || error)
                            })
                        }
                    },
                },
                cancel: {
                    label: "Reload latest",
                    onClick: () => {
                        Promise.resolve()
                            .then(() => clearLocalState(proj._id).catch(() => {}))
                            .then(() => {
                                if (serverProject?.canvasState) {
                                    return snapshotToCache(proj._id, serverProject.canvasState, serverProject.currentImageUrl || null, serverProject.revision).catch(() => {})
                                }
                            })
                            .finally(() => {
                                if (typeof window !== "undefined") window.location.reload()
                            })
                    },
                },
            })
        }
    }, [createProjectRevisionMut, forkLiveCanvasToProject])

    // Stable indirection so the per-project sync manager always calls the latest
    // conflict handler without being torn down when it changes identity.
    const onConflictRef = useRef(() => {})
    onConflictRef.current = handleConflict

    // Proactive concurrent-device handler (driven by the presence heartbeat).
    // Unlike handleConflict (which only fires once a flush is REJECTED), this
    // fires the moment a different device opens the same project — before either
    // side has overwritten the other.
    //   - Newcomer (joined later): immediately fork our work into its own project
    //     and move this tab to it, so the two devices never compete for one project.
    //   - Keeper (joined first): just inform the user; we keep the original.
    const handleConcurrentDevice = useCallback(async ({ others, isNewcomer }) => {
        const proj = projectRef.current
        if (!proj) return
        const otherLabel = others?.[0]?.deviceLabel || "another device"

        if (!isNewcomer) {
            // We were here first — keep the original project, just let the user know.
            toast.info("This project was opened on another device.", {
                id: "canvas-presence",
                duration: 8000,
                description: `${otherLabel} is now editing a separate copy, so your work here is safe.`,
            })
            return
        }

        // We're the newcomer: fork our state into its own project and switch to it.
        toast.loading("This project is open on another device — saving your work as a separate copy…", {
            id: "canvas-presence",
            duration: Infinity,
        })
        const forkProjectId = await forkLiveCanvasToProject("this device's copy")
        if (forkProjectId) {
            toast.success("Saved as a separate copy to avoid conflicts.", {
                id: "canvas-presence",
                duration: 6000,
                description: "Opening your own copy so both devices keep their work.",
            })
            // Let the toast paint, then move this tab onto the fork. A full
            // navigation rebinds the editor cleanly to the new project id.
            setTimeout(() => {
                if (typeof window !== "undefined") window.location.href = `/editor/${forkProjectId}`
            }, 1200)
        } else {
            // Couldn't fork (e.g. free-plan project limit) — fall back to a warning
            // so the user can resolve it manually; the reactive flush-conflict path
            // is still the backstop if they keep editing.
            toast.warning("This project is open on another device.", {
                id: "canvas-presence",
                duration: Infinity,
                dismissible: true,
                description: "Editing it here too may overwrite the other device's changes. Couldn't auto-create a separate copy (project limit reached).",
            })
        }
    }, [forkLiveCanvasToProject])

    const onConcurrentRef = useRef(() => {})
    onConcurrentRef.current = handleConcurrentDevice

    const saveCanvasState = useCallback(async ({ rethrow = false, immediate = false } = {}) => {
        const canvas = canvasInstanceRef.current
        const proj = projectRef.current
        if (!canvas || !proj) return

        const canvasJSON = serializeCanvasState(canvas)
        const currentImageUrl = getPrimaryRemoteImageUrl(canvas)
        let fullState = {
            ...canvasJSON,
            history: historyRef.current.slice(-MAX_PERSISTED_HISTORY),
            historyIndex: historyIndexRef.current,
        }
        // Large projects (lots of objects + many history entries) can push the
        // serialized JSON over MAX_NEON_STATE_CHARS, which would make Neon
        // reject the write. Trim history with a sliding window — drop the
        // OLDEST entries — until the JSON fits. We always keep at least
        // MIN_PERSISTED_HISTORY_ENTRIES so the user can still undo a few steps
        // even on a very large project.
        //
        // Previously this branch wiped the entire history (history: [],
        // historyIndex: -1), which left the user with no undo at all after the
        // first save of a large project. See Bug G.
        if (fullState.history.length > 0 && JSON.stringify(fullState).length > MAX_NEON_STATE_CHARS) {
            const originalHistoryLength = fullState.history.length
            let persistedHistory = fullState.history
            while (
                persistedHistory.length > MIN_PERSISTED_HISTORY_ENTRIES
                && JSON.stringify({ ...fullState, history: persistedHistory }).length > MAX_NEON_STATE_CHARS
            ) {
                persistedHistory = persistedHistory.slice(1)
            }
            const droppedCount = originalHistoryLength - persistedHistory.length
            if (droppedCount > 0) {
                console.warn(
                    `[canvas] History trimmed for storage: dropped ${droppedCount} oldest entries ` +
                    `(${originalHistoryLength} → ${persistedHistory.length}) to fit within ` +
                    `${MAX_NEON_STATE_CHARS} chars. The in-memory undo stack is unaffected.`
                )
            }
            fullState = {
                ...fullState,
                history: persistedHistory,
            }
        }

        // Delegate persistence to the sync manager: it mirrors to IndexedDB
        // FIRST (durable across reload/disconnect/tab-close), dedups identical
        // states, single-flights, and — when online — writes through the cache to
        // Neon, falling back to a direct Neon write when the cache is unavailable.
        // When offline it keeps the local copy and replays it on reconnect.
        const manager = syncRef.current
        if (!manager) {
            // Manager not constructed yet (very first paint) — write directly so
            // the initial state isn't lost.
            try {
                await directWriteRef.current(fullState, currentImageUrl)
            } catch (error) {
                if (rethrow) throw error
                console.error("Error saving canvas state", error)
            }
            return
        }
        try {
            await manager.save(fullState, currentImageUrl, { immediate, rethrow })
        } catch (error) {
            if (error?.status === 401) {
                console.warn("Transient 401 saving canvas state", error.message)
            } else {
                console.error("Error saving canvas state", error)
            }
            if (rethrow) throw error
        }
    }, [])

    // Keep the direct-Neon writer current so the per-project sync manager (built
    // once) always calls the latest updateProject mutation.
    useEffect(() => {
        directWriteRef.current = async (fullState, currentImageUrl) => {
            const proj = projectRef.current
            if (!proj) return
            await updateProject({
                projectId: proj._id,
                canvasState: fullState,
                ...(currentImageUrl ? { currentImageUrl } : {}),
            })
        }
    }, [updateProject])

    // One sync manager per project. On unmount or projectId change, flush any
    // pending writes (best-effort) and tear down its listeners/timers so the next
    // editor session starts clean.
    useEffect(() => {
        const projectId = project?._id
        if (!projectId) return
        const manager = createCanvasSync({
            projectId,
            snapshotFn: snapshotToCache,
            flushFn: flushToNeon,
            onStatus: handleSyncStatus,
            onConflict: (serverProject) => onConflictRef.current(serverProject),
            initialRevision: Number(projectRef.current?.revision) || 0,
            flushDebounceMs: FLUSH_DEBOUNCE_MS,
            minSnapshotIntervalMs: MIN_SNAPSHOT_INTERVAL_MS,
        })
        syncRef.current = manager
        return () => {
            manager.flushNow()
            manager.destroy()
            // Dismiss the conflict toast so it doesn't persist on navigation
            // (it has duration: Infinity and would follow the user to the dashboard).
            toast.dismiss("canvas-conflict")
            if (syncRef.current === manager) syncRef.current = null
        }
    }, [project?._id, handleSyncStatus])

    // Concurrent-device presence: heartbeat so we learn the instant the same
    // project is opened on another device, and react via the stable ref handler.
    useEffect(() => {
        const projectId = project?._id
        if (!projectId) return
        const channel = createPresenceChannel({
            projectId,
            onConcurrent: (info) => onConcurrentRef.current(info),
        })
        return () => {
            channel.stop()
            toast.dismiss("canvas-presence")
        }
    }, [project?._id])

    // Host the AI Extend background poller. Lives at canvas level (not in the
    // extender panel) so a pending genfill keeps polling and can swap the soft
    // preview for the real result no matter which tool is open — and resumes
    // after a reload (the poller persists pending jobs to sessionStorage).
    // Bind the change journal to this project so the History panel shows the
    // right log (persisted per project in sessionStorage).
    useEffect(() => {
        if (project?._id) setJournalProject(project._id)
    }, [project?._id])

    useEffect(() => {
        const projectId = project?._id
        if (!canvasEditor || !projectId) return
        return registerExtendHost({
            projectId,
            getCanvas: () => canvasEditor,
            save: async (id, currentImageUrl, canvas) => {
                await updateProject({
                    projectId: id,
                    currentImageUrl,
                    canvasState: serializeCanvasState(canvas),
                })
            },
        })
    }, [canvasEditor, project?._id, updateProject])

    // beforeunload + pagehide: capture the very latest state before the user
    // navigates away. The manager beacons it to Redis (surviving unload) and
    // keepalive-flushes to Neon, so edits made inside the autosave window aren't
    // lost; the IndexedDB mirror is the backstop if the beacon is dropped.
    useEffect(() => {
        const projectId = project?._id
        if (!projectId) return

        // Page is going away for good (real navigation / tab close): fire-and-forget
        // IDB update then beacon whatever latest the manager holds. Can't await in
        // unload handlers — the beacon + IDB mirror are the backstops.
        const persistOnUnload = () => {
            saveCanvasState().catch(() => {})
            try { syncRef.current?.beaconUnload() } catch { /* ignore */ }
        }

        const persistWhileAlive = () => {
            // Only flush when there's actually dirty/unsynced state. Skipping
            // when nothing is pending avoids bumping the server revision on every
            // tab/window/screen switch, which was the root cause of spurious
            // "edited elsewhere" conflict notifications in single-session use.
            if (!syncRef.current?.hasPendingSync()) return
            saveCanvasState({ immediate: true }).catch(() => {})
        }

        const onBeforeUnload = () => persistOnUnload()
        const onPageHide = (e) => {
            // A persisted (bfcache) pagehide may be restored alive → keep the
            // manager coherent; a non-persisted one is a real unload.
            if (e?.persisted) persistWhileAlive()
            else persistOnUnload()
        }
        const onVisibility = () => {
            if (typeof document !== "undefined" && document.visibilityState === "hidden") {
                persistWhileAlive()
            }
        }
        window.addEventListener("beforeunload", onBeforeUnload)
        window.addEventListener("pagehide", onPageHide)
        document.addEventListener("visibilitychange", onVisibility)
        return () => {
            window.removeEventListener("beforeunload", onBeforeUnload)
            window.removeEventListener("pagehide", onPageHide)
            document.removeEventListener("visibilitychange", onVisibility)
        }
    }, [project?._id, saveCanvasState])

    useEffect(() => {
        if (!canvasRef.current || !projectRef.current) return

        const initGen = ++initGenerationRef.current
        let mounted = true

        disposeCanvasInstance()
        historyRef.current = []
        historyIndexRef.current = -1

        const initializeCanvas = async () => {
            if (initGen !== initGenerationRef.current || !mounted) return

            setIsLoading(true)
            const proj = projectRef.current
            if (!proj) return
            const { width, height } = getContainerSize()
            const el = canvasRef.current
            if (!el) return

            const canvas = new Canvas(el, {
                width: width || proj.width, height: height || proj.height,
                backgroundColor: "transparent",
                preserveObjectStacking: true, controlsAboveOverlay: true, selection: true,
                hoverCursor: "move", moveCursor: "move", defaultCursor: "default",
                allowTouchScrolling: false, renderOnAddRemove: false, skipTargetFind: false,
            })

            if (initGen !== initGenerationRef.current || !mounted) {
                canvas.dispose()
                return
            }

            canvasInstanceRef.current = canvas
            canvas.setDimensions({ width: width || proj.width, height: height || proj.height }, { backstoreOnly: false })

            // Read-through: if the server cache has a newer snapshot than what's
            // in Neon (because a previous session ended with pending debounced
            // writes that haven't been flushed yet), prefer that. Otherwise the
            // user would see an old version of their work after a reload.
            let rawCanvasState = proj.canvasState
            let effectiveCurrentImageUrl = proj.currentImageUrl
            let bestUpdatedAt = Number(proj.updatedAt) || 0
            // The revision the WINNING content is based on. Neon content → the
            // project's current revision; an unflushed Redis/IDB snapshot → the
            // revision IT was based on. The sync manager must flush against this,
            // not always the current Neon revision, or replaying older unflushed
            // content would clobber a newer concurrent write.
            let effectiveBaseRevision = Number(proj.revision) || 0
            try {
                const cachedSnapshot = await fetchCachedSnapshot(proj._id)
                if (cachedSnapshot?.canvasState) {
                    // Prefer the server-stamped time (comparable to Neon's
                    // server-set updatedAt) so a skewed client clock can't make a
                    // Redis snapshot wrongly win or lose against Neon. Older
                    // snapshots without it fall back to the client time.
                    const cachedUpdatedAt = Number(cachedSnapshot.serverUpdatedAt ?? cachedSnapshot.updatedAt) || 0
                    if (cachedUpdatedAt > bestUpdatedAt) {
                        rawCanvasState = cachedSnapshot.canvasState
                        if (cachedSnapshot.currentImageUrl) {
                            effectiveCurrentImageUrl = cachedSnapshot.currentImageUrl
                        }
                        bestUpdatedAt = cachedUpdatedAt
                        if (Number.isFinite(Number(cachedSnapshot.baseRevision))) {
                            effectiveBaseRevision = Number(cachedSnapshot.baseRevision)
                        }
                    }
                }
            } catch (cacheError) {
                console.warn("[canvas] cached snapshot lookup failed:", cacheError?.message || cacheError)
            }

            // Offline backstop: a prior session may have ended offline (or closed
            // before the unload beacon landed), leaving the freshest work only in
            // the local IndexedDB mirror. If it's newer than both Neon and Redis,
            // restore from it — the sync manager replays it to the server on
            // reconnect / next edit.
            try {
                const localState = await loadLocalState(proj._id)
                // Only let the local mirror override server state when it holds
                // UNSYNCED work (dirty). A clean local copy already reached the
                // server, so prefer the server — this stops a skewed client clock
                // from clobbering newer server state across devices.
                if (localState?.fullState && localState.dirty !== false) {
                    const localUpdatedAt = Number(localState.updatedAt) || 0
                    if (localUpdatedAt > bestUpdatedAt) {
                        rawCanvasState = localState.fullState
                        if (localState.currentImageUrl) {
                            effectiveCurrentImageUrl = localState.currentImageUrl
                        }
                        bestUpdatedAt = localUpdatedAt
                        // Only adopt the local IDB revision if it's >= the
                        // project's current server revision. If the local
                        // revision is OLDER, the data was already flushed (e.g.
                        // via beacon on tab close) but IDB wasn't cleaned up.
                        // Adopting the stale revision would make the first flush
                        // send baseRevision=OLD → server has NEW → spurious conflict.
                        const localRev = Number(localState.baseRevision)
                        const serverRev = Number(proj.revision) || 0
                        if (Number.isFinite(localRev) && localRev >= serverRev) {
                            effectiveBaseRevision = localRev
                        }
                    }
                }
            } catch (localError) {
                console.warn("[canvas] local snapshot lookup failed:", localError?.message || localError)
            }

            // Tell the sync manager which revision the loaded content is based on,
            // so its first flush is checked against the right baseline.
            try { syncRef.current?.setBaseRevision(effectiveBaseRevision) } catch { /* manager may not exist yet */ }

            const canvasState = normalizeCanvasState(rawCanvasState)
            const persistedHistory = Array.isArray(rawCanvasState?.history) ? rawCanvasState.history : null
            let hasRestoredViewport = false

            if (!canvasState && (effectiveCurrentImageUrl || proj.originalImageUrl)) {
                try {
                    const imageUrl = effectiveCurrentImageUrl || proj.originalImageUrl
                    const fabricImage = await FabricImage.fromURL(imageUrl, { crossOrigin: "anonymous" })
                    fitImageInsideProject(fabricImage, proj)
                    canvas.add(fabricImage)
                } catch (error) { console.error("Error loading project image:", error) }
            }

            if (canvasState) {
                let loadedFromState = false
                try {
                    await canvas.loadFromJSON(canvasState.canvas || canvasState)
                    // Restore the "grade background" intent so it keeps tracking after reload.
                    canvas.__phosmithGradeBackground = Boolean(canvasState.gradeBackground)
                    removeExpansionFramesFromCanvas(canvas)
                    if (canvasState.viewport) { setViewportState(canvas, canvasState.viewport, { x: proj.width / 2, y: proj.height / 2 }); hasRestoredViewport = true }
                    const imageUrl = effectiveCurrentImageUrl || proj.originalImageUrl
                    await hydrateCanvasImages(canvas, imageUrl, {
                        forcePrimaryImageUrl: true,
                        canvasSize: { width: proj.width, height: proj.height },
                    })
                    for (const obj of canvas.getObjects()) {
                        if (obj?.type?.toLowerCase() === 'image' && obj.filters?.length) {
                            try {
                                obj.applyFilters?.()
                            } catch (filterError) {
                                // Silently swallowing here once hid a real curves regression
                                // for hours. Keep going (don't break the load) but log so
                                // the next failure is diagnosable.
                                console.error('[canvas] applyFilters failed for image:', filterError)
                            }
                        }
                    }
                    canvas.requestRenderAll()
                    loadedFromState = canvas.getObjects().length > 0
                } catch (error) { console.error("Error loading canvas state: ", error) }

                // Fallback: if loadFromJSON threw or produced an empty canvas (e.g. a
                // saved filter type Fabric can no longer enliven), still show the project
                // image so the user doesn't stare at a blank canvas.
                if (!loadedFromState) {
                    const imageUrl = effectiveCurrentImageUrl || proj.originalImageUrl
                    if (imageUrl && canvas.getObjects().length === 0) {
                        try {
                            const fallbackImage = await FabricImage.fromURL(imageUrl, { crossOrigin: "anonymous" })
                            fitImageInsideProject(fallbackImage, proj)
                            canvas.add(fallbackImage)
                            canvas.requestRenderAll()
                        } catch (error) { console.error("Fallback image load failed:", error) }
                    }
                }
            }

            // Always fit the project to the current viewport on load so the
            // canvas content is centered with proper margins, regardless of
            // the viewport state that was saved (which may have been for a
            // different window/container size).
            createInitialViewport(canvas)
            if (initGen !== initGenerationRef.current || !mounted) {
                canvas.dispose()
                canvasInstanceRef.current = null
                return
            }

            canvas.renderOnAddRemove = true
            canvas.calcOffset()
            canvas.requestRenderAll()
            setCanvasEditor(canvas)

            const isExpansionMode = () =>
                activeToolRef.current === 'ai_extender' ||
                Boolean(canvas.__expansionMode)

            const isTypingTarget = (target) => {
                if (!target) return false
                const tag = target.tagName
                if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true
                if (target.isContentEditable) return true
                return false
            }

            const isPanModifierActive = () =>
                spacePressedRef.current ||
                handToolActiveRef.current

            const isMiddleButtonDrag = (event) =>
                event?.button === 1 ||
                event?.buttons === 4

            const shouldStartPan = (opt) => {
                if (isExpansionMode()) return false
                const event = opt?.e
                if (isMiddleButtonDrag(event)) return true
                if (opt?.target) return false
                return isPanModifierActive()
            }

            const applyCursorForMode = () => {
                if (isExpansionMode()) {
                    canvas.skipTargetFind = false
                    canvas.defaultCursor = 'default'
                    canvas.hoverCursor = 'default'
                    canvas.moveCursor = 'default'
                    canvas.upperCanvasEl.style.cursor = 'default'
                    return
                }
                const wantsPan = isPanModifierActive()
                if (
                    canvas.__pixelToolActive ||
                    activeToolRef.current === 'mask' ||
                    activeToolRef.current === 'erase'
                ) {
                    const cursor = wantsPan ? 'grab' : 'crosshair'
                    canvas.skipTargetFind = true
                    canvas.defaultCursor = cursor
                    canvas.hoverCursor = cursor
                    canvas.moveCursor = cursor
                    canvas.upperCanvasEl.style.cursor = cursor
                    return
                }
                canvas.skipTargetFind = false
                canvas.defaultCursor = wantsPan ? 'grab' : 'default'
                canvas.hoverCursor = 'move'
                canvas.moveCursor = 'move'
                canvas.upperCanvasEl.style.cursor = wantsPan ? 'grab' : 'default'
            }

            const endPanning = () => {
                isPanningRef.current = false
                lastPointerRef.current = null
                applyCursorForMode()
            }

            const resetPanInputState = ({ resetHandTool = false } = {}) => {
                isPanningRef.current = false
                lastPointerRef.current = null
                ctrlPressedRef.current = false
                spacePressedRef.current = false
                if (resetHandTool) {
                    handToolActiveRef.current = false
                    setIsHandToolActive(false)
                }
                applyCursorForMode()
            }

            const handleKeyDown = (event) => {
                if (isExpansionMode()) return
                if (event.key === ' ' && !event.repeat && !isTypingTarget(event.target)) {
                    spacePressedRef.current = true
                    applyCursorForMode()
                    event.preventDefault()
                    return
                }
                if (event.key === 'Control' && !event.repeat) {
                    ctrlPressedRef.current = true
                }
            }
            const handleKeyUp = (event) => {
                if (event.key === ' ') {
                    spacePressedRef.current = false
                    endPanning()
                    return
                }
                if (event.key === 'Control') {
                    ctrlPressedRef.current = false
                }
            }
            const handleMouseDown = (opt) => {
                if (shouldStartPan(opt)) {
                    isPanningRef.current = true
                    lastPointerRef.current = { x: opt.e.clientX, y: opt.e.clientY }
                    // Cursor is a DOM style change (no canvas render needed). The
                    // viewport hasn't moved yet, so a render here would paint nothing
                    // new — the first mouse:move pans and renders.
                    canvas.upperCanvasEl.style.cursor = 'grabbing'
                    opt.e.preventDefault()
                    opt.e.stopPropagation()
                }
            }
            const handleMouseMove = (opt) => {
                if (isExpansionMode() || !isPanningRef.current || !lastPointerRef.current) return
                const deltaX = opt.e.clientX - lastPointerRef.current.x
                const deltaY = opt.e.clientY - lastPointerRef.current.y
                canvas.relativePan(new Point(deltaX, deltaY))
                lastPointerRef.current = { x: opt.e.clientX, y: opt.e.clientY }
                canvas.requestRenderAll()
            }
            const handleMouseUp = () => {
                if (isExpansionMode()) return
                endPanning()
            }
            canvas.__setHandToolActive = (active) => {
                handToolActiveRef.current = Boolean(active)
                applyCursorForMode()
            }
            canvas.__syncPanCursor = applyCursorForMode
            const handleMouseWheel = (opt) => {
                if (isExpansionMode()) return
                if (!(opt.e.ctrlKey || ctrlPressedRef.current)) return
                opt.e.preventDefault()
                const zoom = clamp(canvas.getZoom() * Math.pow(0.999, opt.e.deltaY), MIN_ZOOM, MAX_ZOOM)
                const pointer = canvas.getViewportPoint(opt.e)
                canvas.zoomToPoint(new Point(pointer.x, pointer.y), zoom)
                canvas.requestRenderAll()
            }
            const handleWindowPointerUp = () => endPanning()
            const handleWindowBlur = () => resetPanInputState({ resetHandTool: true })
            const handleVisibilityChange = () => {
                if (document.visibilityState === 'hidden') {
                    resetPanInputState({ resetHandTool: true })
                }
            }

            window.addEventListener('keydown', handleKeyDown)
            window.addEventListener('keyup', handleKeyUp)
            window.addEventListener('pointerup', handleWindowPointerUp)
            window.addEventListener('blur', handleWindowBlur)
            document.addEventListener('visibilitychange', handleVisibilityChange)
            canvas.on('mouse:down', handleMouseDown)
            canvas.on('mouse:move', handleMouseMove)
            canvas.on('mouse:up', handleMouseUp)
            canvas.on('mouse:wheel', handleMouseWheel)

            canvas.__cleanupInfiniteWorkspace = () => {
                window.removeEventListener('keydown', handleKeyDown)
                window.removeEventListener('keyup', handleKeyUp)
                window.removeEventListener('pointerup', handleWindowPointerUp)
                window.removeEventListener('blur', handleWindowBlur)
                document.removeEventListener('visibilitychange', handleVisibilityChange)
                canvas.off('mouse:down', handleMouseDown)
                canvas.off('mouse:move', handleMouseMove)
                canvas.off('mouse:up', handleMouseUp)
                canvas.off('mouse:wheel', handleMouseWheel)
                canvas.off('after:render', syncViewportChrome)
                delete canvas.__syncPanCursor
            }
            canvas.__undoCanvasState = () => undoCanvasState()
            canvas.__redoCanvasState = () => redoCanvasState()
            canvas.__pushHistoryState = (meta) => pushHistoryState(canvas, meta)
            canvas.__saveCanvasState = (opts) => saveCanvasState(opts)
            canvas.__getHistoryState = () => ({
                canUndo: historyIndexRef.current > 0,
                canRedo: historyIndexRef.current < historyRef.current.length - 1,
            })
            canvas.__fitCanvasToProject = (size) => {
                fitProjectToViewport(canvas, size)
                canvas.calcOffset()
                canvas.requestRenderAll()
                syncProjectFrame()
                syncPreviewZoomState(canvas)
            }
            canvas.__resetCanvasView = () => {
                fitProjectToViewport(canvas)
                canvas.calcOffset()
                canvas.requestRenderAll()
                syncProjectFrame()
                syncPreviewZoomState(canvas)
            }
            canvas.__setPreviewZoom = (percent) => setCanvasPreviewZoom(canvas, percent)
            canvas.__getPreviewZoom = () => readPreviewZoomPercent(canvas)
            // Doodle ↔ image binding — strokes drawn on a photo follow it when
            // it moves/scales/rotates. Exposed so the Resize tool (which scales
            // the image programmatically) can snapshot before and replay after.
            canvas.__bindDoodles = (image) => bindDoodlesToImage(canvas, image)
            canvas.__followDoodles = (image) => followDoodles(canvas, image)

            const syncProjectFrame = () => {
                const proj = projectRef.current
                if (!proj?.width || !proj?.height) {
                    if (projectFrameStyleRef.current.width !== 0) {
                        projectFrameStyleRef.current = { left: 0, top: 0, width: 0, height: 0 }
                        setProjectFrameStyle(projectFrameStyleRef.current)
                        setIsProjectFrameVisible(false)
                    }
                    return
                }
                const vpt = canvas.viewportTransform || [1, 0, 0, 1, 0, 0]
                const zoom = vpt[0] || 1
                const left = vpt[4]
                const top = vpt[5]
                const width = proj.width * zoom
                const height = proj.height * zoom
                const previous = projectFrameStyleRef.current
                if (
                    previous.left === left &&
                    previous.top === top &&
                    previous.width === width &&
                    previous.height === height
                ) return
                projectFrameStyleRef.current = { left, top, width, height }
                setProjectFrameStyle(projectFrameStyleRef.current)
                setIsProjectFrameVisible(true)
            }
            canvas.__syncProjectFrame = syncProjectFrame
            const syncViewportChrome = () => {
                syncProjectFrame()
                syncPreviewZoomState(canvas)
            }
            canvas.on('after:render', syncViewportChrome)
            syncViewportChrome()

            if (persistedHistory?.length) {
                historyRef.current = persistedHistory.slice(-MAX_PERSISTED_HISTORY)
                historyIndexRef.current = Math.min(
                    Math.max(0, rawCanvasState.historyIndex ?? historyRef.current.length - 1),
                    historyRef.current.length - 1
                )
            } else {
                // Seed the undo baseline without journaling — opening a project
                // is not a user change ("Canvas edit" used to appear on every
                // fresh project open).
                pushHistoryState(canvas, { silent: true })
            }
            emitHistoryChange(canvas)
            setIsLoading(false)
        }

        initializeCanvas()
        return () => {
            mounted = false
            initGenerationRef.current += 1
            disposeCanvasInstance()
        }
    }, [
        project?._id,
        createInitialViewport,
        disposeCanvasInstance,
        fitProjectToViewport,
        pushHistoryState,
        redoCanvasState,
        saveCanvasState,
        setCanvasEditor,
        setCanvasPreviewZoom,
        setViewportState,
        syncPreviewZoomState,
        undoCanvasState,
    ])

    useEffect(() => {
        const canvas = canvasInstanceRef.current
        if (!canvas) return
        canvas.__undoCanvasState = () => undoCanvasState()
        canvas.__redoCanvasState = () => redoCanvasState()
        canvas.__pushHistoryState = (meta) => pushHistoryState(canvas, meta)
        canvas.__saveCanvasState = () => saveCanvasState()
        canvas.__getHistoryState = () => ({
            canUndo: historyIndexRef.current > 0,
            canRedo: historyIndexRef.current < historyRef.current.length - 1,
        })
    }, [canvasEditor, undoCanvasState, redoCanvasState, pushHistoryState, saveCanvasState])

    // ─── Megashader filter wiring ────────────────────────────────────────────
    // The megashader is a stateful pipeline shared via the window event bus
    // (see useMaskLayers in /hooks). Whenever the mask layer stack changes,
    // we find the primary Fabric image on the canvas and install/update the
    // MegashaderFilter. The filter itself is a 2D passthrough that delegates
    // to a private WebGL2 context (see src/lib/megashader/), so installing it
    // doesn't disturb Fabric's filter chain for the curves/grade filters.
    //
    // BOTH refs are component-level so they survive `useEffect` re-runs
    // (which happen when `canvasEditor` changes). Pre-fix, the
    // `getLastAppliedStackRef` was a plain object declared INSIDE the
    // effect, so each re-run created a fresh one starting at `{ chain: [] }`
    // — the first `handleGlobalAlpha` after a re-run would re-apply with
    // an empty stack and silently drop the megashader. The component-level
    // `useRef` keeps the value stable across the effect's re-mounts.
    //
    // Step 10.2 — `recompileTimerRef` is a component-level `useRef`
    // holding the debounce timer for `handleLayersChanged`. Without
    // this, dragging a slider (e.g. per-layer exposure) fires
    // `phosmith:mask-layers-changed` on every step; the GLSL compiler
    // (50-200 ms for a complex chain) re-runs for every step, blocking
    // the main thread. The 150 ms debounce coalesces a rapid burst
    // into one recompile after the user pauses.
    const megashaderAlphaRef = useRef(1)
    // "Show mask" overlay + global invert — chain-wide render options driven
    // by the Mask tool via window events; mirror globalAlpha's ref pattern.
    const megashaderOverlayRef = useRef(false)
    const megashaderInvertRef = useRef(false)
    const getLastAppliedStackRef = useRef(/** @type {import('@/lib/megashader/mask-types').MaskStack} */ ({ chain: [] }))
    const recompileTimerRef = useRef(/** @type {ReturnType<typeof setTimeout> | null} */ (null))
    // Draft live-preview (grade drags): GPU overlay + rAF fast path.
    const previewSessionRef = useRef(/** @type {any} */ (null))
    const previewCommitTimerRef = useRef(/** @type {ReturnType<typeof setTimeout> | null} */ (null))
    const maskApplyRafRef = useRef(/** @type {number | null} */ (null))
    const lastStructuralSigRef = useRef(/** @type {string | null} */ (null))
    useEffect(() => {
        if (typeof window === 'undefined') return undefined

        const findPrimaryImage = () => {
            const canvas = canvasInstanceRef.current
            if (!canvas) return null
            const objects = canvas.getObjects?.() || []
            return objects.find(
                (obj) => obj?.type?.toLowerCase?.() === 'image' && !isPhosmithMaskOverlay(obj)
            ) || null
        }

        // The actual recompile — factored out so the debounced wrapper
        // and `handleGlobalAlpha` both call the same code path.
        const doApply = (stack) => {
            const image = findPrimaryImage()
            if (!image) return
            // Lazy-load the megashader module so the production editor bundle
            // never pulls in the GLSL compiler unless the dev test panel
            // (or a Step 2+ tool) actually subscribes.
            import('@/lib/megashader')
                .then((mod) => {
                    // Split the dynamic-import failure (module genuinely absent
                    // in a non-test build — safe to ignore) from a failure
                    // INSIDE applyMegashaderFilter. The latter used to be
                    // swallowed here, which silently hid a real render bug (the
                    // MegashaderFilter constructor threw on every call, so no
                    // mask layer ever rendered). Surface apply errors loudly.
                    try {
                        mod.applyMegashaderFilter(image, stack, {
                            globalMaskAlpha: megashaderAlphaRef.current,
                            globalInvert: megashaderInvertRef.current,
                            maskOverlay: megashaderOverlayRef.current,
                        })
                        canvasInstanceRef.current?.requestRenderAll?.()
                    } catch (err) {
                        console.error('[megashader] applyMegashaderFilter failed:', err)
                    }
                })
                .catch(() => { /* noop — module not available in non-test paths */ })
        }

        // ── Draft-preview session ─────────────────────────────────────
        // The exact pipeline (doApply → applyFilters → megashader render →
        // readPixels → getImageData) moves ~5 full-resolution buffers per
        // tick — linear in megapixels, so grade drags on large photos drop
        // to a slideshow. During a drag we instead render the megashader
        // from a capped-resolution snapshot into a DOM <canvas> overlaid
        // exactly on the image (GPU work + one small readback per frame,
        // independent of source size), and run the exact full-res pipeline
        // ONCE when the drag goes idle. Lightroom-style draft preview.
        const MAX_PREVIEW_DIM = 1280
        const PREVIEW_COMMIT_IDLE_MS = 160

        const startPreviewSession = () => {
            const canvas = canvasInstanceRef.current
            const image = findPrimaryImage()
            if (!canvas || !image || !canvas.wrapperEl) return null
            // An axis-aligned overlay can't represent a rotated image —
            // fall back to the exact per-tick pipeline (correct, slower).
            if (Math.abs((image.angle || 0) % 360) > 0.01) return null

            // Clean source = all prior filters, WITHOUT the megashader.
            // When the megashader is the only filter this is simply the
            // original element (free); otherwise re-run the chain once with
            // the megashader disabled — a single full-res cost at drag
            // START, not per tick.
            let cleanSource = image._originalElement
            const hasOtherFilters = Array.isArray(image.filters)
                && image.filters.some((f) => f && f.type !== 'Megashader')
            if (hasOtherFilters) {
                const mega = image.filters.find((f) => f && f.type === 'Megashader')
                if (mega) {
                    mega.enabled = false
                    try { image.applyFilters() } catch { /* keep going */ }
                    mega.enabled = true
                }
                cleanSource = image._element || cleanSource
            }
            const srcW = cleanSource?.naturalWidth || cleanSource?.width || 0
            const srcH = cleanSource?.naturalHeight || cleanSource?.height || 0
            if (!srcW || !srcH) return null

            const scale = Math.min(1, MAX_PREVIEW_DIM / Math.max(srcW, srcH))
            const srcCanvas = document.createElement('canvas')
            srcCanvas.width = Math.max(1, Math.round(srcW * scale))
            srcCanvas.height = Math.max(1, Math.round(srcH * scale))
            const sctx = srcCanvas.getContext('2d')
            if (!sctx) return null
            sctx.imageSmoothingEnabled = true
            sctx.imageSmoothingQuality = 'high'
            sctx.drawImage(cleanSource, 0, 0, srcCanvas.width, srcCanvas.height)

            const overlayEl = document.createElement('canvas')
            overlayEl.className = 'phosmith-megashader-preview'
            overlayEl.style.position = 'absolute'
            overlayEl.style.pointerEvents = 'none'
            // Under the interaction canvas (selection borders, brush cursor
            // stay visible), over the lower canvas that shows the image.
            canvas.wrapperEl.insertBefore(overlayEl, canvas.upperCanvasEl || null)
            const session = {
                overlayEl,
                ctx: overlayEl.getContext('2d'),
                srcCanvas,
                image,
            }
            previewSessionRef.current = session
            return session
        }

        const removePreviewOverlay = (session) => {
            try { session.overlayEl.remove() } catch { /* already gone */ }
        }

        const endPreviewSession = (commit) => {
            const session = previewSessionRef.current
            if (!session) return
            previewSessionRef.current = null
            if (previewCommitTimerRef.current !== null) {
                clearTimeout(previewCommitTimerRef.current)
                previewCommitTimerRef.current = null
            }
            if (!commit) {
                removePreviewOverlay(session)
                return
            }
            // Exact full-res commit; drop the overlay only after fabric has
            // painted the committed pixels so the swap is seamless.
            doApply(getLastAppliedStackRef.current)
            const canvas = canvasInstanceRef.current
            let removed = false
            const removeOnce = () => {
                if (removed) return
                removed = true
                removePreviewOverlay(session)
            }
            if (canvas?.once) canvas.once('after:render', removeOnce)
            // Safety net — never leave a stale overlay glued to the DOM.
            setTimeout(removeOnce, 600)
        }

        const renderPreviewFrame = (stack) => {
            const session = previewSessionRef.current
            const canvas = canvasInstanceRef.current
            if (!session || !canvas) return
            const { image, overlayEl, ctx, srcCanvas } = session
            if (!ctx || !canvas.getObjects?.().includes(image)) {
                endPreviewSession(false)
                return
            }
            import('@/lib/megashader').then((mod) => {
                // Guard: session may have committed while the module loaded.
                if (previewSessionRef.current !== session) return
                try {
                    // Position the overlay on the image's on-screen quad.
                    // oCoords are fabric's own viewport-space (CSS px)
                    // corners — exact under any pan/zoom, refreshed here so
                    // a mid-drag zoom can't leave the overlay stranded.
                    image.setCoords()
                    const { tl, tr, bl } = image.oCoords || {}
                    if (!tl || !tr || !bl) { endPreviewSession(false); return }
                    const cssW = Math.max(1, tr.x - tl.x)
                    const cssH = Math.max(1, bl.y - tl.y)
                    overlayEl.style.left = `${tl.x}px`
                    overlayEl.style.top = `${tl.y}px`
                    overlayEl.style.width = `${cssW}px`
                    overlayEl.style.height = `${cssH}px`
                    // Backing store: preview-res is enough while dragging —
                    // scale to the smaller of screen size and preview size.
                    const dpr = window.devicePixelRatio || 1
                    const bw = Math.max(1, Math.min(Math.round(cssW * dpr), srcCanvas.width))
                    const bh = Math.max(1, Math.min(Math.round(cssH * dpr), srcCanvas.height))
                    if (overlayEl.width !== bw) overlayEl.width = bw
                    if (overlayEl.height !== bh) overlayEl.height = bh

                    const result = mod.renderMegashader(srcCanvas, stack, {
                        globalMaskAlpha: megashaderAlphaRef.current,
                        globalInvert: megashaderInvertRef.current,
                        maskOverlay: megashaderOverlayRef.current,
                    })
                    if (!result) return
                    ctx.imageSmoothingEnabled = true
                    ctx.clearRect(0, 0, overlayEl.width, overlayEl.height)
                    ctx.drawImage(result, 0, 0, overlayEl.width, overlayEl.height)
                } catch (err) {
                    console.warn('[megashader] preview frame failed, committing full-res:', err)
                    endPreviewSession(true)
                }
            }).catch((err) => {
                console.warn('[megashader] preview module load failed:', err)
                endPreviewSession(false)
            })
        }

        const schedulePreviewCommit = () => {
            if (previewCommitTimerRef.current !== null) {
                clearTimeout(previewCommitTimerRef.current)
            }
            previewCommitTimerRef.current = setTimeout(() => {
                previewCommitTimerRef.current = null
                endPreviewSession(true)
            }, PREVIEW_COMMIT_IDLE_MS)
        }

        // Structural signature — mirrors `computeCacheKey()` in
        // src/lib/megashader/megashader-compiler.js (length|kinds|ops).
        // Grade values (gamma/wheels/curves/adjust sliders) are uniforms
        // and never change it. Deliberately NOT imported from
        // '@/lib/megashader': that module stays lazy-loaded (see doApply)
        // so the editor bundle never pulls in the GLSL compiler. Keep in
        // sync with computeCacheKey if mask kinds/ops semantics change.
        const structuralSignature = (stack) => {
            const chain = Array.isArray(stack?.chain) ? stack.chain : []
            return `${chain.length}|${chain.map((e) => e?.layer?.kind).join(',')}|${chain.map((e) => e?.op).join(',')}`
        }

        // Uniform-only fast path: one trailing rAF per frame; it reads the
        // latest stack at fire time, so the drag's release value always
        // lands and a pending frame can never apply a stale snapshot.
        // Inside the rAF, prefer the draft-preview session (GPU-only,
        // resolution-capped, instantaneous at any source size); fall back
        // to the exact per-tick pipeline when a session can't be built
        // (no image, rotated image, missing wrapper).
        const scheduleImmediateApply = () => {
            if (maskApplyRafRef.current !== null) return
            maskApplyRafRef.current = requestAnimationFrame(() => {
                maskApplyRafRef.current = null
                const stack = getLastAppliedStackRef.current
                const session = previewSessionRef.current || startPreviewSession()
                if (session) {
                    renderPreviewFrame(stack)
                    schedulePreviewCommit()
                } else {
                    doApply(stack)
                }
            })
        }

        // Step 10.2: debounced entry point. Cancels any pending
        // recompile and schedules a new one 150 ms out. The 150 ms
        // window is short enough to feel instant on slider release
        // but long enough to coalesce a fast slider drag into a
        // single recompile.
        const RECOMPILE_DEBOUNCE_MS = 150
        const handleLayersChanged = () => {
            // A structural change mid-drag invalidates the preview's
            // compiled assumptions — drop the overlay; the debounced
            // full-res apply below repaints the truth.
            endPreviewSession(false)
            if (recompileTimerRef.current !== null) {
                clearTimeout(recompileTimerRef.current)
            }
            recompileTimerRef.current = setTimeout(() => {
                recompileTimerRef.current = null
                // Apply the LATEST stack, not the triggering event's — uniform
                // drags may have landed via the fast path while this debounce
                // was pending, and re-applying an older snapshot would revert them.
                doApply(getLastAppliedStackRef.current)
            }, RECOMPILE_DEBOUNCE_MS)
        }

        const handleGlobalAlpha = (event) => {
            const value = event?.detail?.value
            if (typeof value !== 'number') return
            megashaderAlphaRef.current = Math.max(0, Math.min(1, value))
            // Alpha is a uniform, not a shader-struct change — skip
            // the debounce so the slider feels live. The shader
            // recompile is gated on the LRU cache, so if the program
            // was already compiled for this exact stack, the second
            // call is just a `applyFilters` re-run.
            scheduleImmediateApply()
        }

        // "Show mask" overlay + global invert — chain-wide render options.
        // Like alpha, they're uniforms (no recompile), so re-apply live
        // against the last-applied stack.
        const handleOverlay = (event) => {
            megashaderOverlayRef.current = Boolean(event?.detail?.value)
            scheduleImmediateApply()
        }
        const handleInvert = (event) => {
            megashaderInvertRef.current = Boolean(event?.detail?.value)
            scheduleImmediateApply()
        }

        const wrapped = (event) => {
            const stack = event?.detail?.stack || { chain: [] }
            getLastAppliedStackRef.current = stack
            // Uniform-only edits (grade slider drags: gamma/wheels/curves/
            // adjust) keep the same length|kinds|ops signature — no recompile,
            // so take the live rAF preview path. A changed signature (layer
            // add/remove/reorder/op) is structural → debounced full apply.
            const sig = structuralSignature(stack)
            const structural = lastStructuralSigRef.current === null || sig !== lastStructuralSigRef.current
            lastStructuralSigRef.current = sig
            if (structural) {
                handleLayersChanged()
            } else {
                scheduleImmediateApply()
            }
        }

        window.addEventListener('phosmith:mask-layers-changed', wrapped)
        window.addEventListener('phosmith:mask-global-alpha', handleGlobalAlpha)
        window.addEventListener('phosmith:mask-overlay', handleOverlay)
        window.addEventListener('phosmith:mask-invert', handleInvert)

        return () => {
            window.removeEventListener('phosmith:mask-layers-changed', wrapped)
            window.removeEventListener('phosmith:mask-global-alpha', handleGlobalAlpha)
            window.removeEventListener('phosmith:mask-overlay', handleOverlay)
            window.removeEventListener('phosmith:mask-invert', handleInvert)
            // Cancel the live-preview rAF/commit timers and drop any
            // overlay so a mid-drag unmount can't leave a stale <canvas>
            // glued to the DOM or fire an apply against a torn-down renderer.
            if (maskApplyRafRef.current !== null) {
                cancelAnimationFrame(maskApplyRafRef.current)
                maskApplyRafRef.current = null
            }
            endPreviewSession(false)
            // Step 10.2: clear any pending debounced recompile so a
            // canvasEditor change mid-debounce doesn't fire a stale
            // apply against a torn-down renderer.
            if (recompileTimerRef.current !== null) {
                clearTimeout(recompileTimerRef.current)
                recompileTimerRef.current = null
            }
        }
    }, [canvasEditor])

    // Register the UI-decoupled mask command surface so the in-app agent can
    // drive masking headlessly (NOT wired to any agent yet — see
    // src/lib/agent/). Resolves the active primary image from the live canvas
    // on each call so commands always target the current image.
    useEffect(() => {
        let unregisterMask = () => {}
        let unregisterCrop = () => {}
        let unregisterCollage = () => {}
        let cancelled = false
        Promise.all([
            import('@/lib/agent/command-registry'),
            import('@/lib/agent/mask-commands'),
            import('@/lib/agent/crop-commands'),
            import('@/lib/agent/collage-commands'),
        ]).then(([reg, mask, crop, collage]) => {
            if (cancelled) return
            const getPrimaryImage = () => {
                const canvas = canvasInstanceRef.current
                if (!canvas) return null
                const objects = canvas.getObjects?.() || []
                return objects.find(
                    (obj) => obj?.type?.toLowerCase?.() === 'image' && !isPhosmithMaskOverlay(obj)
                ) || null
            }
            const getCanvas = () => canvasInstanceRef.current
            const getProject = () => projectRef.current
            unregisterMask = reg.registerDomain('mask', mask.createMaskCommands({ getPrimaryImage }))
            unregisterCrop = reg.registerDomain('crop', crop.createCropCommands({ getPrimaryImage, getCanvas }))
            unregisterCollage = reg.registerDomain('collage', collage.createCollageCommands({ getCanvas, getProject }))
        }).catch(() => { /* agent layer optional */ })
        return () => { cancelled = true; unregisterMask(); unregisterCrop(); unregisterCollage() }
    }, [])

    // Track the last-hydrated URL so we skip redundant re-hydrations when
    // the parent re-renders (e.g. during sidebar resize) without the image
    // URL actually changing. This prevents the "image refreshing" flicker.
    const lastHydratedUrlRef = useRef(null)

    useEffect(() => {
        const canvas = canvasInstanceRef.current
        const imageUrl = project?.currentImageUrl || project?.originalImageUrl
        if (!canvas || !imageUrl) return

        // Guard: skip if we've already hydrated this exact URL
        if (lastHydratedUrlRef.current === imageUrl) return
        lastHydratedUrlRef.current = imageUrl

        let cancelled = false
        hydrateCanvasImages(canvas, imageUrl, {
            forcePrimaryImageUrl: true,
            canvasSize: { width: project.width, height: project.height },
        }).then(() => {
            if (!cancelled) canvas.requestRenderAll()
        })

        return () => {
            cancelled = true
        }
    }, [canvasEditor, project?.currentImageUrl, project?.originalImageUrl, project?.width, project?.height])

    useEffect(() => {
        const canvas = canvasInstanceRef.current
        if (!canvas?.upperCanvasEl) return

        if (activeTool === 'ai_extender') {
            isPanningRef.current = false
            ctrlPressedRef.current = false
            spacePressedRef.current = false
            handToolActiveRef.current = false
            setIsHandToolActive(false)
            lastPointerRef.current = null
            canvas.__expansionMode = true
            canvas.skipTargetFind = false
            canvas.defaultCursor = 'default'
            canvas.hoverCursor = 'default'
            canvas.moveCursor = 'default'
            if (canvas.upperCanvasEl) canvas.upperCanvasEl.style.cursor = 'default'
        } else if (canvas.__expansionMode) {
            canvas.__expansionMode = false
            canvas.__syncPanCursor?.()
        }

        // Drawing mode management (handled in draw.jsx useEffect, but ensure cleanup here)
        if (activeTool !== 'draw' && canvas.isDrawingMode) {
            canvas.isDrawingMode = false
        }
        // Mask/Erase manage their own crosshair + skipTargetFind via
        // usePixelMaskTool's canvas lock; don't stomp it here or the brush cursor
        // flickers back to the move cursor on tool entry.
        if (
            activeTool !== 'draw' &&
            activeTool !== 'ai_extender' &&
            activeTool !== 'mask' &&
            activeTool !== 'erase'
        ) {
            canvas.skipTargetFind = false
            canvas.hoverCursor = 'move'
            canvas.moveCursor = 'move'
            canvas.__syncPanCursor?.()
        }
    }, [activeTool, canvasEditor])

    // Keep doodles glued to the image they were drawn on. On pointer-down over an
    // image we snapshot every stroke sitting on it (relative to the image); as the
    // image is dragged/scaled/rotated we replay that relationship so the strokes
    // travel with the photo as one. The programmatic Resize tool handles its own
    // bind/follow via canvas.__bindDoodles / __followDoodles.
    useEffect(() => {
        if (!canvasEditor) return undefined
        const onPointerDown = (opt) => {
            if (opt?.target?.type?.toLowerCase() === 'image') {
                bindDoodlesToImage(canvasEditor, opt.target)
            }
        }
        const onImageTransform = (opt) => {
            if (opt?.target?.type?.toLowerCase() === 'image') {
                followDoodles(canvasEditor, opt.target)
            }
        }
        canvasEditor.on('mouse:down', onPointerDown)
        canvasEditor.on('object:moving', onImageTransform)
        canvasEditor.on('object:scaling', onImageTransform)
        canvasEditor.on('object:rotating', onImageTransform)
        canvasEditor.on('object:modified', onImageTransform)
        return () => {
            canvasEditor.off('mouse:down', onPointerDown)
            canvasEditor.off('object:moving', onImageTransform)
            canvasEditor.off('object:scaling', onImageTransform)
            canvasEditor.off('object:rotating', onImageTransform)
            canvasEditor.off('object:modified', onImageTransform)
        }
    }, [canvasEditor])

    const toggleHandTool = useCallback(() => {
        const canvas = canvasInstanceRef.current
        // Disabled while painting (Mask/Erase) or expanding — the hand tool would
        // pan-drag while the brush is also painting. Hold Space to pan instead.
        if (
            !canvas ||
            activeToolRef.current === 'ai_extender' ||
            activeToolRef.current === 'mask' ||
            activeToolRef.current === 'erase'
        ) return
        const next = !handToolActiveRef.current
        canvas.__setHandToolActive?.(next)
        setIsHandToolActive(next)
    }, [])

    useEffect(() => {
        const handleHandHotkey = (event) => {
            if (event.repeat) return
            if (event.key !== 'h' && event.key !== 'H') return
            const target = event.target
            if (
                target &&
                (target.tagName === 'INPUT' ||
                    target.tagName === 'TEXTAREA' ||
                    target.tagName === 'SELECT' ||
                    target.isContentEditable)
            ) return
            if (event.metaKey || event.ctrlKey || event.altKey) return
            event.preventDefault()
            toggleHandTool()
        }
        window.addEventListener('keydown', handleHandHotkey)
        return () => window.removeEventListener('keydown', handleHandHotkey)
    }, [toggleHandTool])

    // ─── Image drag-and-drop onto canvas ───
    useEffect(() => {
        const container = containerRef.current
        if (!container) return

        const handleDragOver = (e) => {
            e.preventDefault()
            e.dataTransfer.dropEffect = 'copy'
        }

        const handleDrop = async (e) => {
            e.preventDefault()
            const canvas = canvasInstanceRef.current
            if (!canvas) return

            const files = Array.from(e.dataTransfer?.files || []).filter(f => f.type.startsWith('image/'))
            if (files.length === 0) return

            // Use a ref so this handler always sees the latest project, even if the
            // effect isn't re-bound (e.g. switching projects with same dimensions).
            await addImageFilesToCanvas(canvas, files, projectRef.current)
        }

        container.addEventListener('dragover', handleDragOver)
        container.addEventListener('drop', handleDrop)
        return () => {
            container.removeEventListener('dragover', handleDragOver)
            container.removeEventListener('drop', handleDrop)
        }
    }, [canvasEditor, project?.width, project?.height])

    useEffect(() => {
        if (!canvasEditor) return
        let saveTimeout = null
        let historyTimeout = null
        let historyDueAt = Infinity

        // Sooner-wins scheduling with one exception: a pending push for the
        // SAME action (matching coalesceKey — e.g. consecutive text:changed
        // keystrokes) is debounced properly, the timer resets so exactly one
        // undo state lands per typing pause. Across different actions a
        // sooner pending push survives (a 900ms text push must never cancel
        // an immediate push from another event), while an equal-or-sooner new
        // push replaces it so the freshest label wins.
        let historyPendingKey = null
        const scheduleHistoryPush = (meta, delay = 0) => {
            const due = Date.now() + delay
            const key = meta?.coalesceKey || null
            if (historyTimeout) {
                const samePending = key !== null && historyPendingKey === key
                if (!samePending && historyDueAt < due) return
                clearTimeout(historyTimeout)
            }
            historyPendingKey = key
            historyDueAt = due
            historyTimeout = setTimeout(() => {
                historyTimeout = null
                historyDueAt = Infinity
                historyPendingKey = null
                pushHistoryState(canvasEditor, meta)
            }, delay)
        }

        // Tiered autosave (see MAJOR_SAVE_DEBOUNCE_MS / MINOR_SAVE_TRICKLE_MS):
        // majors debounce-reset the fast timer; minors only arm the slow
        // trickle when NOTHING is pending — whatever save is already scheduled
        // carries them, and repeated nudges must not keep pushing the deadline
        // out (a fidgety user would otherwise never save).
        const scheduleSave = (tier) => {
            if (tier === 'minor') {
                if (saveTimeout) return
                saveTimeout = setTimeout(() => {
                    saveTimeout = null
                    saveCanvasState()
                }, MINOR_SAVE_TRICKLE_MS)
                return
            }
            clearTimeout(saveTimeout)
            saveTimeout = setTimeout(() => {
                saveTimeout = null
                saveCanvasState()
            }, MAJOR_SAVE_DEBOUNCE_MS)
        }

        const makeChangeHandler = (eventName) => (event) => {
            // Guard: when restoring from undo/redo, canvas events fire (object:added,
            // object:modified, etc.) but they must NOT push a new history entry —
            // otherwise the redo stack is truncated immediately after an undo.
            if (isRestoringRef.current) return
            if (isExpansionFrameLike(event?.target)) return
            if (isPhosmithMaskOverlay(event?.target)) return

            const change = describeCanvasChange(eventName, event)
            // Zero-delta gesture (click without movement) — nothing happened.
            // Checked BEFORE the grade mirror so a sub-epsilon event can't
            // mutate the background with no save scheduled to persist it.
            if (change.significance === 'none') return

            // When "color grade background" is on, mirror the photo's grade onto the
            // canvas background. Gated to image edits (skip text/shape moves) and to
            // when a background actually exists; change-detected inside.
            if (
                canvasEditor.__phosmithGradeBackground &&
                canvasEditor.backgroundImage &&
                event?.target?.type?.toLowerCase?.() === 'image'
            ) {
                try { syncBackgroundGrade(canvasEditor, true, event.target) } catch { /* ignore */ }
            }
            // Sub-threshold nudge: a real state change that must persist, but
            // not a "main change" — no undo entry, no journal noise, no fast
            // network save. The trickle (or any already-pending save / the
            // unload beacon) carries it; the next major push captures it in
            // the undo stack.
            if (change.significance === 'minor') {
                scheduleSave('minor')
                return
            }

            scheduleHistoryPush(
                { label: change.label, coalesceKey: change.coalesceKey },
                eventName === 'text:changed' ? TEXT_HISTORY_DEBOUNCE_MS : 0,
            )
            scheduleSave('major')
        }

        const handlers = [
            ['object:modified', makeChangeHandler('object:modified')],
            ['object:added', makeChangeHandler('object:added')],
            ['object:removed', makeChangeHandler('object:removed')],
            ['text:changed', makeChangeHandler('text:changed')],
        ]
        handlers.forEach(([name, handler]) => canvasEditor.on(name, handler))

        return () => {
            clearTimeout(saveTimeout)
            clearTimeout(historyTimeout)
            handlers.forEach(([name, handler]) => canvasEditor.off(name, handler))
        }
    }, [canvasEditor, project?._id, pushHistoryState, saveCanvasState])

    useEffect(() => {
        const handleResize = () => {
            if (resizeFrameRef.current) cancelAnimationFrame(resizeFrameRef.current)
            resizeFrameRef.current = requestAnimationFrame(() => {
                resizeFrameRef.current = null
                const canvas = canvasInstanceRef.current
                const proj = projectRef.current
                if (!canvas || !proj || !containerRef.current) return

                const prevWidth = canvas.getWidth()
                const prevHeight = canvas.getHeight()
                const nextWidth = containerRef.current.clientWidth
                const nextHeight = containerRef.current.clientHeight
                if (!nextWidth || !nextHeight) return

                if (prevWidth === nextWidth && prevHeight === nextHeight) {
                    canvas.calcOffset()
                    return
                }

                canvas.setDimensions({ width: nextWidth, height: nextHeight }, { backstoreOnly: false })

                // Always re-fit the project to the new viewport so the canvas
                // content stays centered after sidebar resizes, tool panel
                // toggles, or window resizes.
                fitProjectToViewport(canvas)
                canvas.calcOffset()
                // Render SYNCHRONOUSLY (not requestRenderAll) here: setDimensions
                // above resets the <canvas> element size, which clears its bitmap.
                // requestRenderAll would repaint on the NEXT frame, leaving one
                // blank frame per resize step — visible as flicker while the user
                // drags the sidebar resizer. renderAll repaints in the same frame,
                // so the clear is never shown.
                canvas.renderAll()

                // Sync the project frame overlay and the zoom percentage HUD
                if (typeof canvas.__syncProjectFrame === 'function') {
                    canvas.__syncProjectFrame()
                }
                syncPreviewZoomState(canvas)
            })
        }

        const resizeObserver = typeof ResizeObserver !== "undefined" && containerRef.current
            ? new ResizeObserver(handleResize)
            : null

        resizeObserver?.observe(containerRef.current)
        window.addEventListener("resize", handleResize)
        handleResize()
        return () => {
            resizeObserver?.disconnect()
            window.removeEventListener("resize", handleResize)
            if (resizeFrameRef.current) cancelAnimationFrame(resizeFrameRef.current)
            resizeFrameRef.current = null
        }
    }, [fitProjectToViewport, project?._id, syncPreviewZoomState])

    useEffect(() => {
        const canvas = canvasInstanceRef.current
        const proj = projectRef.current
        if (!canvas || !proj?.width || !proj?.height) return
        fitProjectToViewport(canvas, proj)
        canvas.calcOffset()
        canvas.requestRenderAll()
    }, [fitProjectToViewport, project?._id, project?.width, project?.height])

    // Capture the current on-screen project frame as the compare baseline. We
    // grab the region straight from Fabric's already-composited lower canvas
    // (the "after" pixels the user sees) rather than re-rendering — so capture
    // never disturbs the live canvas and the baseline aligns 1:1 with the
    // projectFrameStyle rect used to position the overlay. Selection handles /
    // brush previews live on the upper canvas, so the lower canvas is clean.
    const captureCompareBaseline = useCallback(() => {
        const canvas = canvasInstanceRef.current
        const frame = projectFrameStyleRef.current
        if (!canvas || !frame?.width || !frame?.height) {
            setCompareBaselineUrl(null)
            return
        }
        try {
            const el = canvas.getElement?.() || canvas.lowerCanvasEl
            if (!el?.width) { setCompareBaselineUrl(null); return }
            // Backing store may be devicePixelRatio-scaled vs the CSS width Fabric
            // reports; map screen-space frame rect into backing-store pixels.
            const scale = el.width / (canvas.getWidth() || el.width)
            const sx = Math.max(0, Math.round(frame.left * scale))
            const sy = Math.max(0, Math.round(frame.top * scale))
            const sw = Math.min(el.width - sx, Math.round(frame.width * scale))
            const sh = Math.min(el.height - sy, Math.round(frame.height * scale))
            if (sw < 1 || sh < 1) { setCompareBaselineUrl(null); return }
            const off = document.createElement("canvas")
            off.width = sw
            off.height = sh
            off.getContext("2d").drawImage(el, sx, sy, sw, sh, 0, 0, sw, sh)
            setCompareBaselineUrl(off.toDataURL("image/png"))
        } catch (err) {
            // Tainted canvas (non-CORS image) or disposed canvas — disable compare.
            console.warn("[canvas] compare baseline capture failed:", err?.message || err)
            setCompareBaselineUrl(null)
        }
    }, [])

    // Re-capture the session baseline whenever the active tool changes (or the
    // canvas first finishes loading). "Before" = the state the current tool
    // inherited; "After" = whatever the user does next. Deferred to the next
    // frame so any pending render/fit has settled and projectFrameStyleRef is
    // current. Leaving compare mode on tool switch avoids a stale overlay.
    useEffect(() => {
        setIsComparing(false)
        if (!canvasEditor || isLoading) return undefined
        const raf = requestAnimationFrame(() => captureCompareBaseline())
        return () => cancelAnimationFrame(raf)
    }, [activeTool, canvasEditor, isLoading, captureCompareBaseline])

    // Safety net: releasing the pointer anywhere ends a hold-to-compare, even if
    // the pointerup lands outside the button.
    useEffect(() => {
        if (!isComparing) return undefined
        const end = () => setIsComparing(false)
        window.addEventListener("pointerup", end)
        window.addEventListener("pointercancel", end)
        return () => {
            window.removeEventListener("pointerup", end)
            window.removeEventListener("pointercancel", end)
        }
    }, [isComparing])

    const canCompare = Boolean(compareBaselineUrl) && !isBusy && isProjectFrameVisible
    const startCompare = useCallback((event) => {
        event.preventDefault()
        event.stopPropagation()
        // Capture the pointer so holding then sliding off the button keeps the
        // before-view up until release (no premature flip on a small jiggle).
        try { event.currentTarget.setPointerCapture(event.pointerId) } catch { /* ignore */ }
        setIsComparing(true)
    }, [])
    const endCompare = useCallback(() => setIsComparing(false), [])
    const handleCompareKeyDown = useCallback((event) => {
        if (event.key === " " || event.key === "Enter") {
            event.preventDefault()
            setIsComparing(true)
        }
    }, [])

    const previewSliderValue = clamp(previewZoomPercent, MIN_PREVIEW_ZOOM_PERCENT, MAX_PREVIEW_ZOOM_PERCENT)
    const canAdjustPreview = Boolean(canvasEditor)

    const applyPreviewZoomPercent = (percent) => {
        const nextPercent = clamp(Number(percent) || 100, MIN_PREVIEW_ZOOM_PERCENT, MAX_PREVIEW_ZOOM_PERCENT)
        setCanvasPreviewZoom(canvasInstanceRef.current, nextPercent)
    }

    const adjustPreviewZoomPercent = (delta) => {
        applyPreviewZoomPercent(previewZoomPercentRef.current + delta)
    }

    const handlePreviewZoomChange = (event) => {
        applyPreviewZoomPercent(event.target.value)
    }

    const stopPreviewControlPropagation = (event) => {
        event.stopPropagation()
    }

    return (
        <div ref={containerRef} className='relative h-full min-h-0 w-full overflow-hidden editor-canvas-host'>
            {/* Dot grid */}
            <div className='absolute inset-0 pointer-events-none editor-canvas-grid' />

            {isProjectFrameVisible && !isBusy && (
                <div
                    className="editor-canvas-project-texture pointer-events-none absolute"
                    style={{
                        left: `${projectFrameStyle.left}px`,
                        top: `${projectFrameStyle.top}px`,
                        width: `${projectFrameStyle.width}px`,
                        height: `${projectFrameStyle.height}px`,
                    }}
                />
            )}

            <div className='absolute inset-0 editor-canvas-fabric-layer'>
                <canvas id='canvas' className='rounded-xl editor-canvas-surface' ref={canvasRef} />
            </div>

            {/* Before/After compare overlay — the session baseline stretched over
                the live canvas's project frame. Shown only while Compare is held. */}
            {isComparing && compareBaselineUrl && isProjectFrameVisible && (
                <>
                    <img
                        src={compareBaselineUrl}
                        alt=""
                        aria-hidden="true"
                        className="editor-canvas-compare-overlay pointer-events-none absolute"
                        style={{
                            left: `${projectFrameStyle.left}px`,
                            top: `${projectFrameStyle.top}px`,
                            width: `${projectFrameStyle.width}px`,
                            height: `${projectFrameStyle.height}px`,
                        }}
                    />
                    <div className="editor-canvas-compare-pill" role="status" aria-live="polite">Before</div>
                </>
            )}

            {!isBusy && (
                <button
                    type="button"
                    onClick={toggleHandTool}
                    className="editor-icon-button absolute z-10 flex items-center justify-center"
                    style={{
                        bottom: 16,
                        right: 16,
                        width: 36,
                        height: 36,
                        background: isHandToolActive ? 'var(--accent-primary)' : 'var(--bg-elevated)',
                        color: isHandToolActive ? '#03050A' : 'var(--text-primary)',
                        borderColor: isHandToolActive ? 'var(--accent-primary)' : 'var(--border-default)',
                    }}
                    title={isHandToolActive ? 'Hand tool on — click to exit (H or Space)' : 'Hand tool — pan the canvas (H or hold Space)'}
                    aria-pressed={isHandToolActive}
                >
                    <Hand className="h-4 w-4" />
                </button>
            )}

            {isProjectFrameVisible && !isBusy && (
                <div
                    className="editor-canvas-project-frame pointer-events-none absolute"
                    style={{
                        left: `${projectFrameStyle.left}px`,
                        top: `${projectFrameStyle.top}px`,
                        width: `${projectFrameStyle.width}px`,
                        height: `${projectFrameStyle.height}px`,
                    }}
                />
            )}

            {!isBusy && (
                <button
                    type="button"
                    className={`editor-canvas-compare-button${isComparing ? " is-active" : ""}`}
                    onPointerDown={canCompare ? startCompare : undefined}
                    onPointerUp={endCompare}
                    onKeyDown={canCompare ? handleCompareKeyDown : undefined}
                    onKeyUp={endCompare}
                    disabled={!canCompare}
                    aria-pressed={isComparing}
                    title={canCompare ? "Hold to compare before / after" : "Compare (no changes yet)"}
                >
                    <ArrowLeftRight className="h-3.5 w-3.5" aria-hidden="true" />
                    <span>{isComparing ? "Before" : "Compare"}</span>
                </button>
            )}

            <div
                className="editor-canvas-preview-controls"
                aria-label="Preview size"
                onPointerDown={stopPreviewControlPropagation}
                onMouseDown={stopPreviewControlPropagation}
                hidden={isBusy}
                style={isBusy ? { display: 'none' } : undefined}
            >
                <button
                    type="button"
                    className="editor-canvas-preview-button"
                    onClick={() => adjustPreviewZoomPercent(-PREVIEW_ZOOM_STEP_PERCENT)}
                    disabled={!canAdjustPreview}
                    title="Shrink preview"
                    aria-label="Shrink preview"
                >
                    <ZoomOut className="h-3.5 w-3.5" />
                </button>
                <input
                    className="editor-canvas-preview-slider"
                    type="range"
                    min={MIN_PREVIEW_ZOOM_PERCENT}
                    max={MAX_PREVIEW_ZOOM_PERCENT}
                    step="1"
                    value={previewSliderValue}
                    onChange={handlePreviewZoomChange}
                    disabled={!canAdjustPreview}
                    aria-label="Preview size"
                />
                <button
                    type="button"
                    className="editor-canvas-preview-button"
                    onClick={() => adjustPreviewZoomPercent(PREVIEW_ZOOM_STEP_PERCENT)}
                    disabled={!canAdjustPreview}
                    title="Enlarge preview"
                    aria-label="Enlarge preview"
                >
                    <ZoomIn className="h-3.5 w-3.5" />
                </button>
                <button
                    type="button"
                    className="editor-canvas-preview-button"
                    onClick={() => canvasInstanceRef.current?.__resetCanvasView?.()}
                    disabled={!canAdjustPreview}
                    title="Fit preview"
                    aria-label="Fit preview"
                >
                    <Maximize2 className="h-3.5 w-3.5" />
                </button>
                <output className="editor-canvas-preview-percent" aria-live="polite">
                    {previewZoomPercent}%
                </output>
            </div>

            {(syncStatus === "offline" || syncStatus === "error" || syncStatus === "saving" || syncStatus === "conflict") && (
                <div
                    className="absolute left-1/2 z-30 flex items-center gap-2"
                    style={{
                        top: 12,
                        transform: "translateX(-50%)",
                        pointerEvents: "none",
                        padding: "5px 12px",
                        background: "var(--bg-elevated, #0a0d14)",
                        border: "2px solid",
                        borderColor:
                            syncStatus === "offline" ? "var(--accent-amber, #f5b945)"
                                : (syncStatus === "error" || syncStatus === "conflict") ? "var(--accent-coral, #ff6b5e)"
                                    : "var(--accent-primary, #38e0c8)",
                        boxShadow: "3px 3px 0 0 rgba(0,0,0,0.55)",
                        borderRadius: 6,
                        fontFamily: "var(--font-mono, monospace)",
                        fontSize: 11,
                        fontWeight: 700,
                        letterSpacing: "0.08em",
                        textTransform: "uppercase",
                        color: "var(--text-primary, #f5f7fa)",
                        whiteSpace: "nowrap",
                    }}
                    role="status"
                    aria-live="polite"
                >
                    <span
                        style={{
                            width: 8,
                            height: 8,
                            borderRadius: "50%",
                            background:
                                syncStatus === "offline" ? "var(--accent-amber, #f5b945)"
                                    : (syncStatus === "error" || syncStatus === "conflict") ? "var(--accent-coral, #ff6b5e)"
                                        : "var(--accent-primary, #38e0c8)",
                        }}
                    />
                    {syncStatus === "offline"
                        ? "Offline · saved locally"
                        : syncStatus === "conflict"
                            ? "Edited elsewhere"
                            : syncStatus === "error"
                                ? "Sync retrying…"
                                : "Syncing…"}
                </div>
            )}

            {isLoading && (
                <div className='neo-loader-surface absolute inset-0 z-40 flex items-center justify-center'>
                    <AuroraLoader message="Loading canvas" />
                </div>
            )}
        </div>
    )
}

export default React.memo(CanvasEditor)
