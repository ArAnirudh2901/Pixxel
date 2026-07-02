"use client"

import { Bot, Eraser, Expand, Eye, ImagePlus, Maximize2, Palette, PanelLeft, PanelRight, Pen, Scissors, Sliders, Text, Crop, ArrowLeft, ChevronDown, Check, Copy, Download, Loader2, Save, Undo2, Redo2, ZoomIn, Keyboard, LayoutGrid, AudioLines } from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { useCanvas, useDynamicAccent } from '../../../../../../context/context'
import usePlanAccess from '../../../../../../hooks/usePlanAccess'
import UpgradeModel from '@/components/upgradeModel'
import { addImageFilesToCanvas } from '@/lib/canvas-images'
import ProBadge from '@/components/pro-badge'
import PixxelWordmark from '@/components/phosmith-wordmark'
import ShortcutsGuide from '@/components/neo/ShortcutsGuide'
import { motion, AnimatePresence } from 'framer-motion'
import { snapshotCanvasToBlobSafe, isTaintError } from '@/lib/canvas-snapshot'

const EXPORT_PRESETS = [
    { id: 'png', label: 'PNG', description: 'Lossless · Best quality', format: 'png', quality: 1 },
    { id: 'jpeg90', label: 'JPEG 90%', description: 'Lossy · Great quality', format: 'jpeg', quality: 0.9 },
    { id: 'jpeg80', label: 'JPEG 80%', description: 'Lossy · Smaller file', format: 'jpeg', quality: 0.8 },
    { id: 'webp90', label: 'WebP 90%', description: 'Modern · Smallest file', format: 'webp', quality: 0.9 },
]

// Resolution multipliers exposed in the export menu. 1× is "as displayed",
// 2× / 3× re-render at higher resolution via fabric's toCanvasElement scale arg.
// Capped at 3× to keep memory bounded — a 4K canvas at 3× is already ~150 MB
// of pixel data, which is the practical ceiling for browser canvas exports.
const SCALE_OPTIONS = [
    { id: '1x', label: '1×', value: 1 },
    { id: '2x', label: '2×', value: 2 },
    { id: '3x', label: '3×', value: 3 },
]

const TOOLS = [
    { id: "resize", label: "Resize", icon: Expand },
    { id: "crop", label: "Crop", icon: Crop },
    { id: "images", label: "Images", icon: ImagePlus },
    { id: "adjust", label: "Adjust", icon: Sliders },
    { id: "draw", label: "Draw", icon: Pen },
    { id: "erase", label: "Erase", icon: Eraser },
    { id: "mask", label: "Mask", icon: Scissors },
    { id: "text", label: "Text", icon: Text },
    { id: "pixel_stretch", label: "Stretch", icon: AudioLines },
    { id: "ai_background", label: "AI BG", icon: Palette, proOnly: true },
    { id: "ai_extender", label: "Extender", icon: Maximize2, proOnly: true },
    { id: "ai_edit", label: "AI Edit", icon: Eye, proOnly: true },
    { id: "ai_agent", label: "Agent", icon: Bot },
    { id: "collage", label: "Collage", icon: LayoutGrid },
]

const ALL_TOOLS = [...TOOLS]

const EditorTopbar = ({ project, onToggleSidebar, isSidebarOpen = false, isNarrowViewport = false }) => {

    const router = useRouter()
    const exportMenuRef = useRef(null)
    const addImageInputRef = useRef(null)
    // Horizontally-scrollable tool row at narrow viewports — the active tool
    // is scrolled into view so users on tablets/laptops never lose track of it.
    const toolsScrollRef = useRef(null)

    const [showUpgradeModel, setShowUpgradeModel] = useState(false)
    const [restrictedTool, setRestrictedTool] = useState(null)
    const [showExportMenu, setShowExportMenu] = useState(false)
    const [canUndo, setCanUndo] = useState(false)
    const [canRedo, setCanRedo] = useState(false)
    const [showShortcuts, setShowShortcuts] = useState(false)
    const [isSaving, setIsSaving] = useState(false)
    const [isExporting, setIsExporting] = useState(false)
    const [exportScale, setExportScale] = useState(1)
    // "idle" | "copying" | "copied" — drives the inline state of the Copy item.
    const [copyState, setCopyState] = useState('idle')

    const triggerAddImage = useCallback(() => {
        addImageInputRef.current?.click()
    }, [])

    useEffect(() => {
        const isTypingTarget = (target) => {
            if (!target) return false
            const tag = target.tagName
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true
            return Boolean(target.isContentEditable)
        }
        const onKeyDown = (event) => {
            if (event.repeat || isTypingTarget(event.target)) return
            if ((event.key === 'I' || event.key === 'i') && event.shiftKey && !event.metaKey && !event.ctrlKey && !event.altKey) {
                event.preventDefault()
                triggerAddImage()
                return
            }
            if (event.key === '?' || (event.key === '/' && event.shiftKey)) {
                event.preventDefault()
                setShowShortcuts((prev) => !prev)
            }
        }
        window.addEventListener('keydown', onKeyDown)
        return () => window.removeEventListener('keydown', onKeyDown)
    }, [triggerAddImage])

    const { canvasEditor, activeTool, onToolChange } = useCanvas()
    const { hasAccess, isPro, isPlanReady } = usePlanAccess()
    const { accent, accentRgb, textOnAccent } = useDynamicAccent()

    const exportResolutionLabel = useMemo(() => {
        if (!project?.width || !project?.height) return ''
        return `${project.width} × ${project.height}`
    }, [project?.width, project?.height])

    // Reset the transient Copy-to-clipboard state whenever the menu closes,
    // so reopening always shows the default "Copy to clipboard" affordance
    // instead of a stale "Copied!" from the previous session.
    useEffect(() => {
        if (!showExportMenu) setCopyState('idle')
    }, [showExportMenu])

    // Auto-scroll the active tool into view inside the (potentially overflowing)
    // tools row. Without this, a user on a 1024–1280 viewport who picks "Agent"
    // via the keyboard or radial menu would see the row still scrolled to the
    // left, with the active button invisible. `block: "nearest"` keeps the row
    // stable when the active tool is already in view (no jitter).
    useEffect(() => {
        const row = toolsScrollRef.current
        if (!row) return
        const activeBtn = row.querySelector(`[data-tool-id="${activeTool}"]`)
        if (!activeBtn) return
        try {
            activeBtn.scrollIntoView({ inline: 'nearest', block: 'nearest', behavior: 'smooth' })
        } catch {
            // Older Safari without smooth behavior support — fall back to instant.
            activeBtn.scrollIntoView({ inline: 'nearest', block: 'nearest' })
        }
    }, [activeTool])

    useEffect(() => {
        if (!showExportMenu) return undefined

        const handleClickOutside = (event) => {
            if (exportMenuRef.current && !exportMenuRef.current.contains(event.target)) {
                setShowExportMenu(false)
            }
        }
        // Escape closes the menu. Without this the only way to dismiss it was
        // clicking outside, which is an accessibility miss for keyboard users.
        const handleKeyDown = (event) => {
            if (event.key === 'Escape') {
                event.preventDefault()
                setShowExportMenu(false)
            }
        }

        window.addEventListener('mousedown', handleClickOutside)
        window.addEventListener('keydown', handleKeyDown)
        return () => {
            window.removeEventListener('mousedown', handleClickOutside)
            window.removeEventListener('keydown', handleKeyDown)
        }
    }, [showExportMenu])

    useEffect(() => {
        if (!canvasEditor) {
            const resetHistoryState = () => {
                setCanUndo(false)
                setCanRedo(false)
            }
            if (typeof queueMicrotask === 'function') queueMicrotask(resetHistoryState)
            else setTimeout(resetHistoryState, 0)
            return undefined
        }

        const syncHistory = () => {
            const state = canvasEditor.__getHistoryState?.()
            // OR in the Mask/Erase tool's per-stroke stack so the buttons stay
            // enabled while it owns undo/redo (its handlers fall back to the
            // global history when its own stack is empty). Fixes the Redo button
            // staying disabled after a mask-stack undo.
            const maskUndo = Boolean(canvasEditor.__maskCanUndo)
            const maskRedo = Boolean(canvasEditor.__maskCanRedo)
            // The Crop tool owns a crop-box preview stack while crop mode is
            // active (same fall-through-to-global contract as the mask stack).
            const cropUndo = Boolean(canvasEditor.__cropCanUndo)
            const cropRedo = Boolean(canvasEditor.__cropCanRedo)
            setCanUndo(Boolean(state?.canUndo) || maskUndo || cropUndo)
            setCanRedo(Boolean(state?.canRedo) || maskRedo || cropRedo)
        }

        syncHistory()
        canvasEditor.on('history:changed', syncHistory)
        window.addEventListener('phosmith:mask-history-changed', syncHistory)
        return () => {
            canvasEditor.off('history:changed', syncHistory)
            window.removeEventListener('phosmith:mask-history-changed', syncHistory)
        }
    }, [canvasEditor])

    const handleUndo = async () => {
        // The Crop tool owns a crop-box preview stack while crop mode is active.
        // Revert the last preview change first; it returns false (and does
        // nothing) once the stack is empty, so we fall through to canvas history.
        if (canvasEditor?.__cropToolUndo?.()) return
        // When the Mask/Erase tool is mounted it owns a per-stroke undo stack and
        // exposes __maskToolUndo (the same path as ⌘Z). Route the button through
        // it so a one-shot action like "Select Subject" is reverted by the SAME
        // stack the keyboard undo uses — it falls back to the global canvas
        // history once the local stack is empty.
        if (canvasEditor?.__maskToolUndo) {
            canvasEditor.__maskToolUndo()
            await canvasEditor.__saveCanvasState?.()
            return
        }
        if (!canvasEditor?.__undoCanvasState) return
        const didUndo = await canvasEditor.__undoCanvasState()
        if (!didUndo) toast.message('Nothing to undo')
        else await canvasEditor.__saveCanvasState?.()
    }

    const handleRedo = async () => {
        if (canvasEditor?.__cropToolRedo?.()) return
        if (canvasEditor?.__maskToolRedo) {
            canvasEditor.__maskToolRedo()
            await canvasEditor.__saveCanvasState?.()
            return
        }
        if (!canvasEditor?.__redoCanvasState) return
        const didRedo = await canvasEditor.__redoCanvasState()
        if (!didRedo) toast.message('Nothing to redo')
        else await canvasEditor.__saveCanvasState?.()
    }

    const handleBackToDashboard = () => {
        router.push("/dashboard")
    }

    const handleSave = async () => {
        if (!canvasEditor?.__saveCanvasState || isSaving) return
        setIsSaving(true)
        const toastId = toast.loading('Saving project...')
        try {
            await canvasEditor.__saveCanvasState({ rethrow: true })
            toast.success('Project saved', { id: toastId })
        } catch (error) {
            console.error('[Save] Failed:', error)
            const msg = error?.message || String(error) || 'Unknown error'
            // Neon's per-document size limit is 1 MB. If saved state exceeds that
            // (large data URLs from uploaded images), the user needs to know — that's
            // the root cause of "my changes are suddenly gone".
            const isSizeError = /too large|maximum.*size|1MB|1 MB/i.test(msg)
            toast.error(
                isSizeError
                    ? 'Save failed: project too large. Re-upload any embedded images so they go through ImageKit.'
                    : `Save failed: ${msg}`,
                { id: toastId, duration: 8000 }
            )
        } finally {
            setIsSaving(false)
        }
    }

    const handleToolChange = (toolId) => {
        if (isPlanReady && !isPro && !hasAccess(toolId)) {
            setRestrictedTool(toolId)
            setShowUpgradeModel(true)
            return
        }
        onToolChange(toolId)
    }

    // Canvas flattening (tight bounding-box render, viewport handling, restore
    // logic, and tainted-canvas recovery) lives in `@/lib/canvas-snapshot` so the
    // export path and the AI agent's image-context capture share ONE definition of
    // "what the canvas looks like". snapshotCanvasToBlobSafe(canvasEditor, opts).

    const handleExport = async ({ format, quality }) => {
        if (!canvasEditor || !project || isExporting) return
        setIsExporting(true)
        const scaleLabel = exportScale > 1 ? ` (${exportScale}×)` : ''
        const toastId = toast.loading(`Exporting as ${format.toUpperCase()}${scaleLabel}…`)

        try {
            const { blob } = await snapshotCanvasToBlobSafe(canvasEditor, { project, scale: exportScale, format, quality })
            const blobUrl = URL.createObjectURL(blob)
            const link = document.createElement('a')
            link.href = blobUrl
            // Tag the filename with the scale so users can tell e.g. a 2× PNG
            // apart from a 1× PNG without having to read the dimensions.
            const scaleSuffix = exportScale > 1 ? `@${exportScale}x` : ''
            link.download = `${project.title || 'export'}${scaleSuffix}.${format === 'jpeg' ? 'jpg' : format}`
            document.body.appendChild(link)
            link.click()
            link.remove()
            // Release the blob URL on the next tick — Safari needs the link click first.
            setTimeout(() => URL.revokeObjectURL(blobUrl), 0)

            setShowExportMenu(false)
            toast.success(`Exported as ${format.toUpperCase()}${scaleLabel}`, { id: toastId })
        } catch (error) {
            console.error('[Export] Failed:', error)
            const message = isTaintError(error)
                ? "Export blocked: an image on the canvas can't be read for security (it loaded without cross-origin access). Re-add it from the Images panel and try again."
                : `Export failed: ${error?.message || 'Unknown error'}`
            toast.error(message, { id: toastId })
        } finally {
            setIsExporting(false)
        }
    }

    const handleCopyToClipboard = async () => {
        if (!canvasEditor || !project || isExporting) return
        // Browsers only reliably accept image/png in the clipboard. Force PNG
        // even if the user selected JPEG/WebP — quality is moot for clipboard.
        if (typeof navigator === 'undefined' ||
            typeof navigator.clipboard?.write !== 'function' ||
            typeof window.ClipboardItem !== 'function') {
            toast.error('Clipboard images are not supported in this browser')
            return
        }

        setIsExporting(true)
        setCopyState('copying')
        const toastId = toast.loading('Copying to clipboard…')

        try {
            const { blob } = await snapshotCanvasToBlobSafe(canvasEditor, { project, scale: exportScale, format: 'png', quality: 1 })
            await navigator.clipboard.write([new window.ClipboardItem({ 'image/png': blob })])
            setCopyState('copied')
            toast.success('Copied to clipboard', { id: toastId })
            // Auto-revert the "Copied!" affordance after a beat so the user can
            // copy again without reopening the menu.
            setTimeout(() => setCopyState('idle'), 1600)
        } catch (error) {
            console.error('[Copy] Failed:', error)
            setCopyState('idle')
            // Most common failure: document not focused (Firefox) or permission denied.
            const msg = /not\s*focused|focus/i.test(error?.message || '')
                ? 'Click the page first, then try Copy again'
                : `Copy failed: ${error?.message || 'Unknown error'}`
            toast.error(msg, { id: toastId })
        } finally {
            setIsExporting(false)
        }
    }

    return (
        <>
            {/* ── Top Navigation Bar ──
                Layout contract (the core fix for tablet/laptop overlap):
                  • LEFT (auto, overflow-hidden): logo, optional back, title, badges. Title
                    truncates with a fluid clamp() so it never pushes the middle off-screen.
                    Content-sized so it takes exactly what it needs.
                  • MIDDLE (minmax(0,1fr), overflow-x-auto): tool buttons. Takes ALL
                    remaining space and scrolls horizontally when its content can't fit.
                    This is what stops the left/right sections from overlapping with tools.
                  • RIGHT (auto, flex-none): utility icons + Export. Never shrinks below its
                    content width — Export is the primary action, must always be reachable.
                Left and right are content-sized (auto) so they never get squeezed to zero,
                and the center absorbs all remaining space and scrolls when needed. */}
            {/* Spacing scale (uniform across every section so the row reads as one
                rhythm instead of three different ones):
                  768–1023:  gap-1.5  (6px)
                  1024–1279: gap-2    (8px)
                  1280–1699: gap-2.5  (10px)
                  1700+:     gap-3    (12px)
                These map onto common laptop / tablet / desktop densities — at the
                15" 16:10 size (≈1728 logical) the scale is 12px, comfortable for a
                Retina display without being airy. The same tokens are reused for
                root, left, middle, and right so every adjacent pair of elements
                sits the same distance apart. Dividers contribute only their 1px
                width; per-element mx-* margins were removed to keep spacing
                gap-only and predictable. */}
            <div className="editor-topbar grid items-center gap-1.5 lg:gap-2 xl:gap-2.5 min-[1700px]:gap-3 px-2 lg:px-3 min-w-0 w-full" style={{ gridTemplateColumns: 'auto minmax(0, 1fr) auto' }}>

                {/* Left section: navigation chrome + project identity */}
                <div className="flex items-center justify-start gap-1.5 lg:gap-2 xl:gap-2.5 min-[1700px]:gap-3 min-w-0 overflow-hidden">
                    {/* Sidebar toggle — visible only when the parent is in overlay-mode
                        (<lg viewport), where the sidebar is hidden by default. Icon flips
                        to PanelRight when the agent sidebar (right side) is active so the
                        affordance points at the side it affects. */}
                    {onToggleSidebar && (
                        <motion.button
                            onClick={onToggleSidebar}
                            className="editor-icon-button flex lg:hidden items-center justify-center flex-none"
                            title={isSidebarOpen ? 'Hide tools panel' : 'Show tools panel'}
                            aria-label={isSidebarOpen ? 'Hide tools panel' : 'Show tools panel'}
                            aria-expanded={isSidebarOpen}
                            aria-controls="editor-sidebar"
                        >
                            {activeTool === 'ai_agent'
                                ? <PanelRight className="h-4 w-4" />
                                : <PanelLeft className="h-4 w-4" />}
                        </motion.button>
                    )}

                    <Link
                        href="/dashboard"
                        className="editor-logo-button flex items-center justify-center flex-none"
                        title="Go to dashboard"
                        aria-label="Go to dashboard"
                    >
                        <PixxelWordmark showText={false} height={24} markScale={2.1} />
                    </Link>

                    {/* Back arrow — redundant with the logo link (both go to /dashboard).
                        Hidden below xl to save ~40px of width on laptops/tablets. */}
                    <motion.button
                        onClick={handleBackToDashboard}
                        className="editor-icon-button hidden xl:flex items-center justify-center flex-none"
                        title="Back to projects"
                        aria-label="Back to projects"
                    >
                        <ArrowLeft className="h-4 w-4" />
                    </motion.button>

                    <div className="hidden xl:block h-5 w-px flex-none" style={{ background: 'var(--border-default)' }} />

                    {/* Title with fluid truncation. clamp(60px, 12vw, 280px) scales
                        smoothly from a 60px stub on narrow tablets up to 280px on
                        ultrawide, with no discrete jumps that re-flow the row. */}
                    <span className="text-sm font-semibold truncate flex-none"
                        style={{ color: 'var(--text-primary)', maxWidth: 'clamp(60px, 12vw, 280px)' }}
                        title={project.title}>
                        {project.title}
                    </span>

                    <ProBadge size="sm" />

                </div>

                {/* Center: Tool buttons.
                    flex-1 + min-w-0 + overflow-x-auto is the magic combination — middle
                    takes ALL remaining space (so the right section can't be pushed off
                    screen) and overflows horizontally when content is too wide.
                    Active tool auto-scrolls into view via the effect below. */}
                <div
                    className="editor-topbar-tools flex items-center justify-center gap-1 lg:gap-1.5 xl:gap-2 min-[1700px]:gap-2.5 min-[2000px]:gap-3 overflow-x-auto scrollbar-hide max-w-[100vw]"
                    ref={toolsScrollRef}
                >
                    {ALL_TOOLS.map((tool) => {
                        const Icon = tool.icon
                        const isActive = activeTool === tool.id
                        const hasToolAccess = hasAccess(tool.id)

                        return (
                            <React.Fragment key={tool.id}>
                                {/* Visual dividers to create tool groups */}
                                {(tool.id === 'erase' || tool.id === 'ai_background') && (
                                    <div className="h-5 w-px flex-none" style={{ background: 'var(--border-default)' }} />
                                )}
                                <button
                                    onClick={() => handleToolChange(tool.id)}
                                    data-tool-id={tool.id}
                                    className={`tool-btn ${isActive ? 'tool-btn--active' : ''} ${!hasToolAccess ? 'tool-btn--locked' : ''}`}
                                    aria-label={tool.label}
                                    aria-pressed={isActive}
                                    title={tool.label}
                                >
                                    <Icon className="h-4 w-4 flex-none" />
                                    {/* Icon-only by default (tooltips + the radial menu carry the
                                        labels). Text labels only appear at 2000px+, where all 12
                                        labeled tools genuinely fit alongside the title + actions
                                        without crowding. Below that the rail stays clean and
                                        evenly spaced. */}
                                    <span className="hidden min-[2000px]:inline">{tool.label}</span>
                                </button>
                            </React.Fragment>
                        )
                    })}
                </div>

                {/* Right: Actions — content-sized so Export is always reachable.
                    overflow-visible keeps the export dropdown from being clipped
                    by the topbar bounds. Non-essential icons (zoom reset, keyboard
                    shortcuts) collapse out at narrow widths since they have
                    keyboard shortcuts (⌘0, ?). */}
                <div className="flex items-center justify-end gap-1.5 lg:gap-2 xl:gap-2.5 min-[1700px]:gap-3 flex-none" style={{ overflow: 'visible' }}>
                    {/* Undo / Redo — always visible (core workflow) */}
                    <motion.button
                        onClick={handleUndo}
                        disabled={!canUndo}
                        className="editor-icon-button flex items-center justify-center disabled:opacity-35 disabled:pointer-events-none flex-none"
                        title="Undo (⌘Z)"
                        aria-label="Undo"
                    >
                        <Undo2 className="h-3.5 w-3.5" />
                    </motion.button>

                    <motion.button
                        onClick={handleRedo}
                        disabled={!canRedo}
                        className="editor-icon-button flex items-center justify-center disabled:opacity-35 disabled:pointer-events-none flex-none"
                        title="Redo (⌘⇧Z)"
                        aria-label="Redo"
                    >
                        <Redo2 className="h-3.5 w-3.5" />
                    </motion.button>

                    <div className="hidden lg:block h-5 w-px flex-none" style={{ background: 'var(--border-default)' }} />

                    {/* Hidden file picker — used by the add-image button below */}
                    <input
                        ref={addImageInputRef}
                        type="file"
                        accept="image/*"
                        multiple
                        className="hidden"
                        onChange={async (e) => {
                            const files = Array.from(e.target.files || [])
                            e.target.value = ''
                            if (files.length === 0) return
                            await addImageFilesToCanvas(canvasEditor, files, project)
                        }}
                    />
                    {/* Add image — keyboard-accessible via Shift+I, hide at <lg
                        to keep the row tight. Tablet users get the same affordance
                        from inside the Images tool panel. */}
                    <motion.button
                        onClick={() => addImageInputRef.current?.click()}
                        disabled={!canvasEditor}
                        className="editor-icon-button hidden lg:flex items-center justify-center disabled:opacity-35 flex-none"
                        title="Add image (Shift + I)"
                        aria-label="Add image"
                    >
                        <ImagePlus className="h-3.5 w-3.5" />
                    </motion.button>

                    <div className="hidden xl:block h-5 w-px flex-none" style={{ background: 'var(--border-default)' }} />

                    {/* Reset View — hide below xl (⌘0 keyboard shortcut still works) */}
                    <motion.button
                        onClick={() => canvasEditor?.__resetCanvasView?.()}
                        className="editor-icon-button hidden xl:flex items-center justify-center flex-none"
                        title="Reset view"
                        aria-label="Reset view"
                    >
                        <ZoomIn className="h-3.5 w-3.5" />
                    </motion.button>

                    {/* Save — always visible (it's the core "don't lose your work" button) */}
                    <motion.button
                        onClick={handleSave}
                        disabled={isSaving}
                        className="editor-icon-button flex items-center justify-center flex-none disabled:opacity-50 disabled:cursor-wait"
                        title={isSaving ? 'Saving…' : 'Save (⌘S)'}
                        aria-label={isSaving ? 'Saving' : 'Save project'}
                    >
                        {isSaving
                            ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            : <Save className="h-3.5 w-3.5" />}
                    </motion.button>

                    {/* Keyboard shortcuts — always visible; also openable via the ? key */}
                    <motion.button
                        onClick={() => setShowShortcuts(true)}
                        className="editor-icon-button flex items-center justify-center flex-none"
                        title="Keyboard shortcuts (?)"
                        aria-label="Show keyboard shortcuts"
                    >
                        <Keyboard className="h-3.5 w-3.5" />
                    </motion.button>

                    <div className="h-5 w-px flex-none" style={{ background: 'var(--border-default)' }} />

                    {/* Export dropdown */}
                    <div className="relative flex-none" ref={exportMenuRef}>
                        <motion.button
                            onClick={() => setShowExportMenu(prev => !prev)}
                            disabled={isExporting}
                            aria-haspopup="menu"
                            aria-expanded={showExportMenu}
                            aria-controls="export-menu"
                            aria-label={isExporting ? 'Exporting…' : 'Open export menu'}
                            className="flex items-center gap-1.5 editor-interactive pill-control"
                            style={{
                                background: accent,
                                border: '2px solid #F4F4F5',
                                color: textOnAccent,
                                boxShadow: '4px 4px 0 0 #F4F4F5',
                                fontFamily: 'var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace)',
                                letterSpacing: '0.12em',
                                textTransform: 'uppercase',
                                fontWeight: 800,
                                fontSize: '0.7rem',
                                padding: '0.5rem 0.85rem',
                                borderRadius: 0,
                                opacity: isExporting ? 0.55 : 1,
                                cursor: isExporting ? 'wait' : 'pointer',
                                transition: 'background 400ms cubic-bezier(0.22, 1, 0.36, 1), color 400ms cubic-bezier(0.22, 1, 0.36, 1)',
                            }}
                        >
                            {isExporting
                                ? <Loader2 className="h-3.5 w-3.5 animate-spin" strokeWidth={2.5} />
                                : <Download className="h-3.5 w-3.5" strokeWidth={2.5} />}
                            Export
                            <ChevronDown className={`h-3 w-3 transition-transform duration-200 ${showExportMenu ? 'rotate-180' : ''}`} />
                        </motion.button>

                        <AnimatePresence>
                            {showExportMenu && (
                                <motion.div
                                    id="export-menu"
                                    role="menu"
                                    aria-label="Export options"
                                    className="absolute right-0 top-full mt-2 z-50 overflow-hidden"
                                    style={{
                                        width: 'clamp(240px, 22vw, 280px)',
                                        background: '#000',
                                        border: '2px solid rgba(244, 244, 245, 0.85)',
                                        borderRadius: '4px',
                                        boxShadow: `5px 5px 0 0 rgba(${accentRgb}, 0.85), inset 0 1px 0 rgba(255, 255, 255, 0.08)`,
                                        transformOrigin: 'top right',
                                    }}
                                    initial={{ opacity: 0, y: -8, scale: 0.96 }}
                                    animate={{ opacity: 1, y: 0, scale: 1 }}
                                    exit={{ opacity: 0, y: -8, scale: 0.96 }}
                                    transition={{ duration: 0.15, ease: [0.4, 0, 0.2, 1] }}
                                >
                                    {/* Header strip */}
                                    <div
                                        className="flex items-center justify-between gap-2"
                                        style={{
                                            padding: '0.5rem 0.65rem',
                                            borderBottom: '1.5px solid rgba(244, 244, 245, 0.55)',
                                            background: 'linear-gradient(180deg, rgba(244, 244, 245, 0.045), transparent)',
                                        }}
                                    >
                                        <span
                                            style={{
                                                fontFamily: 'var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace)',
                                                fontSize: '0.6rem',
                                                fontWeight: 800,
                                                letterSpacing: '0.1em',
                                                textTransform: 'uppercase',
                                                color: '#F4F4F5',
                                            }}
                                        >
                                            Download
                                        </span>
                                        {exportResolutionLabel && (
                                            <span
                                                style={{
                                                    fontFamily: 'var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace)',
                                                    fontSize: '0.55rem',
                                                    fontWeight: 600,
                                                    letterSpacing: '0.06em',
                                                    color: 'rgba(244, 244, 245, 0.5)',
                                                }}
                                            >
                                                {exportScale > 1
                                                    ? `${exportResolutionLabel} · ${exportScale}×`
                                                    : exportResolutionLabel}
                                            </span>
                                        )}
                                    </div>

                                    {/* Scale selector row — picks the resolution multiplier
                                        applied to ALL downloads + Copy in this menu session. */}
                                    <div
                                        role="radiogroup"
                                        aria-label="Export resolution"
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'space-between',
                                            gap: '0.4rem',
                                            padding: '0.4rem 0.55rem',
                                            borderBottom: '1.5px solid rgba(244, 244, 245, 0.18)',
                                        }}
                                    >
                                        <span
                                            style={{
                                                fontFamily: 'var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace)',
                                                fontSize: '0.55rem',
                                                fontWeight: 700,
                                                letterSpacing: '0.1em',
                                                textTransform: 'uppercase',
                                                color: 'rgba(244, 244, 245, 0.55)',
                                            }}
                                        >
                                            Scale
                                        </span>
                                        <div style={{ display: 'flex', gap: '0.25rem' }}>
                                            {SCALE_OPTIONS.map((opt) => {
                                                const selected = exportScale === opt.value
                                                return (
                                                    <button
                                                        key={opt.id}
                                                        type="button"
                                                        role="radio"
                                                        aria-checked={selected}
                                                        onClick={() => setExportScale(opt.value)}
                                                        disabled={isExporting}
                                                        style={{
                                                            minWidth: '2rem',
                                                            padding: '0.2rem 0.45rem',
                                                            borderRadius: '2px',
                                                            border: selected
                                                                ? `1.5px solid ${accent}`
                                                                : '1.5px solid rgba(244, 244, 245, 0.3)',
                                                            background: selected ? accent : 'transparent',
                                                            color: selected ? textOnAccent : '#F4F4F5',
                                                            fontFamily: 'var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace)',
                                                            fontSize: '0.6rem',
                                                            fontWeight: 800,
                                                            letterSpacing: '0.06em',
                                                            cursor: isExporting ? 'wait' : 'pointer',
                                                            transition: 'background 300ms cubic-bezier(0.22, 1, 0.36, 1), border-color 300ms, color 300ms',
                                                        }}
                                                    >
                                                        {opt.label}
                                                    </button>
                                                )
                                            })}
                                        </div>
                                    </div>

                                    {/* Export items */}
                                    <div style={{ padding: '0.35rem' }}>
                                        {/* Copy to clipboard — sits at the top because it's
                                            the fastest "I just want this image" path. */}
                                        <motion.button
                                            key="copy-to-clipboard"
                                            type="button"
                                            role="menuitem"
                                            initial={{ opacity: 0, x: 6 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ duration: 0.15 }}
                                            onClick={handleCopyToClipboard}
                                            disabled={isExporting}
                                            aria-label={copyState === 'copied' ? 'Copied to clipboard' : 'Copy image to clipboard'}
                                            className="neo-export-item"
                                            style={{
                                                display: 'flex',
                                                width: '100%',
                                                alignItems: 'center',
                                                gap: '0.6rem',
                                                padding: '0.5rem 0.55rem',
                                                borderRadius: '3px',
                                                border: '1.5px solid transparent',
                                                background: 'transparent',
                                                cursor: isExporting ? 'wait' : 'pointer',
                                                textAlign: 'left',
                                                transition: 'background 120ms, border-color 120ms, box-shadow 120ms',
                                                marginBottom: '0.2rem',
                                            }}
                                            onMouseEnter={e => {
                                                e.currentTarget.style.background = `rgba(${accentRgb}, 0.12)`
                                                e.currentTarget.style.borderColor = `rgba(${accentRgb}, 0.5)`
                                                e.currentTarget.style.boxShadow = `2px 2px 0 0 rgba(${accentRgb}, 0.4)`
                                            }}
                                            onMouseLeave={e => {
                                                e.currentTarget.style.background = 'transparent'
                                                e.currentTarget.style.borderColor = 'transparent'
                                                e.currentTarget.style.boxShadow = 'none'
                                            }}
                                            onMouseDown={e => {
                                                e.currentTarget.style.transform = 'translate(2px, 2px)'
                                                e.currentTarget.style.boxShadow = 'none'
                                            }}
                                            onMouseUp={e => {
                                                e.currentTarget.style.transform = ''
                                                e.currentTarget.style.boxShadow = `2px 2px 0 0 rgba(${accentRgb}, 0.4)`
                                            }}
                                        >
                                            <div
                                                style={{
                                                    display: 'grid',
                                                    placeItems: 'center',
                                                    width: '1.65rem',
                                                    height: '1.65rem',
                                                    flexShrink: 0,
                                                    borderRadius: '3px',
                                                    border: '1.5px solid rgba(244, 244, 245, 0.5)',
                                                    background: copyState === 'copied' ? '#9BF95B' : accent,
                                                    color: copyState === 'copied' ? '#03050A' : textOnAccent,
                                                    boxShadow: '1.5px 1.5px 0 0 rgba(244, 244, 245, 0.3)',
                                                    transition: 'background 300ms cubic-bezier(0.22, 1, 0.36, 1)',
                                                }}
                                            >
                                                {copyState === 'copying'
                                                    ? <Loader2 className="h-3 w-3 animate-spin" strokeWidth={2.5} />
                                                    : copyState === 'copied'
                                                        ? <Check className="h-3 w-3" strokeWidth={2.8} />
                                                        : <Copy className="h-3 w-3" strokeWidth={2.5} />}
                                            </div>
                                            <div style={{ minWidth: 0 }}>
                                                <div
                                                    style={{
                                                        fontFamily: 'var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace)',
                                                        fontSize: '0.65rem',
                                                        fontWeight: 700,
                                                        letterSpacing: '0.06em',
                                                        textTransform: 'uppercase',
                                                        color: '#F4F4F5',
                                                        lineHeight: 1.3,
                                                    }}
                                                >
                                                    {copyState === 'copied' ? 'Copied!' : 'Copy to clipboard'}
                                                </div>
                                                <div
                                                    style={{
                                                        fontSize: '0.58rem',
                                                        fontWeight: 500,
                                                        color: 'rgba(161, 168, 180, 0.75)',
                                                        lineHeight: 1.3,
                                                        marginTop: '1px',
                                                    }}
                                                >
                                                    PNG · paste into Slack, Notion, anywhere
                                                </div>
                                            </div>
                                        </motion.button>

                                        {/* Divider between the clipboard shortcut and the download presets. */}
                                        <div
                                            aria-hidden="true"
                                            style={{
                                                height: '1px',
                                                margin: '0.25rem 0.1rem 0.35rem',
                                                background: 'rgba(244, 244, 245, 0.12)',
                                            }}
                                        />

                                        {EXPORT_PRESETS.map((preset, idx) => (
                                            <motion.button
                                                key={preset.id}
                                                type="button"
                                                role="menuitem"
                                                initial={{ opacity: 0, x: 6 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                transition={{ delay: (idx + 1) * 0.04, duration: 0.15 }}
                                                onClick={() => handleExport(preset)}
                                                disabled={isExporting}
                                                aria-label={`Download as ${preset.label}${exportScale > 1 ? ` at ${exportScale}× resolution` : ''}`}
                                                className="neo-export-item"
                                                style={{
                                                    display: 'flex',
                                                    width: '100%',
                                                    alignItems: 'center',
                                                    gap: '0.6rem',
                                                    padding: '0.5rem 0.55rem',
                                                    borderRadius: '3px',
                                                    border: '1.5px solid transparent',
                                                    background: 'transparent',
                                                    cursor: isExporting ? 'wait' : 'pointer',
                                                    textAlign: 'left',
                                                    transition: 'background 120ms, border-color 120ms, box-shadow 120ms',
                                                }}
                                                onMouseEnter={e => {
                                                    e.currentTarget.style.background = `rgba(${accentRgb}, 0.12)`
                                                    e.currentTarget.style.borderColor = `rgba(${accentRgb}, 0.5)`
                                                    e.currentTarget.style.boxShadow = `2px 2px 0 0 rgba(${accentRgb}, 0.4)`
                                                }}
                                                onMouseLeave={e => {
                                                    e.currentTarget.style.background = 'transparent'
                                                    e.currentTarget.style.borderColor = 'transparent'
                                                    e.currentTarget.style.boxShadow = 'none'
                                                }}
                                                onMouseDown={e => {
                                                    e.currentTarget.style.transform = 'translate(2px, 2px)'
                                                    e.currentTarget.style.boxShadow = 'none'
                                                }}
                                                onMouseUp={e => {
                                                    e.currentTarget.style.transform = ''
                                                    e.currentTarget.style.boxShadow = `2px 2px 0 0 rgba(${accentRgb}, 0.4)`
                                                }}
                                            >
                                                {/* Icon tile */}
                                                <div
                                                    style={{
                                                        display: 'grid',
                                                        placeItems: 'center',
                                                        width: '1.65rem',
                                                        height: '1.65rem',
                                                        flexShrink: 0,
                                                        borderRadius: '3px',
                                                        border: '1.5px solid rgba(244, 244, 245, 0.5)',
                                                        background: accent,
                                                        color: textOnAccent,
                                                        boxShadow: '1.5px 1.5px 0 0 rgba(244, 244, 245, 0.3)',
                                                        transition: 'background 300ms cubic-bezier(0.22, 1, 0.36, 1)',
                                                    }}
                                                >
                                                    <Download className="h-3 w-3" strokeWidth={2.5} />
                                                </div>

                                                {/* Label */}
                                                <div style={{ minWidth: 0 }}>
                                                    <div
                                                        style={{
                                                            fontFamily: 'var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace)',
                                                            fontSize: '0.65rem',
                                                            fontWeight: 700,
                                                            letterSpacing: '0.06em',
                                                            textTransform: 'uppercase',
                                                            color: '#F4F4F5',
                                                            lineHeight: 1.3,
                                                        }}
                                                    >
                                                        {preset.label}
                                                    </div>
                                                    <div
                                                        style={{
                                                            fontSize: '0.58rem',
                                                            fontWeight: 500,
                                                            color: 'rgba(161, 168, 180, 0.75)',
                                                            lineHeight: 1.3,
                                                            marginTop: '1px',
                                                        }}
                                                    >
                                                        {preset.description}
                                                    </div>
                                                </div>
                                            </motion.button>
                                        ))}
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>
            </div>

            <UpgradeModel
                isOpen={showUpgradeModel}
                onClose={() => {
                    setShowUpgradeModel(false)
                    setRestrictedTool(null)
                }}
                restrictedTool={restrictedTool}
                isPro={isPro}
                reason={
                    restrictedTool === "export"
                        ? "Free plan is limited to 20 exports per month. Upgrade to Pro for unlimited exports"
                        : undefined
                }
            />

            <ShortcutsGuide open={showShortcuts} onClose={() => setShowShortcuts(false)} />
        </>
    )
}

export default EditorTopbar
