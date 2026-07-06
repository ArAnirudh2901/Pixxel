"use client"

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Eraser, Lasso, Loader2, MousePointerClick, Sparkles, Wand2, X } from 'lucide-react'
import { toast } from 'sonner'
import { motion } from 'framer-motion'
import { useCanvas } from '../../../../../../../context/context'
import usePlanAccess from '../../../../../../../hooks/usePlanAccess'
import usePixelMaskTool, { MIN_BRUSH, MAX_BRUSH } from '../../../../../../../hooks/usePixelMaskTool'
import {
    BrushSizeControl,
    LabeledSlider,
    MaskActionButtons,
    ModeToggle,
    TipCard,
    ToolEmptyState,
} from './_pixel-tool-ui'
import { buildImageKitBackgroundRemovalUrls } from '@/lib/imagekit-ai'
import { getRoutingMode, setRoutingMode, subscribeRouting } from '@/lib/ai-routing'


const MAX_BG_DIMENSION = 1600
const PRO_BG_TOOL = 'ai_background'



const buildBackgroundRemovalUrls = (project) => {
    // Prefer originalImageUrl — currentImageUrl may already contain AI transforms
    // (e-bgremove, e-upscale, e-genfill, etc.) that ImageKit rejects with a 400
    // when chained with another e-bgremove. The original is always the clean source.
    const original = project?.originalImageUrl
    const current = project?.currentImageUrl
    const imageUrl = (original?.includes('ik.imagekit.io') ? original : current) || original || current
    return buildImageKitBackgroundRemovalUrls(imageUrl, {
        width: project?.width,
        height: project?.height,
        maxDimension: MAX_BG_DIMENSION,
    })
}

const loadImageElement = (src) =>
    new Promise((resolve, reject) => {
        const img = new Image()
        img.crossOrigin = 'anonymous'
        img.onload = () => resolve(img)
        img.onerror = () => reject(new Error('Failed to load processed image'))
        img.src = src
    })

const getReadableResponseText = async (response) => {
    const ikError = response.headers.get('ik-error') || ''
    const text = await response.text().catch(() => '')
    const body = text
        .replace(/<[^>]*>/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
    return [ikError, body]
        .filter(Boolean)
        .join(': ')
        .slice(0, 180)
}

// Poll ImageKit's bgremove endpoint. While the asset is still being prepared it
// returns an intermediate (non-image) response; once ready it streams the PNG.
const fetchProcessedImage = async (url, { attempts = 7, signal, onStatus } = {}) => {
    let delay = 1200
    let lastError = null
    for (let attempt = 0; attempt < attempts; attempt += 1) {
        if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')

        let response = null
        try {
            response = await fetch(url, { mode: 'cors', cache: 'no-store', signal })
        } catch (error) {
            // Only transient/network errors are retried; aborts bubble immediately.
            if (error.name === 'AbortError') throw error
            lastError = error
        }

        if (response) {
            const contentType = response.headers.get('content-type') || ''
            const intermediate = response.headers.get('is-intermediate-response') === 'true'

            if (response.ok && contentType.startsWith('image/') && !intermediate) {
                const blob = await response.blob()
                return URL.createObjectURL(blob)
            }
            // A definitive HTTP error (not 202 Accepted / 425 Too Early, which mean
            // "still preparing") fails fast — throwing OUTSIDE the try so it isn't
            // swallowed and retried for the full ~25s budget.
            if (!response.ok && response.status >= 400 && response.status !== 425 && response.status !== 202) {
                const detail = await getReadableResponseText(response)
                throw new Error(`Background service error (${response.status})${detail ? `: ${detail}` : ''}`)
            }
            // Otherwise it's an intermediate "still preparing" response — fall through and retry.
        }

        if (attempt < attempts - 1) {
            onStatus?.(`AI is isolating the subject… (${attempt + 2}/${attempts})`)
            await new Promise((resolve) => setTimeout(resolve, delay))
            delay = Math.min(delay + 1200, 6000)
        }
    }
    throw lastError || new Error('Background removal timed out — try again in a moment')
}

const EraseControls = ({ project, dominantColor }) => {
    const { canvasEditor, processingMessage, setProcessingMessage, registerProcessingAbort } = useCanvas()
    const { hasAccess } = usePlanAccess()

    // "Circle to Remove" flips the brush into a deferred lasso: draw a freehand
    // loop, its interior is highlighted as a pending selection, then you
    // Generative-Fill / Erase it. Off → the usual instant erase brush.
    const [circleSelect, setCircleSelect] = useState(false)
    const tool = usePixelMaskTool({
        canvasEditor,
        defaultMode: 'erase',
        supportsMagic: true,
        deferApply: circleSelect,
        inferRegion: false,
        showOverlay: circleSelect,
        livePreview: circleSelect,
        fillEnclosed: circleSelect,
    })
    const [isAutoErasing, setIsAutoErasing] = useState(false)
    const [maskServiceStatus, setMaskServiceStatus] = useState(null) // null | 'checking' | 'warm' | 'cold' | 'warming' | 'unavailable'
    const [inpaintMode, setInpaintMode] = useState(() => getRoutingMode('inpaint'))

    useEffect(() => {
        setInpaintMode(getRoutingMode('inpaint'))
        return subscribeRouting(() => setInpaintMode(getRoutingMode('inpaint')))
    }, [])
    const abortRef = useRef(null)
    const warmupTriggeredRef = useRef(false)
    const { setMagic, setMode, setObjectSelect, discardPending } = tool

    // Circle-to-Remove is mutually exclusive with the click/flood modes.
    const toggleCircleSelect = useCallback(() => {
        setCircleSelect((on) => {
            const next = !on
            if (next) {
                setObjectSelect(false)
                setMagic(false)
            } else {
                discardPending?.()
            }
            return next
        })
    }, [setObjectSelect, setMagic, discardPending])

    // If a click/flood mode is turned on (here or via the radial menu), drop out
    // of Circle-to-Remove and clear any pending lasso selection.
    useEffect(() => {
        if ((tool.objectSelect || tool.magic) && circleSelect) {
            setCircleSelect(false)
            discardPending?.()
        }
    }, [tool.objectSelect, tool.magic, circleSelect, discardPending])

    // Proactively check mask service health and trigger warmup on first mount
    useEffect(() => {
        if (warmupTriggeredRef.current) return
        warmupTriggeredRef.current = true
        let cancelled = false
        ;(async () => {
            try {
                setMaskServiceStatus('checking')
                const resp = await fetch('/api/ai/warmup', { signal: AbortSignal.timeout(5000) })
                if (cancelled) return
                if (!resp.ok) {
                    setMaskServiceStatus('unavailable')
                    return
                }
                const data = await resp.json()
                if (!data.configured) {
                    setMaskServiceStatus('unavailable')
                    return
                }
                if (data.allWarm) {
                    setMaskServiceStatus('warm')
                    return
                }
                // Models not all loaded — trigger warmup in background
                setMaskServiceStatus('warming')
                fetch('/api/ai/warmup', { method: 'POST' })
                    .then((r) => r.ok ? r.json() : null)
                    .then(() => { if (!cancelled) setMaskServiceStatus('warm') })
                    .catch(() => { if (!cancelled) setMaskServiceStatus('cold') })
            } catch {
                if (!cancelled) setMaskServiceStatus('cold')
            }
        })()
        return () => { cancelled = true }
    }, [])

    const canUseAi = hasAccess(PRO_BG_TOOL)
    const backgroundRemovalUrls = useMemo(() => buildBackgroundRemovalUrls(project), [project])

    useEffect(() => () => abortRef.current?.abort(), [])

    // React to radial-menu sub-actions (Brush / Magic / Auto BG / Restore).
    const handleAutoEraseRef = useRef(null)
    useEffect(() => {
        const onSub = (event) => {
            const { toolId, subId } = event.detail || {}
            if (toolId !== 'erase' || !subId) return
            setCircleSelect(false)
            if (subId === 'magic') setMagic(true)
            else if (subId === 'brush') { setMagic(false); setObjectSelect(false); setMode('erase') }
            else if (subId === 'restore') { setMagic(false); setObjectSelect(false); setMode('restore') }
            else if (subId === 'auto') handleAutoEraseRef.current?.()
        }
        window.addEventListener('phosmith:tool-sub', onSub)
        return () => window.removeEventListener('phosmith:tool-sub', onSub)
        // setMagic/setMode/setObjectSelect are stable and handleAutoEraseRef is a
        // ref, so the listener binds once instead of re-binding every render.
    }, [setMagic, setMode, setObjectSelect])

    const handleAutoErase = useCallback(async () => {
        if (!canvasEditor || isAutoErasing) return
        if (!canUseAi) {
            toast.error('AI auto-erase is a Pro feature')
            return
        }
        if (!backgroundRemovalUrls.length) {
            toast.error('Auto-erase needs an ImageKit-hosted image')
            return
        }

        abortRef.current?.abort()
        const controller = new AbortController()
        abortRef.current = controller

        setIsAutoErasing(true)
        setProcessingMessage('AI is isolating the subject…')
        registerProcessingAbort?.(controller)
        let objectUrl = null
        try {
            let lastError = null
            for (let index = 0; index < backgroundRemovalUrls.length; index += 1) {
                try {
                    objectUrl = await fetchProcessedImage(backgroundRemovalUrls[index], {
                        signal: controller.signal,
                        onStatus: setProcessingMessage,
                    })
                    break
                } catch (error) {
                    lastError = error
                    if (error.name === 'AbortError') throw error
                    const canTryFallback =
                        index < backgroundRemovalUrls.length - 1 &&
                        /Background service error \(400\)|rejected|400/i.test(error?.message || '')
                    if (!canTryFallback) throw error
                    setProcessingMessage('Retrying background removal…')
                }
            }
            if (!objectUrl) throw lastError || new Error('Auto-erase failed')
            const imageEl = await loadImageElement(objectUrl)
            const applied = tool.applyAlphaMask(imageEl)
            if (applied) toast.success('Background erased — refine with the brush')
            else toast.error('Could not apply auto-erase to this image')
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.warn('[erase] auto-erase failed:', error)
                toast.error(error?.message || 'Auto-erase failed')
            }
        } finally {
            if (objectUrl) URL.revokeObjectURL(objectUrl)
            setProcessingMessage(null)
            setIsAutoErasing(false)
            if (abortRef.current === controller) abortRef.current = null
        }
    }, [canvasEditor, isAutoErasing, canUseAi, backgroundRemovalUrls, setProcessingMessage, tool])

    // Keep the ref pointing at the latest handler so the sub-action listener can
    // call it without re-binding on every dependency change.
    useEffect(() => { handleAutoEraseRef.current = handleAutoErase }, [handleAutoErase])

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
                icon={Eraser}
                title="No image on canvas"
                subtitle="Add an image first, then use the erase tool"
            />
        )
    }

    const autoBusy = isAutoErasing || Boolean(processingMessage)
    const hasFillSelection = tool.hasMask || tool.hasPending
    const fillBusy = tool.isObjectRunning && tool.objectPhase === 'filling'

    return (
        <div className="space-y-4 overflow-y-auto pr-1 panel-scroll">
            {/* AI auto-erase — one-click background removal (CapCut-style) */}
            <div className="space-y-2">
                <label className="panel-label">Auto-Erase Background</label>
                <button
                    type="button"
                    onClick={handleAutoErase}
                    disabled={autoBusy || !canUseAi}
                    className="flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2.5 text-xs font-semibold editor-interactive disabled:opacity-50"
                    style={{ background: 'var(--accent-primary)', color: '#03050A', border: 'none', boxShadow: 'var(--shadow-glow)' }}
                >
                    {isAutoErasing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                    {isAutoErasing ? 'Erasing…' : 'Erase Background with AI'}
                </button>
                {!canUseAi && (
                    <p className="text-[11px]" style={{ color: 'var(--accent-warning)' }}>
                        ⚠ Pro feature — upgrade to auto-erase backgrounds
                    </p>
                )}
                {canUseAi && !backgroundRemovalUrls.length && (
                    <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
                        Needs an ImageKit-hosted image. You can still erase manually below.
                    </p>
                )}
            </div>

            {/* AI object eraser — SAM 3: click an object, the model segments
                the WHOLE object under the pointer and erases it. Click more
                objects to erase each (multi-subject by accumulation). */}
            <div className="space-y-2" style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
                <div className="flex items-center justify-between gap-2">
                    <label className="panel-label" style={{ margin: 0 }}>AI Object Remover</label>
                    <div className="mask-fill-modes" style={{ marginTop: 0 }}>
                        {(['auto', 'client', 'server']).map((mode) => (
                            <button
                                key={mode}
                                type="button"
                                onClick={() => setRoutingMode('inpaint', mode)}
                                className={`mask-fill-mode-btn ${inpaintMode === mode ? 'mask-fill-mode-btn--active' : ''}`}
                                title={mode === 'client' ? 'LaMa (local mask service)'
                                    : mode === 'server' ? 'Stable Diffusion (Hugging Face)'
                                    : 'LaMa first, SD fallback'}
                            >
                                {mode === 'auto' ? 'Auto' : mode === 'client' ? 'Local' : 'Cloud'}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Warmup status banner */}
                {maskServiceStatus === 'warming' && (
                    <div className="flex items-center gap-2 rounded-lg px-3 py-2 text-[10px]"
                        style={{ background: 'rgba(251, 191, 36, 0.08)', border: '1px solid rgba(251,191,36,0.2)', color: 'rgba(251,191,36,0.9)' }}>
                        <Loader2 className="h-3 w-3 animate-spin shrink-0" />
                        AI models are warming up in the background… first click will be fast once ready
                    </div>
                )}
                {maskServiceStatus === 'cold' && (
                    <div className="flex items-center gap-2 rounded-lg px-3 py-2 text-[10px]"
                        style={{ background: 'rgba(251, 191, 36, 0.06)', border: '1px solid rgba(251,191,36,0.15)', color: 'rgba(251,191,36,0.7)' }}>
                        ⚡ First click may take 30–90s while AI models load
                    </div>
                )}
                {maskServiceStatus === 'unavailable' && (
                    <div className="flex items-center gap-2 rounded-lg px-3 py-2 text-[10px]"
                        style={{ background: 'rgba(239, 68, 68, 0.06)', border: '1px solid rgba(239,68,68,0.15)', color: 'rgba(239,68,68,0.7)' }}>
                        ⚠ AI service not connected — object remover unavailable
                    </div>
                )}

                <motion.button
                    type="button"
                    onClick={() => tool.setObjectSelect(!tool.objectSelect)}
                    whileTap={{ scale: 0.97 }}
                    className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left editor-interactive"
                    style={{
                        background: tool.objectSelect ? 'rgba(124, 58, 237, 0.1)' : 'transparent',
                        border: `1px solid ${tool.objectSelect ? 'rgba(124,58,237,0.6)' : 'var(--border-subtle)'}`,
                        color: tool.objectSelect ? '#C4B5FD' : 'var(--text-secondary)',
                    }}
                >
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg shrink-0"
                        style={{
                            background: tool.objectSelect ? 'rgba(124,58,237,0.15)' : 'var(--bg-elevated)',
                            border: `1px solid ${tool.objectSelect ? 'rgba(124,58,237,0.5)' : 'var(--border-default)'}`,
                        }}>
                        {tool.isObjectRunning
                            ? <Loader2 className="h-4 w-4 animate-spin" />
                            : <Sparkles className="h-4 w-4" />}
                    </div>
                    <div className="min-w-0">
                        <div className="text-xs font-semibold">
                            {tool.isObjectRunning
                                ? (tool.objectPhase === 'filling' ? 'Filling background…' : 'Detecting object…')
                                : tool.objectSelect ? 'Click-to-remove: ON' : 'Click-to-remove: OFF'}
                        </div>
                        <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                            {tool.isObjectRunning
                                ? (tool.objectPhase === 'filling'
                                    ? 'AI is generating the background texture'
                                    : 'SAM 3 is segmenting the object you clicked')
                                : 'SAM 3 detects the object, AI fills the background — click each subject to remove'}
                        </div>
                    </div>
                </motion.button>
            </div>



            {/* Circle to Remove — freehand lasso: draw a loop, its interior is
                highlighted as a selection, then Generative-Fill / Erase it. */}
            <div className="space-y-2" style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
                <motion.button
                    type="button"
                    onClick={toggleCircleSelect}
                    whileTap={{ scale: 0.97 }}
                    className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left editor-interactive"
                    style={{
                        background: circleSelect ? 'rgba(56, 189, 248, 0.1)' : 'transparent',
                        border: `1px solid ${circleSelect ? 'rgba(56,189,248,0.6)' : 'var(--border-subtle)'}`,
                        color: circleSelect ? '#7DD3FC' : 'var(--text-secondary)',
                    }}
                >
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg shrink-0"
                        style={{
                            background: circleSelect ? 'rgba(56,189,248,0.15)' : 'var(--bg-elevated)',
                            border: `1px solid ${circleSelect ? 'rgba(56,189,248,0.5)' : 'var(--border-default)'}`,
                        }}>
                        <Lasso className="h-4 w-4" />
                    </div>
                    <div className="min-w-0">
                        <div className="text-xs font-semibold">
                            {circleSelect ? 'Circle to Remove: ON' : 'Circle to Remove: OFF'}
                        </div>
                        <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                            Draw a loop around something — its inside highlights, then erase or generative-fill it
                        </div>
                    </div>
                </motion.button>
            </div>

            <div className="space-y-2" style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
                <label className="panel-label">Generative Fill</label>
                <motion.button
                    type="button"
                    onClick={() => tool.generativeFill()}
                    disabled={!hasFillSelection || tool.isObjectRunning}
                    whileTap={{ scale: hasFillSelection && !tool.isObjectRunning ? 0.97 : 1 }}
                    className="flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2.5 text-xs font-semibold editor-interactive disabled:opacity-50"
                    style={{ background: 'rgba(124, 58, 237, 0.12)', border: '1px solid rgba(124,58,237,0.6)', color: '#C4B5FD' }}
                >
                    {fillBusy
                        ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        : <Wand2 className="h-3.5 w-3.5" />}
                    {fillBusy ? 'Filling…' : 'Generative Fill'}
                </motion.button>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    {hasFillSelection
                        ? 'Fills the selected area with generated background'
                        : 'Paint or circle the area to replace, then fill it'}
                </p>
            </div>

            {/* Mode */}
            <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
                <ModeToggle mode={tool.mode} setMode={tool.setMode} altActive={tool.altActive} />
            </div>

            {/* Brush controls — hidden when a click mode (magic / AI object) is active */}
            {!tool.magic && !tool.objectSelect && (
                <div className="space-y-3" style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
                    <BrushSizeControl value={tool.brushSize} setValue={tool.setBrushSize} min={MIN_BRUSH} max={MAX_BRUSH} dominantColor={dominantColor} />
                    {!circleSelect && (
                        <>
                            <LabeledSlider label="Hardness" value={tool.hardness} min={1} max={100} suffix="%" onChange={tool.setHardness} dominantColor={dominantColor} />
                            <LabeledSlider label="Flow" value={tool.flow} min={5} max={100} suffix="%" onChange={tool.setFlow} dominantColor={dominantColor} />
                        </>
                    )}

                    {circleSelect ? (
                        tool.hasPending ? (
                            <div className="grid grid-cols-2 gap-2">
                                <button
                                    type="button"
                                    onClick={() => tool.commitErase()}
                                    disabled={tool.isObjectRunning}
                                    className="flex items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium editor-interactive disabled:opacity-50"
                                    style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', color: 'var(--text-secondary)' }}
                                >
                                    <Eraser className="h-3.5 w-3.5" /> Erase
                                </button>
                                <button
                                    type="button"
                                    onClick={() => tool.discardPending()}
                                    disabled={tool.isObjectRunning}
                                    className="flex items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium editor-interactive disabled:opacity-50"
                                    style={{ background: 'transparent', border: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}
                                >
                                    <X className="h-3.5 w-3.5" /> Clear
                                </button>
                            </div>
                        ) : (
                            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                                Draw a loop around what you want to remove — its inside highlights, then Generative Fill or Erase it
                            </p>
                        )
                    ) : (
                        <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                            {tool.hasMask
                                ? 'Erases the painted area and fills it with AI-generated background'
                                : 'Paint over (or draw a shape around) what to remove, then fill it with AI'}
                        </p>
                    )}
                </div>
            )}

            {/* Edge feather */}
            <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
                <LabeledSlider label="Edge Feather" value={tool.feather} min={0} max={50} suffix="px" onChange={tool.setFeather} dominantColor={dominantColor} />
            </div>

            <MaskActionButtons
                hasMask={tool.hasMask}
                undoDepth={tool.undoDepth}
                redoDepth={tool.redoDepth}
                onUndo={tool.undo}
                onRedo={tool.redo}
                onInvert={tool.invert}
                onClear={tool.clear}
            />

            <TipCard>
                <p>• <strong>Paint</strong> over what you want to erase — the exact stroke is removed when you release</p>
                <p>• <strong>Circle to Remove</strong>: draw a freehand loop — the enclosed area is highlighted, then Generative Fill or Erase it</p>
                <p>• <strong>Generative Fill</strong>: scribble over (or draw a shape around) something, then hit the button — AI erases inside and fills the background</p>
                <p>• <strong>AI object remover</strong>: click an object and AI removes it + fills the background — click each subject to remove several</p>
                <p>• Hold <strong>Alt</strong> to temporarily switch Erase ↔ Restore</p>
                <p>• <strong>[</strong> / <strong>]</strong> resize the brush; raise feather for soft edges</p>
            </TipCard>

            <div className="flex items-center gap-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                <MousePointerClick className="h-3 w-3" />
                Non-destructive — restore brings pixels back anytime
            </div>
        </div>
    )
}

export default EraseControls
