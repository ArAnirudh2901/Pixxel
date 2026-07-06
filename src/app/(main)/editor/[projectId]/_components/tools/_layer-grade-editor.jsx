'use client'

/**
 * LayerGradeEditor — the "pro" colour-grading controls for a single megashader
 * layer (or a full-frame base/global layer): a **gamma** slider, a **tone-curve**
 * editor (RGB / R / G / B / Luma) with a histogram, and **3-way colour wheels**
 * (shadows / midtones / highlights). It is the on-screen surface for the
 * per-layer grading the engine already renders (`glsl-fragments.js`
 * `buildLayerAdjustFunction`).
 *
 * Reuse-first: the curve maths is the Fabric-free `curve-lut.js`; the colour
 * conversions are `color-utils.js`; the sliders are the shared `ProRulerSlider`.
 * The curve-editor / colour-wheel markup is ported from the production Adjust
 * tool (`adjust.jsx`) and shares its CSS (`.adjust-curve-*`, `.adjust-wheel-*`).
 *
 * Data contract (matches `mask-types.js#sanitiseLayer`):
 *   - gamma   : number, identity 1, 0.2..2.2 — written via `onUpdate({ gamma })`.
 *   - wheels  : `wheelShadows` / `wheelMidtones` / `wheelHighlights`, each a
 *               signed `[r,g,b]` offset in -1..1 — `onUpdate({ wheelShadows })`.
 *   - curves  : `{ master, r, g, b }` of `{x,y}` points in 0..1 — applied via
 *               `onApplyCurve(layer.id, curves)` (builds the packed LUT, clears
 *               on identity). Curves are disabled if `onApplyCurve` is absent.
 *
 * @module _layer-grade-editor
 */

import React, { useEffect, useRef, useState } from 'react'
import { RotateCcw } from 'lucide-react'
import { ProRulerSlider } from '@/components/editor/ProRulerSlider'
import { buildCurveSvgPath, arePointsIdentity } from '@/lib/curve-lut'
import { hsvToRgb } from '@/lib/color-utils'

/* ── shared geometry / helpers ─────────────────────────────────────────────── */
const CURVE_GRAPH = { left: 8, top: 8, right: 92, bottom: 92 }
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))
const cloneIdentity = () => [{ x: 0, y: 0 }, { x: 1, y: 1 }]

// Sort/clamp curve points, pin the endpoints to x=0 / x=1 (ported from adjust.jsx).
const sanitizeCurvePoints = (raw) => {
    const arr = Array.isArray(raw) ? raw : []
    const cleaned = arr
        .filter((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y))
        .map((p) => ({ x: clamp(p.x, 0, 1), y: clamp(p.y, 0, 1) }))
        .sort((a, b) => a.x - b.x)
    if (cleaned.length < 2) return cloneIdentity()
    cleaned[0] = { x: 0, y: cleaned[0].y }
    cleaned[cleaned.length - 1] = { x: 1, y: cleaned[cleaned.length - 1].y }
    return cleaned
}

// Tone-curve channels. `ch` is the key in the engine's `curves` object; RGB and
// Luma both edit the `master` curve (Luma is just a luminance-framed alias).
const CURVE_CHANNELS = [
    { id: 'rgb', label: 'RGB', color: '#e5e7eb', histogramChannels: ['red', 'green', 'blue'], ch: 'master' },
    { id: 'red', label: 'Red', color: '#ff5d65', histogramChannels: ['red'], ch: 'r' },
    { id: 'green', label: 'Green', color: '#64d989', histogramChannels: ['green'], ch: 'g' },
    { id: 'blue', label: 'Blue', color: '#69a7ff', histogramChannels: ['blue'], ch: 'b' },
    { id: 'luma', label: 'Luminance', color: '#d8dde7', histogramChannels: ['luma'], ch: 'master' },
]
const HISTOGRAM_SERIES = {
    red: '#ff5d65', green: '#64d989', blue: '#69a7ff', luma: '#d8dde7',
}

const CURVE_HIT_RADIUS_VB = 4.5
const CURVE_POINT_MIN_GAP = 0.015

const clientToCurvePoint = (svgEl, clientX, clientY) => {
    if (!svgEl) return null
    const rect = svgEl.getBoundingClientRect()
    if (!rect.width || !rect.height) return null
    const vbX = ((clientX - rect.left) / rect.width) * 100
    const vbY = ((clientY - rect.top) / rect.height) * 100
    const x = (vbX - CURVE_GRAPH.left) / (CURVE_GRAPH.right - CURVE_GRAPH.left)
    const y = 1 - (vbY - CURVE_GRAPH.top) / (CURVE_GRAPH.bottom - CURVE_GRAPH.top)
    return { x: clamp(x, 0, 1), y: clamp(y, 0, 1), vbX, vbY }
}

const findNearestPointIndex = (points, vbX, vbY) => {
    let best = -1
    let bestDist = CURVE_HIT_RADIUS_VB
    for (let i = 0; i < points.length; i++) {
        const p = points[i]
        const px = CURVE_GRAPH.left + p.x * (CURVE_GRAPH.right - CURVE_GRAPH.left)
        const py = CURVE_GRAPH.bottom - p.y * (CURVE_GRAPH.bottom - CURVE_GRAPH.top)
        const d = Math.hypot(vbX - px, vbY - py)
        if (d < bestDist) { bestDist = d; best = i }
    }
    return best
}

// Histogram → SVG path inside the curve viewBox (log-scaled, ported from adjust.jsx).
const buildHistogramPaths = (series) => {
    if (!Array.isArray(series) || series.length === 0) return null
    const maxValue = Math.max(...series)
    if (!maxValue) return null
    const { left, right, top, bottom } = CURVE_GRAPH
    const width = right - left
    const height = bottom - top
    const maxLog = Math.log1p(maxValue)
    const points = series.map((value, index) => {
        const x = left + (index / Math.max(1, series.length - 1)) * width
        const y = bottom - clamp(Math.log1p(value) / maxLog, 0, 1) * height
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
    }).join(' ')
    return { line: points, fill: `${points} L ${right} ${bottom} L ${left} ${bottom} Z` }
}

/* ── colour-wheel maths: signed RGB offset ⇆ 2-D pad position ───────────────── */
// A 3-way wheel is a luma-weighted vec3 RGB offset (-1..1) per tonal range. We
// map the pad's polar position to a zero-sum CHROMA offset (no luminance shift),
// and a separate "luma" slider sets the equal (r=g=b) component. The hue basis
// places R at 0°, G at 120°, B at 240°; because Σcos = Σsin = 0 the forward map
// is automatically zero-sum and the inverse is exact (see notes inline).
const TWO_PI_3 = (2 * Math.PI) / 3
const HUE_BASIS = [
    [Math.cos(0), Math.sin(0)],
    [Math.cos(TWO_PI_3), Math.sin(TWO_PI_3)],
    [Math.cos(2 * TWO_PI_3), Math.sin(2 * TWO_PI_3)],
]
const MAX_TINT = 0.6   // chroma offset at the rim of the wheel
const MAX_LUMA = 0.5   // luma offset at the ends of the luma slider

// pad (x,y in -1..1, y up) + luma → engine offset [r,g,b]
const xyToOffset = (x, y, luma) =>
    HUE_BASIS.map(([bx, by]) => clamp((x * bx + y * by) * MAX_TINT + luma, -1, 1))

// engine offset → pad (x,y); luma is the equal component (mean of the offset)
const offsetToXY = (off) => {
    const m = (off[0] + off[1] + off[2]) / 3
    let x = 0
    let y = 0
    for (let i = 0; i < 3; i++) { x += (off[i] - m) * HUE_BASIS[i][0]; y += (off[i] - m) * HUE_BASIS[i][1] }
    const k = 1.5 * MAX_TINT // Σbx² = Σby² = 3/2 (see note above)
    return [clamp(x / k, -1, 1), clamp(y / k, -1, 1)]
}
const offsetLuma = (off) => (off[0] + off[1] + off[2]) / 3

const isWheelActive = (off) => Array.isArray(off) && off.some((v) => Math.abs(v) > 1e-4)

/* ── tone-curve graph (one channel) ────────────────────────────────────────── */
function CurveGraph({ points, color, histogramChannels, histogram, onChange }) {
    const svgRef = useRef(null)
    const dragRef = useRef(null)
    const [activeIndex, setActiveIndex] = useState(null)
    const [hoverIndex, setHoverIndex] = useState(null)

    const path = buildCurveSvgPath(points, CURVE_GRAPH)

    useEffect(() => {
        if (activeIndex === null) return undefined
        const handleMove = (event) => {
            const drag = dragRef.current
            if (!drag || !svgRef.current) return
            const loc = clientToCurvePoint(svgRef.current, event.clientX, event.clientY)
            if (!loc) return
            const next = drag.points.map((p) => ({ ...p }))
            const isFirst = drag.index === 0
            const isLast = drag.index === next.length - 1
            const prev = next[drag.index - 1]
            const succ = next[drag.index + 1]
            const minX = isFirst ? 0 : (prev ? prev.x + CURVE_POINT_MIN_GAP : 0)
            const maxX = isLast ? 1 : (succ ? succ.x - CURVE_POINT_MIN_GAP : 1)
            next[drag.index] = {
                x: isFirst ? 0 : isLast ? 1 : clamp(loc.x, minX, maxX),
                y: clamp(loc.y, 0, 1),
            }
            dragRef.current = { ...drag, points: next }
            onChange(sanitizeCurvePoints(next))
        }
        const handleUp = () => {
            const drag = dragRef.current
            if (drag) onChange(sanitizeCurvePoints(drag.points))
            dragRef.current = null
            setActiveIndex(null)
        }
        window.addEventListener('pointermove', handleMove)
        window.addEventListener('pointerup', handleUp)
        window.addEventListener('pointercancel', handleUp)
        return () => {
            window.removeEventListener('pointermove', handleMove)
            window.removeEventListener('pointerup', handleUp)
            window.removeEventListener('pointercancel', handleUp)
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [activeIndex])

    const beginDrag = (index, event) => {
        if (event.button !== 0 && event.button !== undefined) return
        event.preventDefault(); event.stopPropagation()
        dragRef.current = { index, points: points.map((p) => ({ ...p })) }
        setActiveIndex(index)
    }

    const handlePointerDownGraph = (event) => {
        if (event.button !== 0 && event.button !== undefined) return
        const svg = svgRef.current
        if (!svg) return
        const loc = clientToCurvePoint(svg, event.clientX, event.clientY)
        if (!loc) return
        const existing = findNearestPointIndex(points, loc.vbX, loc.vbY)
        if (existing >= 0) { beginDrag(existing, event); return }
        event.preventDefault(); event.stopPropagation()
        const sanitized = sanitizeCurvePoints([...points, { x: loc.x, y: loc.y }])
        let inserted = sanitized.findIndex((p) => Math.abs(p.x - loc.x) < 1e-4 && Math.abs(p.y - loc.y) < 1e-4)
        if (inserted < 0) inserted = Math.max(1, sanitized.length - 2)
        dragRef.current = { index: inserted, points: sanitized }
        onChange(sanitized)
        setActiveIndex(inserted)
    }

    const handleDoubleClickPoint = (index) => (event) => {
        event.preventDefault(); event.stopPropagation()
        if (index === 0 || index === points.length - 1) return
        onChange(sanitizeCurvePoints(points.filter((_, i) => i !== index)))
    }

    const handleKeyDownPoint = (index) => (event) => {
        const STEP = event.shiftKey ? 0.04 : 0.008
        let dy = 0
        if (event.key === 'ArrowUp') dy = STEP
        else if (event.key === 'ArrowDown') dy = -STEP
        else return
        event.preventDefault()
        const next = points.map((p, i) => (i === index ? { ...p, y: clamp(p.y + dy, 0, 1) } : p))
        onChange(sanitizeCurvePoints(next))
    }

    const isRgb = histogramChannels.length > 1

    return (
        <div className="adjust-curve-graph">
            <svg ref={svgRef} viewBox="0 0 100 100" className="adjust-curve-svg" preserveAspectRatio="none" onPointerDown={handlePointerDownGraph}>
                {[20, 40, 60, 80].map((line) => (
                    <React.Fragment key={line}>
                        <line x1={line} y1="6" x2={line} y2="94" className="adjust-curve-grid" />
                        <line x1="6" y1={line} x2="94" y2={line} className="adjust-curve-grid" />
                    </React.Fragment>
                ))}
                {histogramChannels.map((key) => {
                    const paths = buildHistogramPaths(histogram?.[key])
                    if (!paths) return null
                    return (
                        <g key={key} className={`adjust-curve-histogram-series ${isRgb ? 'is-rgb' : 'is-single'}`} style={{ '--curve-color': HISTOGRAM_SERIES[key] }} pointerEvents="none">
                            <path d={paths.fill} className="adjust-curve-histogram-fill" />
                            <path d={paths.line} className="adjust-curve-histogram-line" />
                        </g>
                    )
                })}
                <line x1={CURVE_GRAPH.left} y1={CURVE_GRAPH.bottom} x2={CURVE_GRAPH.right} y2={CURVE_GRAPH.top} className="adjust-curve-diagonal" pointerEvents="none" />
                <path d={path} className="adjust-curve-path" style={{ '--curve-color': color }} pointerEvents="none" />
                {points.map((p, index) => {
                    const cx = CURVE_GRAPH.left + p.x * (CURVE_GRAPH.right - CURVE_GRAPH.left)
                    const cy = CURVE_GRAPH.bottom - p.y * (CURVE_GRAPH.bottom - CURVE_GRAPH.top)
                    return (
                        <g key={index} className={`adjust-curve-point-group ${activeIndex === index ? 'is-active' : ''} ${hoverIndex === index ? 'is-hover' : ''}`} style={{ '--curve-color': color }}>
                            <circle
                                cx={cx} cy={cy} r="4.5" className="adjust-curve-point-hit"
                                tabIndex={0}
                                onPointerDown={(e) => beginDrag(index, e)}
                                onDoubleClick={handleDoubleClickPoint(index)}
                                onKeyDown={handleKeyDownPoint(index)}
                                onPointerEnter={() => setHoverIndex(index)}
                                onPointerLeave={() => setHoverIndex((cur) => (cur === index ? null : cur))}
                            />
                            <circle cx={cx} cy={cy} r="1.7" className="adjust-curve-point" pointerEvents="none" />
                        </g>
                    )
                })}
            </svg>
        </div>
    )
}

/* ── one colour wheel (a tonal range) ──────────────────────────────────────── */
function ColorWheel({ label, value, onChange }) {
    const padRef = useRef(null)
    const dragRef = useRef(false)
    const off = Array.isArray(value) && value.length === 3 ? value : [0, 0, 0]
    const [px, py] = offsetToXY(off)
    const luma = offsetLuma(off)
    const hueDeg = ((Math.atan2(py, px) * 180) / Math.PI + 360) % 360
    const tint = hsvToRgb(hueDeg, 1, 1)

    const updateFromEvent = (event) => {
        const rect = padRef.current?.getBoundingClientRect()
        if (!rect) return
        const cx = rect.left + rect.width / 2
        const cy = rect.top + rect.height / 2
        let nx = (event.clientX - cx) / (rect.width / 2)
        let ny = -(event.clientY - cy) / (rect.height / 2)
        const r = Math.hypot(nx, ny)
        if (r > 1) { nx /= r; ny /= r }
        onChange(xyToOffset(nx, ny, luma))
    }
    const handlePointerDown = (event) => {
        event.preventDefault()
        dragRef.current = true
        updateFromEvent(event)
        const move = (e) => { if (dragRef.current) updateFromEvent(e) }
        const up = () => {
            dragRef.current = false
            window.removeEventListener('pointermove', move)
            window.removeEventListener('pointerup', up)
            window.removeEventListener('pointercancel', up)
        }
        window.addEventListener('pointermove', move)
        window.addEventListener('pointerup', up)
        window.addEventListener('pointercancel', up)
    }

    const active = isWheelActive(off)
    return (
        <div className="adjust-color-wheel-card grade-wheel">
            <div className="adjust-color-wheel-top">
                <div><span>{label}</span></div>
                <button
                    type="button"
                    className="adjust-curve-reset"
                    disabled={!active}
                    onClick={() => onChange([0, 0, 0])}
                    title={`Reset ${label}`}
                >
                    <RotateCcw aria-hidden="true" />
                </button>
            </div>
            <div
                ref={padRef}
                className="adjust-wheel-picker"
                role="button"
                aria-label={`${label} colour balance`}
                tabIndex={0}
                onPointerDown={handlePointerDown}
            >
                <span style={{ left: `${50 + px * 43}%`, top: `${50 - py * 43}%`, background: `rgb(${tint.r}, ${tint.g}, ${tint.b})` }} />
            </div>
            <ProRulerSlider
                variant="instrument"
                value={Math.round((luma / MAX_LUMA) * 100)}
                min={-100} max={100} step={1}
                label="Luma"
                onPreview={(v) => onChange(xyToOffset(px, py, (v / 100) * MAX_LUMA))}
                onCommit={(v) => onChange(xyToOffset(px, py, (v / 100) * MAX_LUMA))}
                visual={{ fill: 'rgba(60,72,86,0.48)', accent: '#9aa7b8', trackBg: 'rgba(14,16,20,0.98)' }}
            />
        </div>
    )
}

/* ── the editor ────────────────────────────────────────────────────────────── */
export const LayerGradeEditor = React.memo(function LayerGradeEditor({ layer, onUpdate, onApplyCurve, histogram, dominantColor = '#53d8ff' }) {
    const [activeChannel, setActiveChannel] = useState('rgb')
    const channel = CURVE_CHANNELS.find((c) => c.id === activeChannel) || CURVE_CHANNELS[0]

    const curves = layer.curves || {}
    const curvesEnabled = typeof onApplyCurve === 'function'
    const channelPoints = sanitizeCurvePoints(curves[channel.ch] || cloneIdentity())
    const channelDirty = !arePointsIdentity(channelPoints)

    const applyChannel = (chKey, nextPoints) => {
        if (!curvesEnabled) return
        const merged = {
            master: curves.master || cloneIdentity(),
            r: curves.r || cloneIdentity(),
            g: curves.g || cloneIdentity(),
            b: curves.b || cloneIdentity(),
            [chKey]: nextPoints,
        }
        onApplyCurve(layer.id, merged)
    }

    const gammaPct = Math.round((typeof layer.gamma === 'number' ? layer.gamma : 1) * 100)
    const wheels = [
        { key: 'wheelShadows', label: 'Shadows' },
        { key: 'wheelMidtones', label: 'Midtones' },
        { key: 'wheelHighlights', label: 'Highlights' },
    ]
    const anyGrade = gammaPct !== 100
        || channelDirty
        || wheels.some((w) => isWheelActive(layer[w.key]))
        || CURVE_CHANNELS.some((c) => !arePointsIdentity(sanitizeCurvePoints(curves[c.ch] || cloneIdentity())))

    const resetAll = () => {
        onUpdate({ gamma: 1, wheelShadows: [0, 0, 0], wheelMidtones: [0, 0, 0], wheelHighlights: [0, 0, 0] })
        if (curvesEnabled) onApplyCurve(layer.id, { master: cloneIdentity(), r: cloneIdentity(), g: cloneIdentity(), b: cloneIdentity() })
    }

    return (
        <div className="grade-editor" style={{ borderTop: '1px dashed var(--border-subtle)' }}>
            <div className="grade-head">
                <span className="grade-cap">Grade · curves · wheels</span>
                <button type="button" onClick={resetAll} disabled={!anyGrade} className="adjust-curve-reset" title="Reset gamma, curves and wheels">
                    <RotateCcw aria-hidden="true" /> Reset
                </button>
            </div>

            {/* Gamma */}
            <div className="grade-gamma-row">
                <ProRulerSlider
                    variant="instrument"
                    value={gammaPct}
                    min={20} max={220} step={1}
                    label="Gamma" suffix="%"
                    onPreview={(v) => onUpdate({ gamma: v / 100 })}
                    onCommit={(v) => onUpdate({ gamma: v / 100 })}
                    visual={{ fill: 'rgba(58,48,88,0.5)', accent: '#b8a6f0', trackBg: 'rgba(18,16,26,0.98)' }}
                />
                <input
                    type="number" min={20} max={220} value={gammaPct}
                    className="grade-num"
                    onChange={(e) => {
                        const n = Number(e.target.value)
                        if (Number.isFinite(n)) onUpdate({ gamma: clamp(n, 20, 220) / 100 })
                    }}
                    aria-label="Gamma percent"
                />
            </div>

            {/* Tone curves */}
            <div className="adjust-curve-card" style={{ '--curve-color': channel.color }}>
                <div className="adjust-curve-toolbar">
                    <span className="adjust-curve-toolbar-label">Curve</span>
                    <label className="adjust-curve-select-wrap">
                        <select
                            value={channel.id}
                            onChange={(e) => setActiveChannel(e.target.value)}
                            className="adjust-curve-select"
                            style={{ '--curve-color': channel.color }}
                            aria-label="Curve channel"
                            disabled={!curvesEnabled}
                        >
                            {CURVE_CHANNELS.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
                        </select>
                    </label>
                    <button type="button" className="adjust-curve-reset" disabled={!curvesEnabled || !channelDirty} onClick={() => applyChannel(channel.ch, cloneIdentity())} title={`Reset ${channel.label}`}>
                        <RotateCcw aria-hidden="true" />
                    </button>
                </div>
                <CurveGraph
                    points={channelPoints}
                    color={channel.color}
                    histogramChannels={channel.histogramChannels}
                    histogram={histogram}
                    onChange={(next) => applyChannel(channel.ch, next)}
                />
                <p className="adjust-curve-hint">
                    {curvesEnabled ? 'Click to add a point · drag to shape · ↑/↓ nudge · double-click to remove' : 'Curves unavailable here'}
                </p>
            </div>

            {/* 3-way colour wheels */}
            <div className="grade-cap">Colour wheels</div>
            <div className="grade-wheels">
                {wheels.map((w) => (
                    <ColorWheel
                        key={w.key}
                        label={w.label}
                        value={layer[w.key]}
                        onChange={(off) => onUpdate({ [w.key]: off })}
                    />
                ))}
            </div>
        </div>
    )
})

export default LayerGradeEditor
