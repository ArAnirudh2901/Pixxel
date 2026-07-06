"use client"

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
    ChevronDown, ChevronUp, Eye, EyeOff, Lock, LockOpen, Minus, Plus,
    Eraser, Paintbrush, RotateCcw, Redo2, Trash2, Undo2,
} from 'lucide-react'
import { ProRulerSlider } from '@/components/editor/ProRulerSlider'
import { LayerGradeEditor } from './_layer-grade-editor.jsx'
import { hexToRgba } from '@/lib/color-utils'

/**
 * Shared presentational pieces for the pixel-paint tools (Mask + Erase) so both
 * panels look and behave identically. Logic lives in usePixelMaskTool.
 */

const MODES = [
    { id: 'erase', label: 'Erase', icon: Eraser, hint: 'Paint to hide areas' },
    { id: 'restore', label: 'Restore', icon: Paintbrush, hint: 'Paint hidden areas back' },
]

export function ModeToggle({ mode, setMode, altActive }) {
    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between">
                <label className="panel-label">Brush Mode</label>
                {altActive && (
                    <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded"
                        style={{ background: 'rgba(6,184,212,0.15)', color: 'var(--accent-primary)' }}>
                        ALT · inverted
                    </span>
                )}
            </div>
            <div className="grid grid-cols-2 gap-1.5">
                {MODES.map((m) => {
                    const Icon = m.icon
                    const isActive = mode === m.id
                    return (
                        <motion.button
                            key={m.id}
                            type="button"
                            onClick={() => setMode(m.id)}
                            whileTap={{ scale: 0.97 }}
                            className="flex flex-col items-center gap-1.5 rounded-lg px-2 py-2.5 editor-interactive"
                            style={{
                                background: isActive ? 'rgba(6, 184, 212, 0.1)' : 'transparent',
                                border: `1px solid ${isActive ? 'var(--accent-primary)' : 'var(--border-subtle)'}`,
                                color: isActive ? 'var(--accent-primary)' : 'var(--text-secondary)',
                            }}
                        >
                            <Icon className="h-4 w-4" />
                            <span className="text-xs font-semibold">{m.label}</span>
                            <span className="text-[9px] leading-tight text-center" style={{ color: 'var(--text-muted)' }}>{m.hint}</span>
                        </motion.button>
                    )
                })}
            </div>
        </div>
    )
}

export function BrushSizeControl({ value, setValue, min, max, dominantColor }) {
    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between">
                <label className="panel-label">Brush Size</label>
                <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
                    [ / ] to resize
                </span>
            </div>
            <div className="flex items-center gap-2">
                <button
                    type="button"
                    onClick={() => setValue(Math.max(min, value - 5))}
                    className="flex items-center justify-center w-8 h-11 rounded-lg editor-interactive shrink-0"
                    style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
                >
                    <Minus className="h-3.5 w-3.5" />
                </button>
                <ProRulerSlider
                    className="flex-1 min-w-0"
                    label="Size"
                    value={value}
                    min={min}
                    max={max}
                    step={1}
                    suffix="px"
                    onChange={setValue}
                    visual={{
                        fill: 'rgba(47, 143, 203, 0.45)',
                        accent: dominantColor || '#5eb8ff',
                        trackBg: 'rgba(18, 22, 30, 0.96)',
                    }}
                />
                <button
                    type="button"
                    onClick={() => setValue(Math.min(max, value + 5))}
                    className="flex items-center justify-center w-8 h-11 rounded-lg editor-interactive shrink-0"
                    style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
                >
                    <Plus className="h-3.5 w-3.5" />
                </button>
            </div>
        </div>
    )
}

export function LabeledSlider({ label, value, min, max, step = 1, suffix = '%', onChange, dominantColor }) {
    return (
        <div className="space-y-1.5">
            <label className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{label}</label>
            <ProRulerSlider
                label={label}
                value={value}
                min={min}
                max={max}
                step={step}
                suffix={suffix}
                onChange={onChange}
                visual={{
                    fill: 'rgba(47, 143, 203, 0.45)',
                    accent: dominantColor || '#5eb8ff',
                    trackBg: 'rgba(18, 22, 30, 0.96)',
                }}
            />
        </div>
    )
}

export function MaskActionButtons({ hasMask, undoDepth, redoDepth, onUndo, onRedo, onInvert, onClear }) {
    return (
        <div className="space-y-2" style={{ borderTop: '1px solid rgba(244,244,245,0.06)', paddingTop: '12px' }}>
            <label className="panel-label">Actions</label>
            <div className="grid grid-cols-2 gap-2">
                <button
                    type="button"
                    onClick={onUndo}
                    disabled={undoDepth === 0}
                    className="mask-btn"
                >
                    <Undo2 className="h-3.5 w-3.5" />
                    Undo
                </button>
                <button
                    type="button"
                    onClick={onRedo}
                    disabled={redoDepth === 0}
                    className="mask-btn"
                >
                    <Redo2 className="h-3.5 w-3.5" />
                    Redo
                </button>
            </div>
            <button
                type="button"
                onClick={onInvert}
                disabled={!hasMask}
                className="mask-btn w-full"
            >
                <RotateCcw className="h-3.5 w-3.5" />
                Invert
            </button>
            <button
                type="button"
                onClick={onClear}
                disabled={!hasMask}
                className="mask-btn mask-btn--danger w-full"
            >
                <Trash2 className="h-3.5 w-3.5" />
                Reset
            </button>
        </div>
    )
}

export function ToolEmptyState({ icon: Icon, title, subtitle }) {
    return (
        <div className="mask-empty-state">
            <Icon className="mask-empty-state__icon" />
            <p className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>{title}</p>
            <p className="text-[11px] mt-1" style={{ color: 'var(--text-muted)' }}>{subtitle}</p>
        </div>
    )
}

export function TipCard({ title = 'Tips', children }) {
    return (
        <div className="mask-tip-card">
            <p className="font-semibold mb-2" style={{ color: 'var(--text-secondary)', fontSize: '11px' }}>{title}</p>
            <div className="space-y-1.5">
                {children}
            </div>
        </div>
    )
}

/* ─── Megashader chain UI (Step 2) ─────────────────────────────────────── */

/**
 * Visual badge showing which megashader mask kind a layer is. Each kind
 * has a distinct colour so the chain is scannable. Used in `MaskChainCard`
 * and the test panel.
 */
const KIND_META = {
    luminance:  { label: 'Luminance', color: '#facc15', step: 2 },
    color:      { label: 'Color',     color: '#f97316', step: 2 },
    linear:     { label: 'Linear',    color: '#22d3ee', step: 3 },
    radial:     { label: 'Radial',    color: '#a78bfa', step: 3 },
    smartBrush: { label: 'Smart Brush', color: '#34d399', step: 4 },
    semantic:   { label: 'AI Subject', color: '#f472b6', step: 5 },
    depth:      { label: 'Depth Map', color: '#60a5fa', step: 6 },
    lasso:      { label: 'Lasso',     color: '#06b8d4', step: null },
    brush:      { label: 'Brush',     color: '#c084fc', step: null },
    path:       { label: 'Pen Path',  color: '#53d8ff', step: null },
}

export const getKindMeta = (kind) => KIND_META[kind] || { label: kind, color: '#94a3b8', step: null }

const OPS = [
    { id: 'add', label: 'Add' },
    { id: 'subtract', label: 'Subtract' },
    { id: 'intersect', label: 'Intersect' },
    // Photoshop-parity blend modes (all compile to real GLSL now).
    { id: 'screen', label: 'Screen' },
    { id: 'lighten', label: 'Lighten' },
    { id: 'darken', label: 'Darken' },
    { id: 'overlay', label: 'Overlay' },
]

// Per-layer output modes (root-cause #1). Matches FILL_MODES in mask-types.
const FILL_MODE_OPTIONS = [
    { id: 'fill', label: 'Fill' },
    { id: 'adjust', label: 'Adjust' },
    { id: 'erase', label: 'Erase' },
]

// Convert a {r,g,b} (0..1) fill colour to a #rrggbb hex string for the
// native colour input, and back.
const fillColorToHex = (c) => {
    const ch = (v) => Math.max(0, Math.min(255, Math.round((typeof v === 'number' ? v : 0) * 255)))
    const h = (n) => n.toString(16).padStart(2, '0')
    const col = c || { r: 1, g: 0, b: 0.6 }
    return `#${h(ch(col.r))}${h(ch(col.g))}${h(ch(col.b))}`
}
const hexToFillColor = (hex) => {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex || '')
    if (!m) return { r: 1, g: 0, b: 0.6 }
    const n = parseInt(m[1], 16)
    return { r: ((n >> 16) & 255) / 255, g: ((n >> 8) & 255) / 255, b: (n & 255) / 255 }
}

/**
 * Step 8 — Universal per-layer image-adjustment editor. Renders the four
 * adjustment sliders (Exposure/Contrast/Saturation/Brightness) that are
 * available on EVERY mask layer regardless of kind. The math happens in
 * the GLSL `applyLayerAdjust_<slot>(rgb)` function emitted by
 * `buildLayerAdjustFunction` in glsl-fragments.js; this component is
 * just a presentational wrapper.
 *
 * The sliders are also a "Reset" button that zeros all four fields in
 * one click — common workflow when iterating on adjustments.
 */
function LayerAdjustEditor({ layer, onUpdate, dominantColor }) {
    const reset = () => onUpdate({
        exposure: 0, contrast: 0, saturation: 0, brightness: 0,
        highlights: 0, shadows: 0, whites: 0, blacks: 0, temperature: 0, tint: 0,
    })
    const anyActive = (
        layer.exposure || layer.contrast || layer.saturation || layer.brightness
        || layer.highlights || layer.shadows || layer.whites || layer.blacks
        || layer.temperature || layer.tint
    )
    return (
        <div className="space-y-2 pt-2" style={{ borderTop: '1px dashed var(--border-subtle)' }}>
            <div className="flex items-center justify-between">
                <span
                    className="text-[9px] font-bold uppercase tracking-wider"
                    style={{ color: 'var(--text-muted)' }}
                >
                    Adjustments (this layer)
                </span>
                <button
                    type="button"
                    onClick={reset}
                    disabled={!anyActive}
                    className="text-[9px] px-1.5 py-0.5 rounded editor-interactive disabled:opacity-30"
                    style={{
                        background: 'var(--bg-elevated)',
                        border: '1px solid var(--border-subtle)',
                        color: 'var(--text-secondary)',
                    }}
                >
                    Reset
                </button>
            </div>
            <LabeledSlider
                label="Exposure"
                value={Math.round((layer.exposure ?? 0) * 10) / 10}
                min={-3} max={3} step={0.1} suffix=" EV"
                onChange={(v) => onUpdate({ exposure: v })}
                dominantColor={dominantColor}
            />
            <LabeledSlider
                label="Contrast"
                value={Math.round(layer.contrast ?? 0)}
                min={-100} max={100} step={1} suffix="%"
                onChange={(v) => onUpdate({ contrast: v })}
                dominantColor={dominantColor}
            />
            <LabeledSlider
                label="Saturation"
                value={Math.round(layer.saturation ?? 0)}
                min={-100} max={100} step={1} suffix="%"
                onChange={(v) => onUpdate({ saturation: v })}
                dominantColor={dominantColor}
            />
            <LabeledSlider
                label="Vibrance"
                value={Math.round(layer.vibrance ?? 0)}
                min={-100} max={100} step={1} suffix="%"
                onChange={(v) => onUpdate({ vibrance: v })}
                dominantColor={dominantColor}
            />
            <LabeledSlider
                label="Brightness"
                value={Math.round(layer.brightness ?? 0)}
                min={-100} max={100} step={1} suffix="%"
                onChange={(v) => onUpdate({ brightness: v })}
                dominantColor={dominantColor}
            />

            <div className="text-[8px] font-bold uppercase tracking-wider pt-1" style={{ color: 'var(--text-muted)' }}>Tone</div>
            <LabeledSlider
                label="Highlights"
                value={Math.round(layer.highlights ?? 0)}
                min={-100} max={100} step={1} suffix="%"
                onChange={(v) => onUpdate({ highlights: v })}
                dominantColor={dominantColor}
            />
            <LabeledSlider
                label="Shadows"
                value={Math.round(layer.shadows ?? 0)}
                min={-100} max={100} step={1} suffix="%"
                onChange={(v) => onUpdate({ shadows: v })}
                dominantColor={dominantColor}
            />
            <LabeledSlider
                label="Whites"
                value={Math.round(layer.whites ?? 0)}
                min={-100} max={100} step={1} suffix="%"
                onChange={(v) => onUpdate({ whites: v })}
                dominantColor={dominantColor}
            />
            <LabeledSlider
                label="Blacks"
                value={Math.round(layer.blacks ?? 0)}
                min={-100} max={100} step={1} suffix="%"
                onChange={(v) => onUpdate({ blacks: v })}
                dominantColor={dominantColor}
            />

            <div className="text-[8px] font-bold uppercase tracking-wider pt-1" style={{ color: 'var(--text-muted)' }}>White Balance</div>
            <LabeledSlider
                label="Temperature"
                value={Math.round(layer.temperature ?? 0)}
                min={-100} max={100} step={1} suffix=""
                onChange={(v) => onUpdate({ temperature: v })}
                dominantColor={dominantColor}
            />
            <LabeledSlider
                label="Tint"
                value={Math.round(layer.tint ?? 0)}
                min={-100} max={100} step={1} suffix=""
                onChange={(v) => onUpdate({ tint: v })}
                dominantColor={dominantColor}
            />

            <div className="text-[8px] font-bold uppercase tracking-wider pt-1" style={{ color: 'var(--text-muted)' }}>Detail</div>
            <LabeledSlider
                label="Texture"
                value={Math.round(layer.texture ?? 0)}
                min={-100} max={100} step={1} suffix="%"
                onChange={(v) => onUpdate({ texture: v })}
                dominantColor={dominantColor}
            />
            <LabeledSlider
                label="Dehaze"
                value={Math.round(layer.dehaze ?? 0)}
                min={-100} max={100} step={1} suffix="%"
                onChange={(v) => onUpdate({ dehaze: v })}
                dominantColor={dominantColor}
            />
        </div>
    )
}

/**
 * Per-kind parameter editor. Renders sliders for luminance; a colour
 * swatch + sliders for color; and a "Coming in Step N" placeholder for
 * the kinds Step 2 doesn't yet implement. The parent passes a callback
 * for value updates — this component is presentational.
 *
 * Step 8: every kind editor now ALSO renders `LayerAdjustEditor`
 * underneath the kind-specific controls, so adjustments are universal.
 */
export function KindParamEditor({ layer, onUpdate, onApplyCurve, histogram, dominantColor, imageSize }) {
    const meta = getKindMeta(layer.kind)
    let kindSpecific = null
    if (layer.kind === 'luminance') {
        kindSpecific = (
            <>
                <LabeledSlider
                    label="Min brightness"
                    value={Math.round((layer.min ?? 0) * 255)}
                    min={0} max={254} step={1} suffix=""
                    onChange={(v) => onUpdate({ min: Math.min(v / 255, (layer.max ?? 0.5) - 0.004) })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="Max brightness"
                    value={Math.round((layer.max ?? 0.5) * 255)}
                    min={1} max={255} step={1} suffix=""
                    onChange={(v) => onUpdate({ max: Math.max(v / 255, (layer.min ?? 0) + 0.004) })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="Softness"
                    value={Math.round((layer.softness ?? 0.1) * 100)}
                    min={0} max={50} step={1} suffix="%"
                    onChange={(v) => onUpdate({ softness: v / 100 })}
                    dominantColor={dominantColor}
                />
            </>
        )
    } else if (layer.kind === 'color') {
        const target = layer.target || { h: 0, s: 1, b: 1 }
        const setTarget = (patch) => onUpdate({ target: { ...target, ...patch } })
        // Render a 6×6×6 RGB preview so the user can see what the target is
        // without bringing in a full colour picker. (Step 7 polish can swap
        // this for @uiw/react-color-colorful when the design system is ready.)
        const swatch = `hsl(${target.h}, ${Math.round(target.s * 100)}%, ${Math.round(target.b * 50)}%)`
        kindSpecific = (
            <>
                <div className="flex items-center gap-2">
                    <div
                        className="w-6 h-6 rounded border shrink-0"
                        style={{ background: swatch, borderColor: 'var(--border-subtle)' }}
                        title={`HSL ${Math.round(target.h)}°, ${Math.round(target.s * 100)}%, ${Math.round(target.b * 100)}%`}
                    />
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        Target colour
                    </span>
                </div>
                <LabeledSlider
                    label="Hue"
                    value={Math.round(target.h || 0)}
                    min={0} max={360} step={1} suffix="°"
                    onChange={(v) => setTarget({ h: v })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="Saturation"
                    value={Math.round((target.s || 0) * 100)}
                    min={0} max={100} step={1} suffix="%"
                    onChange={(v) => setTarget({ s: v / 100 })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="Brightness"
                    value={Math.round((target.b || 0) * 100)}
                    min={0} max={100} step={1} suffix="%"
                    onChange={(v) => setTarget({ b: v / 100 })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="Tolerance"
                    value={Math.round((layer.tolerance ?? 0.15) * 100)}
                    min={1} max={50} step={1} suffix="%"
                    onChange={(v) => onUpdate({ tolerance: v / 100 })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="Softness"
                    value={Math.round((layer.softness ?? 0.1) * 100)}
                    min={0} max={50} step={1} suffix="%"
                    onChange={(v) => onUpdate({ softness: v / 100 })}
                    dominantColor={dominantColor}
                />
            </>
        )
    } else if (layer.kind === 'linear') {
        const p1 = layer.p1 || { x: 0, y: 0 }
        const p2 = layer.p2 || { x: 100, y: 0 }
        const setP1 = (patch) => onUpdate({ p1: { ...p1, ...patch } })
        const setP2 = (patch) => onUpdate({ p2: { ...p2, ...patch } })
        const w = imageSize?.width || 1000
        const h = imageSize?.height || 1000
        kindSpecific = (
            <>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    Drag on the canvas, or type exact coordinates below.
                </p>
                <LabeledSlider
                    label="P1 X"
                    value={Math.round(p1.x)}
                    min={0} max={w} step={1} suffix="px"
                    onChange={(v) => setP1({ x: v })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="P1 Y"
                    value={Math.round(p1.y)}
                    min={0} max={h} step={1} suffix="px"
                    onChange={(v) => setP1({ y: v })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="P2 X"
                    value={Math.round(p2.x)}
                    min={0} max={w} step={1} suffix="px"
                    onChange={(v) => setP2({ x: v })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="P2 Y"
                    value={Math.round(p2.y)}
                    min={0} max={h} step={1} suffix="px"
                    onChange={(v) => setP2({ y: v })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="Position"
                    value={Math.round((layer.position ?? 0.5) * 100)}
                    min={0} max={100} step={1} suffix="%"
                    onChange={(v) => onUpdate({ position: v / 100 })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="Feather"
                    value={Math.round((layer.feather ?? 0.1) * 100)}
                    min={0} max={50} step={1} suffix="%"
                    onChange={(v) => onUpdate({ feather: v / 100 })}
                    dominantColor={dominantColor}
                />
            </>
        )
    } else if (layer.kind === 'radial') {
        const center = layer.center || { x: 0, y: 0 }
        const radius = layer.radius || { x: 100, y: 100 }
        const setCenter = (patch) => onUpdate({ center: { ...center, ...patch } })
        const setRadius = (patch) => onUpdate({ radius: { ...radius, ...patch } })
        const w = imageSize?.width || 1000
        const h = imageSize?.height || 1000
        // Rotation: store radians internally; display degrees for users.
        const rotationDeg = Math.round(((layer.rotation ?? 0) * 180) / Math.PI)
        kindSpecific = (
            <>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    Drag a bounding box on the canvas, or type exact values below.
                </p>
                <LabeledSlider
                    label="Center X"
                    value={Math.round(center.x)}
                    min={0} max={w} step={1} suffix="px"
                    onChange={(v) => setCenter({ x: v })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="Center Y"
                    value={Math.round(center.y)}
                    min={0} max={h} step={1} suffix="px"
                    onChange={(v) => setCenter({ y: v })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="Radius X"
                    value={Math.round(radius.x)}
                    min={1} max={w} step={1} suffix="px"
                    onChange={(v) => setRadius({ x: Math.max(1, v) })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="Radius Y"
                    value={Math.round(radius.y)}
                    min={1} max={h} step={1} suffix="px"
                    onChange={(v) => setRadius({ y: Math.max(1, v) })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="Rotation"
                    value={rotationDeg}
                    min={0} max={359} step={1} suffix="°"
                    onChange={(v) => onUpdate({ rotation: (v * Math.PI) / 180 })}
                    dominantColor={dominantColor}
                />
                <LabeledSlider
                    label="Feather"
                    value={Math.round((layer.feather ?? 0.1) * 100)}
                    min={0} max={50} step={1} suffix="%"
                    onChange={(v) => onUpdate({ feather: v / 100 })}
                    dominantColor={dominantColor}
                />
            </>
        )
    } else if (layer.kind === 'smartBrush') {
        kindSpecific = (
            <>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    Bilateral filter applied to the painted alpha. Larger
                    radius = more neighbours contribute. Smaller colour
                    sigma = stricter edge adherence.
                </p>
                <LabeledSlider
                    label="Filter Radius"
                    value={layer.filterRadius ?? 3}
                    min={1} max={8} step={1} suffix=" px"
                    onChange={(v) => onUpdate({ filterRadius: v })}
                />
                <LabeledSlider
                    label="Color Sigma"
                    value={Math.round((layer.sigmaColor ?? 0.15) * 100)}
                    min={1} max={100} step={1} suffix="%"
                    onChange={(v) => onUpdate({ sigmaColor: v / 100 })}
                />
                <LabeledSlider
                    label="Space Sigma"
                    value={Math.round((layer.sigmaSpace ?? 2) * 10)}
                    min={1} max={80} step={1} suffix=" px"
                    onChange={(v) => onUpdate({ sigmaSpace: v / 10 })}
                />
            </>
        )
    } else if (layer.kind === 'semantic' || layer.kind === 'depth') {
        // Step 5 + 6 editors: just the feather/softness slider.
        const featherKey = layer.kind === 'semantic' ? 'feather' : 'softness'
        kindSpecific = (
            <LabeledSlider
                label={featherKey === 'feather' ? 'Feather' : 'Softness'}
                value={Math.round((layer[featherKey] ?? 0.1) * 100)}
                min={0} max={50} step={1} suffix="%"
                onChange={(v) => onUpdate({ [featherKey]: v / 100 })}
                dominantColor={dominantColor}
            />
        )
    } else if (layer.kind === 'lasso') {
        // Per-region feather — each lasso layer owns its edge softness, so
        // editing this only affects THIS selection (not every region).
        kindSpecific = (
            <LabeledSlider
                label="Feather"
                value={Math.round((layer.feather ?? 0.05) * 100)}
                min={0} max={40} step={1} suffix="%"
                onChange={(v) => onUpdate({ feather: v / 100 })}
                dominantColor={dominantColor}
            />
        )
    } else if (layer.kind === 'brush') {
        // The brush's soft edge is baked into its texture at paint time, so
        // there's no live feather here — repaint with a new Edge Feather to
        // change it. Adjustments below still apply per-region.
        kindSpecific = (
            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                Painted selection. Edge feather is baked per region — repaint to
                change it.
            </p>
        )
    } else if (layer.kind === 'path') {
        // Pen/Bézier path — live per-region feather, same as the lasso (the
        // rasterised path alpha is softened in the shader by this value).
        kindSpecific = (
            <LabeledSlider
                label="Feather"
                value={Math.round((layer.feather ?? 0.04) * 100)}
                min={0} max={40} step={1} suffix="%"
                onChange={(v) => onUpdate({ feather: v / 100 })}
                dominantColor={dominantColor}
            />
        )
    }

    if (kindSpecific) {
        return (
            <div className="space-y-2 pt-2" style={{ borderTop: '1px dashed var(--border-subtle)' }}>
                <div className="space-y-2">{kindSpecific}</div>
                <LayerAdjustEditor layer={layer} onUpdate={onUpdate} dominantColor={dominantColor} />
                <LayerGradeEditor
                    layer={layer}
                    onUpdate={onUpdate}
                    onApplyCurve={onApplyCurve}
                    histogram={histogram}
                    dominantColor={dominantColor}
                />
            </div>
        )
    }
    // Unknown / future kind: read-only placeholder.
    return (
        <div
            className="pt-2 text-[10px] text-center"
            style={{ borderTop: '1px dashed var(--border-subtle)', color: 'var(--text-muted)' }}
        >
            <span
                className="inline-block text-[8px] font-bold px-1.5 py-0.5 rounded mr-1.5"
                style={{ background: 'rgba(124,58,237,0.2)', color: '#A78BFA' }}
            >
                COMING
            </span>
            {meta.label} — editor lands in Step {meta.step}
        </div>
    )
}

/**
 * One card in the megashader chain. Shows the layer kind, op, and
 * common controls (opacity, visible, invert, move, delete). The kind-
 * specific editor is collapsed by default — click the chevron to expand.
 *
 * Used by both the dev test panel and the production Mask tool's
 * "Mask Layers" section.
 */
/** Mask kinds whose selection lives in a texture and can therefore have its
 *  boundary grown/shrunk (mirrors TEXTURE_BACKED_KINDS in mask-grow.js). */
const GROWABLE_KINDS = ['semantic', 'lasso', 'brush', 'smartBrush']
const GROW_UI_MAX_PX = 60

/**
 * "Boundary" slider: extend (+) or shrink (−) a texture-backed selection's
 * edge by N pixels. Regenerating the mask texture isn't per-frame cheap, so
 * the slider tracks locally and COMMITS on release — and because the commit
 * is absolute (derived from the layer's pristine base texture every time),
 * scrubbing back to 0 restores the original AI-detected edge exactly.
 */
function BoundaryGrowControl({ layer, locked, onCommit }) {
    const committed = Math.round(layer.growPx || 0)
    const [local, setLocal] = useState(null)
    const value = local ?? committed
    const commit = () => {
        if (local == null) return
        const px = local
        setLocal(null)
        if (px !== committed) onCommit(px)
    }
    return (
        <div className="mask-param-row mt-1.5">
            <span
                className="mask-param-label"
                title="Extend (+) or shrink (−) this selection's boundary, in image pixels. Works on AI-detected subject masks too; 0 restores the original edge."
            >
                Boundary
            </span>
            <input
                type="range"
                min={-GROW_UI_MAX_PX}
                max={GROW_UI_MAX_PX}
                step={1}
                disabled={locked}
                value={value}
                onChange={(e) => setLocal(Number(e.target.value))}
                onMouseUp={commit}
                onTouchEnd={commit}
                onKeyUp={(e) => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'Enter') commit() }}
                onBlur={commit}
                className="mask-range flex-1"
            />
            <span className="mask-param-value">{value > 0 ? `+${value}` : value}px</span>
        </div>
    )
}

export const MaskChainCard = React.memo(function MaskChainCard({
    entry, index, total, isFirst,
    onUpdate, onRemove, onMove, onSetOp, onSetFillMode,
    selected, onSelect,
    dominantColor, imageSize, onExpandBoundary,
    onApplyCurve, histogram,
}) {
    const layer = entry.layer
    const meta = getKindMeta(layer.kind)
    const locked = !!layer.lock
    const fillMode = layer.fillMode || 'adjust'
    // New selection layers default to 'fill' and start expanded so the user
    // immediately sees the fill controls (rather than a collapsed, dead-
    // looking card — the original "I added a mask and nothing happened" UX).
    const [expanded, setExpanded] = useState(fillMode !== 'adjust')

    return (
        <motion.div
            layout
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.18 }}
            onClick={() => onSelect?.(layer.id)}
            className={`mask-chain-card ${selected ? 'mask-chain-card--selected' : ''}`}
        >
            <div className="flex items-center gap-1.5">
                <span
                    className="text-[9px] font-mono w-4 text-center"
                    style={{ color: 'var(--text-muted)' }}
                >
                    {index}
                </span>
                <span
                    className="mask-kind-badge"
                    style={{ background: hexToRgba(meta.color, 0.12), color: meta.color }}
                >
                    {meta.label}
                </span>
                <span
                    className="text-[10px] font-semibold flex-1 truncate"
                    style={{ color: 'var(--text-primary)' }}
                    title={layer.label}
                >
                    {layer.label}
                </span>
                <button
                    type="button"
                    onClick={() => setExpanded((v) => !v)}
                    className="mask-icon-btn"
                    title={expanded ? 'Hide params' : 'Edit params'}
                >
                    {expanded
                        ? <ChevronUp className="h-3 w-3" />
                        : <ChevronDown className="h-3 w-3" />}
                </button>
                <button
                    type="button"
                    onClick={() => onUpdate({ visible: layer.visible === false })}
                    className={`mask-icon-btn ${layer.visible !== false ? 'mask-icon-btn--accent' : ''}`}
                    title={layer.visible === false ? 'Show' : 'Hide'}
                >
                    {layer.visible === false
                        ? <EyeOff className="h-3 w-3" />
                        : <Eye className="h-3 w-3" />}
                </button>
                <button
                    type="button"
                    onClick={() => onMove(layer.id, 'up')}
                    disabled={isFirst}
                    className="mask-icon-btn"
                    title="Move up"
                >
                    <ChevronUp className="h-3 w-3" />
                </button>
                <button
                    type="button"
                    onClick={() => onMove(layer.id, 'down')}
                    disabled={index === total - 1}
                    className="mask-icon-btn"
                    title="Move down"
                >
                    <ChevronDown className="h-3 w-3" />
                </button>
                <button
                    type="button"
                    onClick={() => onUpdate({ lock: !locked })}
                    className={`mask-icon-btn ${locked ? 'mask-icon-btn--accent' : ''}`}
                    title={locked ? 'Unlock layer' : 'Lock layer (freeze params)'}
                >
                    {locked
                        ? <Lock className="h-3 w-3" />
                        : <LockOpen className="h-3 w-3" />}
                </button>
                <button
                    type="button"
                    onClick={() => onRemove(layer.id)}
                    className="mask-icon-btn mask-icon-btn--danger"
                    title="Remove layer"
                >
                    <Trash2 className="h-3 w-3" />
                </button>
            </div>

            {!isFirst && (
                <select
                    value={entry.op}
                    disabled={locked}
                    onChange={(e) => onSetOp(layer.id, e.target.value)}
                    className="mask-select mt-1.5"
                >
                    {OPS.map((op) => (
                        <option key={op.id} value={op.id}>{op.label}</option>
                    ))}
                </select>
            )}

            {/* Output mode: fill (visible selection) / adjust (recolour) /
                erase (cut). Root-cause #1 — a 'fill' layer is visible without
                any adjustment. */}
            <div className="mask-fill-modes mt-1.5">
                {FILL_MODE_OPTIONS.map((m) => {
                    const active = fillMode === m.id
                    return (
                        <button
                            key={m.id}
                            type="button"
                            disabled={locked}
                            onClick={() => (onSetFillMode ? onSetFillMode(layer.id, m.id) : onUpdate({ fillMode: m.id }))}
                            className={`mask-fill-mode-btn ${active ? 'mask-fill-mode-btn--active' : ''}`}
                        >
                            {m.label}
                        </button>
                    )
                })}
            </div>

            {fillMode === 'fill' && (
                <div className="mask-param-row mt-1.5">
                    <span className="mask-param-label">Tint</span>
                    <input
                        type="color"
                        disabled={locked}
                        value={fillColorToHex(layer.fillColor)}
                        onChange={(e) => onUpdate({ fillColor: hexToFillColor(e.target.value) })}
                        className="mask-color-input disabled:opacity-50"
                        title="Fill colour"
                    />
                    <span className="mask-param-label" style={{ minWidth: 'auto' }}>Strength</span>
                    <input
                        type="range"
                        min="0" max="100"
                        disabled={locked}
                        value={Math.round((layer.fillStrength ?? 0.5) * 100)}
                        onChange={(e) => onUpdate({ fillStrength: Number(e.target.value) / 100 })}
                        className="mask-range flex-1"
                    />
                    <span className="mask-param-value">
                        {Math.round((layer.fillStrength ?? 0.5) * 100)}
                    </span>
                </div>
            )}

            <div className="mask-param-row mt-1.5">
                <span className="mask-param-label" title="Overall strength of this mask (Lightroom 'Density')">
                    Density
                </span>
                <input
                    type="range"
                    min="0" max="100"
                    disabled={locked}
                    value={Math.round((layer.opacity ?? 1) * 100)}
                    onChange={(e) => onUpdate({ opacity: Number(e.target.value) / 100 })}
                    className="mask-range flex-1"
                />
                <span className="mask-param-value">
                    {Math.round((layer.opacity ?? 1) * 100)}
                </span>
            </div>
            {typeof onExpandBoundary === 'function'
                && GROWABLE_KINDS.includes(layer.kind)
                && layer.maskTextureKey && (
                <BoundaryGrowControl
                    layer={layer}
                    locked={locked}
                    onCommit={(px) => onExpandBoundary(layer.id, px)}
                />
            )}

            <label className="mask-toggle mt-2">
                <input
                    type="checkbox"
                    checked={!!layer.inverted}
                    disabled={locked}
                    onChange={(e) => onUpdate({ inverted: e.target.checked })}
                />
                Invert
            </label>

            <AnimatePresence initial={false}>
                {expanded && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.18 }}
                        className="overflow-hidden"
                        style={{ pointerEvents: locked ? 'none' : 'auto', opacity: locked ? 0.5 : 1 }}
                    >
                        <KindParamEditor layer={layer} onUpdate={onUpdate} onApplyCurve={onApplyCurve} histogram={histogram} dominantColor={dominantColor} imageSize={imageSize} />
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    )
})

/**
 * Lightroom-style 256-bucket luminance histogram with the min/max range
 * shaded. Pure presentational — the parent passes a precomputed
 * histogram (from `computeImageHistogram`) and the current range.
 *
 * Dimensions are tight (88×36) so it fits above a slider without
 * dominating the panel.
 */
export function LuminanceHistogram({ histogram, min, max, softness }) {
    if (!histogram || !histogram.luma || histogram.luma.length === 0) {
        return (
            <div
                className="rounded-md text-[9px] text-center py-2"
                style={{ background: 'var(--bg-elevated)', color: 'var(--text-muted)', border: '1px solid var(--border-subtle)' }}
            >
                No histogram
            </div>
        )
    }
    const w = 88
    const h = 36
    const maxCount = Math.max(1, ...histogram.luma)
    const minBucket = Math.max(0, Math.min(255, Math.round(min * 255)))
    const maxBucket = Math.max(0, Math.min(255, Math.round(max * 255)))
    const softPx = Math.max(0, Math.round(softness * 255))
    const bars = []
    const barWidth = w / 256
    for (let i = 0; i < 256; i += 1) {
        const barH = (histogram.luma[i] / maxCount) * h
        const inRange = i >= minBucket && i <= maxBucket
        const inSoft = i >= Math.max(0, minBucket - softPx) && i <= Math.min(255, maxBucket + softPx)
        const fill = inRange
            ? 'var(--accent-primary)'
            : inSoft
                ? 'rgba(6,184,212,0.35)'
                : 'rgba(148,163,184,0.4)'
        // Even-distribution formula: bar i sits at i * (w/256), so the
        // rightmost bar ends exactly at w (no clipping at the right edge).
        const x = i * barWidth
        bars.push(
            <rect
                key={i}
                x={x.toFixed(2)}
                y={(h - barH).toFixed(2)}
                width={Math.max(0.3, barWidth).toFixed(2)}
                height={Math.max(0.5, barH).toFixed(2)}
                style={{ fill }}
            />,
        )
    }
    return (
        <svg
            viewBox={`0 0 ${w} ${h}`}
            width={w}
            height={h}
            preserveAspectRatio="none"
            style={{ display: 'block', width: '100%', height: h }}
        >
            {bars}
        </svg>
    )
}
