"use client"

import React, { useCallback, useEffect, useRef, useState } from "react"
import Colorful from "@uiw/react-color-colorful"
import { toast } from "sonner"
import { filters, Gradient, Rect } from "fabric"
import { LineChart, RotateCcw, SlidersHorizontal, Sparkles, WandSparkles } from "lucide-react"
import { ProRulerSlider } from "@/components/editor/ProRulerSlider"
import { PhosmithCurvesFilter, buildLut, buildCurveSvgPath, arePointsIdentity } from "@/lib/curves-filter"
import { useCanvas } from "../../../../../../../context/context"
import {
    buildImageKitChainedTransformUrl,
    ensureCurrentImageKitEndpoint,
    isImageKitUrl,
    normalizeImageKitUrl,
} from "../../../../../../lib/imagekit-ai"

const TEMP_WARM = "#ffb45f"
const TEMP_COOL = "#72b7ff"
const VIGNETTE_LAYER_NAME = "phosmith-vignette-overlay"
const HISTOGRAM_BUCKETS = 256
const HISTOGRAM_SAMPLE_SIZE = 260
const CURVE_GRAPH = {
    left: 8,
    top: 8,
    right: 92,
    bottom: 92,
}

const clamp = (v, min, max) => Math.max(min, Math.min(max, v))
const stripHex = (hex) => String(hex || "000000").replace("#", "").slice(0, 6).padEnd(6, "0")
const signedToken = (value) => (value < 0 ? `N${Math.abs(value)}` : `${value}`)
const imageKitOpacity = (opacity) => Math.round(clamp(opacity, 0, 100) * 0.99)
    .toString(10)
    .padStart(2, "0")

const hexToRgb = (hex) => {
    const clean = stripHex(hex)
    return {
        r: parseInt(clean.slice(0, 2), 16),
        g: parseInt(clean.slice(2, 4), 16),
        b: parseInt(clean.slice(4, 6), 16),
    }
}

const rgbToHex = (r, g, b) =>
    `#${[r, g, b].map((value) => clamp(Math.round(value), 0, 255).toString(16).padStart(2, "0")).join("")}`

const rgbToHsv = ({ r, g, b }) => {
    const rn = r / 255
    const gn = g / 255
    const bn = b / 255
    const max = Math.max(rn, gn, bn)
    const min = Math.min(rn, gn, bn)
    const delta = max - min
    let h = 0
    if (delta) {
        if (max === rn) h = ((gn - bn) / delta) % 6
        else if (max === gn) h = (bn - rn) / delta + 2
        else h = (rn - gn) / delta + 4
        h *= 60
    }
    if (h < 0) h += 360
    return { h, s: max === 0 ? 0 : delta / max, v: max }
}

const hsvToRgb = (h, s, v) => {
    const c = v * s
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1))
    const m = v - c
    let rp = 0
    let gp = 0
    let bp = 0
    if (h < 60) [rp, gp, bp] = [c, x, 0]
    else if (h < 120) [rp, gp, bp] = [x, c, 0]
    else if (h < 180) [rp, gp, bp] = [0, c, x]
    else if (h < 240) [rp, gp, bp] = [0, x, c]
    else if (h < 300) [rp, gp, bp] = [x, 0, c]
    else [rp, gp, bp] = [c, 0, x]
    return {
        r: (rp + m) * 255,
        g: (gp + m) * 255,
        b: (bp + m) * 255,
    }
}

const isSharpnessFilter = (filter) => {
    if (filter?.type !== filters.Convolute.type || !Array.isArray(filter.matrix) || filter.matrix.length !== 9) return false
    const matrix = filter.matrix.map(Number)
    const edge = -matrix[1]
    return (
        edge >= 0 &&
        Math.abs(matrix[0]) < 0.001 &&
        Math.abs(matrix[2]) < 0.001 &&
        Math.abs(matrix[6]) < 0.001 &&
        Math.abs(matrix[8]) < 0.001 &&
        Math.abs(matrix[1] - matrix[3]) < 0.001 &&
        Math.abs(matrix[1] - matrix[5]) < 0.001 &&
        Math.abs(matrix[1] - matrix[7]) < 0.001 &&
        Math.abs(matrix[4] - (1 + 4 * edge)) < 0.01
    )
}

const SLIDER_CONFIGS = [
    { group: "Tone", key: "exposure", label: "Exposure", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Tone", key: "brightness", label: "Brightness", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Tone", key: "contrast", label: "Contrast", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Tone", key: "highlights", label: "Highlights", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Tone", key: "shadows", label: "Shadows", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Tone", key: "whites", label: "Whites", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Tone", key: "blacks", label: "Blacks", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Tone", key: "gamma", label: "Gamma", min: 20, max: 220, step: 1, defaultValue: 100, suffix: "%" },
    { group: "Tone", key: "fade", label: "Fade", min: 0, max: 100, step: 1, defaultValue: 0 },

    { group: "Color", key: "temperature", label: "Temperature", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Color", key: "tint", label: "Tint", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Color", key: "saturation", label: "Saturation", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Color", key: "vibrance", label: "Vibrance", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Color", key: "hue", label: "Hue", min: -180, max: 180, step: 1, defaultValue: 0, suffix: "º" },
    { group: "Color", key: "red", label: "Red", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Color", key: "green", label: "Green", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Color", key: "blue", label: "Blue", min: -100, max: 100, step: 1, defaultValue: 0 },

    { group: "Detail", key: "clarity", label: "Clarity", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Detail", key: "sharpness", label: "Sharpness", min: 0, max: 100, step: 1, defaultValue: 0 },
    { group: "Detail", key: "blur", label: "Blur", min: 0, max: 100, step: 1, defaultValue: 0 },
    { group: "Detail", key: "noise", label: "Noise", min: 0, max: 100, step: 1, defaultValue: 0 },
    { group: "Detail", key: "grain", label: "Grain", min: 0, max: 100, step: 1, defaultValue: 0 },
    { group: "Detail", key: "pixelate", label: "Pixelate", min: 1, max: 32, step: 1, defaultValue: 1 },
    { group: "Detail", key: "mono", label: "B&W Mix", min: 0, max: 100, step: 1, defaultValue: 0 },
    { group: "Detail", key: "vignette", label: "Vignette", min: -100, max: 100, step: 1, defaultValue: 0 },
    { group: "Detail", key: "vignetteSize", label: "Vignette Size", min: 20, max: 95, step: 1, defaultValue: 62, suffix: "%" },
    { group: "Detail", key: "vignetteFeather", label: "Feather", min: 0, max: 100, step: 1, defaultValue: 70, suffix: "%" },
]

const WHEEL_CONFIGS = [
    // NOTE: "multiply" can't be used here — Fabric's BlendColor.applyTo2d bakes
    // alpha into the color BEFORE multiplying (tr = color * alpha), producing
    // near-black pixels instead of a subtle tint. "overlay" gives a similar
    // shadow-darkening effect that actually works in the Canvas2D pipeline.
    { key: "shadowTone", colorKey: "shadowColor", label: "Shadows", defaultColor: "#2342ff", mode: "overlay" },
    { key: "midtoneTone", colorKey: "midtoneColor", label: "Midtones", defaultColor: "#00c2a8", mode: "overlay" },
    { key: "highlightTone", colorKey: "highlightColor", label: "Highlights", defaultColor: "#ffd76d", mode: "screen" },
    { key: "colorizeIntensity", colorKey: "colorizeColor", label: "Colorize", defaultColor: "#ff3f7f", mode: "tint" },
]

const CURVE_CHANNELS = [
    { id: "rgb", label: "RGB", shortLabel: "RGB", color: "#e5e7eb", histogramChannels: ["red", "green", "blue"], pointsKey: "curvePointsMaster" },
    { id: "red", label: "Red", shortLabel: "Red", color: "#ff5d65", histogramChannels: ["red"], pointsKey: "curvePointsRed" },
    { id: "green", label: "Green", shortLabel: "Green", color: "#64d989", histogramChannels: ["green"], pointsKey: "curvePointsGreen" },
    { id: "blue", label: "Blue", shortLabel: "Blue", color: "#69a7ff", histogramChannels: ["blue"], pointsKey: "curvePointsBlue" },
    { id: "luma", label: "Luminance", shortLabel: "Luma", color: "#d8dde7", histogramChannels: ["luma"], pointsKey: "curvePointsMaster" },
]

const CURVE_POINTS_KEYS = ["curvePointsMaster", "curvePointsRed", "curvePointsGreen", "curvePointsBlue"]

const LOOKS = [
    { id: "none", label: "Clean", filterClass: null },
    { id: "vibrant", label: "Vibrant", filterClass: null, preset: { contrast: 10, saturation: 18, vibrance: 42, clarity: 12 } },
    { id: "dramatic", label: "Dramatic", filterClass: null, preset: { contrast: 28, highlights: -18, shadows: -24, clarity: 34, vibrance: 18, vignette: 22 } },
    { id: "cinematic", label: "Cinematic", filterClass: null, preset: { contrast: 18, saturation: -8, vibrance: 18, temperature: -10, shadows: -10, highlights: -8, vignette: 16 } },
    { id: "golden", label: "Golden Hour", filterClass: null, preset: { temperature: 30, tint: 8, highlights: 14, shadows: 12, vibrance: 24 } },
    { id: "matte", label: "Matte", filterClass: null, preset: { contrast: -12, fade: 34, blacks: -14, saturation: -8, grain: 12 } },
    { id: "coolfade", label: "Cool Fade", filterClass: null, preset: { temperature: -24, tint: -8, fade: 20, saturation: -10, blue: 10 } },
    { id: "noir", label: "Noir", filterClass: filters.BlackWhite, preset: { contrast: 34, clarity: 24, blacks: 24, grain: 18, vignette: 28 } },
    { id: "vintage", label: "Vintage", filterClass: filters.Vintage },
    { id: "kodachrome", label: "Kodachrome", filterClass: filters.Kodachrome },
    { id: "technicolor", label: "Technicolor", filterClass: filters.Technicolor },
    { id: "polaroid", label: "Polaroid", filterClass: filters.Polaroid },
    { id: "brownie", label: "Brownie", filterClass: filters.Brownie },
    { id: "sepia", label: "Sepia", filterClass: filters.Sepia },
    { id: "blackwhite", label: "B&W", filterClass: filters.BlackWhite },
]

const cloneIdentityPoints = () => [{ x: 0, y: 0 }, { x: 1, y: 1 }]

const DEFAULT_VALUES = {
    ...SLIDER_CONFIGS.reduce((acc, c) => {
        acc[c.key] = c.defaultValue
        return acc
    }, {}),
    ...WHEEL_CONFIGS.reduce((acc, c) => {
        acc[c.key] = 0
        acc[c.colorKey] = c.defaultColor
        return acc
    }, {}),
    ...CURVE_POINTS_KEYS.reduce((acc, key) => {
        acc[key] = cloneIdentityPoints()
        return acc
    }, {}),
    look: "none",
}

const LOOK_PRESET_KEYS = Array.from(new Set(LOOKS.flatMap((look) => Object.keys(look.preset || {}))))

const IMAGEKIT_DEFAULTS = {
    autoContrast: false,
    retouch: false,
    upscale: false,
    grayscale: false,
    sharpen: 0,
    usm: 0,
    urlBlur: 0,
    colorize: 0,
    colorizeColor: "#FF3F7F",
    shadow: false,
    shadowBlur: 10,
    shadowSaturation: 30,
    shadowX: 2,
    shadowY: 2,
    gradient: false,
    gradientAngle: 45,
    gradientFrom: "#000000",
    gradientTo: "#FFFFFF",
    gradientOpacity: 0,
}

const FILTER_GROUPS = ["Tone", "Color", "Curves", "Wheels", "Detail", "ImageKit"]
const ALL_VALUE_KEYS = new Set([...Object.keys(DEFAULT_VALUES)])
const getValuesSignature = (v) => JSON.stringify(v)

const isImageObject = (obj) => obj?.type === "image" || obj?.type === "Image"
const isVisibleImageObject = (obj) => isImageObject(obj) && obj.visible !== false

const getSelectedImage = (canvasEditor) => {
    if (!canvasEditor) return null
    const active = canvasEditor.getActiveObject()
    return isVisibleImageObject(active) ? active : null
}

const getVisibleImages = (canvasEditor) =>
    (canvasEditor?.getObjects?.() || []).filter(isVisibleImageObject)

const getAdjustmentTargets = (canvasEditor) => {
    const active = canvasEditor?.getActiveObject?.()
    if (active?.type === "activeSelection") {
        const picked = active.getObjects?.().filter(isVisibleImageObject) || []
        if (picked.length) return picked
    }
    const selected = getSelectedImage(canvasEditor)
    return selected ? [selected] : getVisibleImages(canvasEditor)
}

const getAdjustmentSourceImage = (canvasEditor) =>
    getSelectedImage(canvasEditor) || getVisibleImages(canvasEditor)[0] || null

const ensureAdjustmentObjectId = (imageObject) => {
    if (!imageObject) return null
    const existing = imageObject.phosmithAdjustmentId || imageObject._phosmithAdjustmentId || imageObject.__uid
    const id = existing || `image-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
    imageObject.phosmithAdjustmentId = id
    imageObject._phosmithAdjustmentId = id
    return id
}

const getImageSrc = (image) =>
    image?.getSrc?.() ||
    image?._originalElement?.src ||
    image?._element?.src ||
    image?.src ||
    ""

// The curves-panel histogram is a SOURCE reference — it shows the tonal distribution
// of the input pixels so the user can see where their data lives while shaping the
// curve. Reading from `_filteredEl` (the post-filter canvas) would make the bars
// shift around as the user drags the curve, which is confusing and non-standard.
// Apple Photos / Lightroom / Photoshop all show the original histogram with the
// curve overlaid on top. Prefer the original element here for the same behavior.
const getHistogramSourceElement = (image) =>
    image?._originalElement ||
    image?.getElement?.() ||
    image?._element ||
    image?._filteredEl ||
    image?._cacheCanvas ||
    null

const emptyHistogram = () => ({
    red: Array(HISTOGRAM_BUCKETS).fill(0),
    green: Array(HISTOGRAM_BUCKETS).fill(0),
    blue: Array(HISTOGRAM_BUCKETS).fill(0),
    luma: Array(HISTOGRAM_BUCKETS).fill(0),
})

const computeImageHistogram = (image) => {
    if (typeof document === "undefined" || !image) return null
    const source = getHistogramSourceElement(image)
    const sourceWidth = Math.round(source?.naturalWidth || source?.videoWidth || source?.width || image.width || 0)
    const sourceHeight = Math.round(source?.naturalHeight || source?.videoHeight || source?.height || image.height || 0)
    if (!source || !sourceWidth || !sourceHeight) return null

    const scale = Math.min(1, HISTOGRAM_SAMPLE_SIZE / Math.max(sourceWidth, sourceHeight))
    const width = Math.max(1, Math.round(sourceWidth * scale))
    const height = Math.max(1, Math.round(sourceHeight * scale))
    const scratch = document.createElement("canvas")
    scratch.width = width
    scratch.height = height
    const ctx = scratch.getContext("2d", { willReadFrequently: true })
    if (!ctx) return null

    try {
        ctx.drawImage(source, 0, 0, width, height)
        const { data } = ctx.getImageData(0, 0, width, height)
        const histogram = emptyHistogram()
        let pixels = 0

        for (let i = 0; i < data.length; i += 4) {
            const alpha = data[i + 3]
            if (alpha < 16) continue
            const r = data[i]
            const g = data[i + 1]
            const b = data[i + 2]
            const luma = clamp(Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b), 0, 255)
            histogram.red[r] += 1
            histogram.green[g] += 1
            histogram.blue[b] += 1
            histogram.luma[luma] += 1
            pixels += 1
        }

        return pixels ? histogram : null
    } catch (error) {
        console.warn("[Adjust] Histogram unavailable for selected image:", error)
        return null
    }
}

const sanitizeCurvePoints = (raw) => {
    const arr = Array.isArray(raw) ? raw : []
    const cleaned = arr
        .filter((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y))
        .map((p) => ({ x: clamp(p.x, 0, 1), y: clamp(p.y, 0, 1) }))
        .sort((a, b) => a.x - b.x)
    if (cleaned.length < 2) return cloneIdentityPoints()
    cleaned[0] = { x: 0, y: cleaned[0].y }
    cleaned[cleaned.length - 1] = { x: 1, y: cleaned[cleaned.length - 1].y }
    return cleaned
}

const normalizeStoredValues = (values) => {
    const next = { ...DEFAULT_VALUES, ...(values || {}) }
    for (const config of SLIDER_CONFIGS) {
        next[config.key] = clamp(Number(next[config.key] ?? config.defaultValue), config.min, config.max)
    }
    for (const config of WHEEL_CONFIGS) {
        next[config.key] = clamp(Number(next[config.key] ?? 0), 0, 100)
        next[config.colorKey] = String(next[config.colorKey] || config.defaultColor)
    }
    for (const key of CURVE_POINTS_KEYS) {
        next[key] = sanitizeCurvePoints(next[key])
    }
    next.look = LOOKS.some((look) => look.id === next.look) ? next.look : "none"
    return next
}

const filterMatchesManagedKey = (filter) =>
    filter?._phosmithAdjustmentManaged === true ||
    ALL_VALUE_KEYS.has(filter?._phosmithAdjustmentKey) ||
    filter?._phosmithAgentFilter === true ||
    [
        filters.Brightness.type,
        filters.Contrast.type,
        filters.Gamma.type,
        filters.BlendColor.type,
        filters.Saturation.type,
        filters.Vibrance.type,
        filters.HueRotation.type,
        filters.Convolute.type,
        filters.Blur.type,
        filters.Noise.type,
        filters.Pixelate.type,
        filters.ColorMatrix.type,
        filters.Vintage.type,
        filters.Kodachrome.type,
        filters.Technicolor.type,
        filters.Polaroid.type,
        filters.Brownie.type,
        filters.Sepia.type,
        filters.BlackWhite.type,
    ].includes(filter?.type)

const getValuesFromImageFilters = (imageObject) => {
    const stored = imageObject?.phosmithAdjustValues || imageObject?._phosmithAdjustValues
    if (stored) return normalizeStoredValues(stored)

    const next = { ...DEFAULT_VALUES }
    for (const f of imageObject?.filters ?? []) {
        if (f?.type === filters.Brightness.type && typeof f.brightness === "number") next.brightness = clamp(Math.round(f.brightness * 100), -100, 100)
        if (f?.type === filters.Contrast.type && typeof f.contrast === "number") next.contrast = clamp(Math.round(f.contrast * 100), -100, 100)
        if (f?.type === filters.Gamma.type && Array.isArray(f.gamma)) next.gamma = clamp(Math.round((f.gamma.reduce((sum, n) => sum + n, 0) / f.gamma.length) * 100), 20, 220)
        if (f?.type === filters.Saturation.type && typeof f.saturation === "number") next.saturation = clamp(Math.round(f.saturation * 100), -100, 100)
        if (f?.type === filters.Vibrance.type && typeof f.vibrance === "number") next.vibrance = clamp(Math.round(f.vibrance * 100), -100, 100)
        if (f?.type === filters.HueRotation.type && typeof f.rotation === "number") next.hue = clamp(Math.round(f.rotation * 180), -180, 180)
        if (f?.type === filters.Blur.type && typeof f.blur === "number") next.blur = clamp(Math.round(f.blur * 100), 0, 100)
        if (f?.type === filters.Noise.type && typeof f.noise === "number") next.noise = clamp(Math.round(f.noise / 6), 0, 100)
        if (f?.type === filters.Pixelate.type && typeof f.blocksize === "number") next.pixelate = clamp(Math.round(f.blocksize), 1, 32)
        if (isSharpnessFilter(f)) next.sharpness = clamp(Math.round(Math.max(0, -f.matrix[1]) * 100), 0, 100)
    }
    return next
}

const markFilter = (filter, key) => {
    filter._phosmithAdjustmentManaged = true
    filter._phosmithAdjustmentKey = key
    return filter
}

const buildChannelMatrix = (values) => {
    const r = 1 + Number(values.red || 0) / 115
    const g = 1 + Number(values.green || 0) / 115
    const b = 1 + Number(values.blue || 0) / 115
    const exposure = Number(values.exposure || 0) / 260
    const whites = Number(values.whites || 0) / 420
    const blacks = -Number(values.blacks || 0) / 520
    const fade = Number(values.fade || 0) / 520
    const offset = exposure + whites + blacks + fade
    const active =
        Math.abs(r - 1) > 0.001 ||
        Math.abs(g - 1) > 0.001 ||
        Math.abs(b - 1) > 0.001 ||
        Math.abs(offset) > 0.001

    if (!active) return null
    return [
        r, 0, 0, 0, offset,
        0, g, 0, 0, offset,
        0, 0, b, 0, offset,
        0, 0, 0, 1, 0,
    ]
}

const buildGammaValues = (values) => {
    const base = Number(values.gamma || 100) / 100
    return [base, base, base]
}

const buildCurvesFilter = (values) => {
    const masterPts = sanitizeCurvePoints(values.curvePointsMaster)
    const redPts = sanitizeCurvePoints(values.curvePointsRed)
    const greenPts = sanitizeCurvePoints(values.curvePointsGreen)
    const bluePts = sanitizeCurvePoints(values.curvePointsBlue)
    if (
        arePointsIdentity(masterPts) &&
        arePointsIdentity(redPts) &&
        arePointsIdentity(greenPts) &&
        arePointsIdentity(bluePts)
    ) {
        return null
    }
    return new PhosmithCurvesFilter({
        lutR: buildLut(redPts),
        lutG: buildLut(greenPts),
        lutB: buildLut(bluePts),
        lutMaster: buildLut(masterPts),
    })
}

const grayscaleMatrix = (amount) => {
    const a = clamp(amount, 0, 100) / 100
    const inv = 1 - a
    return [
        inv + 0.2126 * a, 0.7152 * a, 0.0722 * a, 0, 0,
        0.2126 * a, inv + 0.7152 * a, 0.0722 * a, 0, 0,
        0.2126 * a, 0.7152 * a, inv + 0.0722 * a, 0, 0,
        0, 0, 0, 1, 0,
    ]
}

const buildConvolution = (amount) => {
    const a = Math.abs(amount) / 100
    if (a <= 0) return null
    if (amount < 0) {
        return [a / 9, a / 9, a / 9, a / 9, 1 - (8 * a) / 9, a / 9, a / 9, a / 9, a / 9]
    }
    return [0, -a, 0, -a, 1 + 4 * a, -a, 0, -a, 0]
}

const addBlend = (acc, key, color, mode, alpha) => {
    if (alpha <= 0) return
    acc.push(markFilter(new filters.BlendColor({ color, mode, alpha: clamp(alpha, 0, 1) }), key))
}

const buildFabricFilters = (values) => {
    const next = []
    const look = LOOKS.find((item) => item.id === values.look)

    if (look?.filterClass) next.push(markFilter(new look.filterClass(), "look"))
    if (values.brightness) next.push(markFilter(new filters.Brightness({ brightness: values.brightness / 100 }), "brightness"))
    if (values.contrast) next.push(markFilter(new filters.Contrast({ contrast: values.contrast / 100 }), "contrast"))
    const gamma = buildGammaValues(values)
    if (gamma.some((value) => Math.abs(value - 1) > 0.001)) next.push(markFilter(new filters.Gamma({ gamma }), "gamma"))
    if (values.temperature) addBlend(next, "temperature", values.temperature >= 0 ? TEMP_WARM : TEMP_COOL, "tint", Math.abs(values.temperature) / 280)
    if (values.tint) addBlend(next, "tint", values.tint >= 0 ? "#e879f9" : "#46d68c", "tint", Math.abs(values.tint) / 320)
    if (values.saturation) next.push(markFilter(new filters.Saturation({ saturation: values.saturation / 100 }), "saturation"))
    if (values.vibrance) next.push(markFilter(new filters.Vibrance({ vibrance: values.vibrance / 100 }), "vibrance"))
    if (values.hue) next.push(markFilter(new filters.HueRotation({ rotation: values.hue / 180 }), "hue"))

    const channelMatrix = buildChannelMatrix(values)
    if (channelMatrix) next.push(markFilter(new filters.ColorMatrix({ matrix: channelMatrix }), "rgb-mixer"))

    const curvesFilter = buildCurvesFilter(values)
    if (curvesFilter) next.push(markFilter(curvesFilter, "curves"))

    // ── Highlights / Shadows / Whites / Blacks ──────────────────────────
    // Fabric's BlendColor "multiply" mode does NOT lerp between the original
    // pixel and the blended result — it bakes alpha into the blend color first
    // (tr = color_channel * alpha) and then does pixel*tr/255. With dark blend
    // colors (#000000, #050505, #1b1b1b) this drives every pixel to near-zero
    // (black) at any noticeable alpha. The fix: use a ColorMatrix that linearly
    // scales pixel brightness — `lerp(1.0, darkFactor, strength)` — so the
    // image darkens smoothly instead of crushing to black.
    //
    // Positive values (lighten) still use BlendColor "screen" which works fine
    // because screen with bright colors correctly lifts pixel values.

    if (values.highlights > 0) addBlend(next, "highlights", "#ffffff", "screen", values.highlights / 260)
    if (values.highlights < 0) {
        // Darken highlights: scale brightness toward ~0.7 at max (-100)
        const s = 1 - Math.abs(values.highlights) / 330
        next.push(markFilter(new filters.ColorMatrix({
            matrix: [s, 0, 0, 0, 0, 0, s, 0, 0, 0, 0, 0, s, 0, 0, 0, 0, 0, 1, 0],
        }), "highlights"))
    }
    if (values.shadows > 0) addBlend(next, "shadows", "#cdd8ff", "screen", values.shadows / 360)
    if (values.shadows < 0) {
        // Darken shadows: scale brightness toward ~0.62 at max (-100)
        const s = 1 - Math.abs(values.shadows) / 260
        next.push(markFilter(new filters.ColorMatrix({
            matrix: [s, 0, 0, 0, 0, 0, s, 0, 0, 0, 0, 0, s, 0, 0, 0, 0, 0, 1, 0],
        }), "shadows"))
    }
    if (values.whites > 0) addBlend(next, "whites", "#ffffff", "screen", values.whites / 340)
    if (values.blacks > 0) {
        // Crush blacks: scale brightness toward ~0.67 at max (100)
        const s = 1 - values.blacks / 300
        next.push(markFilter(new filters.ColorMatrix({
            matrix: [s, 0, 0, 0, 0, 0, s, 0, 0, 0, 0, 0, s, 0, 0, 0, 0, 0, 1, 0],
        }), "blacks"))
    }

    for (const wheel of WHEEL_CONFIGS) {
        const amount = Number(values[wheel.key] || 0)
        if (amount > 0) addBlend(next, wheel.key, values[wheel.colorKey], wheel.mode, amount / (wheel.mode === "tint" ? 120 : 260))
    }

    const clarityMatrix = buildConvolution(values.clarity)
    if (clarityMatrix) next.push(markFilter(new filters.Convolute({ opaque: false, matrix: clarityMatrix }), "clarity"))
    if (values.sharpness) {
        const a = values.sharpness / 100
        next.push(markFilter(new filters.Convolute({ opaque: false, matrix: [0, -a, 0, -a, 1 + 4 * a, -a, 0, -a, 0] }), "sharpness"))
    }
    if (values.blur) next.push(markFilter(new filters.Blur({ blur: values.blur / 100 }), "blur"))
    if (values.noise) next.push(markFilter(new filters.Noise({ noise: values.noise * 6 }), "noise"))
    if (values.grain) next.push(markFilter(new filters.Noise({ noise: values.grain * 4 }), "grain"))
    if (values.pixelate !== 1) next.push(markFilter(new filters.Pixelate({ blocksize: values.pixelate }), "pixelate"))
    if (values.mono) next.push(markFilter(new filters.ColorMatrix({ matrix: grayscaleMatrix(values.mono) }), "mono"))

    return next
}

const isVignetteLayer = (obj) =>
    obj?.name === VIGNETTE_LAYER_NAME ||
    obj?.phosmithAdjustmentOverlay === "vignette" ||
    obj?._phosmithAdjustmentOverlay === "vignette"

const removeVignetteLayers = (canvasEditor, targetId = null) => {
    const objects = canvasEditor?.getObjects?.() || []
    const layers = objects.filter((obj) => {
        if (!isVignetteLayer(obj)) return false
        if (!targetId) return true
        const layerTarget = obj.phosmithAdjustmentTargetId || obj._phosmithAdjustmentTargetId
        return layerTarget === targetId || !layerTarget
    })
    layers.forEach((layer) => canvasEditor.remove(layer))
    return layers.length
}

const applyVignetteLayer = (canvasEditor, imageObject, values) => {
    if (!canvasEditor || !imageObject) return
    const targetId = ensureAdjustmentObjectId(imageObject)
    const amount = Number(values.vignette || 0)
    if (!amount) {
        removeVignetteLayers(canvasEditor, targetId)
        return
    }

    removeVignetteLayers(canvasEditor, targetId)

    const bounds = imageObject.getBoundingRect()
    const width = Math.max(1, bounds.width)
    const height = Math.max(1, bounds.height)
    const radius = Math.max(width, height) * (clamp(values.vignetteSize, 20, 95) / 100)
    const alpha = Math.min(0.82, Math.abs(amount) / 125)
    const edgeColor = amount > 0 ? `rgba(0,0,0,${alpha})` : `rgba(255,255,255,${alpha * 0.55})`
    const feather = clamp(values.vignetteFeather, 0, 100) / 100
    const clearStop = Math.max(0.05, Math.min(0.86, 1 - feather * 0.72))

    const vignette = new Rect({
        left: bounds.left,
        top: bounds.top,
        width,
        height,
        originX: "left",
        originY: "top",
        fill: new Gradient({
            type: "radial",
            gradientUnits: "pixels",
            coords: {
                x1: width / 2,
                y1: height / 2,
                r1: radius * clearStop,
                x2: width / 2,
                y2: height / 2,
                r2: radius,
            },
            colorStops: [
                { offset: 0, color: "rgba(0,0,0,0)" },
                { offset: 0.72, color: "rgba(0,0,0,0)" },
                { offset: 1, color: edgeColor },
            ],
        }),
        selectable: false,
        evented: false,
        hasControls: false,
        hasBorders: false,
        objectCaching: false,
        name: VIGNETTE_LAYER_NAME,
        phosmithAdjustmentOverlay: "vignette",
        _phosmithAdjustmentOverlay: "vignette",
        phosmithAdjustmentTargetId: targetId,
        _phosmithAdjustmentTargetId: targetId,
    })

    canvasEditor.add(vignette)
    const imageIndex = canvasEditor.getObjects().indexOf(imageObject)
    if (imageIndex >= 0 && typeof canvasEditor.moveObjectTo === "function") {
        canvasEditor.moveObjectTo(vignette, imageIndex + 1)
    }
}

const applyAdjustmentFilters = (canvasEditor, values, sigRef, { commit = false } = {}) => {
    if (!canvasEditor) return
    const selectedImage = getSelectedImage(canvasEditor)
    const targets = getAdjustmentTargets(canvasEditor)
    if (!targets.length) return
    const normalized = normalizeStoredValues(values)
    const sig = getValuesSignature(normalized)
    const originalActiveObject = canvasEditor.getActiveObject()

    try {
        targets.forEach((img) => {
            const currentFilters = img.filters ?? []
            const preservedFilters = currentFilters.filter((filter) => !filterMatchesManagedKey(filter))
            const managedFilters = buildFabricFilters(normalized)
            img.filters = [...preservedFilters, ...managedFilters]
            img.phosmithAdjustValues = normalized
            img._phosmithAdjustValues = normalized
            img.applyFilters()
            img.set("dirty", true)
            applyVignetteLayer(canvasEditor, img, normalized)
            if (commit) canvasEditor.fire("object:modified", { target: img })
        })
        if (selectedImage) canvasEditor.setActiveObject(selectedImage)
        else if (originalActiveObject) canvasEditor.setActiveObject(originalActiveObject)
        else canvasEditor.discardActiveObject()
        canvasEditor.requestRenderAll()
        sigRef.current = sig
    } catch (e) {
        console.error(e)
    }
}

const FILTER_VISUAL = {
    exposure: { fill: "rgba(104, 92, 62, 0.5)", accent: "#ffe39a", trackBg: "rgba(20, 18, 13, 0.98)" },
    brightness: { fill: "rgba(42, 58, 82, 0.5)", accent: "#d2dae6", trackBg: "rgba(20, 24, 32, 0.98)" },
    contrast: { fill: "rgba(38, 44, 56, 0.52)", accent: "#e4eaf2", trackBg: "rgba(18, 20, 28, 0.98)" },
    highlights: { fill: "rgba(102, 94, 64, 0.5)", accent: "#f6d36b", trackBg: "rgba(22, 20, 14, 0.98)" },
    shadows: { fill: "rgba(46, 58, 86, 0.5)", accent: "#92b7ff", trackBg: "rgba(13, 16, 24, 0.98)" },
    whites: { fill: "rgba(215, 220, 230, 0.32)", accent: "#f8fafc", trackBg: "rgba(21, 24, 30, 0.98)" },
    blacks: { fill: "rgba(16, 18, 24, 0.78)", accent: "#8b95a6", trackBg: "rgba(8, 9, 12, 0.98)" },
    gamma: { fill: "rgba(58, 48, 88, 0.5)", accent: "#b8a6f0", trackBg: "rgba(18, 16, 26, 0.98)" },
    fade: { fill: "rgba(82, 76, 68, 0.46)", accent: "#d2c1aa", trackBg: "rgba(18, 16, 14, 0.98)" },
    temperature: { fill: "rgba(118, 108, 72, 0.52)", accent: "#ebc94a", trackBg: "rgba(22, 24, 18, 0.98)", bottomAccent: "linear-gradient(90deg, #3a8fd4 0%, #e8c04a 100%)" },
    tint: { fill: "rgba(80, 54, 92, 0.5)", accent: "#d58cff", trackBg: "rgba(18, 14, 22, 0.98)", bottomAccent: "linear-gradient(90deg, #45d483 0%, #d879f7 100%)" },
    saturation: { fill: "rgba(92, 48, 62, 0.5)", accent: "#f07878", trackBg: "rgba(20, 16, 18, 0.98)" },
    vibrance: { fill: "rgba(40, 88, 78, 0.5)", accent: "#5ec9b0", trackBg: "rgba(14, 20, 20, 0.98)" },
    hue: { fill: "rgba(88, 82, 48, 0.48)", accent: "#a8e070", trackBg: "rgba(18, 18, 16, 0.98)", bottomAccent: "linear-gradient(90deg, #ff4b4b, #ffd84b, #65e56e, #4bdcff, #6a63ff, #ff4bf2)" },
    red: { fill: "rgba(128, 44, 54, 0.5)", accent: "#ff5d65", trackBg: "rgba(22, 12, 14, 0.98)" },
    green: { fill: "rgba(44, 98, 64, 0.5)", accent: "#64d989", trackBg: "rgba(12, 20, 15, 0.98)" },
    blue: { fill: "rgba(42, 64, 112, 0.5)", accent: "#69a7ff", trackBg: "rgba(12, 16, 24, 0.98)" },
    clarity: { fill: "rgba(45, 62, 72, 0.5)", accent: "#9bd5df", trackBg: "rgba(12, 18, 21, 0.98)" },
    sharpness: { fill: "rgba(38, 62, 88, 0.5)", accent: "#9fc8e8", trackBg: "rgba(14, 18, 24, 0.98)" },
    blur: { fill: "rgba(32, 58, 82, 0.5)", accent: "#7eb8dc", trackBg: "rgba(12, 16, 22, 0.98)" },
    noise: { fill: "rgba(58, 62, 72, 0.5)", accent: "#c8ced8", trackBg: "rgba(16, 17, 20, 0.98)" },
    grain: { fill: "rgba(72, 66, 56, 0.48)", accent: "#c9b89a", trackBg: "rgba(18, 16, 13, 0.98)" },
    pixelate: { fill: "rgba(38, 72, 58, 0.5)", accent: "#8ccfb1", trackBg: "rgba(14, 20, 18, 0.98)" },
    mono: { fill: "rgba(76, 82, 92, 0.48)", accent: "#e5e7eb", trackBg: "rgba(15, 17, 20, 0.98)" },
    vignette: { fill: "rgba(10, 10, 12, 0.74)", accent: "#d4d4d8", trackBg: "rgba(7, 7, 9, 0.98)" },
    vignetteSize: { fill: "rgba(52, 58, 68, 0.5)", accent: "#a9b3c4", trackBg: "rgba(13, 14, 17, 0.98)" },
    vignetteFeather: { fill: "rgba(68, 58, 72, 0.5)", accent: "#d0b8da", trackBg: "rgba(16, 13, 18, 0.98)" },
}

const ColorWheelPicker = ({ color, onChange }) => {
    const wheelRef = useRef(null)
    const hsv = rgbToHsv(hexToRgb(color))
    const pointerAngle = ((hsv.h - 90) * Math.PI) / 180
    const pointerRadius = hsv.s * 43

    const updateColorFromEvent = (event) => {
        const rect = wheelRef.current?.getBoundingClientRect()
        if (!rect) return
        const cx = rect.left + rect.width / 2
        const cy = rect.top + rect.height / 2
        const dx = event.clientX - cx
        const dy = event.clientY - cy
        const radius = Math.max(1, Math.min(rect.width, rect.height) / 2)
        const saturation = clamp(Math.sqrt(dx * dx + dy * dy) / radius, 0, 1)
        const hue = (Math.atan2(dy, dx) * 180) / Math.PI + 90
        const rgb = hsvToRgb((hue + 360) % 360, saturation, 1)
        onChange(rgbToHex(rgb.r, rgb.g, rgb.b))
    }

    const handlePointerDown = (event) => {
        event.preventDefault()
        updateColorFromEvent(event)
        const handleMove = (moveEvent) => updateColorFromEvent(moveEvent)
        const handleUp = () => {
            window.removeEventListener("pointermove", handleMove)
            window.removeEventListener("pointerup", handleUp)
            window.removeEventListener("pointercancel", handleUp)
        }
        window.addEventListener("pointermove", handleMove)
        window.addEventListener("pointerup", handleUp)
        window.addEventListener("pointercancel", handleUp)
    }

    return (
        <div
            ref={wheelRef}
            className="adjust-wheel-picker"
            role="button"
            aria-label="Color wheel"
            tabIndex={0}
            onPointerDown={handlePointerDown}
        >
            <span
                style={{
                    left: `${50 + Math.cos(pointerAngle) * pointerRadius}%`,
                    top: `${50 + Math.sin(pointerAngle) * pointerRadius}%`,
                    background: color,
                }}
            />
        </div>
    )
}

const ColorWheelCard = ({ config, values, onColor, onAmount }) => (
    <div className="adjust-color-wheel-card">
        <div className="adjust-color-wheel-top">
            <div>
                <span>{config.label}</span>
                <strong>{values[config.key]}</strong>
            </div>
            <span className="adjust-color-chip" style={{ background: values[config.colorKey] }} />
        </div>
        <ColorWheelPicker
            color={values[config.colorKey]}
            onChange={(nextColor) => onColor(config.colorKey, nextColor)}
        />
        <Colorful
            color={values[config.colorKey]}
            disableAlpha
            className="adjust-colorful"
            onChange={(color) => onColor(config.colorKey, color.hex)}
        />
        <ProRulerSlider
            variant="instrument"
            value={values[config.key]}
            onPreview={(v) => onAmount(config.key, v, false)}
            onCommit={(v) => onAmount(config.key, v, true)}
            min={0}
            max={100}
            step={1}
            label="Strength"
            visual={{ fill: "rgba(60, 72, 86, 0.48)", accent: values[config.colorKey], trackBg: "rgba(14, 16, 20, 0.98)" }}
        />
    </div>
)

const HISTOGRAM_SERIES = {
    red: { label: "Red", color: "#ff5d65" },
    green: { label: "Green", color: "#64d989" },
    blue: { label: "Blue", color: "#69a7ff" },
    luma: { label: "Luminance", color: "#d8dde7" },
}

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
        const strength = Math.log1p(value) / maxLog
        const y = bottom - clamp(strength, 0, 1) * height
        return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`
    }).join(" ")

    return {
        line: points,
        fill: `${points} L ${right} ${bottom} L ${left} ${bottom} Z`,
    }
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
    if (!Array.isArray(points)) return -1
    let best = -1
    let bestDist = CURVE_HIT_RADIUS_VB
    for (let i = 0; i < points.length; i++) {
        const p = points[i]
        const px = CURVE_GRAPH.left + p.x * (CURVE_GRAPH.right - CURVE_GRAPH.left)
        const py = CURVE_GRAPH.bottom - p.y * (CURVE_GRAPH.bottom - CURVE_GRAPH.top)
        const dx = vbX - px
        const dy = vbY - py
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < bestDist) {
            bestDist = dist
            best = i
        }
    }
    return best
}

const CurveGraph = ({ channel, values, histogram, onBegin, onPreview, onCommit }) => {
    const svgRef = useRef(null)
    const dragRef = useRef(null)
    const [activeIndex, setActiveIndex] = useState(null)
    const [hoverIndex, setHoverIndex] = useState(null)

    const points = sanitizeCurvePoints(values[channel.pointsKey])
    const isRgb = channel.id === "rgb"
    const path = buildCurveSvgPath(points, CURVE_GRAPH)

    const commitPoints = (next, { commit }) => {
        const sanitized = sanitizeCurvePoints(next)
        if (commit) onCommit(channel.pointsKey, sanitized)
        else onPreview(channel.pointsKey, sanitized)
    }

    useEffect(() => {
        if (activeIndex === null) return undefined

        const handleMove = (event) => {
            const drag = dragRef.current
            if (!drag) return
            const svg = svgRef.current
            if (!svg) return
            const loc = clientToCurvePoint(svg, event.clientX, event.clientY)
            if (!loc) return
            const next = drag.points.map((p, i) => ({ ...p }))
            const isFirst = drag.index === 0
            const isLast = drag.index === next.length - 1
            const prev = next[drag.index - 1]
            const succ = next[drag.index + 1]
            const minX = isFirst ? 0 : (prev ? prev.x + CURVE_POINT_MIN_GAP : 0)
            const maxX = isLast ? 1 : (succ ? succ.x - CURVE_POINT_MIN_GAP : 1)
            const newX = isFirst ? 0 : isLast ? 1 : clamp(loc.x, minX, maxX)
            const newY = clamp(loc.y, 0, 1)
            next[drag.index] = { x: newX, y: newY }
            dragRef.current = { ...drag, points: next }
            commitPoints(next, { commit: false })
        }

        const handleUp = () => {
            const drag = dragRef.current
            if (drag) commitPoints(drag.points, { commit: true })
            dragRef.current = null
            setActiveIndex(null)
        }

        window.addEventListener("pointermove", handleMove)
        window.addEventListener("pointerup", handleUp)
        window.addEventListener("pointercancel", handleUp)
        return () => {
            window.removeEventListener("pointermove", handleMove)
            window.removeEventListener("pointerup", handleUp)
            window.removeEventListener("pointercancel", handleUp)
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [activeIndex, channel.pointsKey])

    const beginDrag = (index, event) => {
        if (event.button !== 0 && event.button !== undefined) return
        event.preventDefault()
        event.stopPropagation()
        onBegin?.()
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
        if (existing >= 0) {
            beginDrag(existing, event)
            return
        }
        // Add a new point at this location
        event.preventDefault()
        event.stopPropagation()
        onBegin?.()
        const next = [...points, { x: loc.x, y: loc.y }].sort((a, b) => a.x - b.x)
        const sanitized = sanitizeCurvePoints(next)
        let insertedIndex = sanitized.findIndex((p) => Math.abs(p.x - loc.x) < 0.0001 && Math.abs(p.y - loc.y) < 0.0001)
        if (insertedIndex < 0) insertedIndex = Math.max(1, Math.min(sanitized.length - 2, sanitized.length - 1))
        dragRef.current = { index: insertedIndex, points: sanitized }
        commitPoints(sanitized, { commit: false })
        setActiveIndex(insertedIndex)
    }

    const handleDoubleClickPoint = (index) => (event) => {
        event.preventDefault()
        event.stopPropagation()
        if (index === 0 || index === points.length - 1) return
        onBegin?.()
        const next = points.filter((_, i) => i !== index)
        commitPoints(next, { commit: true })
    }

    return (
        <div className="adjust-curve-graph">
            <svg ref={svgRef} viewBox="0 0 100 100" className="adjust-curve-svg" preserveAspectRatio="none" onPointerDown={handlePointerDownGraph}>
                {[20, 40, 60, 80].map((line) => (
                    <React.Fragment key={line}>
                        <line x1={line} y1="6" x2={line} y2="94" className="adjust-curve-grid" />
                        <line x1="6" y1={line} x2="94" y2={line} className="adjust-curve-grid" />
                    </React.Fragment>
                ))}
                {(channel.histogramChannels || []).map((key) => {
                    const series = HISTOGRAM_SERIES[key]
                    const paths = buildHistogramPaths(histogram?.[key])
                    if (!paths || !series) return null
                    return (
                        <g
                            key={key}
                            className={`adjust-curve-histogram-series ${isRgb ? "is-rgb" : "is-single"}`}
                            style={{ "--curve-color": series.color }}
                            pointerEvents="none"
                        >
                            <path d={paths.fill} className="adjust-curve-histogram-fill" />
                            <path d={paths.line} className="adjust-curve-histogram-line" />
                        </g>
                    )
                })}
                <line
                    x1={CURVE_GRAPH.left}
                    y1={CURVE_GRAPH.bottom}
                    x2={CURVE_GRAPH.right}
                    y2={CURVE_GRAPH.top}
                    className="adjust-curve-diagonal"
                    style={{ "--curve-color": channel.color }}
                    pointerEvents="none"
                />
                <path d={path} className="adjust-curve-path" style={{ "--curve-color": channel.color }} pointerEvents="none" />
                {points.map((p, index) => {
                    const cx = CURVE_GRAPH.left + p.x * (CURVE_GRAPH.right - CURVE_GRAPH.left)
                    const cy = CURVE_GRAPH.bottom - p.y * (CURVE_GRAPH.bottom - CURVE_GRAPH.top)
                    const isActive = activeIndex === index
                    const isHovered = hoverIndex === index
                    return (
                        <g
                            key={index}
                            className={`adjust-curve-point-group ${isActive ? "is-active" : ""} ${isHovered ? "is-hover" : ""}`}
                            style={{ "--curve-color": channel.color }}
                        >
                            <circle
                                cx={cx}
                                cy={cy}
                                r="4.5"
                                className="adjust-curve-point-hit"
                                onPointerDown={(event) => beginDrag(index, event)}
                                onDoubleClick={handleDoubleClickPoint(index)}
                                onPointerEnter={() => setHoverIndex(index)}
                                onPointerLeave={() => setHoverIndex((cur) => (cur === index ? null : cur))}
                            />
                            <circle
                                cx={cx}
                                cy={cy}
                                r="1.7"
                                className="adjust-curve-point"
                                pointerEvents="none"
                            />
                        </g>
                    )
                })}
            </svg>
        </div>
    )
}

const CurveEditorPanel = ({ values, histogram, activeChannel, onChannelChange, onBegin, onPreview, onCommit }) => {
    const channel = CURVE_CHANNELS.find((item) => item.id === activeChannel) || CURVE_CHANNELS[0]
    const channelPoints = sanitizeCurvePoints(values[channel.pointsKey])
    const isChannelDirty = !arePointsIdentity(channelPoints)

    const handleResetChannel = () => {
        onBegin?.()
        onCommit(channel.pointsKey, cloneIdentityPoints())
    }

    return (
        <div className="adjust-curve-card" style={{ "--curve-color": channel.color }}>
            <div className="adjust-curve-card-header">
                <div className="adjust-curve-card-title">
                    <LineChart aria-hidden="true" />
                    <span>Curves</span>
                </div>
                <button
                    type="button"
                    onClick={handleResetChannel}
                    disabled={!isChannelDirty}
                    className="adjust-curve-reset"
                    title={`Reset ${channel.label} channel`}
                >
                    <RotateCcw aria-hidden="true" />
                    Reset
                </button>
            </div>
            <div className="adjust-curve-toolbar">
                <span className="adjust-curve-toolbar-label">Channel</span>
                <label className="adjust-curve-select-wrap">
                    <select
                        value={channel.id}
                        onChange={(event) => onChannelChange(event.target.value)}
                        className="adjust-curve-select"
                        style={{ "--curve-color": channel.color }}
                        aria-label="Curve channel"
                    >
                        {CURVE_CHANNELS.map((item) => (
                            <option key={item.id} value={item.id}>
                                {item.label}
                            </option>
                        ))}
                    </select>
                </label>
            </div>
            <CurveGraph
                channel={channel}
                values={values}
                histogram={histogram}
                onBegin={onBegin}
                onPreview={onPreview}
                onCommit={onCommit}
            />
            <p className="adjust-curve-hint">
                Click the curve to add a point · drag to shape · double-click a point to remove
            </p>
        </div>
    )
}

// AI transforms that are model calls — must be chained as SEPARATE steps (colon-separated).
const AI_TRANSFORM_PREFIXES = ['e-retouch', 'e-upscale']
const isAiTransform = (token) => AI_TRANSFORM_PREFIXES.some((p) => token.startsWith(p))

const buildImageKitTokens = (values) => {
    const tokens = []
    if (values.autoContrast) tokens.push("e-contrast")
    if (values.retouch) tokens.push("e-retouch")
    if (values.upscale) tokens.push("e-upscale")
    if (values.grayscale) tokens.push("e-grayscale")
    if (values.sharpen > 0) tokens.push(`e-sharpen-${values.sharpen}`)
    if (values.usm > 0) tokens.push(`e-usm-2-2-${(values.usm / 100).toFixed(2)}-0.02`)
    if (values.urlBlur > 0) tokens.push(`bl-${values.urlBlur}`)
    if (values.colorize > 0) tokens.push(`e-colorize-co-${stripHex(values.colorizeColor)}_in-${values.colorize}`)
    if (values.shadow) {
        tokens.push(`e-shadow-bl-${values.shadowBlur}_st-${values.shadowSaturation}_x-${signedToken(values.shadowX)}_y-${signedToken(values.shadowY)}`)
    }
    if (values.gradient && values.gradientOpacity > 0) {
        const alpha = imageKitOpacity(values.gradientOpacity)
        tokens.push(`e-gradient-ld-${values.gradientAngle}_from-${stripHex(values.gradientFrom)}${alpha}_to-${stripHex(values.gradientTo)}${alpha}_sp-0.5`)
    }
    return tokens
}

const ImageKitPanel = ({ imageKitValues, setImageKitValues, onApply, isApplying }) => {
    const tokens = buildImageKitTokens(imageKitValues)
    const setValue = (key, value) => setImageKitValues((prev) => ({ ...prev, [key]: value }))
    const toggle = (key) => {
        const nextValue = !imageKitValues[key]
        console.log("[Adjust ImageKit] toggle", {
            key,
            nextValue,
            tokensBefore: tokens,
        })
        setValue(key, nextValue)
    }

    return (
        <div className="adjust-imagekit-panel">
            <div className="adjust-toggle-grid">
                {[
                    ["autoContrast", "Auto contrast"],
                    ["retouch", "AI retouch"],
                    ["upscale", "AI upscale"],
                    ["grayscale", "Grayscale"],
                    ["shadow", "Shadow"],
                    ["gradient", "Gradient"],
                ].map(([key, label]) => (
                    <button
                        key={key}
                        type="button"
                        onClick={() => toggle(key)}
                        className={`adjust-toggle ${imageKitValues[key] ? "is-on" : ""}`}
                    >
                        <span />
                        {label}
                    </button>
                ))}
            </div>

            {[
                { key: "sharpen", label: "URL Sharpen", min: 0, max: 100 },
                { key: "usm", label: "Unsharp Mask", min: 0, max: 100 },
                { key: "urlBlur", label: "URL Blur", min: 0, max: 100 },
                { key: "colorize", label: "Colorize", min: 0, max: 100 },
                { key: "shadowBlur", label: "Shadow Blur", min: 0, max: 15 },
                { key: "shadowSaturation", label: "Shadow Sat", min: 0, max: 100 },
                { key: "shadowX", label: "Shadow X", min: -100, max: 100 },
                { key: "shadowY", label: "Shadow Y", min: -100, max: 100 },
                { key: "gradientAngle", label: "Gradient Angle", min: 0, max: 359, suffix: "º" },
                { key: "gradientOpacity", label: "Gradient Opacity", min: 0, max: 100 },
            ].map((config) => (
                <ProRulerSlider
                    key={config.key}
                    variant="instrument"
                    value={imageKitValues[config.key]}
                    onCommit={(v) => setValue(config.key, v)}
                    min={config.min}
                    max={config.max}
                    step={1}
                    label={config.label}
                    suffix={config.suffix || ""}
                    visual={FILTER_VISUAL.sharpness}
                />
            ))}

            <div className="adjust-mini-color-grid">
                {[
                    ["colorizeColor", "Colorize"],
                    ["gradientFrom", "Gradient A"],
                    ["gradientTo", "Gradient B"],
                ].map(([key, label]) => (
                    <label key={key} className="adjust-rgb-field">
                        <span>{label}</span>
                        <input type="color" value={imageKitValues[key]} onChange={(event) => setValue(key, event.target.value)} />
                    </label>
                ))}
            </div>

            <div className="adjust-token-preview">
                <span>URL chain</span>
                <code>{tokens.length ? tokens.join(",") : "No ImageKit URL transforms selected"}</code>
            </div>

            <button type="button" className="adjust-apply-imagekit" onClick={onApply} disabled={isApplying || tokens.length === 0}>
                {isApplying ? <Sparkles className="h-3.5 w-3.5 animate-spin" /> : <WandSparkles className="h-3.5 w-3.5" />}
                {isApplying ? "Applying..." : "Apply ImageKit URL transforms"}
            </button>
        </div>
    )
}

const AdjustControls = () => {
    const { canvasEditor, setProcessingMessage } = useCanvas()
    const [activeTab, setActiveTab] = useState("Tone")
    const [activeCurveChannel, setActiveCurveChannel] = useState("rgb")
    const [values, setValues] = useState(DEFAULT_VALUES)
    const [curveHistogram, setCurveHistogram] = useState(null)
    const [imageKitValues, setImageKitValues] = useState(IMAGEKIT_DEFAULTS)
    const [isApplyingImageKit, setIsApplyingImageKit] = useState(false)
    const latestRef = useRef(DEFAULT_VALUES)
    const sigRef = useRef(getValuesSignature(DEFAULT_VALUES))
    const committedSigRef = useRef(getValuesSignature(DEFAULT_VALUES))
    const previewFrame = useRef(null)
    const histogramFrameRef = useRef(null)
    const pendingPreviewRef = useRef(null)
    const isInteractingRef = useRef(false)

    const queueCurveHistogramRefresh = useCallback((image) => {
        if (histogramFrameRef.current) {
            cancelAnimationFrame(histogramFrameRef.current)
            histogramFrameRef.current = null
        }

        const targetImage = image || getAdjustmentSourceImage(canvasEditor)
        histogramFrameRef.current = requestAnimationFrame(() => {
            histogramFrameRef.current = null
            setCurveHistogram(targetImage ? computeImageHistogram(targetImage) : null)
        })
    }, [canvasEditor])

    useEffect(() => {
        if (!canvasEditor) return
        const sync = () => {
            if (isInteractingRef.current) return
            const img = getAdjustmentSourceImage(canvasEditor)
            const next = img ? getValuesFromImageFilters(img) : { ...DEFAULT_VALUES }
            queueCurveHistogramRefresh(img)
            pendingPreviewRef.current = null
            if (previewFrame.current) {
                cancelAnimationFrame(previewFrame.current)
                previewFrame.current = null
            }
            latestRef.current = next
            sigRef.current = getValuesSignature(next)
            committedSigRef.current = getValuesSignature(next)
            setValues((cur) => (getValuesSignature(cur) === getValuesSignature(next) ? cur : next))
        }
        sync()
        canvasEditor.on("selection:created", sync)
        canvasEditor.on("selection:updated", sync)
        canvasEditor.on("selection:cleared", sync)
        canvasEditor.on("object:added", sync)
        return () => {
            pendingPreviewRef.current = null
            if (previewFrame.current) cancelAnimationFrame(previewFrame.current)
            if (histogramFrameRef.current) {
                cancelAnimationFrame(histogramFrameRef.current)
                histogramFrameRef.current = null
            }
            canvasEditor.off("selection:created", sync)
            canvasEditor.off("selection:updated", sync)
            canvasEditor.off("selection:cleared", sync)
            canvasEditor.off("object:added", sync)
        }
    }, [canvasEditor, queueCurveHistogramRefresh])

    const handleBeginChange = () => {
        isInteractingRef.current = true
    }

    const applyNextValues = (next, { commit = false, updateState = false } = {}) => {
        latestRef.current = next
        if (updateState) setValues(next)
        const nextSig = getValuesSignature(next)
        applyAdjustmentFilters(canvasEditor, next, sigRef, { commit: commit && nextSig !== committedSigRef.current })
        // Histogram is computed from the source image (see getHistogramSourceElement)
        // so it doesn't change during a drag — no refresh needed on every preview frame.
        if (commit) {
            committedSigRef.current = nextSig
            isInteractingRef.current = false
        }
    }

    const handlePreviewChange = (key, v) => {
        const isPointsKey = CURVE_POINTS_KEYS.includes(key)
        const val = isPointsKey ? v : (Array.isArray(v) ? v[0] : v)
        const prev = latestRef.current
        if (!isPointsKey && prev[key] === val) return
        const next = { ...prev, [key]: val }
        // For curve points, update React state immediately so the SVG path
        // re-renders in real-time while dragging (not just on mouse-up).
        applyNextValues(next, { updateState: isPointsKey })
    }

    const handleCommitChange = (key, v) => {
        const isPointsKey = CURVE_POINTS_KEYS.includes(key)
        const val = isPointsKey ? v : (Array.isArray(v) ? v[0] : v)
        if (previewFrame.current) {
            cancelAnimationFrame(previewFrame.current)
            previewFrame.current = null
        }
        pendingPreviewRef.current = null
        applyNextValues({ ...latestRef.current, [key]: val }, { commit: true, updateState: true })
    }

    const handleColorChange = (key, color) => {
        const next = { ...latestRef.current, [key]: color }
        applyNextValues(next, { commit: true, updateState: true })
    }

    const handleWheelAmount = (key, v, commit) => {
        const val = Array.isArray(v) ? v[0] : v
        const next = { ...latestRef.current, [key]: val }
        applyNextValues(next, { commit, updateState: commit })
    }

    const resolveImageKitUrl = async (url, { source, tokens }) => {
        console.log("[Adjust ImageKit] resolve request", {
            source,
            tokens,
            url,
        })

        const response = await fetch("/api/imagekit/resolve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                url,
                source,
                preset: tokens.join(","),
                maxAttempts: 12,
                retryDelayMs: 5000,
            }),
        })
        const data = await response.json().catch(() => ({}))

        console.log("[Adjust ImageKit] resolve response", {
            source,
            tokens,
            ok: response.ok,
            status: response.status,
            data,
            url,
        })

        if (!response.ok || !data?.success) {
            throw new Error(data?.error || "ImageKit transform did not finish")
        }

        return data.url || url
    }

    const setLook = (lookId) => {
        const look = LOOKS.find((item) => item.id === lookId)
        const next = normalizeStoredValues({
            ...latestRef.current,
            ...LOOK_PRESET_KEYS.reduce((acc, key) => {
                acc[key] = DEFAULT_VALUES[key]
                return acc
            }, {}),
            ...(look?.preset || {}),
            look: lookId,
        })
        applyNextValues(next, { commit: true, updateState: true })
    }

    const reset = () => {
        const next = { ...DEFAULT_VALUES }
        latestRef.current = next
        pendingPreviewRef.current = null
        if (previewFrame.current) {
            cancelAnimationFrame(previewFrame.current)
            previewFrame.current = null
        }
        setValues(next)
        const nextSig = getValuesSignature(next)
        applyAdjustmentFilters(canvasEditor, next, sigRef, { commit: nextSig !== committedSigRef.current })
        queueCurveHistogramRefresh()
        committedSigRef.current = nextSig
        isInteractingRef.current = false
    }

    const applyImageKitTransforms = async () => {
        const img = getSelectedImage(canvasEditor)
        if (!img) {
            toast.error("Select an image first")
            return
        }

        const sourceUrl = getImageSrc(img)
        if (!isImageKitUrl(sourceUrl)) {
            toast.error("ImageKit URL transforms need an ImageKit-hosted image")
            return
        }

        const tokens = buildImageKitTokens(imageKitValues)
        if (!tokens.length) {
            toast.info("Choose at least one ImageKit transform")
            return
        }

        let baseUrl = img._phosmithImageKitAdjustBaseSrc || img.phosmithImageKitAdjustBaseSrc || sourceUrl

        // AI transforms (e-retouch, e-upscale) must be separate chained steps,
        // not comma-joined in a single step. Split them out.
        const aiTokens = tokens.filter(isAiTransform)
        const regularTokens = tokens.filter((t) => !isAiTransform(t))

        // If AI transforms are present and the image belongs to a different
        // ImageKit account, re-upload it so units are charged correctly.
        if (aiTokens.length > 0) {
            try {
                baseUrl = await ensureCurrentImageKitEndpoint(baseUrl, {
                    onStatus: (msg) => setProcessingMessage?.(msg),
                })
            } catch (reuploadErr) {
                console.warn('[Adjust ImageKit] Re-upload failed:', reuploadErr)
                toast.error('Failed to re-upload image to current account: ' + (reuploadErr?.message || ''))
                return
            }
        }

        img._phosmithImageKitAdjustBaseSrc = baseUrl
        img.phosmithImageKitAdjustBaseSrc = baseUrl
        img.phosmithImageKitAdjustValues = imageKitValues

        // Build: existing-transforms : regular-tokens : ai-token-1 : ai-token-2
        const base = normalizeImageKitUrl(baseUrl)
        const allSteps = []
        // Regular transforms as one comma-joined step
        if (regularTokens.length) allSteps.push(regularTokens.join(','))
        // Each AI transform as its own step
        aiTokens.forEach((t) => allSteps.push(t))

        const nextUrl = buildImageKitChainedTransformUrl(base, allSteps)

        console.log("[Adjust ImageKit] apply start", {
            sourceUrl,
            baseUrl,
            base,
            tokens,
            regularTokens,
            aiTokens,
            allSteps,
            nextUrl,
            image: {
                width: img.width,
                height: img.height,
                scaledWidth: img.getScaledWidth?.(),
                scaledHeight: img.getScaledHeight?.(),
            },
        })

        setIsApplyingImageKit(true)
        setProcessingMessage?.("Applying ImageKit URL transforms...")
        try {
            let readyUrl = nextUrl

            // Only poll when AI transforms are present — they're async.
            // Standard transforms (e-contrast, e-sharpen, etc.) are instant
            // and don't need polling. Polling standard URLs can fail due to CORS.
            if (aiTokens.length > 0) {
                setProcessingMessage?.("Waiting for ImageKit AI processing...")
                readyUrl = await resolveImageKitUrl(nextUrl, {
                    source: "adjust-imagekit-apply",
                    tokens,
                })
            }

            await img.setSrc(readyUrl, { crossOrigin: "anonymous" })
            img.set("dirty", true)
            img.setCoords()
            canvasEditor.requestRenderAll()
            canvasEditor.fire("object:modified", { target: img })
            queueCurveHistogramRefresh(img)
            console.log("[Adjust ImageKit] apply complete", {
                readyUrl,
                width: img.width,
                height: img.height,
                scaledWidth: img.getScaledWidth?.(),
                scaledHeight: img.getScaledHeight?.(),
            })
            toast.success("ImageKit transforms applied")
        } catch (error) {
            console.warn("ImageKit adjustment failed:", error)
            // Provide a user-friendly message for extension quota errors
            const msg = error?.message || ''
            if (/extension.?limit|limit.?exceeded|units.?exhausted/i.test(msg)) {
                toast.error('ImageKit AI extension units exhausted. Only free transforms (contrast, sharpen) are available this month.')
            } else {
                toast.error(msg || "ImageKit transform failed")
            }
        } finally {
            setIsApplyingImageKit(false)
            setProcessingMessage?.(null)
        }
    }

    if (!canvasEditor) {
        return (
            <div className="p-4">
                <p className="text-xs" style={{ color: "var(--text-muted)" }}>Load an image to adjust</p>
            </div>
        )
    }

    return (
        <div className="adjust-panel">
            <div className="adjust-panel-header">
                <div className="flex items-center gap-2">
                    <SlidersHorizontal className="h-4 w-4" />
                    <span>Professional Adjust</span>
                </div>
                <button type="button" onClick={reset} className="adjust-reset-button">
                    <RotateCcw className="h-3.5 w-3.5" /> Reset
                </button>
            </div>

            <div className="adjust-tabs" role="tablist" aria-label="Adjustment groups">
                {FILTER_GROUPS.map((group) => (
                    <button
                        key={group}
                        type="button"
                        onClick={() => setActiveTab(group)}
                        className={`adjust-tab ${activeTab === group ? "is-active" : ""}`}
                        data-active={activeTab === group}
                        role="tab"
                        aria-selected={activeTab === group}
                    >
                        {group}
                    </button>
                ))}
            </div>

            {activeTab === "Curves" ? (
                <div className="adjust-control-list">
                    <CurveEditorPanel
                        values={values}
                        histogram={curveHistogram}
                        activeChannel={activeCurveChannel}
                        onChannelChange={setActiveCurveChannel}
                        onBegin={handleBeginChange}
                        onPreview={handlePreviewChange}
                        onCommit={handleCommitChange}
                    />
                </div>
            ) : activeTab === "Wheels" ? (
                <div className="adjust-control-list">
                    <div className="adjust-look-strip">
                        {LOOKS.map((look) => (
                            <button
                                key={look.id}
                                type="button"
                                onClick={() => setLook(look.id)}
                                className={`adjust-look-button ${values.look === look.id ? "is-active" : ""}`}
                            >
                                {look.label}
                            </button>
                        ))}
                    </div>
                    {WHEEL_CONFIGS.map((config) => (
                        <ColorWheelCard
                            key={config.key}
                            config={config}
                            values={values}
                            onColor={handleColorChange}
                            onAmount={handleWheelAmount}
                        />
                    ))}
                </div>
            ) : activeTab === "ImageKit" ? (
                <div className="adjust-control-list">
                    <ImageKitPanel
                        imageKitValues={imageKitValues}
                        setImageKitValues={setImageKitValues}
                        onApply={applyImageKitTransforms}
                        isApplying={isApplyingImageKit}
                    />
                </div>
            ) : (
                <div className="adjust-control-list">
                    {SLIDER_CONFIGS.filter((cfg) => cfg.group === activeTab).map((cfg) => (
                        <ProRulerSlider
                            key={cfg.key}
                            variant="instrument"
                            value={values[cfg.key]}
                            onBegin={handleBeginChange}
                            onPreview={(v) => handlePreviewChange(cfg.key, v)}
                            onCommit={(v) => handleCommitChange(cfg.key, v)}
                            min={cfg.min}
                            max={cfg.max}
                            step={cfg.step}
                            defaultValue={cfg.defaultValue}
                            label={cfg.label}
                            suffix={cfg.suffix ?? ""}
                            visual={FILTER_VISUAL[cfg.key] || FILTER_VISUAL.brightness}
                        />
                    ))}
                </div>
            )}
        </div>
    )
}

export default AdjustControls
