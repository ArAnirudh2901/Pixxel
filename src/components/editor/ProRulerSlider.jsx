"use client"

import React, { useCallback, useEffect, useRef, useState } from "react"
import { RotateCcw } from "lucide-react"

const TICK_COUNT = 51

const clamp = (value, min, max) => Math.max(min, Math.min(max, value))

const RulerTicks = React.memo(function RulerTicks() {
  return (
    <div className="pro-ruler-tick-band" aria-hidden="true">
      <svg className="pro-ruler-tick-svg" viewBox="0 0 400 48" preserveAspectRatio="none">
        {Array.from({ length: TICK_COUNT }, (_, index) => {
          const isMajor = index % 10 === 0
          const isMid = index % 5 === 0 && !isMajor
          const x = (index / (TICK_COUNT - 1)) * 400
          const h = isMajor ? 15 : isMid ? 9 : 4
          const opacity = isMajor ? 0.34 : isMid ? 0.22 : 0.12
          return (
            <line
              key={index}
              x1={x}
              y1={0}
              x2={x}
              y2={h}
              stroke="white"
              strokeWidth={isMajor ? 1 : 0.75}
              opacity={opacity}
            />
          )
        })}
      </svg>
    </div>
  )
})

/**
 * Pro instrument slider — Amount/Temperature (instrument) or Tone tab (studio).
 */
export function ProRulerSlider({
  value,
  min = 0,
  max = 100,
  step = 1,
  label,
  suffix = "",
  disabled = false,
  variant = "instrument",
  visual = {},
  defaultValue,
  onBegin,
  onPreview,
  onCommit,
  onChange,
  className = "",
}) {
  const isStudio = variant === "studio"
  const rootRef = useRef(null)
  const trackRef = useRef(null)
  const thumbRef = useRef(null)
  const valueRef = useRef(null)
  const localRef = useRef(value)
  const draggingRef = useRef(false)
  const previewRafRef = useRef(null)
  const trackRectRef = useRef(null)
  const detachDragRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)

  const fill = visual.fill ?? "rgba(35, 58, 92, 0.52)"
  const accent = visual.accent ?? "#52b4ff"
  const trackBg = visual.trackBg ?? "rgba(20, 24, 32, 0.98)"
  const rail = visual.rail ?? null
  const bottomAccent = visual.bottomAccent ?? null

  const snapValue = useCallback(
    (raw) => clamp(Math.round(raw / step) * step, min, max),
    [min, max, step]
  )

  const valueToPct = useCallback(
    (v) => clamp(((v - min) / (max - min)) * 100, 0, 100),
    [min, max]
  )

  const setPct = useCallback((p) => {
    const pctStr = `${p}%`
    rootRef.current?.style.setProperty("--pro-ruler-pct", pctStr)
    if (thumbRef.current) {
      thumbRef.current.style.left = pctStr
    }
  }, [])

  const applyVisual = useCallback(
    (v, { ratio } = {}) => {
      const snapped = snapValue(v)
      localRef.current = snapped
      const pct = ratio != null ? ratio * 100 : valueToPct(snapped)
      setPct(pct)
      if (valueRef.current) {
        valueRef.current.textContent = `${snapped}${suffix}`
      }
      return snapped
    },
    [snapValue, suffix, setPct, valueToPct]
  )

  useEffect(() => {
    if (draggingRef.current) return
    applyVisual(value)
  }, [value, applyVisual])

  const ratioFromClientX = useCallback((clientX) => {
    const rect = trackRectRef.current ?? trackRef.current?.getBoundingClientRect()
    if (!rect || rect.width <= 0) return valueToPct(localRef.current) / 100
    trackRectRef.current = rect
    const x = clamp(clientX - rect.left, 0, rect.width)
    return x / rect.width
  }, [valueToPct])

  const schedulePreview = useCallback(
    (snapped) => {
      const handler = onPreview ?? onChange
      if (!handler) return
      if (previewRafRef.current) return
      previewRafRef.current = requestAnimationFrame(() => {
        previewRafRef.current = null
        handler(snapped)
      })
    },
    [onChange, onPreview]
  )

  const moveDrag = useCallback(
    (clientX) => {
      if (!draggingRef.current) return
      const ratio = ratioFromClientX(clientX)
      const raw = min + ratio * (max - min)
      const snapped = applyVisual(raw, { ratio })
      schedulePreview(snapped)
    },
    [applyVisual, max, min, ratioFromClientX, schedulePreview]
  )

  const endDrag = useCallback(() => {
    if (!draggingRef.current) return
    draggingRef.current = false
    trackRectRef.current = null
    setIsDragging(false)
    detachDragRef.current?.()
    detachDragRef.current = null
    if (previewRafRef.current) {
      cancelAnimationFrame(previewRafRef.current)
      previewRafRef.current = null
    }
    const v = localRef.current
    applyVisual(v)
    onCommit?.(v)
    onChange?.(v)
  }, [applyVisual, onChange, onCommit])

  const attachDragListeners = useCallback(() => {
    detachDragRef.current?.()
    const onMove = (e) => moveDrag(e.clientX)
    const onUp = () => endDrag()
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
    window.addEventListener("pointercancel", onUp)
    detachDragRef.current = () => {
      window.removeEventListener("pointermove", onMove)
      window.removeEventListener("pointerup", onUp)
      window.removeEventListener("pointercancel", onUp)
    }
  }, [endDrag, moveDrag])

  const handlePointerDown = useCallback(
    (e) => {
      if (disabled) return
      e.preventDefault()
      e.stopPropagation()
      trackRectRef.current = trackRef.current?.getBoundingClientRect() ?? null
      draggingRef.current = true
      setIsDragging(true)
      onBegin?.()
      attachDragListeners()
      moveDrag(e.clientX)
      try {
        trackRef.current?.setPointerCapture(e.pointerId)
      } catch {
        /* ignore */
      }
    },
    [attachDragListeners, disabled, moveDrag, onBegin]
  )

  useEffect(
    () => () => {
      detachDragRef.current?.()
      if (previewRafRef.current) cancelAnimationFrame(previewRafRef.current)
    },
    []
  )

  const handleKeyDown = (e) => {
    if (disabled) return
    const mult = e.shiftKey ? 10 : 1
    let next = null
    if (e.key === "ArrowRight" || e.key === "ArrowUp") next = localRef.current + step * mult
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") next = localRef.current - step * mult
    if (e.key === "PageUp") next = localRef.current + step * 10
    if (e.key === "PageDown") next = localRef.current - step * 10
    if (e.key === "Home") next = min
    if (e.key === "End") next = max
    if (next === null) return
    e.preventDefault()
    const v = snapValue(next)
    onBegin?.()
    applyVisual(v)
    onCommit?.(v)
    onChange?.(v)
  }

  // Reset this single slider to its default. Reused by double-click-to-reset on
  // the track and the explicit reset affordance shown when the value is off-default.
  const hasDefault = defaultValue != null
  const isOffDefault = hasDefault && Number(value) !== Number(defaultValue)
  const handleReset = useCallback(
    (e) => {
      e?.preventDefault?.()
      e?.stopPropagation?.()
      if (disabled || !hasDefault) return
      const v = snapValue(defaultValue)
      onBegin?.()
      applyVisual(v)
      onCommit?.(v)
      onChange?.(v)
    },
    [applyVisual, defaultValue, disabled, hasDefault, onBegin, onChange, onCommit, snapValue]
  )

  const resetButton = isOffDefault ? (
    <button
      type="button"
      className="pro-ruler-reset"
      // Guard against the track's pointerdown starting a drag on the same press.
      onPointerDown={(e) => e.stopPropagation()}
      onClick={handleReset}
      tabIndex={disabled ? -1 : 0}
      aria-label={`Reset ${label}`}
      title={`Reset ${label}`}
    >
      <RotateCcw aria-hidden="true" />
    </button>
  ) : null

  const trackStyle = {
    "--pro-ruler-fill": fill,
    "--pro-ruler-accent": accent,
    "--pro-ruler-track": trackBg,
    "--pro-ruler-rail": rail || fill,
    "--pro-ruler-pct": `${valueToPct(value)}%`,
  }

  const trackDecor = (
    <>
      {rail && <div className="pro-ruler-rail" aria-hidden="true" />}
      <div className="pro-ruler-fill" aria-hidden="true" />
      <RulerTicks />
      <div ref={thumbRef} className="pro-ruler-thumb" aria-hidden="true" />
      <span className={`pro-ruler-label ${isStudio ? "pro-ruler-label--center" : ""}`}>{label}</span>
      {!isStudio && (
        <span ref={valueRef} className="pro-ruler-value">
          {value}
          {suffix}
        </span>
      )}
      {bottomAccent && (
        <div className="pro-ruler-bottom-accent" style={{ background: bottomAccent }} aria-hidden="true" />
      )}
    </>
  )

  const sliderClass = `pro-ruler-slider ${isStudio ? "pro-ruler-slider--studio " : ""}${isDragging ? "pro-ruler-slider--dragging " : ""}${disabled ? "pro-ruler-slider--disabled " : ""}${className}`.trim()

  const trackProps = {
    ref: trackRef,
    className: `${isStudio ? "pro-ruler-track" : "pro-ruler-bar pro-ruler-bar--instrument"} pro-ruler-interactive`,
    role: "slider",
    tabIndex: disabled ? -1 : 0,
    "aria-label": label,
    "aria-valuemin": min,
    "aria-valuemax": max,
    "aria-valuenow": value,
    "aria-disabled": disabled,
    onKeyDown: handleKeyDown,
    onPointerDown: handlePointerDown,
    onDoubleClick: hasDefault ? handleReset : undefined,
  }

  if (isStudio) {
    return (
      <div ref={rootRef} className={sliderClass} style={trackStyle}>
        <div className="pro-ruler-bar pro-ruler-bar--studio">
          <div {...trackProps}>{trackDecor}</div>
          <div className="pro-ruler-value-box" aria-hidden="true">
            <span ref={valueRef}>
              {value}
              {suffix}
            </span>
          </div>
          {resetButton}
        </div>
      </div>
    )
  }

  return (
    <div ref={rootRef} className={sliderClass} style={trackStyle}>
      <div {...trackProps}>{trackDecor}</div>
      {resetButton}
    </div>
  )
}

export default ProRulerSlider
