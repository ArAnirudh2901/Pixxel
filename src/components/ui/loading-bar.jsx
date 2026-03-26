"use client"

import React from 'react'
import { cn } from '@/lib/utils'

const LoadingBar = ({
    isVisible = true,
    barKey,
    className,
    trackClassName,
    barClassName,
    barWidthClassName = 'w-1/2',
}) => {
    return (
        <div
            aria-hidden="true"
            className={cn(
                'pointer-events-none transition-opacity duration-200',
                isVisible ? 'opacity-100' : 'opacity-0',
                className
            )}
        >
            <div className={cn('relative h-[2px] overflow-hidden rounded-full bg-white/14', trackClassName)}>
                {isVisible && (
                    <span
                        key={barKey}
                        className={cn(
                            'animate-loading-bar-sweep absolute inset-y-0 left-0 block rounded-full bg-[linear-gradient(90deg,rgba(34,211,238,0),rgba(103,232,249,0.95),rgba(255,255,255,0.98),rgba(34,211,238,0))] shadow-[0_0_16px_rgba(103,232,249,0.55)] transform-gpu will-change-transform',
                            barWidthClassName,
                            barClassName
                        )}
                    />
                )}
            </div>
        </div>
    )
}

export default LoadingBar
