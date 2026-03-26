"use client"

import React, { useState } from 'react'
import useIntersectionObserver from '../../hooks/useIntersectionObserver'

const FeatureCard = ({ icon, title, description, delay = 0 }) => {
    const [ref, isVisible] = useIntersectionObserver()
    const [isHovered, setIsHovered] = useState(false)

    return (
        <div
            ref={ref}
            className={`transition-all duration-700 ease-out ${isVisible ? 'translate-y-0 opacity-100' : 'translate-y-6 opacity-0'}`}
            style={{ transitionDelay: `${delay}ms` }}
        >
            <div
                className={`h-full min-h-[208px] rounded-[22px] border border-white/10 bg-white/[0.045] p-5 shadow-[0_14px_34px_rgba(7,12,26,0.18)] backdrop-blur-xl transition-all duration-300 ${isHovered ? 'border-cyan-300/25 bg-white/[0.075] -translate-y-1 shadow-[0_18px_40px_rgba(8,20,42,0.24)]' : ''}`}
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
            >
                <div className='mb-4 flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-white/8 text-xl shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]'>
                    {icon}
                </div>
                <h3 className='mb-3 text-[1.28rem] font-semibold tracking-tight text-white'>
                    {title}
                </h3>
                <p className='text-[0.97rem] leading-6 text-slate-300'>
                    {description}
                </p>
            </div>
        </div>
    )
}

export default FeatureCard
