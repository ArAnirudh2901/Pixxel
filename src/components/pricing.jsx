"use client"

import { PricingTable } from '@clerk/nextjs'
import React from 'react'
import useIntersectionObserver from '../../hooks/useIntersectionObserver'

const Pricing = () => {
    const [headingRef, headingVisible] = useIntersectionObserver()
    const [tableRef, tableVisible] = useIntersectionObserver()

    return (
        <section className='mx-auto max-w-7xl px-6 py-30' id='pricing'>
            <div
                ref={headingRef}
                className={`mx-auto mb-16 max-w-4xl text-center transition-all duration-700 ease-out ${headingVisible ? 'translate-y-0 opacity-100' : 'translate-y-6 opacity-0'}`}
            >
                <h2 className='text-5xl font-black tracking-tight sm:text-6xl'>
                    <span className='text-white'>
                        Simple{' '}
                    </span>
                    <span className='bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent'>
                        Pricing
                    </span>
                </h2>
                <p className='text-xl text-gray-300 max-w-3xl mx-auto'>
                    Start free and upgrade when you need more power. No hidden fees, cancel anytime.
                </p>
            </div>
            <div
                ref={tableRef}
                className={`pricing-table-shell mx-auto flex max-w-6xl justify-center transition-all duration-700 ease-out delay-200 ${tableVisible ? 'translate-y-0 opacity-100' : 'translate-y-6 opacity-0'}`}
            >
                <PricingTable
                    appearance={{
                        elements: {
                            card: 'drop-shadow-xl',
                        },
                    }}
                />
            </div>
        </section>
    )
}

export default Pricing
