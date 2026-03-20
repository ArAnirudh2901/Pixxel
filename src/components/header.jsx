"use client"

import Image from 'next/image'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import React from 'react'

const Header = () => {

    const pathname = usePathname()

    return (
        <header className='fixed top-6 left-1/2 transform -translate-x-1/2 z-50 text-nowrap'>
            <div className='backdrop-blur-md bg-white/10 border border-white/20 rounded-full px-8 py-3 flex items-center justify-between gap-8'>
                <Link href="/" className='mr-10 md:mr-20 flex items-center gap-2'>
                    <Image src='/Logo.png' alt='Pixxel Logo' width={32} height={32} className="w-8 h-8 object-contain" />
                    <span className="text-xl font-bold text-white tracking-wide">Pixxel</span>
                </Link>
                {pathname === '/' && <div className='hidden md:flex space-x-6'>
                    <Link
                        href='#features'
                        className="text-white font-medium transition-all duration-300 hover:text-cyan-400 cursor-pointer"
                    >
                        Features
                    </Link>

                    <Link
                        href='#pricing'
                        className="text-white font-medium transition-all duration-300 hover:text-cyan-400 cursor-pointer"
                    >
                        Pricing
                    </Link>

                    <Link
                        href='#contact'
                        className="text-white font-medium transition-all duration-300 hover:text-cyan-400 cursor-pointer"
                    >
                        Contact
                    </Link>
                </div>}

                <div className='flex items-center gap-3 ml-10 md:ml-20'>
                    Auth
                </div>
            </div>
        </header>
    )
}

export default Header