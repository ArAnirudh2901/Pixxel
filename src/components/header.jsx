"use client"

import { Show, SignInButton, SignUpButton, UserButton } from '@clerk/nextjs'
import Image from 'next/image'
import Link from 'next/link'
import { usePathname, useSearchParams } from 'next/navigation'
import React, { useEffect, useRef, useState } from 'react'
import { Button } from './ui/button'
import { useStoreUser } from '../../hooks/useStoreUser'
import { LayoutDashboard } from 'lucide-react'

const HEADER_INLINE_LOADING_DURATION_MS = 900
const HEADER_ROUTE_LOADING_FALLBACK_MS = 10000

const Header = () => {
    const pathname = usePathname()
    const searchParams = useSearchParams()
    const [loadingState, setLoadingState] = useState({ mode: 'boot', routeKey: null })
    const [loadingBarKey, setLoadingBarKey] = useState(0)
    const loadingTimeoutRef = useRef(null)
    const routeKey = `${pathname}?${searchParams.toString()}`
    const isHeaderLoading =
        loadingState.mode === 'boot' ||
        loadingState.mode === 'inline' ||
        (loadingState.mode === 'route' && loadingState.routeKey === routeKey)

    useStoreUser()

    useEffect(() => {
        const finishInitialLoading = () => {
            setLoadingState((currentState) => {
                if (currentState.mode !== 'boot') {
                    return currentState
                }

                return { mode: 'idle', routeKey: null }
            })
        }

        if (document.readyState === 'complete') {
            window.requestAnimationFrame(finishInitialLoading)
        } else {
            window.addEventListener('load', finishInitialLoading, { once: true })
        }

        return () => {
            window.removeEventListener('load', finishInitialLoading)

            if (loadingTimeoutRef.current) {
                window.clearTimeout(loadingTimeoutRef.current)
            }
        }
    }, [])

    useEffect(() => {
        if (loadingState.mode !== 'route' || loadingState.routeKey === null || loadingState.routeKey === routeKey) {
            return
        }

        if (loadingTimeoutRef.current) {
            window.clearTimeout(loadingTimeoutRef.current)
            loadingTimeoutRef.current = null
        }
    }, [loadingState.mode, loadingState.routeKey, routeKey])

    const triggerHeaderLoading = (event) => {
        if (!(event.target instanceof Element)) {
            return
        }

        const interactiveElement = event.target.closest('a,button,[role="button"]')

        if (!interactiveElement) {
            return
        }

        setLoadingBarKey((currentKey) => currentKey + 1)

        if (loadingTimeoutRef.current) {
            window.clearTimeout(loadingTimeoutRef.current)
        }

        const anchorElement = interactiveElement.closest('a')

        if (anchorElement) {
            const currentUrl = new URL(window.location.href)
            const nextUrl = new URL(anchorElement.href, currentUrl)
            const isRouteNavigation =
                nextUrl.origin !== currentUrl.origin ||
                nextUrl.pathname !== currentUrl.pathname ||
                nextUrl.search !== currentUrl.search

            if (isRouteNavigation) {
                setLoadingState({ mode: 'route', routeKey })
                loadingTimeoutRef.current = window.setTimeout(() => {
                    loadingTimeoutRef.current = null
                    setLoadingState({ mode: 'idle', routeKey: null })
                }, HEADER_ROUTE_LOADING_FALLBACK_MS)
                return
            }
        }

        setLoadingState({ mode: 'inline', routeKey: null })
        loadingTimeoutRef.current = window.setTimeout(() => {
            loadingTimeoutRef.current = null
            setLoadingState({ mode: 'idle', routeKey: null })
        }, HEADER_INLINE_LOADING_DURATION_MS)
    }

    if (pathname.includes("/editor")) {
        return null;        // Hide the header on the editor page
    }
    return (
        <header className='fixed top-6 left-1/2 transform -translate-x-1/2 z-50 text-nowrap'>
            <div
                onClickCapture={triggerHeaderLoading}
                className='relative overflow-hidden backdrop-blur-md bg-white/10 border border-white/20 rounded-full px-8 py-3 flex items-center justify-between gap-8'
            >
                <Link href="/" className='mr-10 md:mr-20 flex items-center gap-2'>
                    <Image src='/Logo.png' alt='Pixxel Logo' width={32} height={32} className="w-8 h-8 object-contain" />
                    <span className="text-xl font-bold text-white tracking-wide">Pixxel</span>
                </Link>
                {pathname === '/' && <div className='hidden md:flex space-x-6'>
                    <a
                        href='#features'
                        className="text-white font-medium transition-all duration-300 hover:text-cyan-400 cursor-pointer"
                    >
                        Features
                    </a>

                    <a
                        href='#pricing'
                        className="text-white font-medium transition-all duration-300 hover:text-cyan-400 cursor-pointer"
                    >
                        Pricing
                    </a>

                    <a
                        href='#contact'
                        className="text-white font-medium transition-all duration-300 hover:text-cyan-400 cursor-pointer"
                    >
                        Contact
                    </a>

                    <a
                        href='#dashboard'
                        className="text-white font-medium transition-all duration-300 hover:text-cyan-400 cursor-pointer"
                    >
                        Dashboard
                    </a>
                </div>}

                <Show when="signed-out">
                    <div className='ml-10 flex min-h-11 min-w-[9rem] shrink-0 items-center justify-end gap-3 md:ml-20 sm:min-w-[15.5rem]'>
                        <SignInButton >
                            <Button variant='glass' className='hidden h-11 px-5 text-sm font-semibold sm:flex'>
                                Sign In
                            </Button>
                        </SignInButton>
                        <SignUpButton>
                            <Button variant='primary' className='h-11 px-5 text-sm font-semibold'>
                                Get Started
                            </Button>
                        </SignUpButton>
                    </div>
                </Show>
                <Show when="signed-in">
                    <Link href="/dashboard">
                        <Button variant='glass' className="hidden sm:flex">
                            <LayoutDashboard className="h-4 w-4" />
                            <span className='hidden md:flex'>Dashboard</span>
                        </Button>
                    </Link>

                    <div className='ml-4 flex min-h-11 min-w-11 shrink-0 items-center justify-end md:ml-8'>
                        <UserButton userProfileMode="modal" />
                    </div>
                </Show>

                <div
                    aria-hidden="true"
                    className={`pointer-events-none absolute inset-x-4 bottom-0 h-[2px] rounded-full bg-white/14 transition-opacity duration-200 ${isHeaderLoading ? 'opacity-100' : 'opacity-0'}`}
                >
                    {isHeaderLoading && (
                        <span
                            key={loadingBarKey}
                            className='animate-header-loading-sweep absolute inset-y-0 left-0 w-1/2 rounded-full bg-[linear-gradient(90deg,rgba(34,211,238,0),rgba(103,232,249,0.95),rgba(255,255,255,0.98),rgba(34,211,238,0))] shadow-[0_0_16px_rgba(103,232,249,0.55)]'
                        />
                    )}
                </div>
            </div>
        </header>
    )
}

export default Header
