import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server'
import { NextResponse } from 'next/server'

const isProtectedRoutes = createRouteMatcher(["/dashboard(.*)", "/editor(.*)"])

export default clerkMiddleware(async (auth, req) => {
    const { userId } = await auth()

    if (!userId && isProtectedRoutes(req)) {
        const { redirectToSignIn } = await auth()

        return redirectToSignIn()
    }

    return NextResponse.next()
})

export const config = {
    matcher: [
        '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
        '/(api|trpc)(.*)',
    ],
}