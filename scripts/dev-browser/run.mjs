#!/usr/bin/env node
/**
 * run.mjs — execute a Playwright snippet against the real Chrome started by
 * launch-chrome.mjs (CDP attach; never launches or closes the browser).
 *
 * Snippet contract: an .mjs module whose default export is
 *   async ({ browser, context, page, screenshotDir }) => { ... }
 * `page` is the most recently opened tab (or a fresh one if none). Whatever
 * the snippet returns is JSON-printed so callers can assert on it.
 *
 * Usage: bun scripts/dev-browser/run.mjs <snippet.mjs> [...snippet args]
 *        (extra args are available to the snippet via process.argv)
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import path from 'node:path'
import { pathToFileURL, fileURLToPath } from 'node:url'

const CDP_PORT = Number(process.env.DEV_BROWSER_CDP_PORT || 9222)
const HERE = path.dirname(fileURLToPath(import.meta.url))
const SCREENSHOT_DIR = path.join(HERE, 'screenshots')

const snippetPath = process.argv[2]
if (!snippetPath) {
    console.error('usage: bun scripts/dev-browser/run.mjs <snippet.mjs> [...args]')
    process.exit(2)
}

mkdirSync(SCREENSHOT_DIR, { recursive: true })

let browser
try {
    browser = await chromium.connectOverCDP(`http://127.0.0.1:${CDP_PORT}`, { timeout: 20_000 })
} catch (err) {
    console.error(
        `dev-browser: no Chrome on CDP port ${CDP_PORT}. Start it first:\n` +
        '  bun scripts/dev-browser/launch-chrome.mjs\n' +
        `(${err?.message || err})`,
    )
    process.exit(1)
}

try {
    // connectOverCDP exposes the real browser's existing (persistent) context
    // — this is where the user's Clerk session cookies live.
    const context = browser.contexts()[0] || (await browser.newContext())
    const pages = context.pages()
    const page = pages.length > 0 ? pages[pages.length - 1] : await context.newPage()

    const mod = await import(pathToFileURL(path.resolve(snippetPath)).href)
    if (typeof mod.default !== 'function') {
        throw new Error(`${snippetPath} must default-export an async function`)
    }
    const result = await mod.default({ browser, context, page, screenshotDir: SCREENSHOT_DIR })
    if (result !== undefined) console.log(JSON.stringify(result, null, 2))
} catch (err) {
    console.error('dev-browser: snippet failed:', err)
    process.exitCode = 1
} finally {
    // Disconnect only — the CDP-attached browser (and the user's login
    // session) must survive across runs.
    await browser.close().catch(() => {})
}
