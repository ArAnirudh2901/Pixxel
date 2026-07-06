#!/usr/bin/env node
/**
 * launch-chrome.mjs — start the user's REAL Google Chrome with a CDP port for
 * dev-browser automation.
 *
 * Why real Chrome (not Playwright's bundled Chromium): Clerk rejects
 * "Chrome for Testing"-class builds as unsupported browsers, so login — and
 * therefore any authenticated E2E flow — is impossible in the bundled build.
 *
 * Why a DEDICATED profile dir (.chrome-dev-profile, gitignored): Chrome ≥136
 * refuses --remote-debugging-port on the default user profile, and we don't
 * want automation touching the user's daily profile anyway. The dedicated
 * profile persists Clerk cookies, so the user logs in ONCE (headful) and every
 * later run reuses the session.
 *
 * Idempotent: if something is already listening on the CDP port, exits 0.
 *
 * Usage: bun scripts/dev-browser/launch-chrome.mjs [startUrl]
 */
import { spawn } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const CDP_PORT = Number(process.env.DEV_BROWSER_CDP_PORT || 9222)
const CDP_URL = `http://127.0.0.1:${CDP_PORT}`
const CHROME_BIN = process.env.DEV_BROWSER_CHROME_BIN
    || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const PROFILE_DIR = path.join(REPO_ROOT, '.chrome-dev-profile')
const START_URL = process.argv[2] || 'http://localhost:3000'

const cdpAlive = async () => {
    try {
        const res = await fetch(`${CDP_URL}/json/version`, { signal: AbortSignal.timeout(1500) })
        return res.ok
    } catch {
        return false
    }
}

if (await cdpAlive()) {
    console.log(`dev-browser: Chrome already listening on ${CDP_URL} — reusing it.`)
    process.exit(0)
}

mkdirSync(PROFILE_DIR, { recursive: true })
const child = spawn(
    CHROME_BIN,
    [
        `--remote-debugging-port=${CDP_PORT}`,
        `--user-data-dir=${PROFILE_DIR}`,
        '--no-first-run',
        '--no-default-browser-check',
        START_URL,
    ],
    { detached: true, stdio: 'ignore' },
)
child.unref()

// Poll until the CDP endpoint answers (Chrome cold start can take a while).
const deadline = Date.now() + 30_000
while (Date.now() < deadline) {
    if (await cdpAlive()) {
        console.log(`dev-browser: real Chrome up on ${CDP_URL} (profile: ${PROFILE_DIR})`)
        console.log('dev-browser: if the app shows a sign-in screen, log in once — the session persists in the profile.')
        process.exit(0)
    }
    await new Promise((r) => setTimeout(r, 300))
}
console.error(`dev-browser: Chrome did not expose ${CDP_URL} within 30s. Is "${CHROME_BIN}" correct?`)
process.exit(1)
