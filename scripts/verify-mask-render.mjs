#!/usr/bin/env bun
/**
 * Headless-browser verification of the phosmith EDITOR megashader render path.
 * Bundles scripts/mask-render-harness/entry.js and drives it with Playwright
 * Chromium — proves applyMegashaderFilter() renders a LOCALISED effect on a
 * real Fabric canvas through Fabric's Canvas2D filter backend.
 *
 * Usage: bun scripts/verify-mask-render.mjs
 */
import { createServer } from 'node:http'
import { readFile, mkdir, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import path from 'node:path'

const ROOT = path.resolve(import.meta.dir, '..')
const HARNESS_DIR = path.join(ROOT, '.cache', 'mask-render-harness')
const PROFILE_DIR = path.join(ROOT, '.cache', 'playwright-client-ai')
const log = (m) => console.log(`[verify-mask-render] ${m}`)
const die = (m) => { console.error(`[verify-mask-render] ✗ ${m}`); process.exit(1) }
const skip = (m) => { log(`skip — ${m}`); process.exit(0) }

let chromium
try { ({ chromium } = await import('playwright')) }
catch { skip('playwright not installed') }

await mkdir(HARNESS_DIR, { recursive: true })
const build = Bun.spawnSync([
    'bun', 'build', path.join(ROOT, 'scripts/mask-render-harness/entry.js'),
    '--outdir', HARNESS_DIR, '--target=browser', '--splitting',
], { cwd: ROOT })
if (build.exitCode !== 0) die(`bundle failed:\n${build.stderr?.toString().slice(0, 2000)}`)
await writeFile(path.join(HARNESS_DIR, 'index.html'),
    '<!doctype html><meta charset="utf-8"><title>mask-render</title><script type="module" src="./entry.js"></script>')
log('harness bundled')

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm', '.map': 'application/json' }
const server = createServer(async (req, res) => {
    try {
        const url = new URL(req.url, 'http://localhost')
        const rel = url.pathname === '/' ? '/index.html' : url.pathname
        const file = path.join(HARNESS_DIR, path.normalize(rel))
        if (!file.startsWith(HARNESS_DIR) || !existsSync(file)) { res.writeHead(404).end('nf'); return }
        res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' })
        res.end(await readFile(file))
    } catch (e) { res.writeHead(500).end(String(e?.message || e)) }
})
await new Promise((r) => server.listen(0, '127.0.0.1', r))
const port = server.address().port
log(`serving on http://127.0.0.1:${port}`)

let context = null, failed = false
try {
    context = await chromium.launchPersistentContext(PROFILE_DIR, { headless: true, args: ['--enable-unsafe-webgpu', '--enable-gpu'] })
} catch (err) {
    server.close()
    if (/executable doesn't exist|browser is not installed/i.test(err?.message || '')) skip('Chromium not installed')
    die(`launch failed: ${err?.message}`)
}
try {
    const page = await context.newPage()
    page.on('console', (m) => { const t = m.text(); if (t.startsWith('[harness]') || t.includes('[megashader]') || m.type() === 'error' || m.type() === 'warning') log(`browser(${m.type()}): ${t.slice(0, 400)}`) })
    page.on('pageerror', (e) => log(`pageerror: ${String(e).slice(0, 300)}`))
    await page.goto(`http://127.0.0.1:${port}/`)
    await page.waitForFunction(() => window.__harnessReady === true, null, { timeout: 30_000 })
    const report = await page.evaluate(() => window.__maskRender.run())
    for (const c of report.checks) {
        console.log(`[verify-mask-render] ${c.ok ? 'ok' : '✗'} ${c.label} — ${c.detail}`)
        if (!c.ok) failed = true
    }
} catch (err) {
    console.error(`[verify-mask-render] ✗ ${err?.message || err}`)
    failed = true
} finally {
    await context.close().catch(() => {})
    server.close()
}
if (failed) { console.error('\n[verify-mask-render] ✗ editor render path FAILED'); process.exit(1) }
console.log('\n[verify-mask-render] ✓ editor megashader render path verified')
