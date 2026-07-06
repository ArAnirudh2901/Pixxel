#!/usr/bin/env bun
/**
 * Headless-browser verification of the ON-DEVICE AI engine.
 *
 * The in-browser model path (src/lib/client-ai.js — CLIPSeg grounding +
 * Depth Anything via transformers.js) cannot run under bun/node, so the
 * other verify suites can only pin its pure logic. This script closes that
 * gap end-to-end:
 *
 *   1. bundles the REAL production module into a standalone page
 *      (scripts/client-ai-harness/entry.js → bun build, no Next/Clerk/env);
 *   2. serves it locally and drives it with Playwright Chromium;
 *   3. runs runClientAISelfTest() in the browser — the actual models infer
 *      on a synthetic scene with a known answer — and asserts every check.
 *
 * Caching: a persistent Chromium profile under .cache/playwright-client-ai
 * keeps the downloaded model files (~200 MB) across runs, so only the first
 * run pays the download.
 *
 * Skips (exit 0, like the other service-dependent verifies) when Playwright
 * or its Chromium build is not installed:
 *   bun add -d playwright && bunx playwright install chromium
 *
 * Usage: bun scripts/verify-client-ai.mjs
 *   HARNESS_TIMEOUT_MS=900000  override the self-test budget (default 15 min)
 */

import { createServer } from 'node:http'
import { readFile, mkdir, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import path from 'node:path'

const ROOT = path.resolve(import.meta.dir, '..')
const HARNESS_DIR = path.join(ROOT, '.cache', 'client-ai-harness')
const PROFILE_DIR = path.join(ROOT, '.cache', 'playwright-client-ai')
const TIMEOUT_MS = Number(process.env.HARNESS_TIMEOUT_MS || 15 * 60 * 1000)

const log = (msg) => console.log(`[verify-client-ai] ${msg}`)
const die = (msg) => { console.error(`[verify-client-ai] ✗ ${msg}`); process.exit(1) }
const skip = (msg) => { log(`skip — ${msg}`); process.exit(0) }

/* ─── 1. Playwright availability ────────────────────────────────────────── */
let chromium
try {
  ({ chromium } = await import('playwright'))
} catch {
  skip('playwright is not installed (bun add -d playwright && bunx playwright install chromium)')
}

/* ─── 2. Bundle the harness ─────────────────────────────────────────────── */
await mkdir(HARNESS_DIR, { recursive: true })
const build = Bun.spawnSync([
  'bun', 'build', path.join(ROOT, 'scripts/client-ai-harness/entry.js'),
  '--outdir', HARNESS_DIR,
  '--target=browser',
  '--splitting',
], { cwd: ROOT })
if (build.exitCode !== 0) {
  die(`harness bundle failed:\n${build.stderr?.toString().slice(0, 1500)}`)
}
await writeFile(
  path.join(HARNESS_DIR, 'index.html'),
  '<!doctype html><meta charset="utf-8"><title>client-ai harness</title><script type="module" src="./entry.js"></script>',
)
// Serve the version-matched ORT runtime from /ort/ exactly like the Next app
// (public/ort via scripts/setup-ort.mjs) — client-ai.js probes this path and
// pins wasmPaths to it, which is what fixes the "webgpuInit is not a
// function" / "no available backend found" bundler failure.
{
  const { copyFileSync, readdirSync } = await import('node:fs')
  const ortSrc = [
    path.join(ROOT, 'node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist'),
    path.join(ROOT, 'node_modules/onnxruntime-web/dist'),
  ].find((d) => existsSync(d))
  if (ortSrc) {
    await mkdir(path.join(HARNESS_DIR, 'ort'), { recursive: true })
    for (const f of readdirSync(ortSrc)) {
      if (/^ort-wasm-simd-threaded.*\.(mjs|wasm)$/.test(f)) {
        copyFileSync(path.join(ortSrc, f), path.join(HARNESS_DIR, 'ort', f))
      }
    }
  }
}
log('harness bundled')

/* ─── 3. Static server ──────────────────────────────────────────────────── */
const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm',
  '.map': 'application/json',
}
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost')
    const rel = url.pathname === '/' ? '/index.html' : url.pathname
    const file = path.join(HARNESS_DIR, path.normalize(rel))
    if (!file.startsWith(HARNESS_DIR) || !existsSync(file)) {
      res.writeHead(404).end('not found')
      return
    }
    const body = await readFile(file)
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' })
    res.end(body)
  } catch (e) {
    res.writeHead(500).end(String(e?.message || e))
  }
})
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
const port = server.address().port
log(`serving harness on http://127.0.0.1:${port}`)

/* ─── 4. Drive the browser ──────────────────────────────────────────────── */
let context = null
try {
  context = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: true,
    // Ask for WebGPU so grounding/depth get the fast device when the host
    // supports it (the RMBG-1.4 matting engine runs fine on plain WASM).
    // Harmless where unsupported: everything falls back to WASM.
    args: ['--enable-unsafe-webgpu', '--enable-gpu'],
  })
} catch (err) {
  server.close()
  if (/executable doesn't exist|browser is not installed/i.test(err?.message || '')) {
    skip('Chromium not installed for playwright (bunx playwright install chromium)')
  }
  die(`could not launch Chromium: ${err?.message}`)
}

let failed = false
try {
  const page = await context.newPage()
  page.on('console', (msg) => {
    const text = msg.text()
    if (text.startsWith('[selftest]') || text.startsWith('[harness]') || msg.type() === 'error') {
      log(`browser: ${text.slice(0, 200)}`)
    }
  })
  page.on('pageerror', (err) => log(`pageerror: ${String(err).slice(0, 200)}`))

  await page.goto(`http://127.0.0.1:${port}/`)
  await page.waitForFunction(() => window.__harnessReady === true, null, { timeout: 30_000 })
  log('harness ready — running self-test (first run downloads the models)…')

  const report = await Promise.race([
    page.evaluate(() => window.__clientAI.run()),
    new Promise((_, reject) => setTimeout(
      () => reject(new Error(`self-test exceeded ${Math.round(TIMEOUT_MS / 1000)}s`)), TIMEOUT_MS)),
  ])

  log(`device=${report.device} total=${(report.totalMs / 1000).toFixed(1)}s`)
  for (const c of report.checks) {
    const mark = c.ok ? 'ok' : '✗'
    console.log(`[verify-client-ai] ${mark} ${c.label} — ${c.detail}`)
    if (!c.ok) failed = true
  }
  if (report.ground?.error) log(`ground error: ${report.ground.error}`)
} catch (err) {
  console.error(`[verify-client-ai] ✗ ${err?.message || err}`)
  failed = true
} finally {
  await context.close().catch(() => {})
  server.close()
}

if (failed) {
  console.error('\n[verify-client-ai] ✗ on-device AI self-test FAILED')
  process.exit(1)
}
console.log('\n[verify-client-ai] ✓ on-device AI verified in a real browser')
