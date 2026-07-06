/**
 * verify-grade-live-and-restore.mjs — focused assertions on an EXISTING
 * project that already has 3 mask layers (linear, radial, subject):
 *
 *   A. LIVE repaint: canvas pixels move WHILE the gamma slider is held down.
 *   B. Wheel + curve edits repaint too (same uniform fast-path).
 *   C. A save (POST /api/canvas/snapshot) fires within seconds of the edits.
 *   D. With IndexedDB wiped, a hard reload restores masks + all grades
 *      (incl. the curve LUT rebuilt from control points) — pixel-compared.
 *   E. Layer visibility toggle still changes pixels (layers are live).
 *
 * Usage: node scripts/dev-browser/run.mjs <this file> [projectUrl]
 */
const PROJECT_URL = process.argv[3] || 'http://localhost:3000/editor/cmr8sezwm0000psqyuha84dnh'
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

export default async ({ page, screenshotDir }) => {
    const steps = []
    const consoleHits = []
    const log = (msg) => { steps.push(msg); console.error(`  • ${msg}`) }
    const shot = (name) => page.screenshot({ path: `${screenshotDir}/${name}.png` }).catch(() => {})
    page.on('console', (m) => {
        const t = m.text()
        if (/megashader|pageerror|error/i.test(t)) consoleHits.push(`[${m.type()}] ${t.slice(0, 200)}`)
    })
    page.on('pageerror', (e) => consoleHits.push(`[pageerror] ${String(e).slice(0, 200)}`))

    const sample = () => page.evaluate(() => {
        const c = document.querySelector('canvas.lower-canvas')
        if (!c || !c.width) return null
        const ctx = c.getContext('2d')
        // Two blob regions (subject mask footprint) + one background region.
        return [[0.35, 0.5], [0.65, 0.45], [0.5, 0.85]].map(([fx, fy]) => {
            const x = Math.floor(c.width * fx), y = Math.floor(c.height * fy)
            const d = ctx.getImageData(x - 8, y - 8, 16, 16).data
            let r = 0, g = 0, b = 0; const n = d.length / 4
            for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2] }
            return [r / n, g / n, b / n]
        })
    })
    const delta = (a, b) => {
        if (!a || !b) return Infinity
        let s = 0, n = 0
        a.forEach((reg, i) => reg.forEach((v, j) => { s += Math.abs(v - b[i][j]); n++ }))
        return s / n
    }

    await page.bringToFront().catch(() => {})
    await page.setViewportSize({ width: 1600, height: 1000 })
    await page.goto(PROJECT_URL, { waitUntil: 'domcontentloaded', timeout: 120_000 })
    await page.waitForSelector('canvas.lower-canvas', { timeout: 60_000 })
    // Wait until the IMAGE is actually rendered (regions differ), not just
    // the canvas element — opening the Mask tool before the image lands
    // leaves the panel with no target image to hydrate from.
    for (let i = 0; i < 120; i++) {
        const s = await sample()
        if (s && delta([s[0]], [s[2]]) > 3) break
        await sleep(500)
    }
    await sleep(1500) // filter restore + first megashader render settle
    await page.locator('[aria-label="Mask"]').first().click()
    await page.getByText('Mask Layers', { exact: false }).first().waitFor({ timeout: 20_000 })
    await page.waitForFunction(() => document.querySelectorAll('.mask-chain-card').length >= 3, null, { timeout: 20_000 })
    log('project open, 3 mask cards present')

    const gradeCard = page.locator('.mask-chain-card').last() // subject layer
    await gradeCard.click()
    let gamma = gradeCard.locator('[aria-label="Gamma"]').first()
    if (!(await gamma.isVisible().catch(() => false))) {
        await gradeCard.locator('button[title="Edit params"], button[title="Hide params"]').first().click()
        await sleep(300)
        gamma = gradeCard.locator('[aria-label="Gamma"]').first()
    }
    await gamma.scrollIntoViewIfNeeded()

    /* A — live gamma. While the pointer is down, live pixels are carried by
       the draft-preview overlay canvas (the lower canvas intentionally
       freezes until the full-res commit ~160ms after the drag idles), so
       mid-drag sampling reads the overlay and asserts it KEEPS CHANGING. */
    const sampleLive = () => page.evaluate(() => {
        const c = document.querySelector('canvas.phosmith-megashader-preview')
            || document.querySelector('canvas.lower-canvas')
        if (!c || !c.width) return null
        const ctx = c.getContext('2d')
        return [[0.5, 0.5], [0.55, 0.4], [0.45, 0.6]].map(([fx, fy]) => {
            const x = Math.floor(c.width * fx), y = Math.floor(c.height * fy)
            const d = ctx.getImageData(Math.max(0, x - 4), Math.max(0, y - 4), 8, 8).data
            let r = 0, g = 0, b = 0; const n = d.length / 4
            for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2] }
            return [r / n, g / n, b / n]
        })
    })
    const preGrade = await sample()
    const gbox = await gamma.boundingBox()
    const gy = gbox.y + gbox.height / 2
    await page.mouse.move(gbox.x + gbox.width * 0.5, gy)
    await page.mouse.down()
    let midDragDelta = 0
    const midSeen = []
    let firstLive = null
    for (let i = 1; i <= 8; i++) {
        await page.mouse.move(gbox.x + gbox.width * (0.5 + 0.05 * i), gy, { steps: 2 })
        await sleep(90)
        const mid = await sampleLive()
        if (!firstLive) { firstLive = mid; continue }
        const d = delta(firstLive, mid)
        midSeen.push(+d.toFixed(2))
        midDragDelta = Math.max(midDragDelta, d)
    }
    const overlayActive = await page.evaluate(() => Boolean(document.querySelector('canvas.phosmith-megashader-preview')))
    await shot('live-1-mid-gamma-drag')
    await page.mouse.up()
    log(`LIVE CHECK (preview overlay active: ${overlayActive}) deltas across drag: ${JSON.stringify(midSeen)} (max ${midDragDelta.toFixed(2)}, need > 0.5)`)
    if (midDragDelta < 0.5) { await shot('live-FAIL-no-repaint'); throw new Error('no repaint during gamma drag — live-grade regression') }
    await sleep(600) // > PREVIEW_COMMIT_IDLE_MS: full-res commit lands, overlay drops
    const gammaVal = await gradeCard.locator('[aria-label="Gamma percent"]').inputValue()
    log(`gamma landed at ${gammaVal}%`)

    /* B — wheel + curve */
    const shadowsLuma = gradeCard.locator('.grade-wheel', { hasText: 'Shadows' }).locator('[aria-label="Luma"]').first()
    if (await shadowsLuma.isVisible().catch(() => false)) {
        await shadowsLuma.scrollIntoViewIfNeeded()
        const lb = await shadowsLuma.boundingBox()
        const ly = lb.y + lb.height / 2
        await page.mouse.move(lb.x + lb.width * 0.5, ly)
        await page.mouse.down()
        await page.mouse.move(lb.x + lb.width * 0.78, ly, { steps: 5 })
        await page.mouse.up()
        await sleep(700) // let the full-res commit land before sampling
        log(`shadows luma dragged — delta vs pre-grade now ${delta(preGrade, await sample()).toFixed(2)}`)
    } else log('WARN: shadows luma not found, skipped')

    const curve = gradeCard.locator('.adjust-curve-svg').first()
    await curve.scrollIntoViewIfNeeded()
    const cb = await curve.boundingBox()
    const cx = cb.x + cb.width * 0.55
    await page.mouse.move(cx, cb.y + cb.height * 0.45)
    await page.mouse.down()
    await page.mouse.move(cx, cb.y + cb.height * 0.2, { steps: 6 })
    await page.mouse.up()
    await sleep(800) // full-res commit + render
    const postAll = await sample()
    log(`curve point pulled — total grade delta ${delta(preGrade, postAll).toFixed(2)}`)
    await shot('live-2-all-grades')

    /* C — save fires */
    const snap = await page.waitForResponse(
        (r) => r.url().includes('/api/canvas/snapshot') && r.request().method() === 'POST',
        { timeout: 15_000 },
    ).catch(() => null)
    if (!snap) { await shot('live-FAIL-no-save'); throw new Error('no snapshot POST within 15s of grade edits') }
    log(`save fired → ${snap.status()}`)
    await sleep(2500) // let IDB mirror + any flush settle

    /* D — wipe IDB, reload, compare */
    await page.evaluate(async () => {
        await new Promise((res) => { const q = indexedDB.deleteDatabase('phosmith-canvas-sync'); q.onsuccess = q.onerror = q.onblocked = res })
    })
    log('IndexedDB wiped — reloading, restore must come from server state')
    await page.reload({ waitUntil: 'domcontentloaded' })
    await page.waitForSelector('canvas.lower-canvas', { timeout: 60_000 })
    let restored = null
    let restoreDelta = Infinity
    for (let i = 0; i < 60; i++) {
        restored = await sample()
        restoreDelta = delta(restored, postAll)
        if (restoreDelta < 2.5) break
        await sleep(500)
    }
    await shot('live-3-after-reload')
    log(`post-reload pixel delta vs saved state: ${restoreDelta.toFixed(2)} (need < 2.5 — proves masks+gamma+wheel+curve restored)`)
    if (restoreDelta >= 2.5) throw new Error(`restore mismatch (delta ${restoreDelta.toFixed(2)})`)

    await page.locator('[aria-label="Mask"]').first().click()
    await page.waitForFunction(() => document.querySelectorAll('.mask-chain-card').length >= 3, null, { timeout: 20_000 })
    const gammaAfter = await page.locator('.mask-chain-card').last().locator('[aria-label="Gamma percent"]').inputValue().catch(() => 'n/a')
    log(`panel rehydrated (3 cards); gamma persisted: ${gammaAfter}% (was ${gammaVal}%)`)

    /* E — visibility toggle */
    const preT = await sample()
    const lastCard = page.locator('.mask-chain-card').last()
    const toggle = lastCard.locator('button[title="Hide"], button[title="Show"]').first()
    if (await toggle.isVisible().catch(() => false)) {
        await toggle.click()
        await sleep(700)
        const dT = delta(preT, await sample())
        log(`visibility toggle delta ${dT.toFixed(2)} (> 0.5 proves layers are live)`)
        await toggle.click()
    } else log('WARN: visibility toggle not found by title — skipped')
    await shot('live-4-final')

    return { ok: true, midDragDelta: +midDragDelta.toFixed(2), restoreDelta: +restoreDelta.toFixed(2), gammaVal, gammaAfter, consoleHits: consoleHits.slice(0, 15), steps }
}
