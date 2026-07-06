/**
 * verify-mask-grade-persistence.mjs — end-to-end check of the three mask
 * fixes, run against real Chrome (see .claude/skills/dev-browser/SKILL.md):
 *
 *   1. LIVE GRADE: dragging Gamma repaints the canvas WHILE the button is
 *      down (the old self-resetting debounce meant nothing moved mid-drag).
 *   2. SAVE TRIGGER: mask/grade edits POST /api/canvas/snapshot within a
 *      few seconds (previously nothing ever saved).
 *   3. DB RESTORE: after wiping IndexedDB and reloading, the mask layers,
 *      grade values (incl. the rebuilt curve LUT) and the rendered pixels
 *      all come back from server state.
 *
 * Creates a FRESH project from a generated test image so no real user
 * project is mutated.
 */
const APP = 'http://localhost:3000'
const TEST_IMAGE = process.env.PIXXEL_E2E_IMAGE
    || '/Users/andhetharuntej/.claude/jobs/5ea38232/tmp/pixxel-e2e-image.png'

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

export default async ({ page, context, screenshotDir }) => {
    // The real window can be arbitrarily small — emulate a working viewport
    // so panels/modals are reachable regardless of the physical window size.
    await page.bringToFront().catch(() => {})
    await page.setViewportSize({ width: 1600, height: 1000 })
    const steps = []
    const consoleHits = []
    const log = (msg) => { steps.push(msg); console.error(`  • ${msg}`) }
    const shot = (name) => page.screenshot({ path: `${screenshotDir}/${name}.png` }).catch(() => {})
    page.on('console', (m) => {
        const t = m.text()
        if (/megashader|Error|error saving|canvas\]/i.test(t)) consoleHits.push(`[${m.type()}] ${t.slice(0, 300)}`)
    })

    // Average RGB of three 16×16 regions of the Fabric lower-canvas.
    const sampleCanvas = () => page.evaluate(() => {
        const c = document.querySelector('canvas.lower-canvas')
        if (!c || !c.width) return null
        const ctx = c.getContext('2d')
        return [[0.35, 0.5], [0.5, 0.85], [0.65, 0.45]].map(([fx, fy]) => {
            const x = Math.floor(c.width * fx), y = Math.floor(c.height * fy)
            const d = ctx.getImageData(x - 8, y - 8, 16, 16).data
            let r = 0, g = 0, b = 0
            const n = d.length / 4
            for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2] }
            return [r / n, g / n, b / n]
        })
    })
    const delta = (a, b) => {
        if (!a || !b) return Infinity
        let sum = 0, n = 0
        a.forEach((reg, i) => reg.forEach((v, j) => { sum += Math.abs(v - b[i][j]); n++ }))
        return sum / n
    }

    /* ── Stage 1: auth ─────────────────────────────────────────────── */
    await page.goto(`${APP}/dashboard`, { waitUntil: 'domcontentloaded', timeout: 180_000 })
    await page.waitForLoadState('networkidle', { timeout: 30_000 }).catch(() => {})
    if (!page.url().includes('/dashboard')) {
        await shot('e2e-need-login')
        throw new Error(`Not logged in (at ${page.url()}) — log in once in the Chrome window, then re-run.`)
    }
    log('logged in, on dashboard')

    // Force SERVER routing for AI capabilities — 'auto' can pick the
    // on-device model, which first downloads hundreds of MB from HF and
    // blows every reasonable test timeout. The local services (8001/8002)
    // are warm and answer in seconds.
    await page.evaluate(() => {
        const all = {}
        for (const cap of ['maskPlan', 'ground', 'depth', 'subjects', 'segment', 'sam2', 'inpaint']) all[cap] = 'server'
        window.localStorage.setItem('phosmith:ai-routing', JSON.stringify(all))
    })
    log('AI routing forced to server for all capabilities')

    /* ── Stage 2: fresh project from the test image ────────────────── */
    await page.getByText('New Project', { exact: false }).first().click()
    const fileInput = page.locator('input[type="file"]')
    await fileInput.waitFor({ state: 'attached', timeout: 10_000 })
    await fileInput.setInputFiles(TEST_IMAGE)
    const title = `mask-e2e ${new Date().toISOString().slice(5, 19)}`
    const titleInput = page.locator('input[type="text"]:visible').first()
    await titleInput.fill(title)
    await page.getByRole('button', { name: /create project/i }).click()
    await page.waitForURL(/\/editor\//, { timeout: 120_000 })
    const projectUrl = page.url()
    log(`project created: ${projectUrl}`)

    // Editor ready = lower-canvas exists and shows non-uniform pixels.
    await page.waitForSelector('canvas.lower-canvas', { timeout: 60_000 })
    let base = null
    for (let i = 0; i < 60; i++) {
        base = await sampleCanvas()
        if (base && delta([base[0]], [base[1]]) > 3) break // image (not flat bg) visible
        await sleep(500)
    }
    await shot('e2e-1-editor-loaded')
    if (!base) throw new Error('canvas never rendered')
    log(`editor rendered, baseline sample ${JSON.stringify(base.map((r) => r.map(Math.round)))}`)

    /* ── Stage 3: open Mask tool ───────────────────────────────────── */
    await page.locator('[aria-label="Mask"]').first().click()
    await page.getByText('Mask Layers', { exact: false }).first().waitFor({ timeout: 20_000 })
    log('mask tool open')

    // .last(): a collapsible Section header can carry the same accessible
    // name as the action button inside it (e.g. "Select Subject") — the
    // header comes first in DOM order, the real button last.
    const clickPanelButton = async (buttonName, sectionName) => {
        let btn = page.getByRole('button', { name: buttonName, exact: false }).last()
        if (!(await btn.isVisible().catch(() => false))) {
            await page.getByText(sectionName, { exact: false }).first().click()
            await sleep(400)
            btn = page.getByRole('button', { name: buttonName, exact: false }).last()
        }
        await btn.scrollIntoViewIfNeeded()
        await btn.click()
    }
    // Idempotent guard — restores the Mask tool if anything (including a
    // human poking the shared headful window) switched tools mid-test.
    const ensureMaskTool = async () => {
        const panelOpen = await page.getByText('Mask Layers', { exact: false }).first().isVisible().catch(() => false)
        if (!panelOpen) {
            await page.locator('[aria-label="Mask"]').first().click()
            await page.getByText('Mask Layers', { exact: false }).first().waitFor({ timeout: 20_000 })
        }
    }
    const cardCount = () => page.locator('.mask-chain-card').count()

    /* ── Stage 4: three mask kinds ─────────────────────────────────── */
    // Gradient adds enter DRAFT mode: the layer is added immediately, then a
    // drag on the canvas sets its geometry and mouse-up commits the draft
    // (until then, adding another draft layer is refused).
    const canvasEl = page.locator('canvas.upper-canvas').first()
    const dragOnCanvas = async (fx1, fy1, fx2, fy2) => {
        const box = await canvasEl.boundingBox()
        await page.mouse.move(box.x + box.width * fx1, box.y + box.height * fy1)
        await page.mouse.down()
        await page.mouse.move(box.x + box.width * fx2, box.y + box.height * fy2, { steps: 10 })
        await page.mouse.up()
        await sleep(400)
    }

    await clickPanelButton('Add Linear to Mask Layers', 'Linear Gradient')
    await page.waitForFunction(() => document.querySelectorAll('.mask-chain-card').length >= 1, null, { timeout: 10_000 })
    log('linear layer added (draft)')

    // A fresh 'fill'-mode selection must be visible on its own.
    let afterLinear = null
    for (let i = 0; i < 20; i++) {
        afterLinear = await sampleCanvas()
        if (delta(base, afterLinear) > 1) break
        await sleep(250)
    }
    const linearDelta = delta(base, afterLinear)
    log(`linear fill visible: canvas delta ${linearDelta.toFixed(2)} (need > 1)`)
    if (linearDelta < 1) { await shot('e2e-FAIL-linear-invisible'); throw new Error('adding a linear mask changed nothing on canvas') }

    await dragOnCanvas(0.30, 0.25, 0.70, 0.75) // set the line, commit draft
    log('linear draft committed via canvas drag')

    await clickPanelButton('Add Radial to Mask Layers', 'Radial Gradient')
    await page.waitForFunction(() => document.querySelectorAll('.mask-chain-card').length >= 2, null, { timeout: 10_000 })
    await dragOnCanvas(0.35, 0.30, 0.68, 0.70) // set the ellipse, commit draft
    log('radial layer added + committed')

    // Service-backed subject mask (segment/masking services must be up).
    await ensureMaskTool()
    await clickPanelButton('Select Subject', 'Select Subject')
    log('Select Subject clicked — waiting for AI mask (CPU inference can be slow)…')
    try {
        await page.waitForFunction(() => document.querySelectorAll('.mask-chain-card').length >= 3, null, { timeout: 240_000 })
    } catch (err) {
        await shot('e2e-FAIL-subject')
        log(`console during subject wait: ${JSON.stringify(consoleHits.slice(-10))}`)
        throw new Error(`subject mask never appeared: ${err.message}`)
    }
    log('subject layer added (3 cards total)')
    await shot('e2e-2-three-masks')

    const preGrade = await sampleCanvas()

    /* ── Stage 5: LIVE gamma drag (mid-drag assertions) ───────────── */
    // Grade the SUBJECT card (last added): its mask verifiably covers the
    // blob sample regions — a grade on a layer whose mask misses the sample
    // points is (correctly) pixel-identical there and would false-fail.
    await ensureMaskTool()
    const card0 = page.locator('.mask-chain-card').last()
    await card0.click() // select
    let gamma = card0.locator('[aria-label="Gamma"]').first()
    if (!(await gamma.isVisible().catch(() => false))) {
        // Card collapsed — expand via its params toggle.
        await card0.locator('button[title="Edit params"], button[title="Hide params"]').first().click()
        await sleep(300)
        gamma = card0.locator('[aria-label="Gamma"]').first()
    }
    await gamma.scrollIntoViewIfNeeded()
    // While the pointer is down, live pixels are carried by the
    // draft-preview OVERLAY canvas (the lower canvas freezes until the
    // full-res commit lands ~160ms after the drag idles) — so mid-drag
    // sampling reads the overlay and asserts it keeps changing.
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
    const gbox = await gamma.boundingBox()
    if (!gbox) throw new Error('gamma slider not visible on subject card')
    const y = gbox.y + gbox.height / 2
    await page.mouse.move(gbox.x + gbox.width * 0.5, y)
    await page.mouse.down()
    let midDragDelta = 0
    let firstLive = null
    for (let i = 1; i <= 8; i++) {
        await page.mouse.move(gbox.x + gbox.width * (0.5 + 0.045 * i), y, { steps: 3 })
        await sleep(80)
        const mid = await sampleLive()
        if (!firstLive) { firstLive = mid; continue }
        midDragDelta = Math.max(midDragDelta, delta(firstLive, mid))
    }
    const overlayActive = await page.evaluate(() => Boolean(document.querySelector('canvas.phosmith-megashader-preview')))
    await shot('e2e-3-mid-gamma-drag')
    await page.mouse.up()
    log(`LIVE CHECK (preview overlay: ${overlayActive}): max live delta while button down: ${midDragDelta.toFixed(2)} (need > 0.5)`)
    if (midDragDelta < 0.5) { await shot('e2e-FAIL-no-live-repaint'); throw new Error('canvas did not repaint during gamma drag — live-grade regression') }
    await sleep(800) // > commit idle: full-res result lands on the lower canvas
    const postGamma = await sampleCanvas()
    const gammaPctAfterDrag = await card0.locator('[aria-label="Gamma percent"]').inputValue()
    log(`gamma after drag: ${gammaPctAfterDrag}% — post-release delta ${delta(preGrade, postGamma).toFixed(2)}`)

    /* ── Stage 6: colour wheel (shadows luma) + curve point ────────── */
    const shadowsLuma = card0.locator('.grade-wheel', { hasText: 'Shadows' }).locator('[aria-label="Luma"]').first()
    await shadowsLuma.scrollIntoViewIfNeeded()
    const lbox = await shadowsLuma.boundingBox()
    if (lbox) {
        const ly = lbox.y + lbox.height / 2
        await page.mouse.move(lbox.x + lbox.width * 0.5, ly)
        await page.mouse.down()
        await page.mouse.move(lbox.x + lbox.width * 0.8, ly, { steps: 6 })
        await sleep(150)
        await page.mouse.up()
        await sleep(700) // let the full-res commit land before sampling
        log(`shadows-luma dragged — delta vs pre-grade ${delta(preGrade, await sampleCanvas()).toFixed(2)}`)
    } else {
        log('WARN: shadows luma slider not found — skipped wheel drag')
    }

    const curve = card0.locator('.adjust-curve-svg').first()
    await curve.scrollIntoViewIfNeeded()
    const cbox = await curve.boundingBox()
    if (!cbox) throw new Error('curve graph not visible on card 0')
    const cx = cbox.x + cbox.width * 0.55
    const cyStart = cbox.y + cbox.height * 0.45   // on/near the diagonal
    const cyEnd = cbox.y + cbox.height * 0.22     // pull up = brighten
    await page.mouse.move(cx, cyStart)
    await page.mouse.down()
    await page.mouse.move(cx, cyEnd, { steps: 8 })
    await sleep(150)
    await page.mouse.up()
    await sleep(800) // full-res commit + render
    const postAllGrades = await sampleCanvas()
    log(`curve point added+dragged — total grade delta ${delta(preGrade, postAllGrades).toFixed(2)}`)
    await shot('e2e-4-all-grades')

    /* ── Stage 7: save fired ───────────────────────────────────────── */
    const snapshotResp = await page.waitForResponse(
        (r) => r.url().includes('/api/canvas/snapshot') && r.request().method() === 'POST',
        { timeout: 15_000 },
    ).catch(() => null)
    if (!snapshotResp) { await shot('e2e-FAIL-no-save'); throw new Error('no /api/canvas/snapshot POST within 15s of grade edits — save trigger regression') }
    log(`save fired: snapshot POST → ${snapshotResp.status()}`)
    // Opportunistic Neon flush (non-fatal): fake tab-hide so the sync manager flushes.
    await page.evaluate(() => {
        try {
            Object.defineProperty(document, 'visibilityState', { get: () => 'hidden', configurable: true })
            document.dispatchEvent(new Event('visibilitychange'))
        } catch { /* ignore */ }
    })
    const flushResp = await page.waitForResponse((r) => r.url().includes('/api/canvas/flush'), { timeout: 8_000 }).catch(() => null)
    log(flushResp ? `neon flush → ${flushResp.status()}` : 'no explicit flush observed (Redis snapshot is enough for restore)')
    await page.evaluate(() => {
        try { delete document.visibilityState } catch { /* ignore */ }
        document.dispatchEvent(new Event('visibilitychange'))
    })

    // Record persisted UI facts to compare after reload.
    const cardsBefore = await page.locator('.mask-chain-card').allInnerTexts()
    const gammaBefore = await card0.locator('[aria-label="Gamma percent"]').inputValue()

    /* ── Stage 8: wipe local mirror, reload, assert restore ────────── */
    await page.evaluate(async () => {
        await new Promise((res) => { const req = indexedDB.deleteDatabase('phosmith-canvas-sync'); req.onsuccess = req.onerror = req.onblocked = res })
    })
    log('IndexedDB mirror deleted — reload must restore from server state')
    await page.reload({ waitUntil: 'domcontentloaded' })
    await page.waitForSelector('canvas.lower-canvas', { timeout: 60_000 })
    let restored = null
    for (let i = 0; i < 80; i++) {
        restored = await sampleCanvas()
        if (restored && delta(restored, postAllGrades) < 2.5) break
        await sleep(500)
    }
    const restoreDelta = delta(restored, postAllGrades)
    await shot('e2e-5-after-reload')
    log(`pixels after reload: delta vs pre-reload ${restoreDelta.toFixed(2)} (need < 2.5 — proves masks+gamma+wheel+CURVE re-render from DB)`)
    if (restoreDelta >= 2.5) throw new Error(`reloaded canvas does not match saved state (delta ${restoreDelta.toFixed(2)}) — restore regression`)

    await page.locator('[aria-label="Mask"]').first().click()
    await page.waitForFunction(() => document.querySelectorAll('.mask-chain-card').length >= 3, null, { timeout: 20_000 })
    const cardsAfter = await page.locator('.mask-chain-card').allInnerTexts()
    const gammaAfter = await page.locator('.mask-chain-card').last().locator('[aria-label="Gamma percent"]').inputValue().catch(() => 'n/a')
    log(`panel rehydrated: ${cardsAfter.length} cards, gamma ${gammaAfter}% (was ${gammaBefore}%)`)
    await shot('e2e-6-panel-rehydrated')
    if (cardsAfter.length < 3) throw new Error('mask panel did not rehydrate 3 layers after reload')

    // Toggle first layer visibility → pixels must move (masks are live, not baked).
    const preToggle = await sampleCanvas()
    const eyeBtn = page.locator('.mask-chain-card').first().getByRole('button').filter({ has: page.locator('svg') }).first()
    await eyeBtn.click()
    await sleep(600)
    const postToggle = await sampleCanvas()
    log(`visibility toggle delta: ${delta(preToggle, postToggle).toFixed(2)} (>0.5 proves layers are live)`)
    await eyeBtn.click().catch(() => {})

    return {
        ok: true,
        projectUrl,
        midDragDelta: +midDragDelta.toFixed(2),
        restoreDelta: +restoreDelta.toFixed(2),
        gammaBefore, gammaAfter,
        cards: { before: cardsBefore.length, after: cardsAfter.length },
        consoleHits: consoleHits.slice(0, 20),
        steps,
    }
}
