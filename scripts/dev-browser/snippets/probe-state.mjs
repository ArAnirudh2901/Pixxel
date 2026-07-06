/**
 * probe-state.mjs — where are we? URL, title, login state, screenshot.
 */
export default async ({ page, screenshotDir }) => {
    await page.goto('http://localhost:3000/dashboard', { waitUntil: 'domcontentloaded', timeout: 60_000 })
    await page.waitForLoadState('networkidle', { timeout: 30_000 }).catch(() => {})
    await page.screenshot({ path: `${screenshotDir}/probe-state.png` })
    const url = page.url()
    const signInVisible = await page
        .locator('input[name="identifier"], .cl-signIn-root, [data-clerk-sign-in]')
        .first()
        .isVisible()
        .catch(() => false)
    return {
        url,
        title: await page.title(),
        signInVisible,
        loggedIn: url.includes('/dashboard') && !signInVisible,
    }
}
