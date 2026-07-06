---
name: dev-browser
description: Drive the local Pixxel app in the user's REAL Google Chrome via CDP. Use this instead of the built-in chromium/dev-browser skill for anything that needs auth — Clerk rejects Playwright's bundled "Chrome for Testing", so login only works in real Chrome.
---

# dev-browser — real-Chrome automation for Pixxel

Do NOT use the built-in chromium-cli/dev-browser skill for this repo: it runs
Playwright's bundled Chromium, which Clerk blocks as an unsupported browser, so
sign-in (and every authenticated page) is unreachable. This project-local
driver attaches to the user's real Google Chrome over CDP instead.

## How to use

1. **Start Chrome** (idempotent — reuses a running instance):

   ```bash
   bun run browser:launch            # opens http://localhost:3000
   node scripts/dev-browser/launch-chrome.mjs https://example.com   # custom URL
   ```

   Chrome starts headful with a dedicated persistent profile at
   `.chrome-dev-profile/` (gitignored). **First run only:** if the app shows a
   sign-in screen, ask the user to log in once in that window — the Clerk
   session persists in the profile for all later runs.

2. **Run a snippet** against it:

   ```bash
   bun run browser:run <snippet.mjs>
   # = node scripts/dev-browser/run.mjs <snippet.mjs>
   ```

   A snippet is an `.mjs` module:

   ```js
   export default async ({ browser, context, page, screenshotDir }) => {
       await page.goto('http://localhost:3000/dashboard')
       await page.screenshot({ path: `${screenshotDir}/dashboard.png` })
       return { title: await page.title() }   // JSON-printed by the runner
   }
   ```

   `page` is the most recent tab in the real browser's persistent context
   (Clerk cookies included). The runner disconnects when done but never closes
   Chrome — state survives across runs. Screenshots go in
   `scripts/dev-browser/screenshots/` (gitignored).

## Conventions

- Keep reusable snippets in `scripts/dev-browser/snippets/`.
- The dev stack for authenticated editor flows: `bun run dev` (port 3000),
  `services/segment/.venv/bin/python services/segment/main.py` (8001),
  `services/masking/.venv/bin/python services/masking/main.py` (8002).
- Port/binary overrides: `DEV_BROWSER_CDP_PORT`, `DEV_BROWSER_CHROME_BIN`.
- Run `run.mjs` with **node, not bun** — playwright's CDP WebSocket connect
  times out under bun (verified with bun 1.3.11 / playwright 1.60).
- Never point this at the user's default Chrome profile — Chrome ≥136 blocks
  CDP on it, and automation shouldn't touch their daily browsing data anyway.
