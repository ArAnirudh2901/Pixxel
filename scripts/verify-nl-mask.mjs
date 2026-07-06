#!/usr/bin/env node
/**
 * Invariant tests for the NL → mask pipeline.
 *
 * Pure parts (no services needed):
 *   - parseMaskDescription: description → MaskPlan (targets, ops, invert,
 *     fillMode, grow/feather extraction)
 *   - validateMaskPlan: shape/enum/range sanitisation
 *   - pickSubjectInstances: qualifier scoring (position/ordinal/size/color,
 *     plural vs singular)
 *   - growCoverage: boundary extension morphology (dilate/erode distances)
 *
 * Live part (skipped when the mask service is unreachable):
 *   - POST /ground/text with a synthetic image and asserts the phrase binds
 *     to the right region.
 *
 * Usage: bun scripts/verify-nl-mask.mjs
 */

import {
  parseMaskDescription,
  validateMaskPlan,
  pickSubjectInstances,
} from '../src/lib/agent/nl-mask-parser.js'
import { growCoverage } from '../src/lib/mask-grow-core.js'
import {
  beginAgentAction,
  endAgentAction,
  isAgentActing,
  recordChange,
  getChanges,
  clearChanges,
} from '../src/lib/change-journal.js'
import {
  AI_CAPABILITIES,
  getRoutingMode,
  prefersClient,
  resetRoutingPolicy,
  resolveOrder,
  setRoutingMode,
} from '../src/lib/ai-routing.js'
import {
  analyzeCoverage,
  analyzeRange,
  evaluateSelfTest,
  validateDepthOutput,
  validateGroundOutput,
  withTimeout,
} from '../src/lib/client-ai-core.js'

const MASK_SERVICE_URL = (process.env.MASKING_SERVICE_URL || process.env.MASK_SERVICE_URL || 'http://127.0.0.1:8002')
  .trim()
  .replace(/\/+$/, '')

let failures = 0
const check = (label, cond, detail = '') => {
  if (cond) console.log(`[verify-nl-mask] ok ${label}`)
  else {
    failures += 1
    console.error(`[verify-nl-mask] ✗ ${label}${detail ? ` — ${detail}` : ''}`)
  }
}

/* ─── 1. parseMaskDescription ───────────────────────────────────────────── */
{
  const p = (d) => parseMaskDescription(d)

  let plan = p('the dog')
  check('"the dog" → subjects[dog]',
    plan.steps.length === 1 && plan.steps[0].target.type === 'subjects'
    && plan.steps[0].target.labels[0] === 'dog' && plan.fillMode === 'fill',
    JSON.stringify(plan))

  plan = p('Mask the person on the left')
  check('"person on the left" → position qualifier',
    plan.steps[0].target.type === 'subjects'
    && plan.steps[0].target.labels[0] === 'person'
    && plan.steps[0].target.qualifiers?.position === 'left',
    JSON.stringify(plan))

  plan = p('the second dog from the right')
  check('"second dog from the right" → ordinal 2 from right',
    plan.steps[0].target.qualifiers?.ordinal === 2
    && plan.steps[0].target.qualifiers?.ordinalFrom === 'right',
    JSON.stringify(plan))

  plan = p('everything except the sky')
  check('"everything except the sky" → inverted concept',
    plan.invert === true && plan.steps.length === 1
    && plan.steps[0].target.type === 'concept'
    && /sky/.test(plan.steps[0].target.phrase),
    JSON.stringify(plan))

  plan = p('the shadows but not the person')
  check('"shadows but not the person" → luminance + subtract subjects',
    plan.steps.length === 2
    && plan.steps[0].target.type === 'luminance'
    && plan.steps[0].target.region === 'shadows'
    && plan.steps[1].op === 'subtract'
    && plan.steps[1].target.type === 'subjects',
    JSON.stringify(plan))

  plan = p('remove the background')
  check('"remove the background" → erase + depth background',
    plan.fillMode === 'erase'
    && plan.steps[0].target.type === 'depth'
    && plan.steps[0].target.region === 'background',
    JSON.stringify(plan))

  plan = p('the red car')
  check('"the red car" → subjects[car] + color qualifier',
    plan.steps[0].target.type === 'subjects'
    && plan.steps[0].target.labels[0] === 'car'
    && plan.steps[0].target.qualifiers?.color === 'red',
    JSON.stringify(plan))

  plan = p('the red jacket, extend the selection by 12px')
  check('"red jacket, extend by 12px" → concept + grow 12',
    plan.grow === 12
    && plan.steps.length === 1
    && plan.steps[0].target.type === 'concept'
    && /jacket/.test(plan.steps[0].target.phrase)
    && !/extend|12/.test(plan.steps[0].target.phrase),
    JSON.stringify(plan))

  plan = p('the top half')
  check('"the top half" → region top 50%',
    plan.steps[0].target.type === 'region'
    && plan.steps[0].target.area === 'top'
    && Math.abs(plan.steps[0].target.fraction - 0.5) < 1e-6,
    JSON.stringify(plan))

  plan = p('the highlights')
  check('"the highlights" → luminance highlights',
    plan.steps[0].target.type === 'luminance'
    && plan.steps[0].target.region === 'highlights',
    JSON.stringify(plan))

  plan = p('all the green areas')
  check('"all the green areas" → colorRange green',
    plan.steps[0].target.type === 'colorRange'
    && plan.steps[0].target.name === 'green',
    JSON.stringify(plan))

  plan = p('the people with soft edges')
  check('"the people with soft edges" → feather + subjects',
    plan.feather === 0.12
    && plan.steps[0].target.type === 'subjects'
    && plan.steps[0].target.labels[0] === 'person',
    JSON.stringify(plan))

  plan = p('the dog and the cat')
  check('"the dog and the cat" → two add steps',
    plan.steps.length === 2
    && plan.steps[0].target.labels?.[0] === 'dog'
    && plan.steps[1].target.labels?.[0] === 'cat'
    && plan.steps[1].op === 'add',
    JSON.stringify(plan))

  plan = p('everything but the person in the middle')
  check('"everything but the person in the middle" → invert + center',
    plan.invert === true
    && plan.steps[0].target.type === 'subjects'
    && plan.steps[0].target.qualifiers?.position === 'center',
    JSON.stringify(plan))

  plan = p('the glowing neon sign')
  check('unknown nouns fall back to concept grounding',
    plan.steps[0].target.type === 'concept'
    && /neon sign/.test(plan.steps[0].target.phrase),
    JSON.stringify(plan))

  plan = p('erase the black and white dog')
  check('"black and white dog" stays ONE phrase (no bogus conjunction split)',
    plan.steps.length === 1 && plan.fillMode === 'erase',
    JSON.stringify(plan))
}

/* ─── 2. validateMaskPlan ───────────────────────────────────────────────── */
{
  check('empty plan rejected', validateMaskPlan({}).valid === false)
  check('plan with no usable steps rejected',
    validateMaskPlan({ steps: [{ op: 'add', target: { type: 'bogus' } }] }).valid === false)

  const v = validateMaskPlan({
    grow: 9999,
    feather: 3,
    fillMode: 'nonsense',
    invert: 'yes',
    steps: [
      { op: 'subtract', target: { type: 'concept', phrase: 'x'.repeat(500) } },
      { op: 'weird', target: { type: 'depth', region: 'background' } },
    ],
  })
  check('values clamped + coerced',
    v.valid
    && v.plan.grow === 200
    && v.plan.feather === 0.5
    && v.plan.fillMode === 'fill'
    && v.plan.invert === false
    && v.plan.steps[0].op === 'add'           // first step always add
    && v.plan.steps[0].target.phrase.length === 100
    && v.plan.steps[1].op === 'add',          // unknown op coerced
    JSON.stringify(v))
}

/* ─── 3. pickSubjectInstances ───────────────────────────────────────────── */
{
  const instances = [
    { index: 0, label: 'person', bboxImage: [50, 100, 100, 300], centroidImage: [100, 250] },
    { index: 1, label: 'person', bboxImage: [350, 80, 120, 340], centroidImage: [410, 250] },
    { index: 2, label: 'person', bboxImage: [650, 120, 90, 260], centroidImage: [695, 250] },
    { index: 3, label: 'dog', bboxImage: [500, 300, 80, 90], centroidImage: [540, 345] },
  ]

  let r = pickSubjectInstances(instances, { labels: ['person'], qualifiers: { position: 'left' } }, { imageWidth: 800 })
  check('position left → leftmost person', r.picked.length === 1 && r.picked[0].index === 0)

  r = pickSubjectInstances(instances, { labels: ['person'], qualifiers: { ordinal: 2, ordinalFrom: 'right' } }, { imageWidth: 800 })
  check('2nd from the right → middle person', r.picked.length === 1 && r.picked[0].index === 1)

  r = pickSubjectInstances(instances, { labels: ['person'], qualifiers: { size: 'largest' } }, {})
  check('largest person picked', r.picked.length === 1 && r.picked[0].index === 1)

  r = pickSubjectInstances(instances, { labels: ['person'], phrase: 'the people' }, {})
  check('plural phrase selects every person', r.picked.length === 3)

  r = pickSubjectInstances(instances, { labels: ['person'], phrase: 'the person' }, {})
  check('singular + multiple candidates → largest with a disambiguation note',
    r.picked.length === 1 && r.picked[0].index === 1 && /detected/.test(r.note || ''))

  r = pickSubjectInstances(instances, { labels: ['cat'] }, {})
  check('unmatched label → empty (executor falls back to grounding)', r.picked.length === 0)

  const colorOf = (inst) => (inst.index === 2 ? 'red' : 'blue')
  r = pickSubjectInstances(instances, { labels: ['person'], qualifiers: { color: 'red' } }, { colorOf })
  check('color qualifier filters by sampled colour', r.picked.length === 1 && r.picked[0].index === 2)
}

/* ─── 4. growCoverage morphology ────────────────────────────────────────── */
{
  const W = 100
  const H = 100
  const cover = new Uint8ClampedArray(W * H)
  for (let y = 40; y < 60; y += 1) {
    for (let x = 40; x < 60; x += 1) cover[y * W + x] = 255
  }
  const bounds = (data) => {
    let x0 = W; let x1 = -1; let y0 = H; let y1 = -1
    for (let y = 0; y < H; y += 1) {
      for (let x = 0; x < W; x += 1) {
        if (data[y * W + x] > 127) {
          if (x < x0) x0 = x
          if (x > x1) x1 = x
          if (y < y0) y0 = y
          if (y > y1) y1 = y
        }
      }
    }
    return { x0, x1, y0, y1 }
  }

  const same = growCoverage(cover, W, H, 0)
  check('grow 0 is identity', same.every((v, i) => v === cover[i]))

  const grown = bounds(growCoverage(cover, W, H, 5))
  check('grow +5 expands each edge by ~5px',
    Math.abs(grown.x0 - 35) <= 2 && Math.abs(grown.x1 - 64) <= 2
    && Math.abs(grown.y0 - 35) <= 2 && Math.abs(grown.y1 - 64) <= 2,
    JSON.stringify(grown))

  const shrunk = bounds(growCoverage(cover, W, H, -5))
  check('grow -5 contracts each edge by ~5px',
    Math.abs(shrunk.x0 - 45) <= 2 && Math.abs(shrunk.x1 - 54) <= 2
    && Math.abs(shrunk.y0 - 45) <= 2 && Math.abs(shrunk.y1 - 54) <= 2,
    JSON.stringify(shrunk))

  const wiped = growCoverage(cover, W, H, -40)
  check('over-shrink empties the mask without crashing', wiped.every((v) => v < 128))
}

/* ─── 5. change journal (user/agent attribution) ────────────────────────── */
{
  clearChanges()
  recordChange({ label: 'Crop applied' })
  check('plain change attributed to user', getChanges()[0]?.source === 'user')

  beginAgentAction()
  beginAgentAction() // nested agent scopes (fromDescription → addLayer)
  check('agent flag is on inside nested scopes', isAgentActing() === true)
  recordChange({ label: 'mask.fromDescription', domain: 'mask' })
  endAgentAction()
  check('agent flag survives inner scope end', isAgentActing() === true)
  endAgentAction()
  check('agent flag clears with outer scope', isAgentActing() === false)
  const agentEntry = getChanges()[0]
  check('agent-scoped change attributed to agent',
    agentEntry?.source === 'agent' && agentEntry?.domain === 'mask')

  recordChange({ label: 'Manual after agent' })
  check('attribution returns to user after agent scope', getChanges()[0]?.source === 'user')

  check('entries are newest-first', getChanges()[2]?.label === 'Crop applied')
  check('empty labels are ignored', recordChange({ label: '  ' }) === null && getChanges().length === 3)

  const explicit = recordChange({ label: 'forced', source: 'agent' })
  check('explicit source overrides the flag', explicit?.source === 'agent')

  clearChanges()
  check('clearChanges empties the journal', getChanges().length === 0)
}

/* ─── 6. AI routing policy ──────────────────────────────────────────────── */
{
  resetRoutingPolicy()
  check('default mode is auto', getRoutingMode('ground') === 'auto')
  check('auto order prefers server with client fallback',
    JSON.stringify(resolveOrder('ground')) === JSON.stringify(['server', 'client']))

  setRoutingMode('ground', 'client')
  check('client mode flips the attempt order',
    JSON.stringify(resolveOrder('ground')) === JSON.stringify(['client', 'server'])
    && prefersClient('ground') === true)

  setRoutingMode('ground', 'server')
  check('server mode keeps server first',
    resolveOrder('ground')[0] === 'server' && prefersClient('ground') === false)

  setRoutingMode('segment', 'client')
  check('segment is client-capable (RMBG-1.4 in browser)',
    JSON.stringify(resolveOrder('segment')) === JSON.stringify(['client', 'server']))

  check('unknown capabilities resolve server-only',
    JSON.stringify(resolveOrder('definitely-not-a-capability')) === JSON.stringify(['server']))

  setRoutingMode('ground', 'bogus-mode')
  check('invalid modes sanitise to auto', getRoutingMode('ground') === 'auto')

  setRoutingMode('not-a-capability', 'client')
  check('unknown capabilities are ignored', getRoutingMode('not-a-capability') === 'auto')

  check('every capability has at least one side',
    Object.keys(AI_CAPABILITIES).every((cap) => resolveOrder(cap).length >= 1))

  resetRoutingPolicy()
  check('reset restores auto everywhere',
    Object.keys(AI_CAPABILITIES).every((cap) => getRoutingMode(cap) === 'auto'))
}

/* ─── 7. client-ai core validators (the on-device engine's guards) ──────── */
{
  // 8x8 map with a hot 3x3 block.
  const w = 8
  const h = 8
  const map = new Float32Array(w * h).fill(0.05)
  for (let y = 2; y <= 4; y += 1) for (let x = 3; x <= 5; x += 1) map[y * w + x] = 0.9
  const stats = analyzeCoverage(map, w, h, 0.5)
  check('analyzeCoverage finds peak/coverage/bbox',
    Math.abs(stats.peak - 0.9) < 1e-6
    && Math.abs(stats.coverage - 9 / 64) < 1e-6
    && JSON.stringify(stats.bbox) === JSON.stringify([3, 2, 3, 3])
    && stats.finite === true,
    JSON.stringify(stats))

  const nan = Float32Array.from(map)
  nan[0] = NaN
  check('analyzeCoverage flags non-finite outputs', analyzeCoverage(nan, w, h).finite === false)

  check('validateGroundOutput accepts a healthy cut',
    validateGroundOutput({ peak: 0.9, coverage: 0.14, finite: true }).usable === true)
  check('validateGroundOutput rejects broken backends',
    validateGroundOutput({ peak: 0.9, coverage: 0.14, finite: false }).usable === false)
  check('validateGroundOutput rejects whole-frame degenerates',
    validateGroundOutput({ peak: 0.9, coverage: 0.99, finite: true }).usable === false)

  check('validateDepthOutput rejects flat maps',
    validateDepthOutput({ width: 10, height: 10, min: 0.5, max: 0.51, finite: true }, { width: 10, height: 10 }).usable === false)
  check('validateDepthOutput rejects wrong dims',
    validateDepthOutput({ width: 9, height: 10, min: 0, max: 1, finite: true }, { width: 10, height: 10 }).usable === false)
  check('analyzeRange measures spread', (() => {
    const r = analyzeRange(Float32Array.from([0.1, 0.9, 0.4]))
    return Math.abs(r.min - 0.1) < 1e-6 && Math.abs(r.max - 0.9) < 1e-6 && r.finite
  })())

  const good = evaluateSelfTest(
    { ground: { found: true, score: 0.96, bbox: [40, 70, 120, 120] }, depth: { width: 320, height: 240, spread: 0.8 } },
    { cx: 100, cy: 130, w: 320, h: 240 },
  )
  check('evaluateSelfTest passes the golden scene', good.ok === true)

  const bad = evaluateSelfTest(
    { ground: { found: true, score: 0.9, bbox: [200, 10, 50, 50] }, depth: { width: 320, height: 240, spread: 0.0 } },
    { cx: 100, cy: 130, w: 320, h: 240 },
  )
  check('evaluateSelfTest fails off-target masks and flat depth',
    bad.ok === false && bad.checks.filter((c) => !c.ok).length === 2)

  const withSeg = evaluateSelfTest(
    {
      ground: { found: true, score: 0.96, bbox: [40, 70, 120, 120] },
      depth: { width: 320, height: 240, spread: 0.8 },
      segment: { width: 320, height: 240, coverage: 0.11, bbox: [42, 72, 118, 118] },
    },
    { cx: 100, cy: 130, w: 320, h: 240 },
  )
  check('evaluateSelfTest passes a sane background-removal matte',
    withSeg.ok === true && withSeg.checks.length === 6)

  const badSeg = evaluateSelfTest(
    {
      ground: { found: true, score: 0.96, bbox: [40, 70, 120, 120] },
      depth: { width: 320, height: 240, spread: 0.8 },
      segment: { width: 320, height: 240, coverage: 0.97, bbox: [0, 0, 320, 240] },
    },
    { cx: 100, cy: 130, w: 320, h: 240 },
  )
  check('evaluateSelfTest rejects a whole-frame matte',
    badSeg.ok === false)

  const withSam = evaluateSelfTest(
    {
      ground: { found: true, score: 0.96, bbox: [40, 70, 120, 120] },
      depth: { width: 320, height: 240, spread: 0.8 },
      sam: { width: 320, height: 240, coverage: 0.11, bbox: [42, 72, 118, 118] },
    },
    { cx: 100, cy: 130, w: 320, h: 240 },
  )
  check('evaluateSelfTest passes a sane click-select mask',
    withSam.ok === true && withSam.checks.length === 6)

  const badSam = evaluateSelfTest(
    {
      ground: { found: true, score: 0.96, bbox: [40, 70, 120, 120] },
      depth: { width: 320, height: 240, spread: 0.8 },
      sam: { width: 320, height: 240, coverage: 0.12, bbox: [200, 10, 60, 60] },
    },
    { cx: 100, cy: 130, w: 320, h: 240 },
  )
  check('evaluateSelfTest fails an off-click SAM mask',
    badSam.ok === false)

  const timedOut = await withTimeout(new Promise(() => {}), 30, 'hang').then(
    () => false,
    (e) => /timed out/.test(e.message),
  )
  check('withTimeout rejects hung promises', timedOut === true)
}

/* ─── 8. live /ground/text (optional) ───────────────────────────────────── */
const reachable = async () => {
  try {
    const ac = new AbortController()
    const t = setTimeout(() => ac.abort(), 1500)
    const resp = await fetch(`${MASK_SERVICE_URL}/health`, { signal: ac.signal })
    clearTimeout(t)
    return resp.ok
  } catch { return false }
}

const liveGroundCheck = async () => {
  if (!(await reachable())) {
    console.log(`[verify-nl-mask] skip live grounding — mask service at ${MASK_SERVICE_URL} unreachable (bun run mask:dev)`)
    return
  }
  const sharp = (await import('sharp')).default
  const W = 640
  const H = 480
  const CX = 200
  const CY = 240
  const R = 110
  const svg = Buffer.from(
    `<svg width="${W}" height="${H}">
       <rect width="${W}" height="${H}" fill="rgb(120,130,140)"/>
       <circle cx="${CX}" cy="${CY}" r="${R}" fill="rgb(225,30,30)"/>
     </svg>`,
  )
  const image = await sharp(svg).jpeg({ quality: 92 }).toBuffer()

  const form = new FormData()
  form.append('image', new Blob([image], { type: 'image/jpeg' }), 'disc.jpg')
  form.append('phrases', JSON.stringify(['the red circle']))
  const resp = await fetch(`${MASK_SERVICE_URL}/ground/text`, { method: 'POST', body: form })
  if (!resp.ok) {
    check('live grounding HTTP ok', false, `${resp.status} ${await resp.text().catch(() => '')}`)
    return
  }
  const data = await resp.json()
  const r = data.results?.[0]
  check('live: "the red circle" binds', !!r?.found, JSON.stringify(r))
  if (r?.found && Array.isArray(r.bbox)) {
    const [x, y, w, h] = r.bbox
    const containsCenter = CX >= x && CX <= x + w && CY >= y && CY <= y + h
    check('live: mask bbox contains the disc center', containsCenter, JSON.stringify(r.bbox))
    check('live: coverage is plausible for the disc', r.coverage > 0.04 && r.coverage < 0.5, String(r.coverage))
  }
}

await liveGroundCheck()

if (failures > 0) {
  console.error(`\n[verify-nl-mask] ✗ ${failures} check(s) failed`)
  process.exit(1)
}
console.log('\n[verify-nl-mask] ✓ all checks passed')
