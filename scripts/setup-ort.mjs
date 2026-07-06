#!/usr/bin/env node
/**
 * Copy onnxruntime-web's runtime files into public/ort/ so the on-device AI
 * (src/lib/client-ai.js) can load them deterministically.
 *
 * WHY: bundlers (Next/Turbopack included) break onnxruntime's runtime fetch of
 * ort-wasm-simd-threaded.jsep.{mjs,wasm} — the JSEP glue that defines
 * `webgpuInit` — so WebGPU init throws "webgpuInit is not a function" and WASM
 * reports "no available backend found", killing every in-browser model. Serving
 * the exact dist files that match the installed @huggingface/transformers'
 * onnxruntime-web version (the NESTED copy, not the root one) and pointing
 * `env.backends.onnx.wasm.wasmPaths` at them fixes both backends. Same recipe
 * as the mask-studio testbed's setup:ort.
 *
 * Wired into `bun run dev` / `bun run build`; idempotent and fast (skips
 * copies when sizes already match).
 */
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync, readFileSync } from 'node:fs'
import path from 'node:path'

const ROOT = path.resolve(import.meta.dirname, '..')
const CANDIDATES = [
    path.join(ROOT, 'node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist'),
    path.join(ROOT, 'node_modules/onnxruntime-web/dist'),
]
const src = CANDIDATES.find((d) => existsSync(d))
if (!src) {
    console.error('[setup-ort] onnxruntime-web dist not found — run bun install first')
    process.exit(1)
}

const out = path.join(ROOT, 'public/ort')
mkdirSync(out, { recursive: true })

let copied = 0
for (const f of readdirSync(src)) {
    if (!/^ort-wasm-simd-threaded.*\.(mjs|wasm)$/.test(f)) continue
    const from = path.join(src, f)
    const to = path.join(out, f)
    if (existsSync(to) && statSync(to).size === statSync(from).size) continue
    copyFileSync(from, to)
    copied += 1
}

const version = JSON.parse(readFileSync(path.join(src, '../package.json'), 'utf8')).version
console.log(`[setup-ort] public/ort ready (onnxruntime-web ${version}, ${copied} file(s) copied)`)
