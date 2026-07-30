/**
 * raw-preview — extract the embedded JPEG preview from a camera RAW, plus a
 * cheap image-header parser. Ported from the seglab decoder (js/image-raw.js +
 * js/image-io.js), adapted for phosmith's upload-then-edit flow.
 *
 * A DSLR RAW (.nef/.cr2/.arw/.dng/…) is a container: sensor mosaic + heavy
 * metadata + one or more ALREADY-DEVELOPED JPEG previews, usually including one
 * at full sensor resolution. For an EDITOR the full-res preview IS the photo
 * (the camera's own 8-bit render) — so we lift it out and hand a plain
 * image/jpeg Blob to the normal ImageKit-upload + Fabric-edit path. No
 * demosaicing, no server, and only the header + the preview's byte range are
 * ever read (a 33 MB NEF → a ~3 MB JPEG). This fast path covers essentially
 * every modern RAW; preview-less containers return null (caller reports it).
 */

const RAW_EXTS = new Set([
    'nef', 'nrw', 'cr2', 'cr3', 'crw', 'arw', 'sr2', 'srf', 'dng', 'orf', 'rw2',
    'raf', 'pef', 'srw', 'raw', 'rwl', 'iiq', '3fr', 'fff', 'erf', 'mrw', 'mef',
    'mos', 'dcr', 'kdc', 'x3f', 'nrf',
])

const HEADER_BYTES = 2 * 1024 * 1024   // IFD tables sit near the front
const SCAN_BYTES = 24 * 1024 * 1024    // fallback scan cap for non-TIFF containers
const MAX_EDGE = 30000                 // reject absurd (false-positive) dims

// `accept` value for file inputs so RAW files aren't greyed out (they often
// carry application/octet-stream, which `image/*` alone excludes).
export const RAW_ACCEPT = [...RAW_EXTS].map((e) => `.${e}`).join(',')
export const IMAGE_UPLOAD_ACCEPT = `image/*,${RAW_ACCEPT}`

export const rawExtOf = (name = '') => (name.split('.').pop() || '').toLowerCase()

/** True for camera RAW files, by extension or vendor mime type. */
export const isRawFile = (file) =>
    !!file && (RAW_EXTS.has(rawExtOf(file.name || ''))
        || /(^|\/)x-(nikon|canon|sony|adobe-dng|fuji|panasonic|olympus)/i.test(file.type || ''))

// ── cheap image-header parse (dims + EXIF orientation), reads only the head ──

const HEAD = 512 * 1024

const exifOrientation = (dv, base) => {
    try {
        const le = dv.getUint16(base) === 0x4949
        const ifd = base + dv.getUint32(base + 4, le)
        const n = dv.getUint16(ifd, le)
        for (let i = 0; i < n; i += 1) {
            const e = ifd + 2 + i * 12
            if (dv.getUint16(e, le) === 0x0112) return dv.getUint16(e + 8, le)
        }
    } catch { /* malformed EXIF → orientation 1 */ }
    return 0
}

const jpegMeta = (dv) => {
    if (dv.getUint16(0) !== 0xffd8) return null // SOI
    let o = 2
    let orientation = 1
    const len = dv.byteLength
    while (o + 4 < len) {
        if (dv.getUint8(o) !== 0xff) { o += 1; continue }
        const marker = dv.getUint8(o + 1)
        if (marker === 0xff || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd9)) { o += 2; continue }
        const seg = dv.getUint16(o + 2)
        if (marker === 0xe1 && o + 10 < len && dv.getUint32(o + 4) === 0x45786966) { // "Exif"
            orientation = exifOrientation(dv, o + 10) || orientation
        }
        const sof = (marker >= 0xc0 && marker <= 0xc3) || (marker >= 0xc5 && marker <= 0xc7)
            || (marker >= 0xc9 && marker <= 0xcb) || (marker >= 0xcd && marker <= 0xcf)
        if (sof && o + 9 < len) return { w: dv.getUint16(o + 7), h: dv.getUint16(o + 5), orientation }
        o += 2 + seg
    }
    return null
}

const pngMeta = (dv) => {
    if (dv.getUint32(0) !== 0x89504e47 || dv.getUint32(4) !== 0x0d0a1a0a) return null
    if (dv.getUint32(12) !== 0x49484452) return null // IHDR
    return { w: dv.getUint32(16), h: dv.getUint32(20), orientation: 1 }
}

const webpMeta = (dv) => {
    if (dv.getUint32(0) !== 0x52494646 || dv.getUint32(8) !== 0x57454250) return null // RIFF…WEBP
    const cc = dv.getUint32(12)
    if (cc === 0x56503858) { // 'VP8X'
        const w = 1 + (dv.getUint8(24) | (dv.getUint8(25) << 8) | (dv.getUint8(26) << 16))
        const h = 1 + (dv.getUint8(27) | (dv.getUint8(28) << 8) | (dv.getUint8(29) << 16))
        return { w, h, orientation: 1 }
    }
    if (cc === 0x56503820 && dv.getUint8(23) === 0x9d && dv.getUint8(24) === 0x01 && dv.getUint8(25) === 0x2a) {
        return { w: dv.getUint16(26, true) & 0x3fff, h: dv.getUint16(28, true) & 0x3fff, orientation: 1 } // 'VP8 '
    }
    if (cc === 0x5650384c && dv.getUint8(20) === 0x2f) { // VP8L
        const b1 = dv.getUint8(21), b2 = dv.getUint8(22), b3 = dv.getUint8(23), b4 = dv.getUint8(24)
        return { w: 1 + (b1 | ((b2 & 0x3f) << 8)), h: 1 + ((b2 >> 6) | (b3 << 2) | ((b4 & 0x0f) << 10)), orientation: 1 }
    }
    return null
}

const gifMeta = (dv) => {
    if (dv.getUint32(0) !== 0x47494638 || (dv.getUint16(4) !== 0x3761 && dv.getUint16(4) !== 0x3961)) return null
    return { w: dv.getUint16(6, true), h: dv.getUint16(8, true), orientation: 1 }
}

const bmpMeta = (dv) => {
    if (dv.getUint16(0) !== 0x424d || dv.byteLength < 26) return null // BM
    const w = Math.abs(dv.getInt32(18, true))
    const h = Math.abs(dv.getInt32(22, true))
    return w && h ? { w, h, orientation: 1 } : null
}

// AVIF/HEIF store dims in an `ispe` box; decline the frugal path if a
// rotation/mirror box is present rather than guess composition.
const isoImageMeta = (dv) => {
    try {
        if (dv.getUint32(4) !== 0x66747970) return null // ftyp
        let spatial = null, transformed = false
        for (let i = 0; i + 20 <= dv.byteLength; i += 1) {
            const type = dv.getUint32(i + 4)
            if (type === 0x69737065) { const w = dv.getUint32(i + 12); const h = dv.getUint32(i + 16); if (w && h) spatial = { w, h } }
            else if (type === 0x69726f74 || type === 0x696d6972) transformed = true // irot / imir
        }
        return spatial && !transformed ? { ...spatial, orientation: 1 } : null
    } catch { return null }
}

const tiffMeta = (dv) => {
    try {
        const le = dv.getUint16(0) === 0x4949
        if (!le && dv.getUint16(0) !== 0x4d4d) return null
        if (dv.getUint16(2, le) !== 0x2a) return null
        const ifd = dv.getUint32(4, le)
        const n = dv.getUint16(ifd, le)
        let w = 0, h = 0, orientation = 1
        for (let i = 0; i < n; i += 1) {
            const e = ifd + 2 + i * 12
            const tag = dv.getUint16(e, le)
            const type = dv.getUint16(e + 2, le)
            const count = dv.getUint32(e + 4, le)
            if (count !== 1) continue
            const value = type === 3 ? dv.getUint16(e + 8, le) : (type === 4 ? dv.getUint32(e + 8, le) : 0)
            if (tag === 0x0100) w = value
            else if (tag === 0x0101) h = value
            else if (tag === 0x0112) orientation = value || orientation
        }
        return w && h ? { w, h, orientation } : null
    } catch { return null }
}

/** { w, h, orientation } in encoded pixels, or null (unsupported header). */
export const readImageMeta = async (blob) => {
    try {
        const dv = new DataView(await blob.slice(0, HEAD).arrayBuffer())
        return jpegMeta(dv) || pngMeta(dv) || webpMeta(dv)
            || gifMeta(dv) || bmpMeta(dv) || tiffMeta(dv) || isoImageMeta(dv)
    } catch { return null }
}

// ── RAW embedded-preview extraction ──

/** Walk TIFF IFD0 + SubIFDs + the IFD chain; collect JPEG preview (offset,len)
 *  pairs from the JPEGInterchangeFormat tags plus the IFD0 orientation. */
const parseTiff = (dv) => {
    if (dv.byteLength < 8) return null
    const le = dv.getUint16(0) === 0x4949
    if (!le && dv.getUint16(0) !== 0x4d4d) return null
    const u16 = (o) => dv.getUint16(o, le)
    const u32 = (o) => dv.getUint32(o, le)
    if (u16(2) !== 0x2a) return null
    const len = dv.byteLength
    const previews = []
    let orientation = 1
    const seen = new Set()
    const walk = (ifd, depth) => {
        if (ifd <= 0 || ifd + 2 > len || seen.has(ifd) || depth > 8) return
        seen.add(ifd)
        const n = u16(ifd)
        if (ifd + 2 + n * 12 + 4 > len) return
        let off = 0, jlen = 0
        const subs = []
        for (let i = 0; i < n; i += 1) {
            const e = ifd + 2 + i * 12
            const tag = u16(e)
            if (tag === 0x0201) off = u32(e + 8)                 // JPEGInterchangeFormat
            else if (tag === 0x0202) jlen = u32(e + 8)           // …Length
            else if (tag === 0x0112) orientation = u16(e + 8) || orientation
            else if (tag === 0x014a) {                            // SubIFDs
                const cnt = u32(e + 4)
                if (cnt === 1) subs.push(u32(e + 8))
                else { const p = u32(e + 8); for (let k = 0; k < cnt && p + 4 * k + 4 <= len; k += 1) subs.push(u32(p + 4 * k)) }
            }
        }
        if (off > 0 && jlen > 0) previews.push({ off, len: jlen })
        for (const s of subs) walk(s, depth + 1)
        walk(u32(ifd + 2 + n * 12), depth + 1)
    }
    walk(u32(4), 0)
    return { previews, orientation }
}

/** SOF dims of a JPEG at `start` (validates scanned candidates). */
const jpegSof = (dv, start) => {
    let o = start + 2
    const len = dv.byteLength
    while (o + 9 < len) {
        if (dv.getUint8(o) !== 0xff) { o += 1; continue }
        const m = dv.getUint8(o + 1)
        if (m === 0xff || m === 0x01 || (m >= 0xd0 && m <= 0xd9)) { o += 2; continue }
        const sof = (m >= 0xc0 && m <= 0xc3) || (m >= 0xc5 && m <= 0xc7) || (m >= 0xc9 && m <= 0xcb) || (m >= 0xcd && m <= 0xcf)
        if (sof) return { w: dv.getUint16(o + 7), h: dv.getUint16(o + 5) }
        o += 2 + dv.getUint16(o + 2)
    }
    return null
}

/** All sane embedded JPEGs in `bytes` (fallback for CR3/RAF and other containers). */
const scanForPreviews = (bytes) => {
    const out = []
    let i = 0
    while (i + 3 < bytes.length) {
        if (bytes[i] === 0xff && bytes[i + 1] === 0xd8 && bytes[i + 2] === 0xff) {
            const sof = jpegSof(new DataView(bytes.buffer, bytes.byteOffset + i, Math.min(bytes.length - i, 262144)), 0)
            if (sof && sof.w > 320 && sof.h > 320 && sof.w < MAX_EDGE && sof.h < MAX_EDGE) {
                let e = i + 2
                while (e + 1 < bytes.length && !(bytes[e] === 0xff && bytes[e + 1] === 0xd9)) e += 1
                if (e + 1 < bytes.length) { out.push({ off: i, len: e + 2 - i, w: sof.w, h: sof.h }); i = e + 2; continue }
            }
        }
        i += 1
    }
    return out
}

const validSoi = async (blob) => {
    const s = new Uint8Array(await blob.slice(0, 3).arrayBuffer())
    return s[0] === 0xff && s[1] === 0xd8 && s[2] === 0xff
}

// Fill in missing SOF dims for TIFF-pointer candidates (read a small window).
const withDims = async (file, cands) => {
    const out = []
    for (const c of cands) {
        if (c.w && c.h) { out.push(c); continue }
        if (c.off <= 0 || c.off + 3 > file.size) continue
        const meta = await readImageMeta(file.slice(c.off, c.off + Math.min(c.len, 262144), 'image/jpeg'))
        if (meta && meta.w > 0) out.push({ ...c, w: meta.w, h: meta.h })
    }
    return out
}

/**
 * Extract the largest embedded JPEG preview from a camera RAW. Returns a plain
 * image/jpeg Blob (the export truth) plus its dimensions and orientation, or
 * null when the container carries no usable preview.
 * @returns {Promise<{ blob: Blob, width: number, height: number, orientation: number } | null>}
 */
export const extractRawPreview = async (file) => {
    let cands = []
    let orientation = 1
    try {
        const head = new DataView(await file.slice(0, HEADER_BYTES).arrayBuffer())
        const tiff = parseTiff(head)
        if (tiff) { cands = tiff.previews; orientation = tiff.orientation }
    } catch { /* not TIFF → scan */ }
    if (!cands.length) {
        try {
            const bytes = new Uint8Array(await file.slice(0, Math.min(file.size, SCAN_BYTES)).arrayBuffer())
            cands = scanForPreviews(bytes)
        } catch { /* give up below */ }
    }

    const sized = await withDims(file, cands)
    if (!sized.length) return null
    const full = sized.reduce((a, b) => (b.w * b.h > a.w * a.h ? b : a))
    if (full.off + full.len > file.size) return null

    const blob = file.slice(full.off, full.off + full.len, 'image/jpeg')
    if (!(await validSoi(blob))) return null
    return { blob, width: full.w, height: full.h, orientation }
}
