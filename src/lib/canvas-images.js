import { FabricImage } from 'fabric'
import { toast } from 'sonner'
import { stripImageMetadata } from '@/lib/strip-metadata'
import { isRawFile, extractRawPreview, readImageMeta } from '@/lib/raw-preview'

const CASCADE_OFFSET = 32

// Reads a File/Blob as a data URL. Kept as a last-resort fallback for the rare
// case where the ImageKit upload fails — at least the image stays usable in the
// current session. Data URLs balloon the saved canvas state, so we avoid them
// when we can (Neon documents are capped at 1 MB).
const readFileAsDataURL = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result || ''))
    reader.onerror = () => reject(reader.error || new Error('FileReader failed'))
    reader.readAsDataURL(file)
  })

// ImageKit rejects images above 25 MP on serving ("ELIMIT"). We downscale
// anything above 24 MP (safety margin) through an offscreen canvas so the
// uploaded image is always servable.
const IMAGEKIT_MAX_MP = 24_000_000
const MAX_EDGE = 8192

// Encode a resized bitmap to a Blob (OffscreenCanvas off the main thread where
// available, else a DOM canvas). Returns null on failure.
const encodeResized = async (bitmap, nw, nh, type) => {
  const canvas = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(nw, nh)
    : Object.assign(document.createElement('canvas'), { width: nw, height: nh })
  const ctx = canvas.getContext('2d')
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(bitmap, 0, 0, nw, nh)
  if (canvas.convertToBlob) return canvas.convertToBlob({ type, quality: 0.92 })
  return new Promise((res) => canvas.toBlob(res, type, 0.92))
}

// Downscale anything above ImageKit's serving limits. Reads dimensions from the
// header (512 KB) and resizes via createImageBitmap — a hardware-decoded,
// off-main-thread path — so a 50 MP DSLR frame never freezes the tab the way a
// synchronous <img> decode + canvas draw did.
const downscaleIfNeeded = async (file) => {
  if (!file?.type?.startsWith('image/')) return file // canvas blobs are pre-sized

  const meta = await readImageMeta(file)
  let w = meta?.w || 0
  let h = meta?.h || 0
  let probe = null
  if (!w || !h) {
    // Unknown header — decode once off-thread to learn the size.
    probe = await createImageBitmap(file).catch(() => null)
    if (!probe) return file
    w = probe.width
    h = probe.height
  }

  if (w * h <= IMAGEKIT_MAX_MP && w <= MAX_EDGE && h <= MAX_EDGE) {
    probe?.close?.()
    return file // within limits — use original
  }

  // Target dims (proportional)
  let nw = w, nh = h
  if (nw > MAX_EDGE || nh > MAX_EDGE) {
    const s = MAX_EDGE / Math.max(nw, nh)
    nw = Math.round(nw * s); nh = Math.round(nh * s)
  }
  if (nw * nh > IMAGEKIT_MAX_MP) {
    const s = Math.sqrt(IMAGEKIT_MAX_MP / (nw * nh))
    nw = Math.round(nw * s); nh = Math.round(nh * s)
  }

  try {
    // Explicit dest dims keep it correct where resizeWidth is ignored (older Safari).
    const bitmap = probe || await createImageBitmap(file, { resizeWidth: nw, resizeHeight: nh, resizeQuality: 'high' })
    const type = file.type === 'image/png' ? 'image/png' : 'image/jpeg'
    const blob = await encodeResized(bitmap, nw, nh, type)
    bitmap.close?.()
    if (!blob) return file
    return new File([blob], file.name, { type: blob.type, lastModified: Date.now() })
  } catch {
    probe?.close?.()
    return file // fall back to original — ImageKit may reject, but no freeze
  }
}

// Uploads to our /api/imagekit/upload endpoint (auth-gated) and returns the CDN URL.
// This is the path that keeps saved canvas state small enough for Neon's per-doc
// size limit when users add several photos to one project.
const uploadFileToImageKit = async (file) => {
  // Strip EXIF, GPS, XMP, IPTC — binary-level, no re-encoding
  const cleanFile = await stripImageMetadata(file)
  // Downscale if the image exceeds ImageKit's 25 MP serving limit
  const readyFile = await downscaleIfNeeded(cleanFile)
  const formData = new FormData()
  formData.append('file', readyFile)
  formData.append('fileName', file.name || 'upload')
  const response = await fetch('/api/imagekit/upload', {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    throw new Error(`ImageKit upload failed: ${response.status}`)
  }
  const data = await response.json()
  if (!data?.success || !data?.url) {
    throw new Error(data?.error || 'ImageKit upload returned no URL')
  }
  return data.url
}

// Uploads a raw image Blob (e.g. a flattened merge) to ImageKit and returns the
// CDN URL. Throws on failure so callers can fall back to a data URL.
export const uploadImageBlobToImageKit = async (blob, fileName = 'image.png') => {
  if (!blob) throw new Error('No blob to upload')
  // Strip any metadata the browser may have embedded
  const blobFile = blob instanceof File ? blob : new File([blob], fileName, { type: blob.type })
  const cleanBlob = await stripImageMetadata(blobFile)
  const formData = new FormData()
  formData.append('file', cleanBlob, fileName)
  formData.append('fileName', fileName)
  const response = await fetch('/api/imagekit/upload', { method: 'POST', body: formData })
  if (!response.ok) throw new Error(`ImageKit upload failed: ${response.status}`)
  const data = await response.json()
  if (!data?.success || !data?.url) throw new Error(data?.error || 'ImageKit upload returned no URL')
  return data.url
}

// Builds a Fabric image from a URL with an OFF-MAIN-THREAD decode. Fabric's own
// loadImage only waits on img.onload, so the JPEG decode lands on first draw and
// freezes the UI for a 12-50 MP DSLR. HTMLImageElement.decode() runs the decode
// on a background thread; awaiting it means the bitmap is ready before render.
// We keep el.src = url (not createImageBitmap) so getSrc() persists the URL —
// undo/redo and reload recreate the image from serialized src.
export const fabricImageFromUrl = async (url) => {
  try {
    const el = new Image()
    el.crossOrigin = 'anonymous'
    el.decoding = 'async'
    el.src = url
    await el.decode()
    return new FabricImage(el)
  } catch {
    // decode() can reject (some data: URLs / CORS / older engines) — fall back
    // to Fabric's own loader, which is functionally identical, just main-thread.
    return FabricImage.fromURL(url, { crossOrigin: 'anonymous' })
  }
}

export const loadFabricImageFromFile = async (file, { silent = false } = {}) =>
  loadFabricImage(file, { silent })

// Bake an EXIF orientation into pixels. A RAW's rotation lives in the container
// (IFD0), NOT in the extracted preview's own EXIF, and we strip metadata before
// upload — so a portrait shot would arrive sideways unless we rotate it here.
const bakeOrientation = async (blob, orientation) => {
  if (!orientation || orientation === 1) return blob
  const bmp = await createImageBitmap(blob).catch(() => null)
  if (!bmp) return blob
  const w = bmp.width, h = bmp.height
  const swap = orientation >= 5 && orientation <= 8
  const cw = swap ? h : w
  const ch = swap ? w : h
  const canvas = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(cw, ch)
    : Object.assign(document.createElement('canvas'), { width: cw, height: ch })
  canvas.width = cw; canvas.height = ch
  const ctx = canvas.getContext('2d')
  // Canonical EXIF-orientation canvas transforms (w/h are pre-rotation dims).
  switch (orientation) {
    case 2: ctx.transform(-1, 0, 0, 1, w, 0); break   // flip horizontal
    case 3: ctx.transform(-1, 0, 0, -1, w, h); break  // rotate 180
    case 4: ctx.transform(1, 0, 0, -1, 0, h); break   // flip vertical
    case 5: ctx.transform(0, 1, 1, 0, 0, 0); break    // transpose
    case 6: ctx.transform(0, 1, -1, 0, h, 0); break   // rotate 90 CW
    case 7: ctx.transform(0, -1, -1, 0, h, w); break  // transverse
    case 8: ctx.transform(0, -1, 1, 0, 0, w); break   // rotate 90 CCW
    default: break
  }
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(bmp, 0, 0)
  bmp.close?.()
  const out = canvas.convertToBlob
    ? await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.95 })
    : await new Promise((r) => canvas.toBlob(r, 'image/jpeg', 0.95))
  return out || blob
}

// Camera RAW is a container: lift out its full-res embedded JPEG preview (the
// camera's own render) and edit THAT — a normal image/jpeg the rest of the
// pipeline uploads, grades, serializes and restores like any photo. Throws
// RAW_NO_PREVIEW when the container has no usable preview.
const resolveSourceFile = async (file) => {
  if (!isRawFile(file)) return file
  const preview = await extractRawPreview(file).catch(() => null)
  if (!preview?.blob) throw new Error('RAW_NO_PREVIEW')
  const upright = await bakeOrientation(preview.blob, preview.orientation).catch(() => preview.blob)
  const base = (file.name || 'photo').replace(/\.[^.]+$/, '')
  return new File([upright], `${base}.jpg`, { type: 'image/jpeg', lastModified: Date.now() })
}

const loadFabricImage = async (file, { silent }) => {
  const sourceFile = await resolveSourceFile(file)
  // Try ImageKit first — small URL, persistent, CDN-served.
  try {
    const url = await uploadFileToImageKit(sourceFile)
    return await fabricImageFromUrl(url)
  } catch (uploadError) {
    console.warn('[canvas-images] ImageKit upload failed, falling back to data URL:', uploadError)
    if (!silent) {
      toast.warning('Upload service unavailable — image saved locally; refresh may not restore it.')
    }
    const dataUrl = await readFileAsDataURL(sourceFile)
    return await fabricImageFromUrl(dataUrl)
  }
}

const countExistingImages = (canvasEditor) => {
  if (!canvasEditor?.getObjects) return 0
  return canvasEditor
    .getObjects()
    .filter((obj) => obj?.type?.toLowerCase() === 'image').length
}

export const fitNewImageToProject = (fabricImage, projectSize, options = {}) => {
  const pW = Math.max(1, projectSize?.width || 800)
  const pH = Math.max(1, projectSize?.height || 600)
  const iW = Math.max(1, fabricImage.width || 1)
  const iH = Math.max(1, fabricImage.height || 1)
  const scale = Math.min((pW * 0.6) / iW, (pH * 0.6) / iH, 1)
  const stackIndex = Math.max(0, Number(options.stackIndex) || 0)
  const offset = stackIndex * CASCADE_OFFSET

  fabricImage.set({
    left: pW / 2 + offset,
    top: pH / 2 + offset,
    originX: 'center',
    originY: 'center',
    scaleX: scale,
    scaleY: scale,
    selectable: true,
    evented: true,
  })
  fabricImage.setCoords()
}

/**
 * Add an image file to the Fabric canvas (used by topbar upload, drop, paste).
 * Pass { silent: true } when adding many images in a batch — only the batch
 * caller should push history + save.
 */
export async function addImageFileToCanvas(canvasEditor, file, project, options = {}) {
  if (!canvasEditor || !file) return false

  const raw = isRawFile(file)
  if (!file.type.startsWith('image/') && !raw) {
    toast.error('Only image files are supported')
    return false
  }
  // A RAW container is large (20–60 MB), but only its small embedded preview is
  // ever read/uploaded — so the 25 MB cap applies to standard images only. Guard
  // RAW with a generous ceiling against pathological files.
  if (!raw && file.size > 25 * 1024 * 1024) {
    toast.error('Image must be under 25 MB')
    return false
  }
  if (raw && file.size > 200 * 1024 * 1024) {
    toast.error('RAW file is too large')
    return false
  }

  const { silent = false, stackIndex } = options
  const toastId = silent ? null : toast.loading('Adding image...')
  try {
    const img = await loadFabricImage(file, { silent })
    const resolvedStackIndex =
      typeof stackIndex === 'number' ? stackIndex : countExistingImages(canvasEditor)
    fitNewImageToProject(img, project, { stackIndex: resolvedStackIndex })
    canvasEditor.add(img)
    canvasEditor.setActiveObject(img)
    canvasEditor.requestRenderAll()
    if (!silent) {
      canvasEditor.__pushHistoryState?.({ label: 'Added image', domain: 'images' })
      canvasEditor.__saveCanvasState?.()
      toast.success('Image added', { id: toastId })
    }
    return img
  } catch (err) {
    const msg = err?.message === 'RAW_NO_PREVIEW'
      ? 'This RAW has no embedded preview to import'
      : 'Failed to load image'
    if (toastId) toast.error(msg, { id: toastId })
    else toast.error(msg)
    console.error('[canvas-images] Load error:', err)
    return false
  }
}

/**
 * Add many image files at once. Adds them sequentially with a cascade offset
 * and only pushes a single history state at the end.
 */
export async function addImageFilesToCanvas(canvasEditor, files, project) {
  if (!canvasEditor || !files?.length) return 0
  const baseIndex = countExistingImages(canvasEditor)
  const toastId = toast.loading(
    files.length === 1 ? 'Adding image...' : `Adding ${files.length} images...`
  )
  let added = 0
  for (let i = 0; i < files.length; i++) {
    const result = await addImageFileToCanvas(canvasEditor, files[i], project, {
      silent: true,
      stackIndex: baseIndex + i,
    })
    if (result) added += 1
  }
  if (added > 0) {
    canvasEditor.__pushHistoryState?.({ label: 'Added images', domain: 'images', detail: `${added} images` })
    canvasEditor.__saveCanvasState?.()
  }
  if (added === files.length) {
    toast.success(added === 1 ? 'Image added' : `${added} images added`, { id: toastId })
  } else if (added > 0) {
    toast.warning(`Added ${added} of ${files.length} images`, { id: toastId })
  } else {
    toast.error('No images were added', { id: toastId })
  }
  return added
}
