"use client"

import React, { useEffect, useState } from "react"
import dynamic from "next/dynamic"
import { CanvasContext } from "../../../../../../context/context"
import { Bot, Crop, Eraser, Expand, Eye, ImagePlus, Maximize2, Palette, Pen, Scissors, Sliders, Text, LayoutGrid, AudioLines } from "lucide-react"
import { extractDominantColors, getContrastingColor, adjustColorBrightness } from "@/lib/color-extraction"
// Mask + Erase lock canvas interaction synchronously on mount (via usePixelMaskTool):
// they disable selection, swap to a crosshair, and attach the brush cursor. Lazy-loading
// them would leave the canvas selectable during the chunk fetch, so an early drag could
// move the image instead of painting. Keep these two eager (they're small); the heavy
// panels below stay split out.
import MaskControls from "./tools/mask"
import EraseControls from "./tools/erase"
import HistoryPanel from "./history-panel"

// Lazy-load each tool panel so the editor's initial bundle stays small — only the
// active tool's code is fetched (on first use, then cached). The heaviest panels
// (the AI agent, the adjust/curves panel) are split into their own chunks instead
// of shipping with the editor shell. A lightweight skeleton fills the panel during
// the chunk load. ssr:false because every panel is a client-only canvas tool.
const PanelLoading = () => (
    <div className="space-y-3 p-1" aria-busy="true" aria-label="Loading tool">
        <div className="h-8 rounded-lg animate-pulse" style={{ background: "var(--bg-elevated)" }} />
        <div className="h-24 rounded-lg animate-pulse" style={{ background: "var(--bg-elevated)" }} />
        <div className="h-8 rounded-lg animate-pulse" style={{ background: "var(--bg-elevated)" }} />
        <div className="h-8 w-2/3 rounded-lg animate-pulse" style={{ background: "var(--bg-elevated)" }} />
    </div>
)
const lazyTool = (loader) => dynamic(loader, { ssr: false, loading: PanelLoading })

const TextControls = lazyTool(() => import("./tools/text"))
const ResizeControls = lazyTool(() => import("./tools/resize"))
const CropContent = lazyTool(() => import("./tools/crop"))
const AdjustControls = lazyTool(() => import("./tools/adjust"))
const BackgroundControls = lazyTool(() => import("./tools/ai-background"))
const AIExtender = lazyTool(() => import("./tools/ai-extender"))
const AIEdits = lazyTool(() => import("./tools/ai-edit"))
const ImageKitAgent = lazyTool(() => import("./tools/imagekit-agent"))
const ImageManager = lazyTool(() => import("./tools/images"))
const DrawControls = lazyTool(() => import("./tools/draw"))
const CollageControls = lazyTool(() => import("./tools/collage"))
const PixelStretchControls = lazyTool(() => import("./tools/pixel-stretch"))

const TOOL_CONFIGS = {
    resize: { title: "Resize", icon: Expand },
    crop: { title: "Crop", icon: Crop },
    images: { title: "Images", icon: ImagePlus },
    adjust: { title: "Adjust", icon: Sliders },
    draw: { title: "Draw", icon: Pen },
    erase: { title: "Erase", icon: Eraser },
    mask: { title: "Mask", icon: Scissors },
    text: { title: "Text", icon: Text },
    pixel_stretch: { title: "Pixel Stretch", icon: AudioLines },
    ai_background: { title: "AI Background", icon: Palette },
    ai_extender: { title: "AI Extender", icon: Maximize2 },
    ai_edit: { title: "AI Edit", icon: Eye },
    ai_agent: { title: "ImageKit Agent", icon: Bot },
    collage: { title: "Collage", icon: LayoutGrid },
}

const ALL_TOOL_CONFIGS = { ...TOOL_CONFIGS }

export default function EditorSidebar({ project: projectProp, width }) {
    const { activeTool } = React.useContext(CanvasContext)
    const project = projectProp
    const [dominantColor, setDominantColor] = useState('#53D8FF')
    const [contrastingColor, setContrastingColor] = useState('#000000')
    const [lighterColor, setLighterColor] = useState('#9BF95B')

    useEffect(() => {
        if (!project?.currentImageUrl && !project?.originalImageUrl) return

        const imageUrl = project.currentImageUrl || project.originalImageUrl
        extractDominantColors(imageUrl, 1).then(colors => {
            if (colors.length > 0) {
                const primary = colors[0]
                setDominantColor(primary.hex)
                setContrastingColor(getContrastingColor(primary.r, primary.g, primary.b))
                setLighterColor(adjustColorBrightness(primary.hex, 40))
            }
        }).catch(err => {
            console.warn('Color extraction failed:', err)
        })
    }, [project?.currentImageUrl, project?.originalImageUrl])

    if (!activeTool) return (
        <div className="flex flex-col h-full">
            <div className="flex flex-col items-center justify-center flex-1 text-center p-6" style={{ color: 'var(--text-muted)' }}>
                <div className="text-4xl mb-4 opacity-50">🖌️</div>
                <p className="text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>Select a tool to begin</p>
                <p className="text-[11px]">Choose from the toolbar above to start editing</p>
            </div>
            <HistoryPanel />
        </div>
    )

    const config = ALL_TOOL_CONFIGS[activeTool]
    if (!config) return null
    const Icon = config.icon

    const renderContent = () => {
        const colorProps = { dominantColor, contrastingColor, lighterColor }
        switch (activeTool) {
            case "crop": return <CropContent {...colorProps} />
            case "resize": return project ? <ResizeControls project={project} {...colorProps} /> : <div>Loading...</div>
            case "images": return project ? <ImageManager project={project} {...colorProps} /> : <div>Loading...</div>
            case "adjust": return <AdjustControls {...colorProps} />
            case "draw": return <DrawControls {...colorProps} />
            case "erase": return <EraseControls project={project} {...colorProps} />
            case "mask": return <MaskControls {...colorProps} />
            case "text": return <TextControls {...colorProps} />
            case "pixel_stretch": return <PixelStretchControls {...colorProps} />
            case "ai_background": return project ? <BackgroundControls project={project} {...colorProps} /> : <div>Loading...</div>
            case "ai_extender": return project ? <AIExtender project={project} {...colorProps} /> : <div>Loading...</div>
            case "ai_edit": return project ? <AIEdits project={project} {...colorProps} /> : <div>Loading...</div>
            case "ai_agent": return project ? <ImageKitAgent project={project} {...colorProps} /> : <div>Loading...</div>
            case "collage": return project ? <CollageControls project={project} {...colorProps} /> : <div>Loading...</div>
            default: return <div>Tool not available</div>
        }
    }

    // The AI Agent panel owns its own header (brand mark + status + actions),
    // so showing the sidebar's generic "<Tool> · Properties" header on top of
    // it makes the panel look stacked and cluttered. Hide the sidebar header
    // for that one tool; all other tools keep it. The agent also handles its
    // own internal scrolling (chat log scrolls independently of the composer),
    // so the wrapper must NOT scroll/overflow — otherwise its height becomes
    // unbounded and the composer gets pushed off-screen.
    const showSidebarHeader = activeTool !== "ai_agent"
    const contentClass = activeTool === "ai_agent"
        ? "flex-1 min-h-0 flex flex-col overflow-hidden"
        : "panel-scroll flex-1 overflow-y-auto p-4 min-h-0"

    return (
        <aside
            className="editor-sidebar h-full flex flex-col"
            style={width ? { "--editor-sidebar-width": `${width}px` } : undefined}
        >
            {showSidebarHeader && (
                <div className="editor-sidebar-header flex items-center gap-3">
                    <div
                        className="editor-tool-emblem flex items-center justify-center"
                        style={{
                            background: `linear-gradient(180deg, ${lighterColor}, ${dominantColor})`,
                            borderColor: dominantColor,
                            color: contrastingColor,
                        }}
                    >
                        <Icon className="h-4 w-4" />
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{config.title}</h3>
                        <p className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>Properties</p>
                    </div>
                </div>
            )}
            <div className={contentClass}>
                {renderContent()}
            </div>
            {/* The agent panel owns its full height (chat log + composer +
                resizable edits dock) and has its own thread history — pinning
                the journal footer under it squeezes the composer. */}
            {activeTool !== "ai_agent" && <HistoryPanel />}
        </aside>
    )
}
