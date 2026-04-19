"use client"

import { useParams } from 'next/navigation'
import React, { useState } from 'react'
import { useConvexAuth } from 'convex/react'
import { CanvasContext } from '../../../../../context/context'
import { Loader2, Monitor } from 'lucide-react'
import { useConvexQuery } from '../../../../../hooks/useConvexQuery'
import { api } from '../../../../../convex/_generated/api'
import { RingLoader } from 'react-spinners'
import CanvasEditor from './_components/canvas'

const Editor = () => {

    const params = useParams()
    const projectId = params.projectId
    const { isLoading: isAuthLoading, isAuthenticated } = useConvexAuth()

    const [canvasEditor, setCanvasEditor] = useState(null)
    const [processingMessage, setProcessingMessage] = useState(null)

    const [activeTool, setActiveTool] = useState("resize")

    const {
        data: project,
        isLoading: isProjectLoading,
        error,
    } = useConvexQuery(
        api.projects.getProject,
        isAuthenticated ? { projectId } : "skip",
    )
    const isLoading = isAuthLoading || isProjectLoading

    if (isLoading) {
        return (
            <div className='min-h-screen bg-slate-900 flex items-center justify-center'>
                <div className='flex flex-col items-center gap-4'>
                    <Loader2 className='h-8 w-8 animate-spin text-cyan-400' />
                    <p className='text-white/70'>
                        Loading...
                    </p>
                </div>
            </div>
        )
    }

    if (error || !project) {
        return (
            <div className='min-h-screen bg-slate-900 flex items-center justify-center'>
                <div className='text-center'>
                    <h1 className='text-2xl font-bold text-white mb-2'>
                        Project Not Found
                    </h1>
                    <p className='text-white/70'>
                        You may not have access or the project does not exist
                    </p>
                </div>
            </div>
        )
    }

    return (
        <CanvasContext.Provider value={{
            canvasEditor,
            setCanvasEditor,
            activeTool,
            onToolChange: setActiveTool,
            processingMessage,
            setProcessingMessage
        }}>
            <div className='lg:hidden min-h-screen bg-slate-900 flex items-center justify-center p-6'>
                <div className='text-center max-w-md'>
                    <Monitor className='h-16 w-16 text-cyan-400 mx-auto mb-6' />

                    <h1 className='text-2xl font-bold text-white mb-4 text-center'>
                        Desktop Required
                    </h1>

                    <p className='text-white/50 text-sm'>
                        Please use a larger screen to access the full editing experience.
                    </p>
                </div>
            </div>

            <div className='hidden lg:block min-h-screen bg-slate-900'>

                <div className='flex flex-col h-screen'>
                    {processingMessage &&
                        <div className='fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center'>
                            <div className='rounded-lg p-6 flex flex-col items-center gap-4'>
                                <RingLoader color="#fff" />
                                <div className='text-center'>
                                    <p className='text-white font-medium'>
                                        {processingMessage}
                                    </p>
                                    <p className='text-white/70 text-sm mt-1'>
                                        Please wait...
                                    </p>
                                </div>
                            </div>
                        </div>
                    }

                    {/* Top Bar Component */}
                    <div className='flex flex-1 overflow-hidden'>
                        {/* Sidebar Component */}
                        <div className='flex-1 bg-slate-800'>
                            {/* Canvas Component */}
                            <CanvasEditor project={project} />
                        </div>
                    </div>
                </div>
            </div>
        </CanvasContext.Provider>
    )
}

export default Editor
