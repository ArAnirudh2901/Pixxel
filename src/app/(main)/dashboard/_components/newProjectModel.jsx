import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Alert, AlertAction, AlertDescription, AlertTitle } from '@/components/ui/alert'
import React, { useState } from 'react'
import usePlanAccess from '../../../../../hooks/usePlanAccess'
import { useConvexMutation } from '../../../../../hooks/useConvexQuery'
import { api } from '../../../../../convex/_generated/api'
import { Crown, ImageIcon, Loader2, Upload, X } from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { toast } from 'sonner'
import { useRouter } from 'next/navigation'
import UpgradeModel from '@/components/upgradeModel'

const NewProjectModel = ({ isOpen, onClose, currentProjectCount = 0 }) => {

    const [isUploading, setIsUploading] = useState(false)
    const [projectTitle, setProjectTitle] = useState("")
    const [selectedFile, setSelectedFile] = useState(null)
    const [previewUrl, setPreviewUrl] = useState(null)
    const [showUpgradeModel, setShowUpgradeModel] = useState(false)
    const router = useRouter()

    const { isFree, canCreateProject } = usePlanAccess()
    const canCreate = canCreateProject(currentProjectCount)

    const { mutate: createProject } = useConvexMutation(api.projects.create)

    const clearSelectedFile = () => {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl)
        }

        setSelectedFile(null)
        setPreviewUrl(null)
        setIsUploading(false)
        setProjectTitle("")
    }

    const onDrop = (acceptedFiles) => {
        const file = acceptedFiles[0]

        if (file) {
            setSelectedFile(file)
            setPreviewUrl(URL.createObjectURL(file))

            const nameWithoutExtension = file.name.replace(/\.[^/.]+$/, "")
            setProjectTitle(nameWithoutExtension || "Untitled Project")
        }
    }

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            "image/*": [".png", ".jpg", ".webp", ".jpeg", ".gif", ".svg", ".avif", ".apng"]
        },
        maxFiles: 1,
        maxSize: 20 * 1024 * 1024,   // 20mb file size limit
        disabled: !canCreate || isUploading,
    })

    const handleCreateProject = async () => {
        if (!canCreate) {
            setShowUpgradeModel(true)
            return
        }

        if (!selectedFile || !projectTitle.trim()) {
            toast.error("Please select an image or enter a project title")
            return
        }

        setIsUploading(true)

        try {
            const formData = new FormData()
            formData.append("file", selectedFile)
            formData.append("fileName", selectedFile.name)

            const uploadResponse = await fetch("/api/imagekit/upload", {
                method: "POST",
                body: formData
            })

            const uploadData = await uploadResponse.json()

            if (!uploadData.success)
                throw new Error(uploadData.error || "Failed to upload the image")

            // Creating a project in Convex
            const projectId = await createProject({
                title: projectTitle.trim(),
                originalImageUrl: uploadData.url,
                currentImageUrl: uploadData.url,
                thumbnailUrl: uploadData.thumbnailUrl,
                width: uploadData.width || 800,
                height: uploadData.height || 600,
                canvasState: null,
            })

            if (!projectId) {
                return
            }

            toast.success("Project created successfully")
            router.push(`/editor/${projectId}`)

        } catch (error) {
            console.error("Error creating project:", error)
            toast.error(
                error.message || "Failed to create project. Please try again."
            )
        } finally {
            setIsUploading(false)
        }
    }

    return (
        <>

            <Dialog open={isOpen} onOpenChange={(open) => {
                if (!open) {
                    onClose()
                }
            }}>
                <DialogContent className="sm:max-w-2xl">
                    <DialogHeader>
                        <DialogTitle className="text-2xl font-bold text-white">
                            Create New Project
                        </DialogTitle>
                        <DialogDescription className="text-slate-300">
                            Start a fresh canvas from your dashboard. Free accounts can keep up to 3 projects at a time.
                        </DialogDescription>
                    </DialogHeader>

                    <div className='space-y-6'>
                        {isFree && currentProjectCount >= 2 &&
                            (<Alert className='bg-amber-500/10 border-amber-500/20'>
                                <Crown className='h-5 w-5 text-amber-400' />
                                <AlertDescription className="text-amber-400/80">
                                    <div className="font-semibold text-amber-400 mb-1">
                                        {currentProjectCount === 2
                                            ? "Last Free Project"
                                            : "Project Limit Reached"
                                        }
                                    </div>
                                </AlertDescription>
                                {!canCreate && (
                                    <AlertAction>
                                        <Button variant="outline" onClick={onClose}>Close</Button>
                                    </AlertAction>
                                )}
                            </Alert>)}

                        {/* Area for uploading images */}
                        {!selectedFile
                            ? <div
                                {...getRootProps()}
                                className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${isDragActive
                                    ? "border-cyan-400 bg-cyan-400/5"
                                    : "border-white/20 hover:border-white/40"
                                    } ${!canCreate ? "opacity-50 pointer-events-none" : ""}`}
                            >
                                <input {...getInputProps()} />

                                <Upload className='h-12 w-12 text-white/50 mx-auto mb-4' />

                                <h3 className='text-xl font-semibold text-white mb-2'>
                                    {isDragActive ? "Drop your image here" : "Upload an Image"}
                                </h3>

                                <p className='mb-4 whitespace-nowrap text-sm text-white/70'>
                                    {canCreate
                                        ? "Drag and drop your image, or click to browse"
                                        : "Upgrade to Pro to create more projects"}
                                </p>{" "}

                                <p className='whitespace-nowrap text-xs text-white/50'>
                                    Supports all the image formats upto 20MB
                                </p>
                            </div>
                            : <div className='space-y-6'>
                                <div className='relative overflow-hidden rounded-xl border border-white/10 bg-[radial-gradient(circle_at_top,rgba(56,189,248,0.14),transparent_42%),linear-gradient(160deg,rgba(15,23,42,0.92),rgba(15,23,42,0.74))]'>
                                    <img
                                        className='h-64 w-full object-contain p-3'
                                        src={previewUrl}
                                        alt="Uploaded image preview" />

                                    <Button
                                        className='absolute top-2 right-2 bg-black/50 text-white hover:bg-black/100 rounded-md'
                                        variant='ghost'
                                        size='sm'
                                        onClick={clearSelectedFile}
                                    >
                                        <X className='h-4 w-4' />
                                    </Button>
                                </div>
                                <div className='space-y-2'>
                                    <Label htmlFor="project-title" className="text-white">
                                        Project Title
                                    </Label>
                                    <Input
                                        id="project-title"
                                        type="text"
                                        value={projectTitle}
                                        onChange={(e) => setProjectTitle(e.target.value)}
                                        placeholder="Enter project name..."
                                        className={"bg-slate-700 border-white/20 text-white placeholder-white/50 focus:border-cyan-400 focus:ring-cyan-400"}
                                    >
                                    </Input>
                                </div>
                                <div className='bg-slate-700/50 rounded-lg p-4'>
                                    <div className='flex items-center gap-3'>
                                        <ImageIcon className='h-5 w-5 text-cyan-400' />
                                        <div>
                                            <p className='text-white font-medium'>
                                                {selectedFile.name}
                                            </p>
                                            <p className='text-white/70 text-sm'>
                                                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        }

                    </div>

                    {isFree && (
                        <Badge variant='secondary' className="bg-slate-700 text-white/70">
                            {currentProjectCount}/3 projects
                        </Badge>
                    )}
                    <DialogFooter>
                        <Button
                            className="text-white/70 hover:text-white"
                            variant="ghost"
                            onClick={onClose}
                            disabled={isUploading}
                        >
                            Cancel
                        </Button>

                        <Button
                            variant="primary"
                            onClick={handleCreateProject}
                            disabled={!selectedFile || !projectTitle.trim() || isUploading}
                        >
                            {isUploading
                                ? (
                                    <>
                                        <Loader2 className='h-4 w-4 animate-spin' />
                                    </>
                                ) : (
                                    "Create Project"
                                )
                            }
                        </Button>

                    </DialogFooter>
                </DialogContent>
            </Dialog>

            <UpgradeModel
                isOpen={showUpgradeModel}
                onClose={() => { setShowUpgradeModel(false) }}
                restrictedTool="projects"
                reason="Free plan is limited to 3 projects. Upgrade to Pro 
                for unlimited projects and access to all AI editing tools."
            />
        </>
    )
}

export default NewProjectModel
