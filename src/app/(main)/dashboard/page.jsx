"use client"

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useConvexAuth } from 'convex/react'
import Link from 'next/link'
import { api } from '../../../../convex/_generated/api'
import { useConvexMutation, useConvexQuery } from '../../../../hooks/useConvexQuery'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'
import { Calendar, Check, ImageIcon, Loader2, Plus, Trash2, Undo2, X } from 'lucide-react'
import NewProjectModel from './_components/newProjectModel'
import { FastAverageColor } from 'fast-average-color'
import { motion } from 'framer-motion'
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from '@/components/ui/alert-dialog'

const fac = new FastAverageColor()

const loadingCards = Array.from({ length: 6 })



const formatRelativeTime = (timestamp) => {
    if (!timestamp) {
        return 'just now'
    }

    const elapsedMs = timestamp - Date.now()
    const relativeTimeFormat = new Intl.RelativeTimeFormat('en', {
        numeric: 'auto',
    })

    const units = [
        ['year', 1000 * 60 * 60 * 24 * 365],
        ['month', 1000 * 60 * 60 * 24 * 30],
        ['week', 1000 * 60 * 60 * 24 * 7],
        ['day', 1000 * 60 * 60 * 24],
        ['hour', 1000 * 60 * 60],
        ['minute', 1000 * 60],
    ]

    for (const [unit, unitMs] of units) {
        if (Math.abs(elapsedMs) >= unitMs) {
            return relativeTimeFormat.format(
                Math.round(elapsedMs / unitMs),
                unit,
            )
        }
    }

    return 'just now'
}

const getProjectPreview = (project) =>
    project.thumbnailUrl || project.currentImageUrl || project.originalImageUrl

const DEFAULT_ACCENT = { r: 56, g: 189, b: 248, isDark: true } // cyan fallback

const ProjectCard = ({
    project,
    index,
    isSelectionMode,
    isSelected,
    isDeletingThisProject,
    isBulkDeleting,
    onSelect,
    onDelete,
}) => {
    const previewUrl = getProjectPreview(project)
    const [accentState, setAccentState] = useState({
        previewUrl: null,
        color: DEFAULT_ACCENT,
    })

    useEffect(() => {
        if (!previewUrl) {
            return
        }

        let isCancelled = false

        fac.getColorAsync(previewUrl, { algorithm: 'simple', crossOrigin: 'anonymous' })
            .then((color) => {
                if (!isCancelled) {
                    setAccentState({
                        previewUrl,
                        color: {
                            r: color.value[0],
                            g: color.value[1],
                            b: color.value[2],
                            isDark: color.isDark,
                        },
                    })
                }
            })
            .catch(() => {
                if (!isCancelled) {
                    setAccentState({
                        previewUrl,
                        color: DEFAULT_ACCENT,
                    })
                }
            })

        return () => { isCancelled = true }
    }, [previewUrl])

    const accent = accentState.previewUrl === previewUrl ? accentState.color : DEFAULT_ACCENT
    const accentRgb = `${accent.r}, ${accent.g}, ${accent.b}`
    const selectionTintOpacity = accent.isDark ? 0.34 : 0.24
    const selectionGlassStyle = {
        background: `linear-gradient(135deg, rgba(${accentRgb}, ${selectionTintOpacity}) 0%, rgba(${accentRgb}, ${Math.max(selectionTintOpacity - 0.12, 0.12)}) 100%), radial-gradient(circle at top left, rgba(255,255,255, ${accent.isDark ? 0.14 : 0.22}) 0%, transparent 42%)`,
        border: `1px solid rgba(255,255,255, ${accent.isDark ? 0.18 : 0.28})`,
        boxShadow: `inset 0 1px 0 rgba(255,255,255, ${accent.isDark ? 0.1 : 0.16}), inset 0 0 0 1px rgba(${accentRgb}, ${accent.isDark ? 0.22 : 0.16}), 0 10px 30px rgba(${accentRgb}, ${accent.isDark ? 0.14 : 0.1})`,
        backdropFilter: 'blur(18px) saturate(165%)',
        WebkitBackdropFilter: 'blur(18px) saturate(165%)',
    }

    return (
        <motion.article
            initial={{ opacity: 0, scale: 0.9, y: 15 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ 
                duration: 0.4, 
                delay: Math.min(index * 0.05, 0.4),
                ease: [0.23, 1, 0.32, 1] 
            }}
            whileHover={!isSelectionMode ? { y: -4, scale: 1.015 } : {}}
            whileTap={!isSelectionMode ? { scale: 0.98 } : {}}
            onClick={isSelectionMode ? (event) => onSelect(event, project._id) : undefined}
            onKeyDown={isSelectionMode ? (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                    onSelect(event, project._id)
                }
            } : undefined}
            role={isSelectionMode ? 'button' : undefined}
            tabIndex={isSelectionMode ? 0 : undefined}
            aria-pressed={isSelectionMode ? isSelected : undefined}
            className={cn(
                'group/card relative overflow-hidden rounded-2xl border bg-white/[0.02] shadow-[0_8px_30px_rgba(0,0,0,0.12)] transition-colors duration-300 will-change-transform focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/30 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950',
                isSelectionMode && 'cursor-pointer',
                isSelectionMode ? 'border-white/16 bg-white/[0.04]' : 'border-white/12',
            )}
        >
            <div className='relative aspect-[16/10] overflow-hidden bg-slate-950/40'>
                {/* Selected state uses an average-color frosted glass overlay so it stays visible on any thumbnail */}
                <div 
                    className={cn(
                        "pointer-events-none absolute inset-0 z-20 rounded-2xl transition-opacity duration-300",
                        isSelected ? "opacity-100" : "opacity-0"
                    )}
                    style={selectionGlassStyle}
                />

                {previewUrl ? (
                    <div
                        className='absolute inset-0 z-0 bg-cover bg-center transition-transform duration-700 ease-out group-hover/card:scale-105'
                        style={{ backgroundImage: `url(${previewUrl})` }}
                    />
                ) : (
                    <div className='absolute inset-0 z-0 bg-[radial-gradient(circle_at_30%_30%,rgba(56,189,248,0.2),transparent_50%),linear-gradient(160deg,rgba(15,23,42,0.96),rgba(17,24,39,0.85))]' />
                )}

                {/* Bottom scrim — darkens on hover */}
                <div className='absolute inset-x-0 bottom-0 z-10 h-3/4 bg-gradient-to-t from-black/80 via-black/40 to-transparent transition-opacity duration-500 group-hover/card:from-black/90 group-hover/card:via-black/60' />

                {/* Dual-tone inset frame keeps the card edge readable over both bright and dark images */}
                <div
                    className='pointer-events-none absolute inset-0 z-[15] rounded-2xl'
                    style={{
                        boxShadow: 'inset 0 0 0 1px rgba(255,255,255,0.28), inset 0 0 0 2px rgba(2,6,23,0.72), inset 0 1px 0 rgba(255,255,255,0.08)',
                    }}
                />

                <div className='relative z-20 flex h-full flex-col justify-between'>
                    {/* Top actions */}
                    <div className='flex items-start justify-between px-3 pt-3'>
                        <div>
                            {isSelectionMode ? (
                                <button
                                    type="button"
                                    onClick={(event) => onSelect(event, project._id)}
                                    className={cn(
                                        'flex items-center justify-center rounded-full border backdrop-blur-sm transition duration-200',
                                        isSelected
                                            ? 'size-8 border-white/95 bg-white text-slate-950 shadow-[0_12px_30px_rgba(2,6,23,0.28)]'
                                            : 'size-7 border-white/20 bg-black/30 text-white/70',
                                    )}
                                    style={isSelected ? {
                                        borderColor: 'rgba(255,255,255,0.96)',
                                        background: 'linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(244,247,250,0.92) 100%)',
                                        boxShadow: `0 0 0 3px rgba(${accentRgb}, 0.28), 0 14px 34px rgba(2,6,23,0.34), inset 0 1px 0 rgba(255,255,255,1)`,
                                    } : undefined}
                                    aria-label={isSelected ? `Deselect ${project.title}` : `Select ${project.title}`}
                                >
                                    <Check
                                        className={isSelected ? 'h-4 w-4' : 'h-3 w-3'}
                                        strokeWidth={isSelected ? 3.25 : 2.25}
                                    />
                                </button>
                            ) : (
                                <button
                                    type="button"
                                    onClick={(event) => onDelete(event, project._id, project.title)}
                                    disabled={isDeletingThisProject || isBulkDeleting}
                                    className='flex size-7 items-center justify-center rounded-full border border-white/15 bg-black/30 text-white/70 opacity-0 backdrop-blur-sm transition duration-200 group-hover/card:opacity-100 focus-visible:opacity-100 hover:bg-black/50 disabled:opacity-60'
                                    aria-label={`Delete ${project.title}`}
                                >
                                    {isDeletingThisProject ? (
                                        <Loader2 className='h-3 w-3 animate-spin' />
                                    ) : (
                                        <Trash2 className='h-3 w-3' />
                                    )}
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Bottom info — always white */}
                    <div className='px-3.5 pb-3.5'>
                        <h3 className='line-clamp-1 text-lg font-semibold leading-snug text-white [overflow-wrap:anywhere] drop-shadow-[0_1px_8px_rgba(0,0,0,0.8)]'>
                            {project.title}
                        </h3>
                        <p className='mt-0.5 text-xs font-medium text-white/70 drop-shadow-[0_1px_6px_rgba(0,0,0,0.7)]'>
                            Edited {formatRelativeTime(project.updatedAt)}
                        </p>
                    </div>
                </div>
            </div>
        </motion.article>
    )
}

const Dashboard = () => {
    const { isLoading: isAuthLoading, isAuthenticated } = useConvexAuth()
    const [showNewProjectModal, setShowNewProjectModal] = useState(false)
    const [isSelectionMode, setIsSelectionMode] = useState(false)
    const [selectedProjectIds, setSelectedProjectIds] = useState([])
    const [deletingProjectId, setDeletingProjectId] = useState(null)
    const [isBulkDeleting, setIsBulkDeleting] = useState(false)
    const [deleteConfirm, setDeleteConfirm] = useState({ open: false, type: null, projectId: null, projectTitle: null })
    const { data: projects = [], isLoading: isProjectsLoading } = useConvexQuery(
        api.projects.getUserProjects,
        isAuthenticated ? {} : "skip"
    )
    const { mutate: deleteProject } = useConvexMutation(api.projects.deleteProject)
    const { mutate: bulkDeleteProjects } = useConvexMutation(api.projects.bulkDeleteProjects)
    const isLoading = isAuthLoading || isProjectsLoading
    const projectCount = projects.length
    const hasProjects = projectCount > 0
    const prevProjectIdsRef = useRef(null)

    useEffect(() => {
        const currentIds = projects.map((p) => p._id).join(',')
        if (prevProjectIdsRef.current === currentIds) return
        prevProjectIdsRef.current = currentIds

        const availableProjectIds = new Set(projects.map((project) => project._id))
        setSelectedProjectIds((currentSelectedIds) =>
            currentSelectedIds.filter((projectId) => availableProjectIds.has(projectId)),
        )
    }, [projects])

    useEffect(() => {
        if (selectedProjectIds.length === 0 && !hasProjects) {
            setIsSelectionMode(false)
        }
    }, [hasProjects, selectedProjectIds.length])

    const handleSelectionModeToggle = () => {
        setIsSelectionMode((currentValue) => {
            if (currentValue) {
                setSelectedProjectIds([])
            }

            return !currentValue
        })
    }

    const handleProjectSelection = (event, projectId) => {
        event.preventDefault()
        event.stopPropagation()

        setSelectedProjectIds((currentIds) =>
            currentIds.includes(projectId)
                ? currentIds.filter((currentId) => currentId !== projectId)
                : [...currentIds, projectId],
        )
    }

    const handleDeleteProject = async (event, projectId, projectTitle) => {
        event.preventDefault()
        event.stopPropagation()

        if (deletingProjectId || isBulkDeleting) {
            return
        }

        setDeleteConfirm({ open: true, type: 'single', projectId, projectTitle })
    }

    const handleDeleteSelectedProjects = async () => {
        if (selectedProjectIds.length === 0 || isBulkDeleting) {
            return
        }

        const selectedCount = selectedProjectIds.length
        const projectLabel = selectedCount === 1 ? 'project' : 'projects'
        setDeleteConfirm({ open: true, type: 'bulk', projectId: null, projectTitle: `${selectedCount} ${projectLabel}` })
    }

    const performDelete = useCallback(async (type, projectId, projectTitle) => {
        if (type === 'single') {
            setDeletingProjectId(projectId)
            try {
                const result = await deleteProject({ projectId })
                if (result?.success) {
                    setSelectedProjectIds((currentIds) =>
                        currentIds.filter((currentId) => currentId !== projectId),
                    )
                }
            } finally {
                setDeletingProjectId(null)
            }
        } else if (type === 'bulk') {
            setIsBulkDeleting(true)
            try {
                const result = await bulkDeleteProjects({
                    projectIds: selectedProjectIds,
                })
                if (result?.success) {
                    setSelectedProjectIds([])
                    setIsSelectionMode(false)
                }
            } finally {
                setIsBulkDeleting(false)
            }
        }
    }, [deleteProject, bulkDeleteProjects, selectedProjectIds])

    const confirmDelete = useCallback(async () => {
        const { type, projectId, projectTitle } = deleteConfirm
        setDeleteConfirm({ open: false, type: null, projectId: null, projectTitle: null })

        // Show undoable toast with a 5-second delay before actual deletion
        const label = type === 'single' ? `"${projectTitle}"` : projectTitle
        let undone = false

        toast(`Deleted ${label}`, {
            duration: 5000,
            action: {
                label: 'Undo',
                onClick: () => { undone = true },
            },
            onAutoClose: async () => {
                if (undone) return
                await performDelete(type, projectId, projectTitle)
            },
            onDismiss: async () => {
                if (undone) return
                await performDelete(type, projectId, projectTitle)
            },
        })
    }, [deleteConfirm, performDelete])

    return (
        <div className='min-h-[calc(100svh-3rem)] pt-24 pb-8'>
            <div className='mx-auto flex w-full max-w-[1400px] flex-col gap-6 px-4 sm:px-6 lg:px-8'>
                {/* Slim header */}
                <div className='flex items-center justify-between'>
                    <div>
                        <h1 className='text-2xl font-bold tracking-tight text-white sm:text-3xl'>
                            Projects
                        </h1>
                        <div className='mt-1.5 flex items-center gap-4 text-sm text-white/50'>
                            <span className='inline-flex items-center gap-1.5'>
                                <ImageIcon className='h-3.5 w-3.5' />
                                {projectCount} {projectCount === 1 ? 'project' : 'projects'}
                            </span>
                            {hasProjects && (
                                <span className='inline-flex items-center gap-1.5'>
                                    <Calendar className='h-3.5 w-3.5' />
                                    Last created {formatRelativeTime(
                                        Math.max(...projects.map(p => p._creationTime || p.updatedAt))
                                    )}
                                </span>
                            )}
                        </div>
                    </div>

                    <Button
                        onClick={() => setShowNewProjectModal(true)}
                        variant='primary'
                        className='h-10 gap-2 px-4 text-sm'
                    >
                        <Plus className='h-4 w-4' />
                        New Project
                    </Button>
                </div>

                <section className='space-y-4'>
                    {/* Inline toolbar */}
                    {hasProjects && (
                        <div className='flex items-center justify-end gap-2'>
                            {isSelectionMode ? (
                                <>
                                    <span className='text-xs text-white/50 mr-auto'>
                                        {selectedProjectIds.length} selected
                                    </span>

                                    <Button
                                        variant='glass'
                                        className='h-9 px-3 text-xs'
                                        onClick={handleSelectionModeToggle}
                                        disabled={isBulkDeleting}
                                    >
                                        <X className='h-3.5 w-3.5' />
                                        Cancel
                                    </Button>

                                    <Button
                                        variant='destructive'
                                        className='h-9 rounded-full border border-red-400/20 px-3 text-xs text-red-100'
                                        onClick={handleDeleteSelectedProjects}
                                        disabled={selectedProjectIds.length === 0 || isBulkDeleting}
                                    >
                                        {isBulkDeleting ? (
                                            <Loader2 className='h-3.5 w-3.5 animate-spin' />
                                        ) : (
                                            <Trash2 className='h-3.5 w-3.5' />
                                        )}
                                        Delete
                                    </Button>
                                </>
                            ) : (
                                <Button
                                    variant='glass'
                                    className='h-9 px-3 text-xs'
                                    onClick={handleSelectionModeToggle}
                                >
                                    <Check className='h-3.5 w-3.5' />
                                    Select
                                </Button>
                            )}
                        </div>
                    )}

                    {isLoading ? (
                        <section className='grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4'>
                            {loadingCards.map((_, index) => (
                                <div
                                    key={index}
                                    className='overflow-hidden rounded-2xl border border-white/8 bg-white/[0.02]'
                                >
                                    <div className='aspect-[16/10] animate-pulse bg-white/[0.05]' />
                                </div>
                            ))}
                        </section>
                    ) : hasProjects ? (
                        <section className='grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4'>
                            {projects.map((project, index) => {
                                const isSelected = selectedProjectIds.includes(project._id)
                                const isDeletingThisProject = deletingProjectId === project._id
                                const cardContent = (
                                    <ProjectCard
                                        project={project}
                                        index={index}
                                        isSelectionMode={isSelectionMode}
                                        isSelected={isSelected}
                                        isDeletingThisProject={isDeletingThisProject}
                                        isBulkDeleting={isBulkDeleting}
                                        onSelect={handleProjectSelection}
                                        onDelete={handleDeleteProject}
                                    />
                                )

                                if (isSelectionMode) {
                                    return (
                                        <div
                                            key={project._id}
                                            className='group block'
                                        >
                                            {cardContent}
                                        </div>
                                    )
                                }

                                return (
                                    <Link
                                        key={project._id}
                                        href={`/editor/${project._id}`}
                                        className='group block'
                                    >
                                        {cardContent}
                                    </Link>
                                )
                            })}
                        </section>
                    ) : (
                        <div className='flex flex-col items-center gap-4 rounded-2xl border border-dashed border-white/10 py-16 px-6 text-center'>
                            <p className='text-lg font-medium text-white/60'>No projects yet</p>
                            <p className='text-sm text-white/40'>
                                Upload an image to get started.
                            </p>

                            <Button
                                onClick={() => setShowNewProjectModal(true)}
                                variant='primary'
                                className='mt-2 h-10 gap-2 px-4 text-sm'
                            >
                                <Plus className='h-4 w-4' />
                                Create Project
                            </Button>
                        </div>
                    )}
                </section>

                <NewProjectModel
                    isOpen={showNewProjectModal}
                    onClose={() => setShowNewProjectModal(false)}
                    currentProjectCount={projectCount}
                />

                <AlertDialog
                    open={deleteConfirm.open}
                    onOpenChange={(open) => {
                        if (!open) setDeleteConfirm({ open: false, type: null, projectId: null, projectTitle: null })
                    }}
                >
                    <AlertDialogContent>
                        <AlertDialogHeader>
                            <AlertDialogTitle>
                                {deleteConfirm.type === 'bulk'
                                    ? `Delete ${deleteConfirm.projectTitle}?`
                                    : `Delete "${deleteConfirm.projectTitle}"?`
                                }
                            </AlertDialogTitle>
                            <AlertDialogDescription>
                                This action cannot be undone. {deleteConfirm.type === 'bulk'
                                    ? 'All selected projects will be permanently removed.'
                                    : 'This project will be permanently removed.'}
                            </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                            <AlertDialogCancel>Cancel</AlertDialogCancel>
                            <AlertDialogAction onClick={confirmDelete}>
                                Delete
                            </AlertDialogAction>
                        </AlertDialogFooter>
                    </AlertDialogContent>
                </AlertDialog>
            </div>
        </div>
    )
}

export default Dashboard
