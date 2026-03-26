"use client"

import React, { useState } from 'react'
import { useConvexAuth } from 'convex/react'
import Link from 'next/link'
import { api } from '../../../../convex/_generated/api'
import { useConvexQuery } from '../../../../hooks/useConvexQuery'
import { Button } from '@/components/ui/button'
import LoadingBar from '@/components/ui/loading-bar'
import { ArrowUpRight, Clock3, FolderOpen, Plus, Sparkles } from 'lucide-react'
import NewProjectModel from './_components/newProjectModel'

const loadingCards = Array.from({ length: 3 })

const formatProjectDate = (timestamp) => {
    if (!timestamp) {
        return 'Just now'
    }

    return new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
    }).format(new Date(timestamp))
}

const getProjectPreview = (project) =>
    project.thumbnailUrl || project.currentImageUrl || project.originalImageUrl

const Dashboard = () => {
    const { isLoading: isAuthLoading, isAuthenticated } = useConvexAuth()
    const [showNewProjectModal, setShowNewProjectModal] = useState(false)
    const { data: projects = [], isLoading: isProjectsLoading } = useConvexQuery(
        api.projects.getUserProjects,
        isAuthenticated ? {} : "skip"
    )
    const isLoading = isAuthLoading || isProjectsLoading
    const projectCount = projects.length
    const hasProjects = projectCount > 0
    const lastUpdatedLabel =
        hasProjects ? formatProjectDate(projects[0].updatedAt) : 'No edits yet'

    const projectOverviewCards = [
        {
            label: 'Projects',
            value: String(projectCount).padStart(2, '0'),
            helper: projectCount === 1 ? 'Canvas in workspace' : 'Canvases in workspace',
            icon: FolderOpen,
        },
        {
            label: 'Latest update',
            value: lastUpdatedLabel,
            helper: projectCount > 0 ? 'Most recently touched project' : 'Your edits will show up here',
            icon: Clock3,
        },
        {
            label: 'Workspace',
            value: isLoading ? 'Syncing' : projectCount > 0 ? 'Active' : 'Ready',
            helper: 'Minimal dashboard, instant access',
            icon: Sparkles,
        },
    ]

    const emptyStateCards = [
        {
            label: 'Starter canvas',
            value: '1400 x 900',
            helper: 'Balanced landscape workspace for a first pass',
            icon: Sparkles,
        },
        {
            label: 'Project flow',
            value: 'Auto-saved',
            helper: 'Each new canvas lands back in this dashboard',
            icon: Clock3,
        },
        {
            label: 'Editor access',
            value: 'One click',
            helper: 'Create once and continue directly in the editor',
            icon: ArrowUpRight,
        },
    ]

    const overviewCards = hasProjects ? projectOverviewCards : emptyStateCards

    return (
        <div className='min-h-[calc(100svh-3rem)] pt-28 pb-8'>
            <div className='mx-auto flex max-w-6xl flex-col gap-6 px-6'>
                <section className='relative overflow-hidden rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.08),rgba(255,255,255,0.03))] px-5 py-5 shadow-[0_24px_72px_rgba(8,15,40,0.24)] backdrop-blur-xl sm:px-6 sm:py-6'>
                    <div className='pointer-events-none absolute -left-10 top-0 h-32 w-32 rounded-full bg-cyan-400/10 blur-3xl' />
                    <div className='pointer-events-none absolute right-0 top-1/2 h-44 w-44 -translate-y-1/2 rounded-full bg-blue-500/10 blur-3xl' />

                    <div className='relative flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between'>
                        <div className='max-w-2xl'>
                            <div className='inline-flex items-center gap-2 rounded-full border border-white/12 bg-white/[0.06] px-3 py-1 text-[0.68rem] font-semibold uppercase tracking-[0.24em] text-white/60'>
                                <Sparkles className='h-3.5 w-3.5 text-cyan-200' />
                                Creative Workspace
                            </div>

                            <h1 className='mt-4 text-3xl font-black tracking-tight text-white sm:text-4xl xl:text-[3.35rem]'>
                                Your Projects
                            </h1>

                            <p className='mt-2.5 max-w-xl text-sm leading-6 text-white/70 sm:text-base'>
                                Create, organize, and reopen your AI-powered image designs from one calm
                                workspace.
                            </p>

                            {!hasProjects && (
                                <div className='mt-5 flex flex-col items-start gap-3 sm:flex-row sm:items-center'>
                                    <Button
                                        onClick={() => setShowNewProjectModal(true)}
                                        variant='primary'
                                        size='xl'
                                        className='h-12 gap-2 px-6 text-sm'
                                    >
                                        <Plus className='h-5 w-5' />
                                        New Project
                                    </Button>

                                    <p className='text-sm text-white/55'>
                                        Starts with a clean 1400 x 900 canvas.
                                    </p>
                                </div>
                            )}
                        </div>

                        {hasProjects && (
                            <Button
                                onClick={() => setShowNewProjectModal(true)}
                                variant='primary'
                                size='xl'
                                className='h-12 gap-2 px-6 text-sm self-start lg:mt-2'
                            >
                                <Plus className='h-5 w-5' />
                                New Project
                            </Button>
                        )}
                    </div>

                    <div className='relative mt-6 grid gap-4 md:grid-cols-3'>
                        {overviewCards.map(({ label, value, helper, icon: Icon }) => (
                            <div
                                key={label}
                                className='rounded-[22px] border border-white/10 bg-slate-950/20 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] transition duration-300 hover:-translate-y-0.5 hover:border-white/16 hover:bg-white/[0.05]'
                            >
                                <div className='mb-4 flex items-center justify-between'>
                                    <p className='text-xs uppercase tracking-[0.22em] text-white/45'>{label}</p>

                                    <div className='flex size-9 items-center justify-center rounded-full border border-white/10 bg-white/[0.06] text-white/70'>
                                        <Icon className='h-4 w-4' />
                                    </div>
                                </div>

                                <p className='text-[1.85rem] font-semibold text-white'>{value}</p>
                                <p className='mt-1.5 text-sm text-white/55'>{helper}</p>
                            </div>
                        ))}
                    </div>
                </section>

                {isLoading ? (
                    <section className='grid gap-5 md:grid-cols-3'>
                        {loadingCards.map((_, index) => (
                            <div
                                key={index}
                                className='overflow-hidden rounded-[26px] border border-white/10 bg-white/[0.035] p-4 shadow-[0_18px_52px_rgba(8,15,40,0.22)] backdrop-blur-sm'
                            >
                                <div className='mb-4 aspect-[16/9] rounded-[20px] bg-[linear-gradient(160deg,rgba(255,255,255,0.08),rgba(255,255,255,0.02))]' />

                                <div className='space-y-3'>
                                    <div className='h-5 w-2/3 rounded-full bg-white/10' />
                                    <div className='h-4 w-1/2 rounded-full bg-white/[0.08]' />

                                    <div className='pt-2'>
                                        <LoadingBar />
                                    </div>
                                </div>
                            </div>
                        ))}
                    </section>
                ) : hasProjects ? (
                    <>
                        <div className='flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between'>
                            <div>
                                <h2 className='text-2xl font-semibold text-white'>Recent Work</h2>
                                <p className='text-sm text-white/60'>
                                    Open a project to keep editing right where you left off.
                                </p>
                            </div>

                            <p className='text-sm text-white/45'>
                                {projectCount} {projectCount === 1 ? 'project' : 'projects'} in your workspace
                            </p>
                        </div>

                        <section className='grid gap-5 lg:gap-6 md:grid-cols-2 xl:grid-cols-3'>
                            {projects.map((project) => {
                                const previewUrl = getProjectPreview(project)

                                return (
                                    <Link
                                        key={project._id}
                                        href={`/editor/${project._id}`}
                                        className='group block'
                                    >
                                        <article className='overflow-hidden rounded-[28px] border border-white/10 bg-white/[0.035] p-3 shadow-[0_20px_60px_rgba(8,15,40,0.22)] backdrop-blur-sm transition duration-300 hover:-translate-y-1 hover:border-white/18 hover:bg-white/[0.055]'>
                                            <div className='relative aspect-[16/10] overflow-hidden rounded-[22px] border border-white/10 bg-slate-950/40'>
                                                {previewUrl ? (
                                                    <div
                                                        className='absolute inset-0 bg-cover bg-center transition duration-500 group-hover:scale-105'
                                                        style={{ backgroundImage: `url(${previewUrl})` }}
                                                    />
                                                ) : (
                                                    <div className='absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(56,189,248,0.28),transparent_28%),radial-gradient(circle_at_80%_80%,rgba(168,85,247,0.24),transparent_32%),linear-gradient(160deg,rgba(15,23,42,0.96),rgba(17,24,39,0.85))]' />
                                                )}

                                                <div className='absolute inset-0 bg-gradient-to-b from-slate-950/5 via-slate-950/15 to-slate-950/80' />

                                                <div className='relative flex h-full flex-col justify-between p-4'>
                                                    <div className='flex items-start justify-between gap-3'>
                                                        <span className='rounded-full border border-white/15 bg-slate-950/45 px-3 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.22em] text-white/60'>
                                                            Project
                                                        </span>

                                                        <span className='rounded-full border border-white/12 bg-white/[0.08] px-3 py-1 text-xs font-medium text-white/75'>
                                                            {project.width} x {project.height}
                                                        </span>
                                                    </div>

                                                    <div className='flex items-end justify-between gap-4'>
                                                        <div className='min-w-0'>
                                                            <p className='text-xs text-white/60'>
                                                                Updated {formatProjectDate(project.updatedAt)}
                                                            </p>

                                                            <h3 className='mt-2 truncate text-xl font-semibold text-white'>
                                                                {project.title}
                                                            </h3>
                                                        </div>

                                                        <div className='flex size-10 shrink-0 items-center justify-center rounded-full border border-white/12 bg-white/[0.08] text-white/80 transition duration-300 group-hover:border-cyan-200/35 group-hover:bg-cyan-300/10 group-hover:text-cyan-100'>
                                                            <ArrowUpRight className='h-4 w-4' />
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </article>
                                    </Link>
                                )
                            })}
                        </section>
                    </>
                ) : null}

                <NewProjectModel
                    isOpen={showNewProjectModal}
                    onClose={() => setShowNewProjectModal(false)}
                />
            </div>
        </div>
    )
}

export default Dashboard
