import { InfoIcon } from '@phosphor-icons/react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Alert, AlertAction, AlertDescription, AlertTitle } from '@/components/ui/alert'
import React from 'react'
import usePlanAccess from '../../../../../hooks/usePlanAccess'
import { useConvexQuery } from '../../../../../hooks/useConvexQuery'
import { api } from '../../../../../convex/_generated/api'

const NewProjectModel = ({ isOpen, onClose }) => {

    const { isFree, canCreateProject } = usePlanAccess()

    const { data: projects } = useConvexQuery(api.projects.getUserProjects)
    const currentProjectCount = projects?.length || 0
    const canCreate = canCreateProject(currentProjectCount)

    return (
        <Dialog open={isOpen} onOpenChange={(open) => {
            if (!open) {
                onClose()
            }
        }}>
            <DialogContent>
                <DialogHeader>
                    <DialogTitle className="text-2xl font-bold text-white">
                        Create New Project
                    </DialogTitle>
                    <DialogDescription className="text-slate-300">
                        Start a fresh canvas from your dashboard. Free accounts can keep up to 3 projects at a time.
                    </DialogDescription>
                </DialogHeader>

                <div className='space-y-6'>
                    <Alert
                        variant={canCreate ? 'default' : 'destructive'}
                        className='rounded-xl border border-white/10 bg-slate-900/60'
                    >
                        <InfoIcon className='size-4' />
                        <AlertTitle>
                            {canCreate ? 'Project slot available' : 'Project limit reached'}
                        </AlertTitle>
                        <AlertDescription>
                            {isFree
                                ? canCreate
                                    ? `You are using ${currentProjectCount} of 3 free project slots.`
                                    : 'Free accounts can create up to 3 projects. Remove an existing project or upgrade to continue.'
                                : 'Your Pro plan can create new projects without the free-project cap.'}
                        </AlertDescription>
                        {!canCreate && (
                            <AlertAction>
                                <Button variant="outline" onClick={onClose}>Close</Button>
                            </AlertAction>
                        )}
                    </Alert>
                </div>

                {isFree && (
                    <Badge variant='secondary' className="bg-slate-700 text-white/70">
                        {currentProjectCount}/3 projects
                    </Badge>
                )}
                <DialogFooter>
                    <Button variant="outline" onClick={onClose}>Cancel</Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    )
}

export default NewProjectModel
