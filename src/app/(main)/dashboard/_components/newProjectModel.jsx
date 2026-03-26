import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
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
                </DialogHeader>

                <div className='space=y-6'>

                </div>

                {isFree && (
                    <Badge variant='secondary' className="bg-slate-700 text-white/70">
                        {currentProjectCount}/3 projects
                    </Badge>
                )}
                <DialogFooter>Cancel</DialogFooter>
            </DialogContent>
        </Dialog>
    )
}

export default NewProjectModel
