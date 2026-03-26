"use client"

import { useParams } from 'next/navigation'
import React from 'react'

const Editor = () => {
    const { projectId } = useParams()

    return (
        <div>Editor: {projectId}</div>
    )
}

export default Editor
