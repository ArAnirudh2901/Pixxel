"use client"

import { useMutation, useQueries } from "convex/react"
import { getFunctionName } from "convex/server"
import { useEffect, useMemo, useRef, useState } from "react"
import { toast } from "sonner"

export const useConvexQuery = (query, ...args) => {
    const isSkipped = args[0] === "skip"
    const queryArgs = isSkipped ? {} : (args[0] ?? {})
    const queryName = isSkipped ? null : getFunctionName(query)
    const serializedQueryArgs = JSON.stringify(queryArgs)
    const queryRequests = useMemo(
        () => (
            isSkipped
                ? {}
                : {
                    queryResult: {
                        query: queryName,
                        args: JSON.parse(serializedQueryArgs),
                    },
                }
        ),
        [isSkipped, queryName, serializedQueryArgs],
    )
    const results = useQueries(queryRequests)
    const result = isSkipped ? undefined : results.queryResult
    const lastErrorMessageRef = useRef(null)
    const error = result instanceof Error ? result : null
    const data = error || isSkipped ? undefined : result
    const isLoading = !isSkipped && result === undefined

    useEffect(() => {
        if (isSkipped) {
            lastErrorMessageRef.current = null
            return
        }

        if (error) {
            if (lastErrorMessageRef.current !== error.message) {
                toast.error(error.message)
                lastErrorMessageRef.current = error.message
            }
        } else {
            lastErrorMessageRef.current = null
        }
    }, [error, isSkipped])

    return { data, isLoading, error }
}

export const useConvexMutation = (mutation, ...args) => {
    const mutationFn = useMutation(mutation)

    const [data, setData] = useState(undefined)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)

    const mutate = async (...args) => {
        setIsLoading(true)
        setError(null)

        try {
            const response = await mutationFn(...args)
            setData(response)
            return response
        } catch (err) {
            setError(err)
            toast.error(err.message)
        } finally {
            setIsLoading(false)
        }
    }

    return { mutate, data, isLoading, error }
}
