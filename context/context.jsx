import { createContext, useContext } from "react";

export const CanvasContext = createContext(null)

export const useCanvas = () => {
    const context = useContext(CanvasContext)

    if (!context) {
        throw new Error("useCanvas must be used within a CanvasContext.Provider")
    }

    return context
}
