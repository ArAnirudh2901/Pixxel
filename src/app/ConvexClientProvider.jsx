"use client";

import { useAuth } from "@clerk/nextjs";
import { ConvexReactClient } from "convex/react";
import { ConvexProviderWithClerk } from "convex/react-clerk";
import { useConvexAuth, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";
import { useEffect } from "react";

const convex = new ConvexReactClient(process.env.NEXT_PUBLIC_CONVEX_URL);

// Syncs the authenticated Clerk user into the Convex database on login
function UserStoreSync({ children }) {
    const { isAuthenticated } = useConvexAuth();
    const storeUser = useMutation(api.users.store);

    useEffect(() => {
        if (!isAuthenticated) return;
        storeUser().catch(console.error);
    }, [isAuthenticated, storeUser]);

    return children;
}

export function ConvexClientProvider({ children }) {
    return (
        <ConvexProviderWithClerk
            client={convex}
            useAuth={useAuth}
        >
            <UserStoreSync>
                {children}
            </UserStoreSync>
        </ConvexProviderWithClerk>
    );
}
