import { mutation, query } from "./_generated/server";

function getDisplayName(identity) {
    return identity.name?.trim() || identity.email?.split("@")[0] || "Anonymous";
}

export const store = mutation({
    args: {},
    handler: async (ctx) => {
        const identity = await ctx.auth.getUserIdentity();
        if (!identity) {
            throw new Error("Called storeUser without authentication present");
        }

        const displayName = getDisplayName(identity);

        // Check if we've already stored this identity before.
        // Note: If you don't want to define an index right away, you can use
        // ctx.db.query("users")
        //  .filter(q => q.eq(q.field("tokenIdentifier"), identity.tokenIdentifier))
        //  .unique();
        const user = await ctx.db
            .query("users")
            .withIndex("by_token", (q) =>
                q.eq("tokenIdentifier", identity.tokenIdentifier),
            )
            .unique();
        if (user !== null) {
            const updates = {};

            if (user.name !== displayName) {
                updates.name = displayName;
            }

            if (user.email !== identity.email) {
                updates.email = identity.email;
            }

            if (user.imageUrl !== identity.pictureUrl) {
                updates.imageUrl = identity.pictureUrl;
            }

            if (Object.keys(updates).length > 0) {
                await ctx.db.patch(user._id, updates);
            }

            return user._id;
        }
        // If it's a new identity, create a new `User`.
        return await ctx.db.insert("users", {
            name: displayName,
            tokenIdentifier: identity.tokenIdentifier,
            email: identity.email,
            imageUrl: identity.pictureUrl,
            // About the plan and usage of the User
            plan: "free",
            projectsUsed: 0,
            exportsThisMonth: 0,
            createdAt: Date.now(),
            lastActiveAt: Date.now(),
        });
    },
});

export const getCurrentUser = query({
    args: {},
    handler: async (ctx) => {
        const identity = await ctx.auth.getUserIdentity();
        if (!identity) {
            throw new Error("Not Authenticated");
        }

        const user = await ctx.db
            .query("users")
            .withIndex("by_token", (q) =>
                q.eq("tokenIdentifier", identity.tokenIdentifier))
            .unique();

        if (!user) {
            throw new Error("User not found.");
        }

        return user;
    }
});
