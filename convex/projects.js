import { v } from "convex/values";
import { mutation, query } from "./_generated/server";
import { api, internal } from "./_generated/api";
import { getAuthUserId } from "./users";

export const create = mutation({
    args: {
        title: v.string(),
        originalImageUrl: v.optional(v.string()),
        currentImageUrl: v.optional(v.string()),
        thumbnailUrl: v.optional(v.string()),
        width: v.number(),
        height: v.number(),
        canvasState: v.optional(v.any())
    },
    handler: async (ctx, args) => {
        const user = await getAuthUserId(ctx);

        if (user.plan === "free") {
            const projectCount = await ctx.db
                .query("projects")
                .withIndex("by_user", (q) => q.eq("userId", user._id))
                .take(4);

            if (projectCount.length >= 3) {
                throw new Error("Free plan limited to 3 projects. Upgrade to pro for unlimited projects.");
            }
        }

        const projectId = await ctx.db.insert("projects", {
            title: args.title,
            userId: user._id,
            originalImageUrl: args.originalImageUrl,
            currentImageUrl: args.currentImageUrl,
            thumbnailUrl: args.thumbnailUrl,
            width: args.width,
            height: args.height,
            canvasState: args.canvasState ?? null,
            createdAt: Date.now(),
            updatedAt: Date.now(),
        });

        await ctx.db.patch(user._id, {
            projectsUsed: user.projectsUsed + 1,
            lastActiveAt: Date.now(),
        });

        return projectId;
    },
});

export const getUserProjects = query({
    args: {},
    handler: async (ctx) => {
        const user = await getAuthUserId(ctx);

        const projects = await ctx.db
            .query("projects")
            .withIndex("by_user_updated", (q) => q.eq("userId", user._id))
            .order("desc")
            .collect();

        return projects;
    },
});

export const deleteProject = mutation({
    args: {
        projectId: v.id("projects"),
    },
    handler: async (ctx, args) => {
        const user = await getAuthUserId(ctx);

        const project = await ctx.db.get(args.projectId);

        if (!project) {
            throw new Error("Project not found");
        }

        if (project.userId !== user._id) {
            throw new Error("Access denied");
        }

        await ctx.db.delete(args.projectId);

        await ctx.db.patch(user._id, {
            projectsUsed: Math.max(0, user.projectsUsed - 1),
            lastActiveAt: Date.now(),
        });

        return { success: true };
    },
});

export const bulkDeleteProjects = mutation({
    args: {
        projectIds: v.array(v.id("projects")),
    },
    handler: async (ctx, args) => {
        const user = await getAuthUserId(ctx);
        const uniqueProjectIds = [...new Set(args.projectIds)];

        if (uniqueProjectIds.length === 0) {
            return { success: true, deletedCount: 0 };
        }

        const ownedProjects = [];

        for (const projectId of uniqueProjectIds) {
            const project = await ctx.db.get(projectId);

            if (!project) {
                continue;
            }

            if (project.userId !== user._id) {
                throw new Error("Access denied");
            }

            ownedProjects.push(project);
        }

        for (const project of ownedProjects) {
            await ctx.db.delete(project._id);
        }

        if (ownedProjects.length > 0) {
            await ctx.db.patch(user._id, {
                projectsUsed: Math.max(0, user.projectsUsed - ownedProjects.length),
                lastActiveAt: Date.now(),
            });
        }

        return {
            success: true,
            deletedCount: ownedProjects.length,
        };
    },
});

export const getProject = query({
    args: {
        projectId: v.id("projects")
    },
    handler: async (ctx, args) => {
        const user = await getAuthUserId(ctx)

        const project = await ctx.db.get(args.projectId)
        if (!project)
            throw new Error("Project not found")

        if (!user || project.userId !== user._id)
            throw new Error("Access denied")

        return project
    }
})

export const updateProject = mutation({
    args: {
        projectId: v.id("projects"),
        canvasState: v.optional(v.any()),
        width: v.optional(v.number()),
        height: v.optional(v.number()),
        currentImageUrl: v.optional(v.string()),
        thumbnailUrl: v.optional(v.string()),
        activeTransformations: v.optional(v.string()),
        backgroundRemoved: v.optional(v.boolean()),
    },
    handler: async (ctx, args) => {
        const user = await getAuthUserId(ctx)

        const project = await ctx.db.get(args.projectId)
        if (!project)
            throw new Error("Project not found")

        if (!user || project.userId !== user._id)
            throw new Error("Access denied")

        const updateData = {
            updatedAt: Date.now(),
        }

        if (args.canvasState !== undefined)
            updateData.canvasState = args.canvasState

        if (args.width !== undefined)
            updateData.width = args.width

        if (args.height !== undefined)
            updateData.height = args.height

        if (args.currentImageUrl !== undefined)
            updateData.currentImageUrl = args.currentImageUrl

        if (args.thumbnailUrl !== undefined)
            updateData.thumbnailUrl = args.thumbnailUrl

        if (args.activeTransformations !== undefined)
            updateData.activeTransformations = args.activeTransformations

        if (args.backgroundRemoved !== undefined)
            updateData.backgroundRemoved = args.backgroundRemoved

        await ctx.db.patch(args.projectId, updateData)

        await ctx.db.patch(user._id, {
            lastActiveAt: Date.now()
        })

        return args.projectId
    }
})