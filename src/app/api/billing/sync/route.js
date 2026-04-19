import { NextResponse } from "next/server";
import { auth } from "@clerk/nextjs/server";
import { fetchMutation } from "convex/nextjs";
import { api } from "../../../../../convex/_generated/api";

export async function POST() {
    const { userId, has, getToken, sessionClaims } = await auth();

    if (!userId) {
        return NextResponse.json(
            { error: "Unauthorized" },
            { status: 401 },
        );
    }

    const token =
        sessionClaims?.aud === "convex"
            ? await getToken()
            : await getToken({ template: "convex" });

    if (!token) {
        return NextResponse.json(
            { error: "Missing Convex auth token" },
            { status: 500 },
        );
    }

    const plan = has?.({ plan: "pro" }) ? "pro" : "free";

    await fetchMutation(
        api.users.syncPlan,
        { plan },
        { token },
    );

    return NextResponse.json({ plan });
}
