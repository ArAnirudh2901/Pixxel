// convex/auth.config.js
const domain = process.env.CLERK_JWT_ISSUER_DOMAIN;
if (!domain) throw new Error("Missing CLERK_JWT_ISSUER_DOMAIN");

/** @type {import("convex/server").AuthConfig} */
const authConfig = {
    providers: [{ domain, applicationID: "convex" }],
};

export default authConfig;
