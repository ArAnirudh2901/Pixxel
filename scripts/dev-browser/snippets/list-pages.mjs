export default async ({ context }) => {
    return Promise.all(context.pages().map(async (p) => ({
        url: p.url(),
        title: await p.title().catch(() => '?'),
    })))
}
