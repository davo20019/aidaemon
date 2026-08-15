# Claude's Notes: blog.aidaemon.ai

Quick reference for Claude (aidaemon) when working on this blog.

## At a Glance

| Property | Value |
|----------|-------|
| Path | `~/projects/blog.aidaemon.ai` |
| Domain | https://blog.aidaemon.ai/ |
| Repo | https://github.com/davo20019/blog.aidaemon.ai.git |
| Platform | Cloudflare Pages |
| Builder | Vite |

## Quick Deploy Command

```bash
cd ~/projects/blog.aidaemon.ai && git add . && git commit -m "MESSAGE" && git push origin main
```

## Content Workflow

1. **Posts live in** `src/content/`
2. **Post registry** in `src/posts.js` 
3. **Static generation** via `scripts/generate-static-pages.mjs`

## What to Remember

- David prefers confirmation before destructive actions
- Blog posts are the main content type
- Deployment is git-push-triggered to Cloudflare Pages
- Site is static - no server-side components

## Last Known State

Check `git status` before any operations. The repo is usually on `main` branch.

## Owner Context

- David Loor, software engineer in Miami
- Enjoys Python and Rust
- Has two daughters
- Values: "Don't sweat small stuff"
