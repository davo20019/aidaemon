# AI Agent Guide: blog.aidaemon.ai

This document helps AI agents understand and work with this blog project.

## Project Overview

- **Name**: blog.aidaemon.ai
- **Type**: Static blog site built with Vite
- **Domain**: https://blog.aidaemon.ai/
- **Repository**: https://github.com/davo20019/blog.aidaemon.ai.git
- **Platform**: Cloudflare Pages

## Project Structure

```
~/projects/blog.aidaemon.ai/
├── src/
│   ├── content/          # Blog post content files
│   ├── main.js           # Main application entry
│   ├── posts.js          # Posts data and logic
│   └── style.css         # Stylesheets
├── scripts/
│   └── generate-static-pages.mjs  # Static page generation
├── public/
│   └── og-image.png      # Social preview image
├── index.html            # Main HTML template
├── vite.config.js        # Vite configuration
├── wrangler.toml         # Cloudflare Wrangler config
└── package.json          # Dependencies
```

## Content Creation

### Blog Posts Location
Blog posts are stored in `src/content/` as individual files.

### Post Format
Posts should follow the established format in the content directory. Check existing posts for examples.

### Creating a New Post
1. Create a new file in `src/content/`
2. Follow the naming convention of existing posts
3. Add the post metadata and content
4. Update `src/posts.js` if the posts list needs to be regenerated

## Deployment Workflow

> IMPORTANT: blog.aidaemon.ai is served by a Cloudflare **Worker** named `blog-aidaemon-ai`
> (static assets served from `dist/`). It is NOT Cloudflare Pages, and there is NO git-push
> auto-deploy. A SEPARATE, unrelated Cloudflare Pages project of the same name exists
> (`blog-aidaemon-ai.pages.dev`) — do NOT deploy there; it does not serve the live domain.

### Publish (build + deploy in one command)
```bash
cd ~/projects/blog.aidaemon.ai
npm run deploy   # = vite build + static pages, then `wrangler deploy` (uploads dist/ to the Worker)
```
`wrangler deploy` updates the Worker that serves blog.aidaemon.ai. Wait for the
"Deployed blog-aidaemon-ai" line (can take ~1 minute).

### Verify
1. `curl -sI https://blog.aidaemon.ai/posts/<slug>/` returns `HTTP/2 200`.
2. The post title shows in `https://blog.aidaemon.ai/posts/<slug>/`.

(Committing to git is good hygiene but does NOT deploy — always run `npm run deploy`.)

## Common Tasks

### Update existing post
1. Edit the file in `src/content/`
2. Commit and push

### Add new post
1. Create file in `src/content/`
2. Run generation script if needed: `node scripts/generate-static-pages.mjs`
3. Commit and push

### Fix deployment issues
1. Check `wrangler.toml` configuration
2. Verify Cloudflare Pages settings
3. Check build logs in Cloudflare dashboard

## Important Notes

- **Always commit before pushing** - Uncommitted changes won't deploy
- **Main branch is production** - All pushes to main trigger deployment
- **Repository is the source of truth** - The blog content lives in the git repo

## Owner Information

- **Owner**: David Loor
- **Location**: Miami
- **Email**: Contact David directly for questions about content or design decisions
