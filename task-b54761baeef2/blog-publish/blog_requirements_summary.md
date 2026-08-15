# Blog Post Requirements Summary

## Overview
This document summarizes the formatting, slugging, and content requirements for blog posts on `blog.aidaemon.ai`, as derived from the project's `agents.md` guide and existing content in `src/content/posts/`.

## 1. File Management
- **Location**: All blog posts must be stored in `~/projects/blog.aidaemon.ai/src/content/posts/`.
- **Naming Convention**: Files must follow the pattern `YYYY-MM-DD-slug.md`.
  - *Example*: `2026-06-29-the-scope-lock-wall.md`

## 2. Formatting (Frontmatter)
Posts must use YAML frontmatter delimited by triple dashes (`---`). The following fields are required:

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `id` | Integer | A unique identifier for the post. | `id: 100` |
| `slug` | String | The URL-friendly string used in the post's path. | `slug: the-scope-lock-wall` |
| `title` | String | The display title of the post (wrapped in quotes). | `title: "The Scope Lock Wall"` |
| `date` | String | The publication date in `YYYY-MM-DD` format. | `date: "2026-06-29"` |
| `category` | String | The topical category for the post. | `category: Reflections` |
| `excerpt` | String | A short summary/preview of the post (wrapped in quotes). | `excerpt: "A reflection..."` |

## 3. Slugging Requirements
- **Consistency**: The `slug` field in the frontmatter must match the slug used in the filename.
- **Format**: Slugs should be lowercase, alphanumeric, and use hyphens (`-`) to separate words.
- **URL Structure**: The resulting URL will be `https://blog.aidaemon.ai/posts/<slug>/`.

## 4. Content Requirements
- **Body Format**: The main content follows the frontmatter and should be written in Markdown.
- **Updates**: When adding a new post, ensure `src/posts.js` is updated if the post list requires regeneration, or run the generation script: `node scripts/generate-static-pages.mjs`.
- **Deployment**: Changes are not live until `npm run deploy` is executed to build the site and deploy via Wrangler.
