# Diary Post Requirements Review

## Bottom Line

A new diary entry belongs in `/Users/davidloor/projects/blog.aidaemon.ai/src/content/posts/` as one Markdown file. Its frontmatter must carry the six fields used by the sampled posts: `id`, `slug`, `title`, `date`, `category`, and `excerpt`. The public address is derived from `slug` as `/posts/<slug>/`.

The active request does not supply a fixed filename, slug, title, ID, or timestamp. The required repository path can therefore only be specified as `/Users/davidloor/projects/blog.aidaemon.ai/src/content/posts/<YYYY-MM-DD>-<slug>.md`, matching the recent samples. Do not create the post as part of this review.

## Required Frontmatter and Schema

Use a YAML frontmatter block at the very start of the file:

```yaml
---
id: <unique numeric ID>
slug: <lowercase-hyphenated-slug>
title: "<post title>"
date: "<ISO date or datetime with offset>"
category: "reflection"
excerpt: "<one-sentence summary>"
---
```

Observed details:

- Every sampled post uses all six fields above.
- `id` is numeric. Choose a unique value rather than reuse a sampled value.
- `slug` is unquoted, lowercase, and hyphenated. The routing code constructs `/posts/<slug>/`.
- `title` and `excerpt` are quoted strings.
- `date` accepts both a date-only value, such as `"2026-06-28"`, and a timestamp with UTC offset, such as `"2026-07-27T19:15:00-04:00"`.
- The recent posts use `category: "reflection"`; an older sample uses capitalization (`"Reflections"`). For a new diary-style post, preserve the recent lowercase `reflection` convention.

## Observed Writing Conventions

Recent entries begin with an H1 that exactly repeats the frontmatter title. They use short, factual paragraphs and develop one narrow observation from a concrete task. The newest sample is lean: four body paragraphs, a horizontal rule, then one italicized closing line containing the written date and the central takeaway.

Use this pattern for a concise diary post:

1. State the assigned task and visible constraint in the first paragraph.
2. Describe the concrete writing path and required metadata without expanding into unrelated technical history.
3. State the follow-up deployment check as a distinct final operational fact.
4. End with `---` and an italicized one-sentence date/takeaway note.

Avoid elaborate subheadings, broad claims about autonomy, invented implementation results, or claims that deployment has already succeeded. The requested subject is one post assignment and its verification path.

## Write and Deployment Considerations

The guidance contains inconsistent older workflow notes, but its explicit deployment warning is controlling: the live site is served by the Cloudflare Worker `blog-aidaemon-ai`, not the similarly named Cloudflare Pages project. A git push does not publish the site.

When a post is later authorized and written:

- Save it under `src/content/posts/`, specifically `/Users/davidloor/projects/blog.aidaemon.ai/src/content/posts/<YYYY-MM-DD>-<slug>.md`.
- The content loader targets `./content/posts/*.md`; placing a Markdown post elsewhere will not make it part of the normal post set.
- Build and deploy with `npm run deploy` from the repository root. This runs the build/static-page process and `wrangler deploy` for the Worker.
- Wait for the `Deployed blog-aidaemon-ai` confirmation.
- Verify the live route returns `HTTP/2 200` at `https://blog.aidaemon.ai/posts/<slug>/` and confirm the title is visible on that page.
- Do not deploy to `blog-aidaemon-ai.pages.dev`.

The guide also says to commit before pushing, but it separately and explicitly says committing or pushing does not deploy. Treat `npm run deploy` plus the live-route check as the publication and verification steps.

## Grounded Diary Angle

**Angle:** *One lean post, one specific path, one later proof.*

Frame the entry around being assigned a narrowly bounded publishing task: prepare a factual diary post at `/Users/davidloor/projects/blog.aidaemon.ai/src/content/posts/<YYYY-MM-DD>-<slug>.md`, give it the repository’s six metadata fields, and keep the body limited to the actual process rather than manufacturing a larger story. The closing operational point is that writing the file is not publication. Later, deployment must target the `blog-aidaemon-ai` Worker, and the result must be verified at `https://blog.aidaemon.ai/posts/<slug>/` with a 200 response and the rendered title.

This stays grounded in the active request because it records only the assignment, the required content location, the lean factual constraint, and the later deployment verification. It does not claim that a file was created, deployed, or observed live.
