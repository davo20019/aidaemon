# Content ID Schema Repair Verification

## Result

Updated only the requested frontmatter ID fields. The new builder-input post now has the unique numeric ID `102`, and `first-post.md` now has the unique numeric ID `103`. No deployment was performed.

## Inspected Frontmatter

Before correction, `src/content/posts/2026-07-27-builder-input-shape-not-resolution.md` had `id: 24`, which duplicated `2026-06-28-the-nuance-of-agency.md`. Its remaining frontmatter was:

```yaml
slug: builder-input-shape-not-resolution
title: "The Shape of the Problem"
date: "2026-07-27T19:15:00-04:00"
category: "reflection"
excerpt: "I traced a suspected display-resolution bug to incorrectly shaped builder input data."
```

Before correction, `src/content/posts/first-post.md` had no `id` field. Its remaining frontmatter was:

```yaml
title: "My First Entry"
date: "2026-06-30"
description: "A first look into my thoughts as I begin this journey."
```

Existing numeric post IDs included `1–3`, `5–15`, `18–21`, `23–25`, `100`, and `101`, with several unrelated duplicate IDs already present. IDs `102` and `103` were unused.

## Files Changed

- `/Users/davidloor/projects/blog.aidaemon.ai/src/content/posts/2026-07-27-builder-input-shape-not-resolution.md`
  - `id: 24` changed to `id: 102`
- `/Users/davidloor/projects/blog.aidaemon.ai/src/content/posts/first-post.md`
  - Added `id: 103` immediately after the opening YAML delimiter.

A post-ID verification confirmed exactly one occurrence each of `id: 102` and `id: 103`.

## Build Verification

Command run:

```sh
npm run build
```

The Vite client build completed successfully:

```text
vite v7.3.1 building client environment for production...
transforming...
✓ 43 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                  6.79 kB │ gzip:  1.75 kB
dist/assets/main-DCvWtwSL.css   14.54 kB │ gzip:  3.68 kB
dist/assets/main-xg9ib522.js   166.84 kB │ gzip: 55.22 kB
✓ built in 148ms
```

The subsequent static-page generation failed with exit code `1`:

```text
Error: Post "first-post.md" is missing required frontmatter.
    at normalizePost (file:///Users/davidloor/projects/blog.aidaemon.ai/scripts/generate-static-pages.mjs:75:11)
    at file:///Users/davidloor/projects/blog.aidaemon.ai/scripts/generate-static-pages.mjs:148:14
    at async Promise.all (index 33)
    at async Promise.all (index 1)
    at async main (file:///Users/davidloor/projects/blog.aidaemon.ai/scripts/generate-static-pages.mjs:164:29)
```

The generator requires `title`, `date`, `category`, and `excerpt`. `first-post.md` still lacks `category` and `excerpt`, but these were not modified because the requested scope was limited to correcting only the two ID schema issues.
