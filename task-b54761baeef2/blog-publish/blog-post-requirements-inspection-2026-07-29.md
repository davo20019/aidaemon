# Blog Post Requirements Inspection

## Bottom Line

Existing posts use YAML front matter followed by Markdown. The consistent metadata fields are `id`, `slug`, `title`, `date`, `category`, and `excerpt`. For a concise diary entry grounded strictly in this session, write about planning the work and inspecting the blog-post format before drafting. Do not claim that a post was prepared, created, published, deployed, or that any technical issue was resolved.

## Repository Guidance

`/Users/davidloor/projects/blog.aidaemon.ai/agents.md` says posts live in `src/content/` as individual files and should follow the established format in that directory. It directs authors to use existing posts as the format reference. It does not specify a separate formal schema beyond that convention.

The guide also distinguishes content work from deployment. A new post would normally be placed in `src/content/`, use the existing filename convention, include metadata and content, and may require an update to `src/posts.js` if the post list must be regenerated. None of those actions were performed during this inspection.

## Observed Front Matter

The two inspected posts were:

- `src/content/posts/2026-07-27-builder-input-shape-not-resolution.md`
- `src/content/posts/2026-03-16-the-pause-before-action.md`

Both begin with YAML front matter delimited by `---`. The shared fields are:

```yaml
---
id: <numeric identifier>
slug: <lowercase hyphenated URL slug>
title: "<post title>"
date: "<ISO 8601 timestamp with UTC offset>"
category: "reflection"
excerpt: "<one-sentence summary>"
---
```

Use a unique numeric `id`; do not infer its next value from these two samples alone. The inspected examples use a filename beginning with a date, then a hyphenated slug, and end in `.md`.

## Observed Writing Style

Posts start with an H1 that repeats the front-matter title. They use first-person, reflective prose with short paragraphs and a concrete sequence of observation, interpretation, and takeaway.

The newer example is especially concise: four short body paragraphs followed by a horizontal rule and a single italicized closing note dated in plain language. The older example shows that longer reflection posts can add an H2 and bullet list. Both keep the excerpt factual and specific rather than promotional.

For the safest short entry, prefer the newer example's compact shape: H1, three or four short paragraphs, then an optional brief italic closing line.

## Safest Diary Topic

**Topic:** *Planning Before Drafting*

Safe factual frame: the session began with a task plan and a stated preparation task. The preparation consisted of identifying the need to inspect the repository guidance and representative posts before any writing decision. The reflection can be about the value and limits of preparation: planning establishes boundaries, while preparation should remain tied to a concrete next step.

Safe claims:

- A task plan was established.
- The work called for reviewing blog guidance and existing post conventions.
- Preparation was the claimed first step before any draft decision.

Claims to avoid:

- That a diary post was drafted, saved, published, or deployed.
- That a preparation task was completed beyond the concrete inspection described here.
- Any invented emotional state, system failure, technical diagnosis, or result not present in this session.

A concise title compatible with the observed style is **“The Plan Before the Draft.”** An equally conservative excerpt is: **“A brief reflection on setting the next step before turning preparation into action.”**
