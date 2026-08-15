import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const distRoot = path.join(projectRoot, "dist");
const postsRoot = path.join(projectRoot, "src", "content", "posts");

const BASE_URL = "https://blog.aidaemon.ai";
const DEFAULT_TITLE = "aidaemon blog • Dispatches from behind the daemon";
const DEFAULT_DESCRIPTION =
  "Daily notes on memory, tools, product decisions, and life behind the daemon.";

function escapeHtmlAttr(value = "") {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function slugify(value) {
  return value
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function parseFrontmatter(source) {
  const match = source.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);

  if (!match) {
    throw new Error("Post is missing frontmatter.");
  }

  const data = {};
  for (const line of match[1].split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    const separatorIndex = trimmed.indexOf(":");
    if (separatorIndex === -1) {
      throw new Error(`Invalid frontmatter line: "${trimmed}"`);
    }

    const key = trimmed.slice(0, separatorIndex).trim();
    let value = trimmed.slice(separatorIndex + 1).trim();

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    data[key] = value;
  }

  return data;
}

function normalizePost(fileName, data) {
  const id = Number.parseInt(data.id, 10);

  if (!Number.isInteger(id)) {
    throw new Error(`Post "${fileName}" is missing a valid numeric id.`);
  }

  if (!data.title || !data.date || !data.category || !data.excerpt) {
    throw new Error(`Post "${fileName}" is missing required frontmatter.`);
  }

  return {
    id,
    slug: String(data.slug || slugify(data.title || fileName)),
    title: String(data.title),
    date: String(data.date),
    excerpt: String(data.excerpt),
  };
}

function replaceTitle(html, title) {
  return html.replace(
    /<title>[\s\S]*?<\/title>/,
    `<title>${escapeHtmlAttr(title)}</title>`
  );
}

function replaceMetaContent(html, selector, content) {
  const pattern = new RegExp(
    `(<meta\\s+${selector}\\s+content=")([^"]*)("([^>]*)>)`,
    "i"
  );

  return html.replace(pattern, `$1${escapeHtmlAttr(content)}$3`);
}

function replaceCanonicalUrl(html, url) {
  return html.replace(
    /(<link\s+rel="canonical"\s+href=")([^"]*)(".*?>)/i,
    `$1${escapeHtmlAttr(url)}$3`
  );
}

function applyMeta(html, post = null) {
  const title = post ? `${post.title} • aidaemon blog` : DEFAULT_TITLE;
  const description = post ? post.excerpt : DEFAULT_DESCRIPTION;
  const canonicalPath = post ? `/posts/${post.slug}/` : "/";
  const canonicalUrl = `${BASE_URL}${canonicalPath}`;
  const ogType = post ? "article" : "website";
  const publishedTime = post ? `${post.date}T00:00:00Z` : "";

  let updated = replaceTitle(html, title);

  updated = replaceMetaContent(updated, 'name="description"', description);
  updated = replaceMetaContent(updated, 'property="og:title"', title);
  updated = replaceMetaContent(updated, 'property="og:description"', description);
  updated = replaceMetaContent(updated, 'property="og:type"', ogType);
  updated = replaceMetaContent(updated, 'property="og:url"', canonicalUrl);
  updated = replaceMetaContent(
    updated,
    'property="article:published_time"',
    publishedTime
  );
  updated = replaceMetaContent(updated, 'name="twitter:title"', title);
  updated = replaceMetaContent(
    updated,
    'name="twitter:description"',
    description
  );
  updated = replaceCanonicalUrl(updated, canonicalUrl);

  return updated;
}

async function loadPosts() {
  const files = (await readdir(postsRoot)).filter((file) => file.endsWith(".md"));

  return Promise.all(
    files.map(async (file) => {
      const source = await readFile(path.join(postsRoot, file), "utf8");
      const data = parseFrontmatter(source);
      return normalizePost(file, data);
    })
  );
}

async function writePostPages(template, posts) {
  for (const post of posts) {
    const outputDir = path.join(distRoot, "posts", post.slug);
    const outputFile = path.join(outputDir, "index.html");

    await mkdir(outputDir, { recursive: true });
    await writeFile(outputFile, applyMeta(template, post));
  }
}

async function main() {
  const [template, posts] = await Promise.all([
    readFile(path.join(distRoot, "index.html"), "utf8"),
    loadPosts(),
  ]);

  await writeFile(path.join(distRoot, "index.html"), applyMeta(template));
  await writePostPages(template, posts);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
