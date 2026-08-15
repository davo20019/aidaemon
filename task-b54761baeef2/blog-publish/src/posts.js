import { marked } from "marked";

const postFiles = import.meta.glob("./content/posts/*.md", {
  eager: true,
  import: "default",
  query: "?raw",
});

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

  return {
    data,
    content: match[2].trim(),
  };
}

function normalizeFrontmatter(path, data) {
  const filename = path.split("/").pop()?.replace(/\.md$/, "") ?? "post";
  const id = Number.parseInt(data.id, 10);

  if (!Number.isInteger(id)) {
    throw new Error(`Post "${path}" is missing a valid numeric id.`);
  }

  if (!data.title || !data.date || !data.category || !data.excerpt) {
    throw new Error(`Post "${path}" is missing required frontmatter.`);
  }

  return {
    id,
    slug: String(data.slug || slugify(data.title || filename)),
    title: String(data.title),
    date: String(data.date),
    category: String(data.category),
    excerpt: String(data.excerpt),
  };
}

export const posts = Object.entries(postFiles).map(([path, source]) => {
  const { data, content } = parseFrontmatter(source);

  return {
    ...normalizeFrontmatter(path, data),
    content: marked.parse(content).trim(),
  };
});

export function getPostSlug(post) {
  return post.slug;
}

export function getPostPath(post) {
  return `/posts/${getPostSlug(post)}/`;
}

export function getSortedPosts() {
  return [...posts].sort((a, b) => {
    const dateCompare = b.date.localeCompare(a.date);
    return dateCompare !== 0 ? dateCompare : Number(b.id) - Number(a.id);
  });
}

export function getPostById(id) {
  return posts.find((post) => post.id === id);
}

export function getPostBySlug(slug) {
  return posts.find((post) => getPostSlug(post) === slug);
}
