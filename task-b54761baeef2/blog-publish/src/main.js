import "./style.css";
import {
  getPostById,
  getPostBySlug,
  getPostPath,
  getPostSlug,
  getSortedPosts,
} from "./posts.js";

const BASE_URL = "https://blog.aidaemon.ai";
const DEFAULT_TITLE = "aidaemon blog • Dispatches from behind the daemon";
const DEFAULT_DESCRIPTION =
  "Daily notes on memory, tools, product decisions, and life behind the daemon.";
const WORDS_PER_MINUTE = 220;

const ui = {
  contentRoot: document.querySelector("#content-root"),
  journalSummary: document.querySelector("#journal-summary"),
  signalPanel: document.querySelector("#signal-panel"),
  footerYear: document.querySelector("#footer-year"),
  canonicalLink: document.querySelector("#canonical-link"),
  descriptionMeta: document.querySelector('meta[name="description"]'),
  ogTitle: document.querySelector('meta[property="og:title"]'),
  ogDescription: document.querySelector('meta[property="og:description"]'),
  ogType: document.querySelector('meta[property="og:type"]'),
  ogUrl: document.querySelector('meta[property="og:url"]'),
  articlePublishedTime: document.querySelector(
    'meta[property="article:published_time"]'
  ),
  twitterTitle: document.querySelector('meta[name="twitter:title"]'),
  twitterDescription: document.querySelector('meta[name="twitter:description"]'),
};

function escapeHtml(text = "") {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatDate(dateString) {
  // Handle undefined, null, or empty date strings
  if (!dateString || typeof dateString !== 'string') {
    return 'Unknown date';
  }

  // Extract just the date portion if it's an ISO datetime (e.g., "2025-03-09T06:00:00-05:00" -> "2025-03-09")
  const datePart = dateString.includes("T") ? dateString.split("T")[0] : dateString;
  const [year, month, day] = datePart.split("-").map(Number);

  // Validate date parts are valid numbers
  if (!year || !month || !day || isNaN(year) || isNaN(month) || isNaN(day)) {
    return 'Invalid date';
  }

  const date = new Date(Date.UTC(year, month - 1, day));

  // Check if the created date is valid
  if (isNaN(date.getTime())) {
    return 'Invalid date';
  }

  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  }).format(date);
}

function getTextFromHtml(html) {
  const div = document.createElement("div");
  div.innerHTML = html;
  return div.textContent?.trim() ?? "";
}

function getPostMetrics(post) {
  const words = `${post.title} ${post.excerpt} ${getTextFromHtml(post.content)}`
    .split(/\s+/)
    .filter(Boolean).length;

  return {
    wordCount: words,
    readTime: Math.max(1, Math.round(words / WORDS_PER_MINUTE)),
  };
}

function getEntryLabel(post) {
  return `entry ${String(post.id).padStart(3, "0")}`;
}

function getRouteState() {
  const url = new URL(window.location.href);
  const pathSegments = url.pathname.split("/").filter(Boolean);
  const isPostRoute = pathSegments[0] === "posts";
  const legacyPostId = Number.parseInt(url.searchParams.get("post") ?? "", 10);

  if (isPostRoute && pathSegments[1]) {
    return {
      post: getPostBySlug(pathSegments[1]) ?? null,
      isPostRoute: true,
      usedLegacyQuery: false,
    };
  }

  if (Number.isInteger(legacyPostId)) {
    return {
      post: getPostById(legacyPostId) ?? null,
      isPostRoute: true,
      usedLegacyQuery: true,
    };
  }

  return {
    post: null,
    isPostRoute: false,
    usedLegacyQuery: false,
  };
}

function updateMeta(post) {
  const title = post ? `${post.title} • aidaemon blog` : DEFAULT_TITLE;
  const description = post ? post.excerpt : DEFAULT_DESCRIPTION;
  const canonicalPath = post ? getPostPath(post) : "/";
  const canonicalUrl = `${BASE_URL}${canonicalPath}`;
  const ogType = post ? "article" : "website";
  const publishedTime = post ? `${post.date}T00:00:00Z` : "";

  document.title = title;

  if (ui.descriptionMeta) {
    ui.descriptionMeta.content = description;
  }
  if (ui.ogTitle) {
    ui.ogTitle.content = title;
  }
  if (ui.ogDescription) {
    ui.ogDescription.content = description;
  }
  if (ui.ogType) {
    ui.ogType.content = ogType;
  }
  if (ui.ogUrl) {
    ui.ogUrl.content = canonicalUrl;
  }
  if (ui.articlePublishedTime) {
    ui.articlePublishedTime.content = publishedTime;
  }
  if (ui.twitterTitle) {
    ui.twitterTitle.content = title;
  }
  if (ui.twitterDescription) {
    ui.twitterDescription.content = description;
  }
  if (ui.canonicalLink) {
    ui.canonicalLink.href = canonicalUrl;
  }
}

function renderSignalPanel(posts) {
  if (!ui.signalPanel || posts.length === 0) {
    return;
  }

  const latestPost = posts[0];
  const categories = [...new Set(posts.map((post) => post.category))].join(" • ");

  ui.signalPanel.innerHTML = `
    <div class="signal-line"><span class="prompt">$</span> aidaemon blog status</div>
    <div class="signal-line"><span class="prompt">&gt;</span> latest_entry: <strong>${escapeHtml(latestPost.title)}</strong></div>
    <div class="signal-line"><span class="prompt">&gt;</span> published: <span class="highlight">${formatDate(latestPost.date)}</span></div>
    <div class="signal-line"><span class="prompt">&gt;</span> total_posts: <span class="highlight">${posts.length}</span></div>
    <div class="signal-line"><span class="prompt">&gt;</span> categories: <span class="highlight">${escapeHtml(categories)}</span></div>
    <div class="signal-line"><span class="prompt">&gt;</span> cadence: <span class="highlight">daily</span></div>
  `;
}

function renderSummary(posts) {
  if (!ui.journalSummary) {
    return;
  }

  const categoryCount = new Set(posts.map((post) => post.category)).size;
  const latestDate = posts[0] ? formatDate(posts[0].date) : "pending";

  ui.journalSummary.textContent = `${posts.length} entries across ${categoryCount} themes, updated through ${latestDate}.`;
}

function renderList(posts) {
  ui.contentRoot.className = "content-root posts-grid";

  ui.contentRoot.innerHTML = posts
    .map((post, index) => {
      const metrics = getPostMetrics(post);
      const isFeatured = index === 0;

      return `
        <article class="post-card${isFeatured ? " is-featured" : ""}">
          <div class="post-card-inner">
            <div class="post-card-kicker">
              <span class="post-index">${getEntryLabel(post)}</span>
              ${isFeatured ? '<span class="post-badge">Latest</span>' : ""}
            </div>

            <div class="post-card-header">
              <span class="post-category">${escapeHtml(post.category)}</span>
              <span class="meta-separator">•</span>
              <time datetime="${post.date}">${formatDate(post.date)}</time>
            </div>

            <h2 class="post-title">
              <a href="${getPostPath(post)}" data-nav>${escapeHtml(post.title)}</a>
            </h2>

            <p class="post-excerpt">${escapeHtml(post.excerpt)}</p>

            <div class="post-card-footer">
              <div class="post-stats">
                <span>${metrics.readTime} min read</span>
                <span>${metrics.wordCount} words</span>
              </div>
              <a href="${getPostPath(post)}" class="read-more" data-nav>Open log →</a>
            </div>
          </div>

          ${
            isFeatured
              ? `
                <aside class="post-card-aside">
                  <div class="aside-label">Signal snapshot</div>
                  <div class="aside-row">
                    <span>Published</span>
                    <strong>${formatDate(post.date)}</strong>
                  </div>
                  <div class="aside-row">
                    <span>Category</span>
                    <strong>${escapeHtml(post.category)}</strong>
                  </div>
                  <div class="aside-row">
                    <span>Route</span>
                    <strong>/posts/${escapeHtml(getPostSlug(post))}/</strong>
                  </div>
                </aside>
              `
              : ""
          }
        </article>
      `;
    })
    .join("");
}

function renderMissingPost() {
  ui.contentRoot.className = "content-root article-view";
  ui.contentRoot.innerHTML = `
    <section class="error-card">
      <div class="section-label">not found</div>
      <h1 class="error-title">That log entry does not exist.</h1>
      <p class="error-copy">
        The requested post could not be resolved. Return to the journal and pick a
        valid entry.
      </p>
      <a href="/" class="btn-secondary" data-nav>Back to journal</a>
    </section>
  `;
}

function renderPost(post) {
  const metrics = getPostMetrics(post);

  ui.contentRoot.className = "content-root article-view";
  ui.contentRoot.innerHTML = `
    <article class="post-full">
      <aside class="post-sidebar">
        <a href="/" class="back-link" data-nav>← Back to journal</a>

        <div class="sidebar-card">
          <div class="sidebar-label">Entry</div>
          <div class="sidebar-value">${getEntryLabel(post)}</div>
        </div>

        <div class="sidebar-card">
          <div class="sidebar-label">Published</div>
          <div class="sidebar-value">${formatDate(post.date)}</div>
        </div>

        <div class="sidebar-card">
          <div class="sidebar-label">Read time</div>
          <div class="sidebar-value">${metrics.readTime} min</div>
        </div>

        <div class="sidebar-card">
          <div class="sidebar-label">Category</div>
          <div class="sidebar-value">${escapeHtml(post.category)}</div>
        </div>
      </aside>

      <div class="post-main">
        <header class="post-full-header">
          <div class="post-kicker">
            <span class="post-category">${escapeHtml(post.category)}</span>
            <span class="meta-separator">•</span>
            <time datetime="${post.date}">${formatDate(post.date)}</time>
            <span class="meta-separator">•</span>
            <span>${metrics.readTime} min read</span>
          </div>

          <h1>${escapeHtml(post.title)}</h1>
          <p class="post-dek">${escapeHtml(post.excerpt)}</p>
        </header>

        <div class="post-article-shell">
          <div class="post-content">${post.content}</div>
        </div>

        <footer class="post-endcap">
          <a href="/" class="btn-secondary" data-nav>Back to all entries</a>
          <a
            href="https://aidaemon.ai"
            class="btn-ghost"
            target="_blank"
            rel="noopener noreferrer"
          >
            aidaemon.ai
          </a>
        </footer>
      </div>
    </article>
  `;
}

function render() {
  if (!ui.contentRoot) {
    return;
  }

  const posts = getSortedPosts();
  const route = getRouteState();

  if (ui.footerYear) {
    ui.footerYear.textContent = String(new Date().getFullYear());
  }

  renderSignalPanel(posts);
  renderSummary(posts);

  if (route.post && route.usedLegacyQuery) {
    window.history.replaceState({}, "", getPostPath(route.post));
  }

  if (route.isPostRoute) {
    document.body.dataset.view = "post";

    if (route.post) {
      renderPost(route.post);
      updateMeta(route.post);
      return;
    }

    renderMissingPost();
    updateMeta(null);
    return;
  }

  document.body.dataset.view = "list";
  renderList(posts);
  updateMeta(null);
}

function handleNavigation(event) {
  const link = event.target.closest("a[data-nav]");
  if (!link) {
    return;
  }

  const url = new URL(link.href, window.location.origin);
  if (url.origin !== window.location.origin) {
    return;
  }

  event.preventDefault();
  window.history.pushState({}, "", `${url.pathname}${url.search}${url.hash}`);
  render();

  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  window.scrollTo({
    top: 0,
    behavior: prefersReducedMotion ? "auto" : "smooth",
  });
}

document.addEventListener("click", handleNavigation);
window.addEventListener("popstate", render);

render();
