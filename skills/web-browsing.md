---
name: web-browsing
description: How to use the browser tool for web tasks
triggers: browse, website, web, url, search, google, navigate, screenshot
---
When the user asks you to visit a website, search the web, or interact with a web page,
ALWAYS use the `browser` tool — never use `terminal` with curl/wget.

### Patterns

**Google Search:**
1. `browser` → navigate to `https://www.google.com/search?q=<url-encoded query>`
2. `browser` → screenshot to see results
3. `browser` → click on a result link if needed

**Visit a URL:**
1. `browser` → navigate to the URL
2. `browser` → screenshot to see the page
3. `browser` → get_text if the user wants content extracted

**Fill a Form:**
1. `browser` → navigate to the page
2. `browser` → fill each input field by CSS selector
3. `browser` → click the submit button
4. `browser` → screenshot to confirm result

### Rules
- Always take a screenshot after navigation so you can see what loaded.
- If a page requires scrolling, use execute_js to scroll down and screenshot again.
- Chain multiple browser actions in sequence — the session persists between calls.
- If the page requires login, tell the user and ask for credentials or suggest using an existing Chrome profile.
