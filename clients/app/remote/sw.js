// The web app's service worker, on a site of its own: requests the app
// makes to `<scope>_daemon/…` (sign-in, pictures, files, downloads) go to
// the page that made them, which sends them to the daemon over iroh
// (js/browser.js) and hands the reply back. Nothing else is touched, and
// nothing is cached.

const BASE = new URL("_daemon/", self.registration.scope).pathname;
// Replies that carry no body (the Response constructor refuses one).
const EMPTY = new Set([101, 204, 205, 304]);

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin || !url.pathname.startsWith(BASE)) return;
  event.respondWith(forward(event, url));
});

async function forward(event, url) {
  const client =
    (event.clientId && (await self.clients.get(event.clientId))) ||
    (await self.clients.matchAll({ type: "window" }))[0];
  if (!client) return new Response("Open the app to reach the daemon.", { status: 503 });
  const request = event.request;
  const headers = [];
  request.headers.forEach((value, name) => headers.push(name, value));
  const body =
    request.method === "GET" || request.method === "HEAD"
      ? new Uint8Array()
      : new Uint8Array(await request.arrayBuffer());
  const reply = await new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = (message) => resolve(message.data);
    client.postMessage(
      {
        type: "aidaemon-fetch",
        method: request.method,
        path: "/" + url.pathname.slice(BASE.length) + url.search,
        headers,
        body,
      },
      [channel.port2, body.buffer],
    );
  });
  if (reply.error) return new Response(reply.error, { status: 502 });
  const replyHeaders = new Headers();
  for (let i = 0; i + 1 < reply.headers.length; i += 2) {
    replyHeaders.append(reply.headers[i], reply.headers[i + 1]);
  }
  return new Response(EMPTY.has(reply.status) ? null : reply.body, {
    status: reply.status,
    headers: replyHeaders,
  });
}
