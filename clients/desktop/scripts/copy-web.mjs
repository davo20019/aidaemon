// The app ships its own copy of the web client: copy the built page and its
// modules (clients/web/dist) into dist/, at the same paths the gateway
// serves them, so index.html's /client/v1/app/app.js resolves in the shell.
import { copyFileSync, mkdirSync, readdirSync, rmSync } from "node:fs";

const web = new URL("../../web/", import.meta.url);
const dist = new URL("../dist/", import.meta.url);
const app = new URL("client/v1/app/", dist);
rmSync(dist, { recursive: true, force: true });
mkdirSync(app, { recursive: true });
const modules = readdirSync(new URL("dist/", web)).filter((n) => n.endsWith(".js") && n !== "protocol.js");
for (const name of modules) copyFileSync(new URL(`dist/${name}`, web), new URL(name, app));
copyFileSync(new URL("index.html", web), new URL("index.html", dist));
console.log(`copied index.html and ${modules.join(", ")} into clients/desktop/dist/`);
