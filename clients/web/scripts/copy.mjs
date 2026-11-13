// Put the built page where the gateway serves it from (include_str!): the
// page and every compiled module except the type-only protocol.js.
import { copyFileSync, readdirSync } from "node:fs";

const to = new URL("../../../crates/aidaemon-client-gateway/web/", import.meta.url);
const dist = new URL("../dist/", import.meta.url);
const modules = readdirSync(dist).filter((name) => name.endsWith(".js") && name !== "protocol.js");
for (const name of modules) copyFileSync(new URL(name, dist), new URL(name, to));
copyFileSync(new URL("../index.html", import.meta.url), new URL("index.html", to));
console.log(`copied index.html and ${modules.join(", ")} to crates/aidaemon-client-gateway/web/`);
