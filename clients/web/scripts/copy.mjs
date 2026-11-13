// Put the built page where the gateway serves it from (include_str!).
import { copyFileSync } from "node:fs";

const to = new URL("../../../crates/aidaemon-client-gateway/web/", import.meta.url);
copyFileSync(new URL("../dist/app.js", import.meta.url), new URL("app.js", to));
copyFileSync(new URL("../index.html", import.meta.url), new URL("index.html", to));
console.log("copied app.js and index.html to crates/aidaemon-client-gateway/web/");
