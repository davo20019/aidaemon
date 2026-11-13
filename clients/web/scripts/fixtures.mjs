// Run the shared reducer fixtures against the compiled TypeScript reducer
// (dist/state.js, from `tsc -p .`). Each fixture is a list of server frames
// and the state the Rust reducer leaves; this one must leave the same.
// Keys whose value is null count as absent (Rust omits most empty options).
import { readdirSync, readFileSync } from "node:fs";
import { apply, newState } from "../dist/state.js";

const dir = new URL("../../../crates/aidaemon-client-core/tests/fixtures/reducer/", import.meta.url);

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    const out = {};
    for (const key of Object.keys(value).sort()) {
      if (value[key] === null || value[key] === undefined) continue;
      out[key] = canonical(value[key]);
    }
    return out;
  }
  return value;
}

const files = readdirSync(dir).filter((name) => name.endsWith(".golden")).sort();
if (files.length < 8) {
  console.error(`expected at least 8 reducer fixtures, found ${files.length}`);
  process.exit(1);
}
let failed = 0;
for (const name of files) {
  const fixture = JSON.parse(readFileSync(new URL(name, dir), "utf8"));
  const state = newState();
  for (const frame of fixture.frames) apply(state, frame);
  const got = JSON.stringify(canonical(state), null, 1);
  const want = JSON.stringify(canonical(fixture.state), null, 1);
  if (got === want) {
    console.log(`ok   ${name}`);
  } else {
    failed += 1;
    console.log(`FAIL ${name}\n--- want\n${want}\n--- got\n${got}`);
  }
}
console.log(`${files.length - failed} of ${files.length} reducer fixtures pass`);
process.exit(failed ? 1 : 0);
