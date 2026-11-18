// The Rust root from TypeScript: `nsh test` (and `nsh test --target tui`)
// runs this headless with the crate. With no daemon to reach, the app still
// opens on a new conversation.
import { expect, open, test } from "nsh/test";

test("opens on a new conversation", async () => {
  const app = await open(".");
  expect(app.texts()).toContain("What should we work on?");
});
