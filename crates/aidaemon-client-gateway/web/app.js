// The AIdaemon web client's entry: find out where it runs (a browser tab,
// or the desktop shell) and start the screen on it. The pieces:
//   state.ts     the event reducer (no DOM), held to the Rust one by fixtures
//   link.ts      the browser's own socket, sign-in and resume
//   platform.ts  the only file that knows where the client runs
//   ui.ts        the screen and its keys
import { detect } from "./platform.js";
import { Screen } from "./ui.js";
const screen = new Screen(detect());
void screen.start();
