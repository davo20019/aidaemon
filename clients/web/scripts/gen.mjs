// Generate src/protocol.ts from the protocol's JSON Schema. Every type in
// `$defs` is emitted (pairing and HTTP types too, which no frame reaches).
import { readFileSync, writeFileSync } from "node:fs";
import { compile } from "json-schema-to-typescript";

const source = new URL("../../../schemas/client-protocol/v1.schema.json", import.meta.url);
const text = readFileSync(source, "utf8").replaceAll("#/$defs/", "#/definitions/");
const schema = JSON.parse(text);
// The root lists every definition, so each becomes a named type; frames
// are `ClientFrame` and `ServerFrame`.
const root = {
  title: "ProtocolTypes",
  description: "An index of every protocol type (not a wire value).",
  type: "object",
  properties: Object.fromEntries(
    Object.keys(schema.$defs).map((name) => [name, { $ref: `#/definitions/${name}` }]),
  ),
  additionalProperties: false,
  definitions: schema.$defs,
};
const ts = await compile(root, "ProtocolTypes", {
  bannerComment:
    "/* Generated from schemas/client-protocol/v1.schema.json by `npm run gen`. Do not edit. */",
  additionalProperties: false,
  unreachableDefinitions: true,
});
writeFileSync(new URL("../src/protocol.ts", import.meta.url), ts);
console.log("wrote src/protocol.ts");
