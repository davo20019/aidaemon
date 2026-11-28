# qr.js verification

`cases.js` encodes 70 cases with `../public/qr.js`. They cover the pairing URL
(123 bytes, version 8), empty and short strings, UTF-8 (accents, Greek,
Japanese, emoji, a tab and a newline), strings just under and just over a
version's capacity, one string at the full capacity of each version 1 to 40,
all 8 masks forced on the URL, and a forced larger version. Each case is printed
as `<hex utf8> <version> <mask> <modules>`.

The checker is a throwaway Rust crate. It is not part of the workspace, and it
depends on `qrcode = "=0.14.1"` (the version in `Cargo.lock`), `rqrr` and
`image`. For each case it:

1. builds the `qrcode` crate's symbol with the same version, EC level M, byte
   mode and mask (`Bits::push_byte_data`, `ec::construct_codewords`,
   `Canvas::draw_data`, `Canvas::apply_mask`) and compares it with the JS output
   module for module;
2. draws the JS matrix with a 4-module quiet zone, decodes it with `rqrr` and
   checks that the decoded text matches the input.

```
node deploy/eink/qr-test/cases.js | qrcheck
PASS 70  FAIL 0  auto-mask disagreements 8
```

Seven of the eight "auto-mask disagreements" are the cases that force a mask.
The eighth is `"A"`. There the crate picks a different mask because its
dark/light balance penalty is linear (`|dark% * 2 - 100|`), while the standard
scores 10 points per full 5% away from 50%. `qr.js` uses the standard rule. Any
mask decodes correctly.
