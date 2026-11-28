/*
 * qr.js: a small, dependency-free QR code encoder for the e-ink page.
 *
 * It runs on old browsers (the Kindle Scribe's Chrome 80): a classic script,
 * ES5-style syntax, no modules, no optional chaining, no WebAssembly.
 *
 * What it encodes (ISO/IEC 18004 Model 2):
 *   - byte mode, the text as UTF-8;
 *   - error correction level M;
 *   - the smallest version (1..40) that holds the data;
 *   - the mask with the lowest penalty score (the four standard rules).
 *
 * API (global `AidaemonQr`, and `module.exports` under Node for tests):
 *   AidaemonQr.matrix(text[, opts]) -> { size, version, mask, get(x, y) }
 *       get(x, y) is true for a dark module; x is the column, y the row.
 *       The quiet zone is not part of the matrix. `opts` is for tests:
 *       { version: n } forces a version, { mask: n } forces a mask (0..7).
 *   AidaemonQr.drawToCanvas(canvas, text[, moduleSize])
 *       Sizes the canvas and draws black modules on white with a 4-module
 *       quiet zone, on integer pixel coordinates (no anti-aliasing).
 *       Returns the matrix.
 */
(function (root) {
  'use strict';

  // ---------------------------------------------------------------------
  // Error correction level M block structure, per version (index 0 is
  // version 1): [EC codewords per block, blocks in group 1, data codewords
  // per group-1 block, blocks in group 2, data codewords per group-2 block].
  // Group-2 blocks hold one more data codeword than group-1 blocks.
  // ---------------------------------------------------------------------
  var EC_M_BLOCKS = [
    [10, 1, 16, 0, 0], [16, 1, 28, 0, 0], [26, 1, 44, 0, 0], [18, 2, 32, 0, 0],
    [24, 2, 43, 0, 0], [16, 4, 27, 0, 0], [18, 4, 31, 0, 0], [22, 2, 38, 2, 39],
    [22, 3, 36, 2, 37], [26, 4, 43, 1, 44], [30, 1, 50, 4, 51], [22, 6, 36, 2, 37],
    [22, 8, 37, 1, 38], [24, 4, 40, 5, 41], [24, 5, 41, 5, 42], [28, 7, 45, 3, 46],
    [28, 10, 46, 1, 47], [26, 9, 43, 4, 44], [26, 3, 44, 11, 45], [26, 3, 41, 13, 42],
    [26, 17, 42, 0, 0], [28, 17, 46, 0, 0], [28, 4, 47, 14, 48], [28, 6, 45, 14, 46],
    [28, 8, 47, 13, 48], [28, 19, 46, 4, 47], [28, 22, 45, 3, 46], [28, 3, 45, 23, 46],
    [28, 21, 45, 7, 46], [28, 19, 47, 10, 48], [28, 2, 46, 29, 47], [28, 10, 46, 23, 47],
    [28, 14, 46, 21, 47], [28, 14, 46, 23, 47], [28, 12, 47, 26, 48], [28, 6, 47, 34, 48],
    [28, 29, 46, 14, 47], [28, 13, 46, 32, 47], [28, 40, 47, 7, 48], [28, 18, 47, 31, 48]
  ];

  // Format information bits for level M: the two EC-level bits are 00.
  var EC_LEVEL_M_BITS = 0;

  function sizeOf(version) {
    return version * 4 + 17;
  }

  function dataCodewords(version) {
    var b = EC_M_BLOCKS[version - 1];
    return b[1] * b[2] + b[3] * b[4];
  }

  // Number of modules left for data and EC codewords (and remainder bits)
  // once the function patterns are placed. Used to self-check the table.
  function rawDataModules(version) {
    var n = sizeOf(version);
    var result = n * n;
    result -= 3 * 64; // three finder patterns with their separators
    result -= 2 * (n - 16); // two timing patterns (minus finder overlap)
    result -= 31; // two copies of format info, plus the dark module
    if (version >= 2) {
      var align = Math.floor(version / 7) + 2;
      result -= (align * align - 3) * 25; // alignment patterns, minus corners
      result += (align - 2) * 2 * 5; // ...that overlap the timing patterns
      if (version >= 7) result -= 36; // two copies of version info
    }
    return result;
  }

  // Byte mode character-count field width.
  function countBits(version) {
    return version <= 9 ? 8 : 16;
  }

  // ---------------------------------------------------------------------
  // UTF-8
  // ---------------------------------------------------------------------
  function utf8Bytes(text) {
    var out = [];
    for (var i = 0; i < text.length; i++) {
      var c = text.charCodeAt(i);
      // Join a surrogate pair into one code point; a lone surrogate
      // becomes U+FFFD, as TextEncoder does.
      if (c >= 0xd800 && c <= 0xdbff && i + 1 < text.length) {
        var d = text.charCodeAt(i + 1);
        if (d >= 0xdc00 && d <= 0xdfff) {
          c = 0x10000 + ((c - 0xd800) << 10) + (d - 0xdc00);
          i++;
        }
      }
      if (c >= 0xd800 && c <= 0xdfff) c = 0xfffd;
      if (c < 0x80) {
        out.push(c);
      } else if (c < 0x800) {
        out.push(0xc0 | (c >> 6), 0x80 | (c & 63));
      } else if (c < 0x10000) {
        out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
      } else {
        out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 63),
          0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
      }
    }
    return out;
  }

  // ---------------------------------------------------------------------
  // Reed-Solomon over GF(256), field polynomial x^8+x^4+x^3+x^2+1 (0x11d),
  // generator element alpha = 2.
  // ---------------------------------------------------------------------
  var GF_EXP = new Array(512);
  var GF_LOG = new Array(256);
  (function () {
    var x = 1;
    for (var i = 0; i < 255; i++) {
      GF_EXP[i] = x;
      GF_LOG[x] = i;
      x <<= 1;
      if (x & 0x100) x ^= 0x11d;
    }
    // Doubled so gfMul can skip a modulo.
    for (var j = 255; j < 512; j++) GF_EXP[j] = GF_EXP[j - 255];
  })();

  function gfMul(a, b) {
    if (a === 0 || b === 0) return 0;
    return GF_EXP[GF_LOG[a] + GF_LOG[b]];
  }

  // Generator polynomial (x - a^0)(x - a^1)...(x - a^(degree-1)), as its
  // coefficients from the highest power down, without the leading 1.
  function rsGenerator(degree) {
    var poly = [1];
    for (var i = 0; i < degree; i++) {
      var next = new Array(poly.length + 1);
      for (var k = 0; k < next.length; k++) next[k] = 0;
      for (var j = 0; j < poly.length; j++) {
        next[j] ^= poly[j];
        next[j + 1] ^= gfMul(poly[j], GF_EXP[i]);
      }
      poly = next;
    }
    return poly.slice(1);
  }

  // EC codewords: the remainder of data(x) * x^degree divided by the
  // generator, computed as a linear-feedback shift register.
  function rsRemainder(data, generator) {
    var rem = [];
    for (var i = 0; i < generator.length; i++) rem.push(0);
    for (var d = 0; d < data.length; d++) {
      var factor = data[d] ^ rem.shift();
      rem.push(0);
      for (var g = 0; g < generator.length; g++) {
        rem[g] ^= gfMul(generator[g], factor);
      }
    }
    return rem;
  }

  // ---------------------------------------------------------------------
  // Data codewords: mode, count, payload, terminator, padding.
  // ---------------------------------------------------------------------
  function pickVersion(byteCount) {
    for (var v = 1; v <= 40; v++) {
      if (fits(byteCount, v)) return v;
    }
    throw new Error('AidaemonQr: text is too long for a QR code (' + byteCount + ' bytes)');
  }

  function fits(byteCount, version) {
    if (byteCount >= (1 << countBits(version))) return false;
    return 4 + countBits(version) + 8 * byteCount <= dataCodewords(version) * 8;
  }

  function dataCodewordsFor(bytes, version) {
    var bits = [];
    function push(value, length) {
      for (var i = length - 1; i >= 0; i--) bits.push((value >>> i) & 1);
    }
    push(4, 4); // mode indicator 0100: byte mode
    push(bytes.length, countBits(version));
    for (var i = 0; i < bytes.length; i++) push(bytes[i], 8);

    var capacity = dataCodewords(version) * 8;
    push(0, Math.min(4, capacity - bits.length)); // terminator
    push(0, (8 - bits.length % 8) % 8); // pad to a byte boundary

    var words = [];
    for (var b = 0; b < bits.length; b += 8) {
      var w = 0;
      for (var k = 0; k < 8; k++) w = (w << 1) | bits[b + k];
      words.push(w);
    }
    // Alternate pad codewords 11101100 and 00010001 up to capacity.
    for (var p = 0; words.length < capacity / 8; p++) {
      words.push(p % 2 === 0 ? 0xec : 0x11);
    }
    return words;
  }

  // Split into blocks, add EC to each, and interleave: the data codewords
  // column by column across blocks, then the EC codewords the same way.
  function finalCodewords(data, version) {
    var spec = EC_M_BLOCKS[version - 1];
    var ecLen = spec[0];
    var generator = rsGenerator(ecLen);
    var blocks = [];
    var offset = 0;
    var groups = [[spec[1], spec[2]], [spec[3], spec[4]]];
    for (var g = 0; g < 2; g++) {
      for (var n = 0; n < groups[g][0]; n++) {
        var chunk = data.slice(offset, offset + groups[g][1]);
        offset += groups[g][1];
        blocks.push({ data: chunk, ec: rsRemainder(chunk, generator) });
      }
    }

    var out = [];
    var maxData = Math.max(spec[2], spec[4]);
    for (var i = 0; i < maxData; i++) {
      for (var b = 0; b < blocks.length; b++) {
        if (i < blocks[b].data.length) out.push(blocks[b].data[i]);
      }
    }
    for (var e = 0; e < ecLen; e++) {
      for (var c = 0; c < blocks.length; c++) out.push(blocks[c].ec[e]);
    }
    return out;
  }

  // ---------------------------------------------------------------------
  // The matrix. Modules are stored row-major: index y * size + x.
  // ---------------------------------------------------------------------
  function Grid(version) {
    this.version = version;
    this.size = sizeOf(version);
    var total = this.size * this.size;
    this.dark = new Array(total);
    this.isFunction = new Array(total);
    for (var i = 0; i < total; i++) {
      this.dark[i] = false;
      this.isFunction[i] = false;
    }
  }

  Grid.prototype.setFunction = function (x, y, dark) {
    var i = y * this.size + x;
    this.dark[i] = dark;
    this.isFunction[i] = true;
  };

  // Alignment pattern centre coordinates (the same list for rows and
  // columns): 6, then evenly spaced (even steps) ending at size - 7.
  function alignmentPositions(version) {
    if (version === 1) return [];
    var count = Math.floor(version / 7) + 2;
    var size = sizeOf(version);
    // Version 32 is the one exception to the rounding rule.
    var step = version === 32 ? 26
      : Math.ceil((version * 4 + 4) / (count * 2 - 2)) * 2;
    var result = [6];
    for (var pos = size - 7; result.length < count; pos -= step) {
      result.splice(1, 0, pos);
    }
    return result;
  }

  // BCH remainder of `value` (already shifted left by the degree) by `poly`.
  function bchRemainder(value, poly) {
    var degree = -1;
    for (var t = poly; t > 0; t >>= 1) degree++;
    for (var bit = 30; bit >= degree; bit--) {
      if ((value >>> bit) & 1) value ^= poly << (bit - degree);
    }
    return value;
  }

  Grid.prototype.drawFunctionPatterns = function () {
    var size = this.size;
    var x, y, i, j;

    // Timing patterns: row 6 and column 6 alternate, dark on even indexes.
    for (i = 0; i < size; i++) {
      this.setFunction(6, i, i % 2 === 0);
      this.setFunction(i, 6, i % 2 === 0);
    }

    // Finder patterns with their light separators, in three corners.
    var corners = [[3, 3], [size - 4, 3], [3, size - 4]];
    for (var c = 0; c < 3; c++) {
      for (j = -4; j <= 4; j++) {
        for (i = -4; i <= 4; i++) {
          x = corners[c][0] + i;
          y = corners[c][1] + j;
          if (x < 0 || y < 0 || x >= size || y >= size) continue;
          var ring = Math.max(Math.abs(i), Math.abs(j)); // 0..4
          this.setFunction(x, y, ring !== 2 && ring !== 4);
        }
      }
    }

    // Alignment patterns: 5x5, everywhere on the position grid except
    // where they would overlap a finder pattern.
    var pos = alignmentPositions(this.version);
    var last = pos.length - 1;
    for (var a = 0; a < pos.length; a++) {
      for (var b = 0; b < pos.length; b++) {
        if ((a === 0 && b === 0) || (a === 0 && b === last) || (a === last && b === 0)) continue;
        for (j = -2; j <= 2; j++) {
          for (i = -2; i <= 2; i++) {
            this.setFunction(pos[a] + i, pos[b] + j,
              Math.max(Math.abs(i), Math.abs(j)) !== 1);
          }
        }
      }
    }

    // Reserve the format areas (filled in after masking) and the version
    // areas, and set the dark module.
    this.drawFormatBits(0);
    this.drawVersion();
  };

  // 15 format bits: EC level (2) + mask (3), a BCH(15,5) code with
  // generator 0x537, XORed with 0x5412 so the result is never all zero.
  Grid.prototype.drawFormatBits = function (mask) {
    var info = (EC_LEVEL_M_BITS << 3) | mask;
    var bits = ((info << 10) | bchRemainder(info << 10, 0x537)) ^ 0x5412;
    var size = this.size;
    var i;
    function bit(k) { return ((bits >>> k) & 1) === 1; }

    // First copy, around the top-left finder (skipping the timing row
    // and column at index 6).
    for (i = 0; i <= 5; i++) this.setFunction(8, i, bit(i));
    this.setFunction(8, 7, bit(6));
    this.setFunction(8, 8, bit(7));
    this.setFunction(7, 8, bit(8));
    for (i = 9; i < 15; i++) this.setFunction(14 - i, 8, bit(i));

    // Second copy, split between the top-right and bottom-left finders.
    for (i = 0; i < 8; i++) this.setFunction(size - 1 - i, 8, bit(i));
    for (i = 8; i < 15; i++) this.setFunction(8, size - 15 + i, bit(i));
    this.setFunction(8, size - 8, true); // the dark module, always dark
  };

  // Versions 7 and up carry 18 version bits: the version (6) plus a
  // BCH(18,6) code with generator 0x1f25, in two 6x3 blocks.
  Grid.prototype.drawVersion = function () {
    if (this.version < 7) return;
    var bits = (this.version << 12) | bchRemainder(this.version << 12, 0x1f25);
    for (var i = 0; i < 18; i++) {
      var dark = ((bits >>> i) & 1) === 1;
      var a = this.size - 11 + (i % 3);
      var b = Math.floor(i / 3);
      this.setFunction(a, b, dark); // top right
      this.setFunction(b, a, dark); // bottom left
    }
  };

  // Place codeword bits in the two-column zigzag: from the bottom-right
  // corner, up then down, right to left, skipping the vertical timing
  // column and every function module. Leftover modules are remainder
  // bits (0).
  Grid.prototype.drawCodewords = function (codewords) {
    var size = this.size;
    var bitIndex = 0;
    var totalBits = codewords.length * 8;
    for (var right = size - 1; right >= 1; right -= 2) {
      if (right === 6) right = 5;
      var upward = ((right + 1) & 2) === 0;
      for (var v = 0; v < size; v++) {
        var y = upward ? size - 1 - v : v;
        for (var k = 0; k < 2; k++) {
          var x = right - k;
          var i = y * size + x;
          if (this.isFunction[i]) continue;
          if (bitIndex < totalBits) {
            this.dark[i] = ((codewords[bitIndex >>> 3] >>> (7 - (bitIndex & 7))) & 1) === 1;
            bitIndex++;
          }
        }
      }
    }
  };

  // The eight mask conditions; i is the row (y), j the column (x).
  function maskBit(mask, j, i) {
    switch (mask) {
      case 0: return (i + j) % 2 === 0;
      case 1: return i % 2 === 0;
      case 2: return j % 3 === 0;
      case 3: return (i + j) % 3 === 0;
      case 4: return (Math.floor(i / 2) + Math.floor(j / 3)) % 2 === 0;
      case 5: return (i * j) % 2 + (i * j) % 3 === 0;
      case 6: return ((i * j) % 2 + (i * j) % 3) % 2 === 0;
      default: return ((i + j) % 2 + (i * j) % 3) % 2 === 0;
    }
  }

  // XOR the mask over the data modules. Applying it twice undoes it.
  Grid.prototype.applyMask = function (mask) {
    var size = this.size;
    for (var y = 0; y < size; y++) {
      for (var x = 0; x < size; x++) {
        var i = y * size + x;
        if (!this.isFunction[i] && maskBit(mask, x, y)) this.dark[i] = !this.dark[i];
      }
    }
  };

  // ---------------------------------------------------------------------
  // Penalty score (lower is better), with the standard weights.
  // ---------------------------------------------------------------------
  var N1 = 3;
  var N2 = 3;
  var N3 = 40;
  var N4 = 10;

  // Rules 1 and 3 along one line of modules (a row or a column).
  function linePenalty(line) {
    var score = 0;
    var n = line.length;

    // Rule 1: each run of 5+ same-colour modules scores N1 + (run - 5).
    var run = 1;
    for (var i = 1; i <= n; i++) {
      if (i < n && line[i] === line[i - 1]) {
        run++;
      } else {
        if (run >= 5) score += N1 + (run - 5);
        run = 1;
      }
    }

    // Rule 3: dark-light-dark-dark-dark-light-dark (1:1:3:1:1) with four
    // light modules on either side. Modules outside the symbol count as
    // light (the quiet zone).
    function at(k) { return k >= 0 && k < n ? line[k] : false; }
    for (var s = -4; s < n; s++) {
      if (at(s) && !at(s + 1) && at(s + 2) && at(s + 3) && at(s + 4) &&
          !at(s + 5) && at(s + 6)) {
        var lightBefore = !at(s - 1) && !at(s - 2) && !at(s - 3) && !at(s - 4);
        var lightAfter = !at(s + 7) && !at(s + 8) && !at(s + 9) && !at(s + 10);
        if (lightBefore || lightAfter) score += N3;
      }
    }
    return score;
  }

  Grid.prototype.penalty = function () {
    var size = this.size;
    var dark = this.dark;
    var score = 0;
    var x, y;

    for (y = 0; y < size; y++) score += linePenalty(dark.slice(y * size, (y + 1) * size));
    for (x = 0; x < size; x++) {
      var col = [];
      for (y = 0; y < size; y++) col.push(dark[y * size + x]);
      score += linePenalty(col);
    }

    // Rule 2: each 2x2 block of one colour scores N2.
    for (y = 0; y < size - 1; y++) {
      for (x = 0; x < size - 1; x++) {
        var c = dark[y * size + x];
        if (c === dark[y * size + x + 1] && c === dark[(y + 1) * size + x] &&
            c === dark[(y + 1) * size + x + 1]) {
          score += N2;
        }
      }
    }

    // Rule 4: N4 for every full 5% the dark share is away from 50%.
    var darkCount = 0;
    for (var i = 0; i < dark.length; i++) if (dark[i]) darkCount++;
    var percent = darkCount * 100 / dark.length;
    score += Math.floor(Math.abs(percent - 50) / 5) * N4;
    return score;
  };

  // ---------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------
  function matrix(text, opts) {
    opts = opts || {};
    var bytes = utf8Bytes(String(text));
    var version = opts.version || pickVersion(bytes.length);
    if (version < 1 || version > 40 || !fits(bytes.length, version)) {
      throw new Error('AidaemonQr: text does not fit version ' + version);
    }

    var grid = new Grid(version);
    grid.drawFunctionPatterns();
    grid.drawCodewords(finalCodewords(dataCodewordsFor(bytes, version), version));

    var mask = opts.mask;
    if (mask === undefined || mask === null) {
      // Try each mask (with its format bits, which count toward the score)
      // and keep the first one with the lowest penalty.
      var best = Infinity;
      for (var m = 0; m < 8; m++) {
        grid.applyMask(m);
        grid.drawFormatBits(m);
        var p = grid.penalty();
        if (p < best) {
          best = p;
          mask = m;
        }
        grid.applyMask(m); // undo
      }
    }
    grid.applyMask(mask);
    grid.drawFormatBits(mask);

    var size = grid.size;
    var dark = grid.dark;
    return {
      size: size,
      version: version,
      mask: mask,
      get: function (x, y) {
        return x >= 0 && y >= 0 && x < size && y < size && dark[y * size + x];
      }
    };
  }

  var QUIET_ZONE = 4;

  function drawToCanvas(canvas, text, moduleSize) {
    var m = matrix(text);
    var scale = Math.max(1, Math.floor(moduleSize || 4));
    var total = (m.size + QUIET_ZONE * 2) * scale;
    canvas.width = total;
    canvas.height = total;
    var ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, total, total);
    ctx.fillStyle = '#000000';
    for (var y = 0; y < m.size; y++) {
      for (var x = 0; x < m.size; x++) {
        if (m.get(x, y)) {
          ctx.fillRect((x + QUIET_ZONE) * scale, (y + QUIET_ZONE) * scale, scale, scale);
        }
      }
    }
    return m;
  }

  // Self-check: every version's blocks must fill its codeword capacity.
  for (var v = 1; v <= 40; v++) {
    var spec = EC_M_BLOCKS[v - 1];
    var blocks = spec[1] + spec[3];
    if (dataCodewords(v) + blocks * spec[0] !== Math.floor(rawDataModules(v) / 8)) {
      throw new Error('AidaemonQr: bad block table for version ' + v);
    }
  }

  var api = { matrix: matrix, drawToCanvas: drawToCanvas };
  root.AidaemonQr = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : this);
