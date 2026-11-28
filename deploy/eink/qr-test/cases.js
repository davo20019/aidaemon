// Emits QR test cases for the reference checker (see README.md), one per
// line: `<hex utf8> <version> <mask> <modules>`, modules row-major with
// '1' for dark. Run: node deploy/eink/qr-test/cases.js | <checker>
var qr = require('../public/qr.js');

var hex64 = '';
for (var h = 0; h < 64; h++) hex64 += '0123456789abcdef'[(h * 7 + 3) % 16];
var url = 'https://app.aidaemon.ai/pair-eink#n=K7F2&s=9QXM4HTAPZ6CR&k=' + hex64;

var texts = [
  url, '', 'a', 'A', '0', 'hello', 'Hello, World!', 'https://aidaemon.ai',
  '1234567890', 'HELLO WORLD 123', 'café', 'naïve résumé', 'Ελληνικά',
  '日本語のテキスト', 'emoji 😀 🎉', 'tab\there\nnewline',
  'x'.repeat(14), 'x'.repeat(15), 'x'.repeat(122), 'x'.repeat(123),
  url + '&extra=' + 'q'.repeat(40)
];
// One string near the capacity of each version 1..40.
var caps = [14, 26, 42, 62, 84, 106, 122, 152, 180, 213, 251, 287, 331, 362, 412,
  450, 504, 560, 624, 666, 711, 779, 857, 911, 997, 1059, 1125, 1190, 1264, 1370,
  1452, 1538, 1628, 1722, 1809, 1911, 1989, 2099, 2213, 2331];
for (var v = 0; v < caps.length; v++) {
  var s = '';
  for (var k = 0; k < caps[v]; k++) s += String.fromCharCode(33 + (k * 31 + v) % 94);
  texts.push(s);
}

function line(text, m) {
  var bytes = Buffer.from(text, 'utf8').toString('hex');
  var bits = '';
  for (var y = 0; y < m.size; y++) for (var x = 0; x < m.size; x++) bits += m.get(x, y) ? '1' : '0';
  return bytes + ' ' + m.version + ' ' + m.mask + ' ' + bits;
}

var out = [];
texts.forEach(function (t, i) {
  var m = qr.matrix(t);
  if (i >= 21 && m.version !== i - 20) throw new Error('capacity string ' + (i - 20) + ' got version ' + m.version);
  out.push(line(t, m));
});
// Every mask forced, on the URL.
for (var mask = 0; mask < 8; mask++) out.push(line(url, qr.matrix(url, { mask: mask })));
// A larger forced version than needed.
out.push(line('hello', qr.matrix('hello', { version: 10 })));
console.log(out.join('\n'));
