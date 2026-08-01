const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 816, height: 1056 }, deviceScaleFactor: 1 });
  await page.goto('file://' + path.resolve('imerkar-investor-narrative-v5.html'), { waitUntil: 'networkidle' });
  await page.emulateMedia({ media: 'screen' });
  await page.pdf({
    path: '/tmp/imerkar-investor-narrative-v5.pdf',
    width: '8.5in', height: '11in',
    margin: { top: '0in', right: '0in', bottom: '0in', left: '0in' },
    printBackground: true, preferCSSPageSize: true
  });
  await browser.close();
})();
