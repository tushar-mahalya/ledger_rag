import { readdir, readFile } from "node:fs/promises";
import { gzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";

const assets = fileURLToPath(new URL("../dist/assets/", import.meta.url));
let total = 0;
let largest = 0;
for (const file of await readdir(assets)) {
  if (!/\.(js|css)$/.test(file)) continue;
  const source = await readFile(`${assets}/${file}`);
  total += gzipSync(source).length;
  if (file.endsWith(".js")) largest = Math.max(largest, source.length);
}
if (!total || !largest || total > 200 * 1024 || largest > 450 * 1024) {
  throw new Error(
    `Frontend budget exceeded: total gzip ${total} bytes; largest JS ${largest} bytes`,
  );
}
console.log(
  JSON.stringify({
    total_js_css_gzip_bytes: total,
    largest_js_bytes: largest,
    gzip_budget_bytes: 204800,
    js_chunk_budget_bytes: 460800,
  }),
);
