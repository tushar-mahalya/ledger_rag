import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = process.env.LEDGER_API_ORIGIN ?? "http://127.0.0.1:8080";
const url = new URL(backend);
if (
  !["http:", "https:"].includes(url.protocol) ||
  url.username ||
  url.password
) {
  throw new Error("LEDGER_API_ORIGIN must be a credential-free HTTP(S) origin");
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/v1/": { target: url.origin, changeOrigin: true } },
  },
  preview: { port: 4173, strictPort: true },
  build: { sourcemap: false, chunkSizeWarningLimit: 450 },
});
