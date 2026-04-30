import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite + React with a /api proxy so the dev server can talk to the
// FastAPI backend on localhost:8000 without CORS gymnastics.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://backend:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./src/test-setup.ts",
    css: false,
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    exclude: ["node_modules/**", "node_modules.broken/**", "dist/**"],
  },
});
