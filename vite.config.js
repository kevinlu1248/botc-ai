import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://localhost:3001", changeOrigin: true },
      // `ws: true` is required — without it Vite doesn't upgrade the connection
      // and the browser's /ws/stt socket fails with no useful error.
      "/ws": { target: "ws://localhost:3001", ws: true, changeOrigin: true },
    },
  },
});
