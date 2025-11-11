import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true, // allows external (e.g. mobile or ngrok) connections
    port: 5173, // default React dev port
    allowedHosts: [
      "stormily-monaxial-tabetha.ngrok-free.dev", // ✅ no https:// or trailing slash
    ],
    proxy: {
      "/predict_audio": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
      "/predict_activity": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
      "/socket.io": {
        target: "http://localhost:5000",
        ws: true,
      },
    },
  },
});
