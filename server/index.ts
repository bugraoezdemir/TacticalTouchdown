import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { serveStatic } from "./static";
import { createServer } from "http";
import { spawn } from "child_process";
import { createProxyMiddleware } from "http-proxy-middleware";

const app = express();
const httpServer = createServer(app);

// Start Python Backend
console.log("Starting Python Backend...");
const pythonProcess = spawn('python3', ['server/main.py'], {
  stdio: 'inherit'
});

pythonProcess.on('error', (err) => {
  console.error('Failed to start Python backend:', err);
});

process.on('exit', () => {
    pythonProcess.kill();
});

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

export function log(message: string, source = "express") {
  const formattedTime = new Date().toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });

  console.log(`${formattedTime} [${source}] ${message}`);
}

// Proxy API requests to Python
// Mount at /api so requests to /api/... go to http://127.0.0.1:5001/...
app.use('/api', createProxyMiddleware({
  target: 'http://127.0.0.1:5001',
  changeOrigin: true,
  pathRewrite: { '^/api': '' },  // Strip /api prefix when forwarding
  logLevel: 'debug'
}));

(async () => {
  await registerRoutes(httpServer, app);

  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";
    res.status(status).json({ message });
    throw err;
  });

  if (process.env.NODE_ENV === "production") {
    serveStatic(app);
  } else {
    const { setupVite } = await import("./vite");
    await setupVite(httpServer, app);
  }

  const port = parseInt(process.env.PORT || "5000", 10);
  httpServer.listen(
    {
      port,
      host: "0.0.0.0",
      reusePort: true,
    },
    () => {
      log(`serving on port ${port}`);
    },
  );
})();
