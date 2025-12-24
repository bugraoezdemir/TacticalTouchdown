# Multi-stage build for Tactical Touchdown
# Stage 1: Build frontend and install dependencies
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./
COPY vite.config.ts ./
COPY postcss.config.js ./
COPY components.json ./

# Install Node dependencies
RUN npm install

# Copy client source
COPY client ./client
COPY shared ./shared
COPY server ./server
COPY script ./script
COPY vite-plugin-meta-images.ts ./
COPY drizzle.config.ts ./

# Build the app with the custom build script
RUN npm run build

# Stage 2: Runtime environment with Node + Python
FROM node:22-alpine

# Install Python and required system packages
RUN apk add --no-cache python3 py3-pip make g++

WORKDIR /app

# Copy Node dependencies from builder
COPY package*.json ./
RUN npm install --only=production

# Copy built client and server
COPY --from=builder /app/dist ./dist

# Copy server files
COPY server ./server
COPY shared ./shared

# Copy Python dependencies
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --break-system-packages flask flask-cors numpy

# Expose ports
EXPOSE 8080

# Set environment variables
ENV NODE_ENV=production
ENV PORT=8080

# Start the server
CMD ["node", "dist/index.cjs"]
