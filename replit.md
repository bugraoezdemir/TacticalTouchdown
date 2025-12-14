# Tactical Master - Football Coach Simulator

## Overview

Tactical Master is a football (soccer) simulation game where players take on the role of a team coach. The application features a real-time match simulation with a visual pitch display, tactical controls, and game state management. The architecture combines a React frontend for the UI with a Python Flask backend handling the game engine physics and AI decision-making.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React with TypeScript, using Vite as the build tool
- **Routing**: Wouter for lightweight client-side routing
- **State Management**: Zustand store (`gameStore.ts`) for local game state (players, ball, score, time)
- **Data Fetching**: TanStack React Query for server state management
- **UI Components**: Shadcn/ui component library with Radix UI primitives
- **Styling**: Tailwind CSS with custom dark theme optimized for the tactical/gaming aesthetic
- **Animations**: Framer Motion for smooth player and ball movement visualization

### Backend Architecture
- **Dual Server Setup**: Express.js (TypeScript) acts as the primary server, proxying API requests to a Python Flask backend
- **Python Game Engine**: Flask server (`server/main.py`) running on port 5001 handles game logic
- **Game Physics**: `server/game_engine.py` contains the core simulation with Player, Ball, Goalkeeper, and Game classes
- **API Proxy**: Express uses `http-proxy-middleware` to forward `/api/*` requests to the Python backend

### Data Flow
1. Frontend makes API calls to `/api/state`, `/api/tick`, `/api/reset`
2. Express proxies these to Flask backend
3. Flask returns game state as JSON (player positions, ball position, score)
4. Frontend renders the state on a visual pitch using animated components

### Database
- **ORM**: Drizzle ORM configured for PostgreSQL
- **Schema**: Currently minimal - users table with id, username, password
- **Storage**: In-memory storage implementation exists (`MemStorage`) as fallback

### Build System
- **Development**: Vite dev server for frontend, tsx for TypeScript execution
- **Production**: Custom build script using esbuild for server bundling, Vite for client build
- **Output**: Bundled to `dist/` directory with static assets in `dist/public/`

## External Dependencies

### Database
- PostgreSQL (via `DATABASE_URL` environment variable)
- Drizzle Kit for schema migrations

### Python Backend
- Flask for HTTP API
- Flask-CORS for cross-origin requests
- NumPy for game physics calculations

### Frontend Libraries
- React 18 with React DOM
- TanStack React Query for data fetching
- Radix UI primitives for accessible components
- Framer Motion for animations
- Zustand for state management

### Development Tools
- Vite with React plugin and Tailwind CSS plugin
- Replit-specific plugins for development (cartographer, dev-banner, runtime-error-modal)
- TypeScript with strict mode enabled