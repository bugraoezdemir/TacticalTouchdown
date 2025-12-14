import { create } from 'zustand';

export type Team = 'home' | 'away';
export type PlayerRole = 'GK' | 'DEF' | 'MID' | 'FWD';
export type GameStateType = 'playing' | 'goal_scored' | 'corner_kick' | 'goal_kick' | 'throw_in' | 'kickoff';

export interface Player {
  id: number;
  team: Team;
  role: PlayerRole;
  x: number;
  y: number;
  vx: number;
  vy: number;
  hasBall: boolean;
  number: number;
  lateralRole?: string;
  longitudinalRole?: string;
}

export interface Ball {
  x: number;
  y: number;
  vx: number;
  vy: number;
  ownerId: number | null;
}

export interface LastTouch {
  team: Team;
  playerId: number;
}

interface GameStore {
  players: Player[];
  ball: Ball;
  score: { home: number; away: number };
  time: number;
  isPlaying: boolean;
  gameSpeed: number;
  state: GameStateType;
  lastTouch: LastTouch | null;
  restartTeam: Team;

  togglePlay: () => void;
  setGameSpeed: (speed: number) => void;
  resetGame: () => void;
  tick: () => void;
  init: () => void;
}

export const useGameStore = create<GameStore>((set, get) => ({
  players: [],
  ball: { x: 50, y: 50, vx: 0, vy: 0, ownerId: null },
  score: { home: 0, away: 0 },
  time: 0,
  isPlaying: false,
  gameSpeed: 1,
  state: 'playing',
  lastTouch: null,
  restartTeam: 'home',

  togglePlay: () => set((state) => ({ isPlaying: !state.isPlaying })),
  setGameSpeed: (speed) => set({ gameSpeed: speed }),
  
  resetGame: async () => {
      try {
          const res = await fetch('/api/reset', { method: 'POST' });
          const data = await res.json();
          set({ 
            players: data.players,
            ball: data.ball,
            score: data.score,
            time: data.time,
            state: data.state || 'playing',
            lastTouch: data.lastTouch || null,
            restartTeam: data.restartTeam || 'home',
            isPlaying: false 
          });
      } catch (e) {
          console.error("Failed to reset game:", e);
      }
  },

  tick: async () => {
      try {
          const res = await fetch('/api/tick', { method: 'POST' });
          const data = await res.json();
          set({
            players: data.players,
            ball: data.ball,
            score: data.score,
            time: data.time,
            state: data.state || 'playing',
            lastTouch: data.lastTouch || null,
            restartTeam: data.restartTeam || 'home'
          });
      } catch (e) {
          console.error("Failed to tick game:", e);
      }
  },

  init: async () => {
      try {
        const res = await fetch('/api/state');
        if (res.ok) {
            const data = await res.json();
            set({
              players: data.players,
              ball: data.ball,
              score: data.score,
              time: data.time,
              state: data.state || 'playing',
              lastTouch: data.lastTouch || null,
              restartTeam: data.restartTeam || 'home'
            });
        }
      } catch (e) { console.error("Failed to init game:", e) }
  }
}));
