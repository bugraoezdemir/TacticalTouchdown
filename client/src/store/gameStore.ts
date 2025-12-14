import { create } from 'zustand';

export type Team = 'home' | 'away';
export type PlayerRole = 'GK' | 'DEF' | 'MID' | 'FWD';

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
}

export interface Ball {
  x: number;
  y: number;
  vx: number;
  vy: number;
  ownerId: number | null;
}

interface GameState {
  players: Player[];
  ball: Ball;
  score: { home: number; away: number };
  time: number;
  isPlaying: boolean;
  gameSpeed: number;

  togglePlay: () => void;
  setGameSpeed: (speed: number) => void;
  resetGame: () => void;
  tick: () => void;
  init: () => void;
}

export const useGameStore = create<GameState>((set, get) => ({
  players: [],
  ball: { x: 50, y: 50, vx: 0, vy: 0, ownerId: null },
  score: { home: 0, away: 0 },
  time: 0,
  isPlaying: false,
  gameSpeed: 1,

  togglePlay: () => set((state) => ({ isPlaying: !state.isPlaying })),
  setGameSpeed: (speed) => set({ gameSpeed: speed }),
  
  resetGame: async () => {
      try {
          const res = await fetch('/api/reset', { method: 'POST' });
          const data = await res.json();
          set({ ...data, isPlaying: false });
      } catch (e) {
          console.error("Failed to reset game:", e);
      }
  },

  tick: async () => {
      try {
          const res = await fetch('/api/tick', { method: 'POST' });
          const data = await res.json();
          set(data);
      } catch (e) {
          console.error("Failed to tick game:", e);
      }
  },

  init: async () => {
      try {
        const res = await fetch('/api/state');
        if (res.ok) {
            const data = await res.json();
            set(data);
        }
      } catch (e) { console.error("Failed to init game:", e) }
  }
}));
