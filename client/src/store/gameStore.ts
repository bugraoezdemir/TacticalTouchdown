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

export interface Tactics {
  formation: string;
  mentality: string;
  dribbleFrequency: number;
  shootFrequency: number;
  availableFormations: string[];
  availableMentalities: string[];
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
  tactics: Tactics;

  togglePlay: () => void;
  setGameSpeed: (speed: number) => void;
  resetGame: () => void;
  tick: () => void;
  init: () => void;
  setTactics: (updates: Partial<Pick<Tactics, 'formation' | 'mentality' | 'dribbleFrequency' | 'shootFrequency'>>) => void;
}

const defaultTactics: Tactics = {
  formation: '4-4-2',
  mentality: 'normal',
  dribbleFrequency: 1.0,
  shootFrequency: 1.0,
  availableFormations: ['4-4-2', '4-3-3', '3-5-2', '5-3-2'],
  availableMentalities: ['defensive', 'normal', 'offensive'],
};

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
  tactics: defaultTactics,

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
            tactics: data.tactics || defaultTactics,
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
            restartTeam: data.restartTeam || 'home',
            tactics: data.tactics || get().tactics
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
              restartTeam: data.restartTeam || 'home',
              tactics: data.tactics || defaultTactics
            });
        }
      } catch (e) { console.error("Failed to init game:", e) }
  },

  setTactics: (updates) => {
      set((state) => ({
        tactics: { ...state.tactics, ...updates }
      }));
      
      fetch('/api/tactics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates)
      }).catch((e) => {
        console.error("Failed to sync tactics:", e);
      });
  }
}));
