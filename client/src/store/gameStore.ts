import { create } from 'zustand';

export type Team = 'home' | 'away';
export type PlayerRole = 'GK' | 'DEF' | 'MID' | 'FWD';

export interface Player {
  id: number;
  team: Team;
  role: PlayerRole;
  x: number; // 0-100
  y: number; // 0-100
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
  isPlaying: boolean;
  gameSpeed: number;
  score: { home: number; away: number };
  time: number; // in seconds
  
  // Actions
  togglePlay: () => void;
  setGameSpeed: (speed: number) => void;
  resetGame: () => void;
  tick: () => void;
}

const INITIAL_PLAYERS: Player[] = [
  // HOME TEAM (Blue, attacking Right ->)
  { id: 1, team: 'home', role: 'GK', x: 5, y: 50, vx: 0, vy: 0, hasBall: false, number: 1 },
  { id: 2, team: 'home', role: 'DEF', x: 20, y: 20, vx: 0, vy: 0, hasBall: false, number: 4 },
  { id: 3, team: 'home', role: 'DEF', x: 20, y: 50, vx: 0, vy: 0, hasBall: false, number: 5 },
  { id: 4, team: 'home', role: 'DEF', x: 20, y: 80, vx: 0, vy: 0, hasBall: false, number: 3 },
  { id: 5, team: 'home', role: 'MID', x: 40, y: 30, vx: 0, vy: 0, hasBall: false, number: 8 },
  { id: 6, team: 'home', role: 'MID', x: 40, y: 70, vx: 0, vy: 0, hasBall: false, number: 6 },
  { id: 7, team: 'home', role: 'FWD', x: 60, y: 50, vx: 0, vy: 0, hasBall: true, number: 9 }, // Starts with ball

  // AWAY TEAM (Red, attacking Left <-)
  { id: 11, team: 'away', role: 'GK', x: 95, y: 50, vx: 0, vy: 0, hasBall: false, number: 1 },
  { id: 12, team: 'away', role: 'DEF', x: 80, y: 20, vx: 0, vy: 0, hasBall: false, number: 4 },
  { id: 13, team: 'away', role: 'DEF', x: 80, y: 50, vx: 0, vy: 0, hasBall: false, number: 5 },
  { id: 14, team: 'away', role: 'DEF', x: 80, y: 80, vx: 0, vy: 0, hasBall: false, number: 3 },
  { id: 15, team: 'away', role: 'MID', x: 60, y: 30, vx: 0, vy: 0, hasBall: false, number: 8 },
  { id: 16, team: 'away', role: 'MID', x: 60, y: 70, vx: 0, vy: 0, hasBall: false, number: 6 },
  { id: 17, team: 'away', role: 'FWD', x: 40, y: 50, vx: 0, vy: 0, hasBall: false, number: 9 },
];

export const useGameStore = create<GameState>((set, get) => ({
  players: INITIAL_PLAYERS,
  ball: { x: 60, y: 50, vx: 0, vy: 0, ownerId: 7 },
  isPlaying: false,
  gameSpeed: 1,
  score: { home: 0, away: 0 },
  time: 0,

  togglePlay: () => set((state) => ({ isPlaying: !state.isPlaying })),
  setGameSpeed: (speed) => set({ gameSpeed: speed }),
  resetGame: () => set({ 
    players: INITIAL_PLAYERS, 
    ball: { x: 60, y: 50, vx: 0, vy: 0, ownerId: 7 },
    score: { home: 0, away: 0 },
    time: 0,
    isPlaying: false
  }),

  tick: () => {
    const { players, ball, score, time } = get();
    
    // 1. Move Ball
    let newBall = { ...ball };
    let newPlayers = players.map(p => ({ ...p }));
    let newScore = { ...score };
    let goalScored = false;

    if (ball.ownerId === null) {
      newBall.x += newBall.vx;
      newBall.y += newBall.vy;
      // Friction
      newBall.vx *= 0.95;
      newBall.vy *= 0.95;
    } else {
      const owner = newPlayers.find(p => p.id === ball.ownerId);
      if (owner) {
        newBall.x = owner.x + 1; // Slightly ahead
        newBall.y = owner.y;
      }
    }

    // Goal Logic
    if (newBall.x < 0) {
      newScore.away++;
      goalScored = true;
    } else if (newBall.x > 100) {
      newScore.home++;
      goalScored = true;
    }
    
    // Reset if goal
    if (goalScored) {
        set({
            score: newScore,
            ball: { x: 50, y: 50, vx: 0, vy: 0, ownerId: null },
            players: INITIAL_PLAYERS.map(p => ({...p})), // Reset positions
            isPlaying: false 
        });
        return;
    }

    // 2. Move Players
    newPlayers.forEach(player => {
      let targetX = player.x;
      let targetY = player.y;

      // Simple AI Logic
      if (player.hasBall) {
        // Attack Logic
        if (player.team === 'home') {
          targetX = 100; // Goal
          targetY = 50; 
        } else {
          targetX = 0; // Goal
          targetY = 50;
        }

        // Random dribble variation
        targetY += (Math.random() - 0.5) * 10;
        
        // Shoot logic
        const distToGoal = player.team === 'home' ? 100 - player.x : player.x;
        if (distToGoal < 20 && Math.random() < 0.05) {
             player.hasBall = false;
             newBall.ownerId = null;
             newBall.vx = (player.team === 'home' ? 2 : -2);
             newBall.vy = (50 - player.y) * 0.05; // Aim for center
        }

      } else {
        // Defense/Support Logic
        if (ball.ownerId && newPlayers.find(p => p.id === ball.ownerId)?.team === player.team) {
           // Teammate has ball - Move to support
           targetX = newBall.x + (player.team === 'home' ? -10 : 10);
           targetY = newBall.y + (player.id % 2 === 0 ? 10 : -10);
        } else {
           // Opponent has ball or free ball - Chase ball
           // But GK stays home
           if (player.role === 'GK') {
             targetX = player.team === 'home' ? 5 : 95;
             targetY = newBall.y * 0.5 + 25; // Simple tracking
           } else {
             targetX = newBall.x;
             targetY = newBall.y;
           }
        }
      }

      // Move towards target
      const dx = targetX - player.x;
      const dy = targetY - player.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const speed = 0.5; // Player speed

      if (dist > 1) {
        player.x += (dx / dist) * speed;
        player.y += (dy / dist) * speed;
      }
      
      // Collision/Tackle Logic (Simplified)
      if (!player.hasBall && newBall.ownerId !== player.id && newBall.ownerId !== null) {
          const distToBall = Math.sqrt(Math.pow(player.x - newBall.x, 2) + Math.pow(player.y - newBall.y, 2));
          if (distToBall < 2 && Math.random() < 0.1) {
             // Steal!
             const currentOwner = newPlayers.find(p => p.id === newBall.ownerId);
             if (currentOwner) currentOwner.hasBall = false;
             
             player.hasBall = true;
             newBall.ownerId = player.id;
          }
      }
      
      // Pick up loose ball
      if (!player.hasBall && newBall.ownerId === null) {
          const distToBall = Math.sqrt(Math.pow(player.x - newBall.x, 2) + Math.pow(player.y - newBall.y, 2));
          if (distToBall < 2) {
             player.hasBall = true;
             newBall.ownerId = player.id;
          }
      }

      // Boundaries
      player.x = Math.max(0, Math.min(100, player.x));
      player.y = Math.max(0, Math.min(100, player.y));
    });

    set({ players: newPlayers, ball: newBall, time: time + 0.1 });
  }
}));
