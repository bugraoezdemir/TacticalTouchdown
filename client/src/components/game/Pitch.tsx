import { motion, AnimatePresence } from 'framer-motion';
import { useGameStore, Player, Ball, GameStateType } from '@/store/gameStore';
import { cn } from '@/lib/utils';
import { useEffect } from 'react';

const PlayerDot = ({ player }: { player: Player }) => {
  return (
    <motion.div
      className={cn(
        "absolute w-4 h-4 -ml-2 -mt-2 rounded-full border-2 border-white/50 flex items-center justify-center text-[8px] font-bold shadow-sm transition-colors z-10",
        player.team === 'home' ? "bg-blue-500 text-white" : "bg-red-500 text-white",
        player.role === 'GK' && "bg-yellow-500 text-black border-black/20"
      )}
      animate={{ 
        left: `${player.x}%`, 
        top: `${player.y}%`,
        scale: player.hasBall ? 1.2 : 1
      }}
      transition={{ type: "tween", duration: 0.1, ease: "linear" }}
    >
      {player.number}
      {player.hasBall && (
         <div className="absolute -bottom-1 w-1 h-1 bg-white rounded-full animate-ping" />
      )}
    </motion.div>
  );
};

const BallDot = ({ ball }: { ball: Ball }) => {
  return (
    <motion.div
      className="absolute w-2 h-2 -ml-1 -mt-1 bg-white rounded-full shadow-md z-20 border border-black/10"
      animate={{ 
        left: `${ball.x}%`, 
        top: `${ball.y}%` 
      }}
      transition={{ type: "tween", duration: 0.1, ease: "linear" }}
    />
  );
};

const GoalPost = ({ side }: { side: 'left' | 'right' }) => {
  return (
    <div 
      className={cn(
        "absolute w-1 bg-white/80 shadow-lg z-5",
        side === 'left' ? "left-0" : "right-0"
      )}
      style={{
        top: '40%',
        height: '20%'
      }}
    >
      <div className="absolute inset-0 bg-gradient-to-r from-gray-300 to-white rounded-sm" />
      <div className={cn(
        "absolute top-0 h-0.5 w-3 bg-white/80",
        side === 'left' ? "left-0" : "right-0 -translate-x-2"
      )} />
      <div className={cn(
        "absolute bottom-0 h-0.5 w-3 bg-white/80",
        side === 'left' ? "left-0" : "right-0 -translate-x-2"
      )} />
    </div>
  );
};

const getStateMessage = (state: GameStateType, restartTeam: string): string => {
  switch (state) {
    case 'goal_scored':
      return '⚽ GOAL!';
    case 'corner_kick':
      return `🚩 Corner - ${restartTeam === 'home' ? 'Blue' : 'Red'}`;
    case 'goal_kick':
      return `🥅 Goal Kick - ${restartTeam === 'home' ? 'Blue' : 'Red'}`;
    case 'throw_in':
      return `↗️ Throw-in - ${restartTeam === 'home' ? 'Blue' : 'Red'}`;
    case 'kickoff':
      return `⚽ Kickoff - ${restartTeam === 'home' ? 'Blue' : 'Red'}`;
    default:
      return '';
  }
};

const StateOverlay = ({ state, restartTeam }: { state: GameStateType; restartTeam: string }) => {
  if (state === 'playing') return null;
  
  const message = getStateMessage(state, restartTeam);
  const isGoal = state === 'goal_scored';
  
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.8 }}
        className={cn(
          "absolute inset-0 flex items-center justify-center z-30 pointer-events-none",
          isGoal && "bg-black/30"
        )}
      >
        <motion.div
          initial={{ y: -20 }}
          animate={{ y: 0 }}
          className={cn(
            "px-6 py-3 rounded-lg font-bold text-lg shadow-lg",
            isGoal 
              ? "bg-gradient-to-r from-yellow-400 to-orange-500 text-white text-2xl" 
              : "bg-black/70 text-white"
          )}
        >
          {message}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default function Pitch() {
  const { players, ball, isPlaying, tick, gameSpeed, init, state, restartTeam } = useGameStore();

  useEffect(() => {
    init();
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(tick, 50 / gameSpeed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, tick, gameSpeed]);

  return (
    <div className="w-full aspect-[1.5] bg-pitch relative overflow-hidden rounded-lg shadow-2xl border-4 border-white/10 pitch-pattern">
      {/* Goal Posts */}
      <GoalPost side="left" />
      <GoalPost side="right" />
      
      {/* Field Markings */}
      <div className="absolute inset-4 border-2 border-pitch-line/50 rounded-sm pointer-events-none">
        {/* Center Line */}
        <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-pitch-line/50 -translate-x-1/2" />
        {/* Center Circle */}
        <div className="absolute left-1/2 top-1/2 w-32 h-32 border-2 border-pitch-line/50 rounded-full -translate-x-1/2 -translate-y-1/2" />
        {/* Penalty Areas */}
        <div className="absolute left-0 top-1/2 w-24 h-48 border-2 border-pitch-line/50 -translate-y-1/2" />
        <div className="absolute right-0 top-1/2 w-24 h-48 border-2 border-pitch-line/50 -translate-y-1/2" />
        {/* Goal Areas (6-yard box) */}
        <div className="absolute left-0 top-1/2 w-8 h-24 border-2 border-pitch-line/50 -translate-y-1/2" />
        <div className="absolute right-0 top-1/2 w-8 h-24 border-2 border-pitch-line/50 -translate-y-1/2" />
      </div>

      {/* Dynamic Elements */}
      {players.map(p => <PlayerDot key={p.id} player={p} />)}
      <BallDot ball={ball} />
      
      {/* Game State Overlay */}
      <StateOverlay state={state} restartTeam={restartTeam} />
    </div>
  );
}
