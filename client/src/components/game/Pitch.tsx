import { motion } from 'framer-motion';
import { useGameStore, Player, Ball } from '@/store/gameStore';
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

export default function Pitch() {
  const { players, ball, isPlaying, tick, gameSpeed } = useGameStore();

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(tick, 50 / gameSpeed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, tick, gameSpeed]);

  return (
    <div className="w-full aspect-[1.5] bg-pitch relative overflow-hidden rounded-lg shadow-2xl border-4 border-white/10 pitch-pattern">
      {/* Field Markings */}
      <div className="absolute inset-4 border-2 border-pitch-line/50 rounded-sm pointer-events-none">
        {/* Center Line */}
        <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-pitch-line/50 -translate-x-1/2" />
        {/* Center Circle */}
        <div className="absolute left-1/2 top-1/2 w-32 h-32 border-2 border-pitch-line/50 rounded-full -translate-x-1/2 -translate-y-1/2" />
        {/* Penalty Areas */}
        <div className="absolute left-0 top-1/2 w-24 h-48 border-2 border-pitch-line/50 -translate-y-1/2" />
        <div className="absolute right-0 top-1/2 w-24 h-48 border-2 border-pitch-line/50 -translate-y-1/2" />
      </div>

      {/* Dynamic Elements */}
      {players.map(p => <PlayerDot key={p.id} player={p} />)}
      <BallDot ball={ball} />
    </div>
  );
}
