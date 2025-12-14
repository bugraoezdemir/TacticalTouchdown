import { useGameStore } from "@/store/gameStore";

export default function ScoreBoard() {
  const { score, time } = useGameStore();
  
  const minutes = Math.floor(time / 60);
  const seconds = Math.floor(time % 60);

  return (
    <div className="flex items-center justify-between px-8 py-4 bg-background/80 backdrop-blur-md border-b border-white/5 sticky top-0 z-50">
      <div className="flex items-center gap-8">
        <div className="text-2xl font-bold font-sans tracking-tight text-white">
            <span className="text-blue-500">HOME</span> <span className="mx-2 opacity-50">VS</span> <span className="text-red-500">AWAY</span>
        </div>
      </div>

      <div className="absolute left-1/2 -translate-x-1/2 flex flex-col items-center">
        <div className="text-4xl font-mono font-bold tracking-widest bg-black/50 px-6 py-1 rounded-lg border border-white/10 shadow-inner">
          {score.home} - {score.away}
        </div>
        <div className="mt-1 text-sm font-mono text-primary animate-pulse">
            {minutes.toString().padStart(2, '0')}:{seconds.toString().padStart(2, '0')}
        </div>
      </div>

      <div className="text-sm text-muted-foreground font-mono">
        LEAGUE MATCH • WEEK 12
      </div>
    </div>
  );
}
