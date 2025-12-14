import { useGameStore } from "@/store/gameStore";
import { Button } from "@/components/ui/button";
import { Play, Pause, RotateCcw, FastForward } from "lucide-react";

export default function GameControls() {
  const { isPlaying, togglePlay, resetGame, gameSpeed, setGameSpeed } = useGameStore();

  return (
    <div className="flex items-center gap-4 bg-card/50 p-4 rounded-xl border border-white/5 backdrop-blur-sm">
      <Button 
        onClick={togglePlay}
        size="lg"
        className={isPlaying ? "bg-yellow-500 hover:bg-yellow-600 text-black" : "bg-primary hover:bg-primary/90 text-black"}
      >
        {isPlaying ? <Pause className="w-6 h-6 fill-current" /> : <Play className="w-6 h-6 fill-current ml-1" />}
      </Button>

      <Button variant="outline" size="icon" onClick={resetGame}>
        <RotateCcw className="w-4 h-4" />
      </Button>

      <div className="h-8 w-px bg-white/10 mx-2" />

      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground uppercase font-mono">Speed</span>
        <Button 
            variant={gameSpeed === 1 ? "secondary" : "ghost"} 
            size="sm" 
            onClick={() => setGameSpeed(1)}
            className="h-8 w-8 p-0"
        >
            1x
        </Button>
        <Button 
            variant={gameSpeed === 2 ? "secondary" : "ghost"} 
            size="sm" 
            onClick={() => setGameSpeed(2)}
             className="h-8 w-8 p-0"
        >
            2x
        </Button>
      </div>
    </div>
  );
}
