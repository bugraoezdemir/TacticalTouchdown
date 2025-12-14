import Pitch from "@/components/game/Pitch";
import TacticsPanel from "@/components/game/TacticsPanel";
import GameControls from "@/components/game/GameControls";
import ScoreBoard from "@/components/game/ScoreBoard";

export default function GamePage() {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col font-sans selection:bg-primary/30">
      <ScoreBoard />
      
      <main className="flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 max-w-[1800px] mx-auto w-full">
        {/* Main Game Area */}
        <div className="lg:col-span-8 flex flex-col gap-6">
           <div className="relative group">
              <Pitch />
              {/* Overlay Gradient for cinematic effect */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/40 via-transparent to-transparent pointer-events-none rounded-lg" />
           </div>
           
           <GameControls />

           {/* Live Commentary / Log Placeholder */}
           <div className="bg-card/30 border border-white/5 rounded-lg p-4 h-32 overflow-hidden relative">
              <div className="text-xs font-mono text-muted-foreground mb-2 uppercase">Match Log</div>
              <div className="space-y-1 text-sm font-mono opacity-80">
                  <p><span className="text-primary">12:30</span> Match started.</p>
                  <p><span className="text-primary">12:35</span> Player 9 (Home) intercepts the ball.</p>
                  <p><span className="text-primary">12:42</span> Shot on target!</p>
              </div>
              <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-background to-transparent" />
           </div>
        </div>

        {/* Sidebar */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          <TacticsPanel />
          
          {/* Player Stats Placeholder */}
          <div className="bg-card/30 border border-white/5 rounded-lg p-4 flex-1">
             <div className="text-sm font-bold text-muted-foreground mb-4 uppercase tracking-wider">Key Player Stats</div>
             <div className="space-y-2">
                {[1,2,3].map(i => (
                    <div key={i} className="flex items-center justify-between p-2 hover:bg-white/5 rounded transition-colors cursor-pointer group">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400 font-bold text-xs">{9+i}</div>
                            <div>
                                <div className="text-sm font-bold">Striker Name</div>
                                <div className="text-xs text-muted-foreground">Condition: 9{i}%</div>
                            </div>
                        </div>
                        <div className="text-xs font-mono text-green-400 group-hover:text-green-300">8.5</div>
                    </div>
                ))}
             </div>
          </div>
        </div>
      </main>
    </div>
  );
}
