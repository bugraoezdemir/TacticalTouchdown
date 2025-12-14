import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Shield, Sword, Gauge, Target, Users } from "lucide-react";
import { useGameStore } from "@/store/gameStore";

export default function TacticsPanel() {
  const tactics = useGameStore((state) => state.tactics);
  const setTactics = useGameStore((state) => state.setTactics);

  const handleFormationChange = (formation: string) => {
    setTactics({ formation });
  };

  const handleMentalityChange = (mentality: string) => {
    setTactics({ mentality });
  };

  const handleDribbleChange = (value: number[]) => {
    setTactics({ dribbleFrequency: value[0] });
  };

  const handleShootChange = (value: number[]) => {
    setTactics({ shootFrequency: value[0] });
  };

  const getMentalityIcon = (mentality: string) => {
    switch (mentality) {
      case 'defensive': return <Shield className="w-4 h-4" />;
      case 'offensive': return <Sword className="w-4 h-4" />;
      default: return <Users className="w-4 h-4" />;
    }
  };

  const getMentalityColor = (mentality: string, isActive: boolean) => {
    if (!isActive) return 'opacity-50 hover:opacity-100';
    switch (mentality) {
      case 'defensive': return 'bg-blue-500/20 border-blue-500/40 text-blue-400';
      case 'offensive': return 'bg-red-500/20 border-red-500/40 text-red-400';
      default: return 'bg-primary/10 border-primary/20 text-primary';
    }
  };

  return (
    <Card className="h-full bg-card/50 border-white/5 backdrop-blur-sm" data-testid="tactics-panel">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-primary">
          <Gauge className="w-5 h-5" /> Tactical Command
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        
        <div className="space-y-3">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground">Formation</Label>
          <div className="grid grid-cols-2 gap-2">
            {tactics.availableFormations.map((formation) => (
              <Button
                key={formation}
                variant="outline"
                size="sm"
                data-testid={`button-formation-${formation}`}
                className={tactics.formation === formation 
                  ? 'bg-primary/10 border-primary/20 text-primary hover:bg-primary/20' 
                  : 'opacity-50 hover:opacity-100'
                }
                onClick={() => handleFormationChange(formation)}
              >
                {formation}
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-3">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground">Mentality</Label>
          <div className="grid grid-cols-3 gap-2">
            {tactics.availableMentalities.map((mentality) => (
              <Button
                key={mentality}
                variant="outline"
                size="sm"
                data-testid={`button-mentality-${mentality}`}
                className={getMentalityColor(mentality, tactics.mentality === mentality)}
                onClick={() => handleMentalityChange(mentality)}
              >
                <span className="flex items-center gap-1 capitalize">
                  {getMentalityIcon(mentality)}
                  <span className="hidden sm:inline">{mentality}</span>
                </span>
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-4 pt-4 border-t border-white/5">
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label className="flex items-center gap-2">
                <Users className="w-4 h-4 text-yellow-400" /> Dribbling
              </Label>
              <span className="text-xs text-muted-foreground" data-testid="text-dribble-value">
                {tactics.dribbleFrequency < 0.8 ? 'Low' : tactics.dribbleFrequency > 1.2 ? 'High' : 'Normal'}
              </span>
            </div>
            <Slider
              data-testid="slider-dribble"
              value={[tactics.dribbleFrequency]}
              onValueChange={handleDribbleChange}
              min={0.5}
              max={2.0}
              step={0.1}
              className="[&_.bg-primary]:bg-yellow-500"
            />
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <Label className="flex items-center gap-2">
                <Target className="w-4 h-4 text-green-400" /> Shooting
              </Label>
              <span className="text-xs text-muted-foreground" data-testid="text-shoot-value">
                {tactics.shootFrequency < 0.8 ? 'Low' : tactics.shootFrequency > 1.2 ? 'High' : 'Normal'}
              </span>
            </div>
            <Slider
              data-testid="slider-shoot"
              value={[tactics.shootFrequency]}
              onValueChange={handleShootChange}
              min={0.5}
              max={2.0}
              step={0.1}
              className="[&_.bg-primary]:bg-green-500"
            />
          </div>
        </div>

        <div className="pt-4 border-t border-white/5 text-xs text-muted-foreground">
          <p>Current: <span className="text-primary font-medium">{tactics.formation}</span> / <span className="capitalize font-medium">{tactics.mentality}</span></p>
        </div>

      </CardContent>
    </Card>
  );
}
