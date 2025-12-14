import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Shield, Sword, Gauge, Target, Users } from "lucide-react";
import { useGameStore } from "@/store/gameStore";

const defaultFormations = ['4-4-2', '4-3-3', '3-5-2', '5-3-2'];
const defaultMentalities = ['defensive', 'normal', 'offensive'];

export default function TacticsPanel() {
  const tactics = useGameStore((state) => state.tactics);
  const setTactics = useGameStore((state) => state.setTactics);

  const formations = tactics.availableFormations?.length ? tactics.availableFormations : defaultFormations;
  const mentalities = tactics.availableMentalities?.length ? tactics.availableMentalities : defaultMentalities;

  const getMentalityIcon = (mentality: string) => {
    switch (mentality) {
      case 'defensive': return <Shield className="w-4 h-4" />;
      case 'offensive': return <Sword className="w-4 h-4" />;
      default: return <Users className="w-4 h-4" />;
    }
  };

  const getMentalityStyle = (mentality: string, isActive: boolean) => {
    if (!isActive) return 'opacity-60 hover:opacity-100 hover:bg-white/5';
    switch (mentality) {
      case 'defensive': return 'bg-blue-500/20 border-blue-500/50 text-blue-400';
      case 'offensive': return 'bg-red-500/20 border-red-500/50 text-red-400';
      default: return 'bg-primary/20 border-primary/50 text-primary';
    }
  };

  const getDribbleLabel = (value: number) => {
    if (value < 0.8) return 'Low';
    if (value > 1.2) return 'High';
    return 'Normal';
  };

  const getShootLabel = (value: number) => {
    if (value < 0.8) return 'Low';
    if (value > 1.2) return 'High';
    return 'Normal';
  };

  return (
    <Card className="h-full bg-card/50 border-white/10 backdrop-blur-sm" data-testid="tactics-panel">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-primary text-lg">
          <Gauge className="w-5 h-5" /> Tactical Command
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-3">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground font-semibold">
            Formation
          </Label>
          <div className="grid grid-cols-2 gap-2">
            {formations.map((formation) => (
              <Button
                key={formation}
                variant="outline"
                size="sm"
                data-testid={`button-formation-${formation}`}
                className={
                  tactics.formation === formation
                    ? 'bg-primary/20 border-primary/50 text-primary font-bold'
                    : 'opacity-60 hover:opacity-100 hover:bg-white/5'
                }
                onClick={() => setTactics({ formation })}
              >
                {formation}
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-3">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground font-semibold">
            Mentality
          </Label>
          <div className="grid grid-cols-3 gap-2">
            {mentalities.map((mentality) => (
              <Button
                key={mentality}
                variant="outline"
                size="sm"
                data-testid={`button-mentality-${mentality}`}
                className={getMentalityStyle(mentality, tactics.mentality === mentality)}
                onClick={() => setTactics({ mentality })}
              >
                <span className="flex items-center gap-1 capitalize">
                  {getMentalityIcon(mentality)}
                  <span className="hidden sm:inline text-xs">{mentality}</span>
                </span>
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-4 pt-4 border-t border-white/10">
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <Label className="flex items-center gap-2 text-sm">
                <Users className="w-4 h-4 text-yellow-400" /> Dribbling
              </Label>
              <span className="text-xs text-muted-foreground font-medium" data-testid="text-dribble-value">
                {getDribbleLabel(tactics.dribbleFrequency ?? 1.0)}
              </span>
            </div>
            <Slider
              data-testid="slider-dribble"
              value={[tactics.dribbleFrequency ?? 1.0]}
              onValueChange={(value) => setTactics({ dribbleFrequency: value[0] })}
              min={0.5}
              max={2.0}
              step={0.1}
              className="cursor-pointer"
            />
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <Label className="flex items-center gap-2 text-sm">
                <Target className="w-4 h-4 text-green-400" /> Shooting
              </Label>
              <span className="text-xs text-muted-foreground font-medium" data-testid="text-shoot-value">
                {getShootLabel(tactics.shootFrequency ?? 1.0)}
              </span>
            </div>
            <Slider
              data-testid="slider-shoot"
              value={[tactics.shootFrequency ?? 1.0]}
              onValueChange={(value) => setTactics({ shootFrequency: value[0] })}
              min={0.5}
              max={2.0}
              step={0.1}
              className="cursor-pointer"
            />
          </div>
        </div>

        <div className="pt-4 border-t border-white/10 text-sm text-muted-foreground">
          <p>
            Current: <span className="text-primary font-bold">{tactics.formation}</span>
            {' / '}
            <span className="capitalize font-bold">{tactics.mentality}</span>
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
