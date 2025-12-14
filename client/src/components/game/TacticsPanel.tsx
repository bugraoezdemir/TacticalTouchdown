import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Shield, Sword, Gauge, Target } from "lucide-react";

export default function TacticsPanel() {
  return (
    <Card className="h-full bg-card/50 border-white/5 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-primary">
          <Gauge className="w-5 h-5" /> Tactical Command
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        
        {/* Formation */}
        <div className="space-y-3">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground">Formation</Label>
          <div className="grid grid-cols-3 gap-2">
            <Button variant="outline" size="sm" className="bg-primary/10 border-primary/20 text-primary hover:bg-primary/20">4-4-2</Button>
            <Button variant="outline" size="sm" className="opacity-50 hover:opacity-100">4-3-3</Button>
            <Button variant="outline" size="sm" className="opacity-50 hover:opacity-100">3-5-2</Button>
          </div>
        </div>

        {/* Sliders */}
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label className="flex items-center gap-2"><Sword className="w-4 h-4 text-red-400" /> Aggression</Label>
              <span className="text-xs text-muted-foreground">High</span>
            </div>
            <Slider defaultValue={[75]} max={100} step={1} className="[&_.bg-primary]:bg-red-500" />
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <Label className="flex items-center gap-2"><Shield className="w-4 h-4 text-blue-400" /> Defensive Line</Label>
              <span className="text-xs text-muted-foreground">Deep</span>
            </div>
            <Slider defaultValue={[30]} max={100} step={1} className="[&_.bg-primary]:bg-blue-500" />
          </div>
          
           <div className="space-y-2">
            <div className="flex justify-between">
              <Label className="flex items-center gap-2"><Target className="w-4 h-4 text-green-400" /> Passing Risk</Label>
              <span className="text-xs text-muted-foreground">Safe</span>
            </div>
            <Slider defaultValue={[50]} max={100} step={1} className="[&_.bg-primary]:bg-green-500" />
          </div>
        </div>

        {/* Toggles */}
        <div className="space-y-4 pt-4 border-t border-white/5">
           <div className="flex items-center justify-between">
              <Label htmlFor="counter-attack" className="text-sm">Counter Attack</Label>
              <Switch id="counter-attack" defaultChecked />
           </div>
           <div className="flex items-center justify-between">
              <Label htmlFor="offside-trap" className="text-sm">Offside Trap</Label>
              <Switch id="offside-trap" />
           </div>
        </div>

      </CardContent>
    </Card>
  );
}
