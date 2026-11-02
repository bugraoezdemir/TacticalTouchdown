import { useState } from "react";
import type { PlayerDTO, TeamDTO } from "../types/simulation";
import { FORMATIONS } from "../types/simulation";

const MIN = 0.5;
const MAX = 2.0;
const STEP = 0.05;

type Action =
  | "shoot"
  | "pass_"
  | "dribble"
  | "pressing"
  | "roaming"
  | "offensiveness"
  | "direct"
  | "one_two"
  | "attack_call"
  | "pass_range"
  | "breadth"
  | "compactness"
  | "possession"
  | "crossing"
  | "lofted"
  | "risk"
  | "tempo"
  | "press_aggression"
  | "man_marking";
type BiasField =
  | "bias_shoot"
  | "bias_pass"
  | "bias_dribble"
  | "bias_pressing"
  | "bias_roaming"
  | "bias_offensiveness"
  | "bias_direct"
  | "bias_one_two"
  | "bias_attack_call"
  | "bias_pass_range"
  | "bias_breadth"
  | "bias_compactness"
  | "bias_possession"
  | "bias_crossing"
  | "bias_lofted"
  | "bias_risk"
  | "bias_tempo"
  | "bias_press_aggression"
  | "bias_man_marking";

interface Slider {
  key: Action;
  label: string;
  biasField: BiasField;
}

/** Sliders are a relative bias, never a probability: they do not sum to
 *  anything, and the engine clamps each to [0.5, 2.0]. */
const ATTACKING: Slider[] = [
  { key: "shoot", label: "Shoot", biasField: "bias_shoot" },
  { key: "pass_", label: "Pass", biasField: "bias_pass" },
  { key: "dribble", label: "Dribble", biasField: "bias_dribble" },
  { key: "direct", label: "Direct pass", biasField: "bias_direct" },
  { key: "one_two", label: "One-two", biasField: "bias_one_two" },
  { key: "attack_call", label: "Attack call", biasField: "bias_attack_call" },
  { key: "pass_range", label: "Pass range", biasField: "bias_pass_range" },
  { key: "possession", label: "Possession", biasField: "bias_possession" },
  { key: "crossing", label: "Crosses", biasField: "bias_crossing" },
  { key: "lofted", label: "Lofted balls", biasField: "bias_lofted" },
  { key: "risk", label: "Risk", biasField: "bias_risk" },
  { key: "tempo", label: "Tempo", biasField: "bias_tempo" },
  { key: "roaming", label: "Roaming", biasField: "bias_roaming" },
];

const DEFENDING: Slider[] = [
  { key: "pressing", label: "Press numbers", biasField: "bias_pressing" },
  { key: "press_aggression", label: "Press aggression", biasField: "bias_press_aggression" },
  { key: "man_marking", label: "Man marking", biasField: "bias_man_marking" },
];

/** Depth stations a player up or down the pitch. It is applied when stations
 *  are recomputed rather than per tick, so it sits with Shape. */
const SHAPE: Slider[] = [
  { key: "offensiveness", label: "Depth", biasField: "bias_offensiveness" },
  { key: "breadth", label: "Breadth", biasField: "bias_breadth" },
  { key: "compactness", label: "Compactness", biasField: "bias_compactness" },
];

const ALL_SLIDERS = [...SHAPE, ...ATTACKING, ...DEFENDING];

interface Props {
  team: TeamDTO | null;
  side: "home" | "away";
  onTeamChange: (teamId: number, action: Action, value: number) => void;
  onFormationChange: (teamId: number, formation: string) => void;
  onPlayerChange: (
    teamId: number,
    playerId: number,
    field: BiasField,
    value: number | null,
  ) => void;
}

export function TacticsPanel({
  team,
  side,
  onTeamChange,
  onFormationChange,
  onPlayerChange,
}: Props) {
  const [expanded, setExpanded] = useState<number | null>(null);
  const [openAttack, setOpenAttack] = useState(true);
  const [openDefend, setOpenDefend] = useState(false);

  if (!team) {
    return <aside className={`tactics tactics-${side}`} aria-hidden="true" />;
  }

  function teamSliders(sliders: Slider[]) {
    return sliders.map(({ key, label }) => (
      <label key={key} className="slider-row">
        <span className="slider-label">{label}</span>
        <input
          type="range"
          min={MIN}
          max={MAX}
          step={STEP}
          value={team!.tactics[key]}
          onChange={(e) => onTeamChange(team!.team_id, key, Number(e.target.value))}
        />
        <span className="slider-value">{team!.tactics[key].toFixed(2)}&times;</span>
      </label>
    ));
  }

  function playerSliders(p: PlayerDTO, sliders: Slider[]) {
    return sliders.map(({ key, label, biasField }) => {
      const raw = p[biasField] as number | null;
      const value = raw ?? team!.tactics[key];
      return (
        <label key={key} className="slider-row">
          <span className="slider-label">{label}</span>
          <input
            type="range"
            min={MIN}
            max={MAX}
            step={STEP}
            value={value}
            onChange={(e) =>
              onPlayerChange(
                team!.team_id,
                p.player_id,
                biasField,
                Number(e.target.value),
              )
            }
          />
          {/* "team" rather than a number makes it obvious what is inherited. */}
          <span className="slider-value">
            {raw === null ? "team" : `${value.toFixed(2)}×`}
          </span>
        </label>
      );
    });
  }

  return (
    <aside className={`tactics tactics-${side}`}>
      <h2>
        <span className={`team-dot ${side}-dot`} />
        {team.name}
      </h2>

      <label className="formation-row">
        <span className="slider-label">Shape</span>
        <select
          value={team.tactics.formation}
          onChange={(e) => onFormationChange(team.team_id, e.target.value)}
        >
          {FORMATIONS.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
      </label>

      {teamSliders(SHAPE)}

      <section className="tactics-section">
        <button
          type="button"
          className="section-head"
          onClick={() => setOpenAttack((v) => !v)}
          aria-expanded={openAttack}
        >
          <span className="section-mark">{openAttack ? "▾" : "▸"}</span> Attacking
        </button>
        {openAttack && (
          <div className="tactics-team">
            {teamSliders(ATTACKING)}
          </div>
        )}
      </section>

      <section className="tactics-section">
        <button
          type="button"
          className="section-head"
          onClick={() => setOpenDefend((v) => !v)}
          aria-expanded={openDefend}
        >
          <span className="section-mark">{openDefend ? "▾" : "▸"}</span> Defending
        </button>
        {openDefend && (
          <div className="tactics-team">
            {teamSliders(DEFENDING)}
          </div>
        )}
      </section>

      <h3>Players</h3>
      <ul className="tactics-players">
        {team.players.map((p) => {
          const overridden = ALL_SLIDERS.some(({ biasField }) => p[biasField] !== null);
          const open = expanded === p.player_id;
          return (
            <li key={p.player_id} className={overridden ? "has-override" : ""}>
              <button
                type="button"
                className="player-row"
                onClick={() => setExpanded(open ? null : p.player_id)}
                aria-expanded={open}
              >
                <span className="player-pos">{p.position}</span>
                <span className="player-name">{p.name}</span>
                <span className="player-mark">{overridden ? "●" : open ? "▾" : "▸"}</span>
              </button>

              {open && (
                <div className="player-sliders">
                  <h4>Shape</h4>
                  {playerSliders(p, SHAPE)}
                  <h4>Attacking</h4>
                  {playerSliders(p, ATTACKING)}
                  <h4>Defending</h4>
                  {playerSliders(p, DEFENDING)}
                  {overridden && (
                    <button
                      type="button"
                      className="reset-player"
                      onClick={() =>
                        ALL_SLIDERS.forEach(({ biasField }) =>
                          onPlayerChange(team.team_id, p.player_id, biasField, null),
                        )
                      }
                    >
                      Reset to team
                    </button>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
