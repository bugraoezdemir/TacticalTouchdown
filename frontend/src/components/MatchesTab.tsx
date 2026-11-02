import { useState } from "react";
import { StatsTable } from "./StatsTable";
import type { Fixture, PlayerStatsDTO, Tournament } from "../types/simulation";

interface Props {
  tournament: Tournament | null;
}

/** Per-player columns. Ratios are computed here rather than stored, so the
 *  engine keeps only counts and the display decides how to read them. */
const COLUMNS: {
  key: string;
  label: string;
  title: string;
  value: (p: PlayerStatsDTO) => string;
  /** Highlight when a value is poor — the same thresholds the rules use. */
  bad?: (p: PlayerStatsDTO) => boolean;
}[] = [
  { key: "touches", label: "Tch", title: "Touches", value: (p) => String(p.touches) },
  { key: "passes", label: "Pas", title: "Passes", value: (p) => String(p.passes) },
  {
    key: "completion",
    label: "Cmp%",
    title: "Pass completion",
    value: (p) => (p.passes ? `${Math.round((p.passes_completed / p.passes) * 100)}%` : "—"),
  },
  {
    key: "intercepted",
    label: "Int%",
    title: "Share of his passes intercepted (rules fire above 22%)",
    value: (p) => (p.passes ? `${Math.round((p.passes_intercepted / p.passes) * 100)}%` : "—"),
    bad: (p) => p.passes >= 30 && p.passes_intercepted / p.passes > 0.22,
  },
  { key: "dribbles", label: "Drb", title: "Dribbles", value: (p) => String(p.dribbles) },
  {
    key: "dribbleLoss",
    label: "Lost%",
    title: "Share of his dribbles lost (rules fire above 28%)",
    value: (p) => (p.dribbles ? `${Math.round((p.dribbles_lost / p.dribbles) * 100)}%` : "—"),
    bad: (p) => p.dribbles >= 30 && p.dribbles_lost / p.dribbles > 0.28,
  },
  { key: "shots", label: "Sht", title: "Shots", value: (p) => String(p.shots) },
  { key: "onTarget", label: "OnT", title: "Shots on target", value: (p) => String(p.shots_on_target) },
  { key: "goals", label: "Gls", title: "Goals", value: (p) => String(p.goals) },
];

function PlayerTable({ rows, title }: { rows: PlayerStatsDTO[]; title: string }) {
  if (rows.length === 0) {
    return <p className="muted">No player data recorded for {title}.</p>;
  }
  // Busiest first: the players who actually shaped the match.
  const sorted = [...rows].sort((a, b) => b.touches - a.touches);
  return (
    <div className="player-stats-block">
      <h4>{title}</h4>
      <table className="player-stats">
        <thead>
          <tr>
            <th>Player</th>
            <th>Pos</th>
            {COLUMNS.map((c) => (
              <th key={c.key} title={c.title}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((p) => (
            <tr key={p.slot}>
              <td className="player-name">{p.name}</td>
              <td className="muted">{p.position}</td>
              {COLUMNS.map((c) => (
                <td key={c.key} className={c.bad?.(p) ? "stat-bad" : undefined}>
                  {c.value(p)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function MatchesTab({ tournament }: Props) {
  const [openKey, setOpenKey] = useState<string | null>(null);

  if (!tournament) {
    return (
      <p className="muted">
        No tournament yet — create one in the Tournament tab and matches will
        appear here as they are played.
      </p>
    );
  }

  const played = tournament.fixtures.filter((f) => f.played);
  if (played.length === 0) {
    return <p className="muted">No matches played yet.</p>;
  }

  // Newest first: the match you just watched is the one you want to inspect.
  const byDay = new Map<number, Fixture[]>();
  for (const f of played) {
    const list = byDay.get(f.matchday) ?? [];
    list.push(f);
    byDay.set(f.matchday, list);
  }
  const days = [...byDay.keys()].sort((a, b) => b - a);

  return (
    <div className="matches-tab">
      <h2>Matches</h2>
      <p className="muted">
        Every played fixture. Click one for team and player statistics —
        highlighted cells are the ones the adaptation rules react to.
      </p>

      {days.map((day) => (
        <section key={day} className="match-day">
          <h3>Matchday {day}</h3>
          {(byDay.get(day) ?? []).map((f) => {
            const key = `${f.matchday}-${f.home_id}-${f.away_id}`;
            const open = openKey === key;
            return (
              <div key={key} className="match-entry">
                <button
                  className={
                    f.involves_user ? "match-line user-fixture" : "match-line"
                  }
                  onClick={() => setOpenKey(open ? null : key)}
                >
                  <span className="fx-home">{f.home_name}</span>
                  <span className="fx-score">
                    {f.home_goals} – {f.away_goals}
                  </span>
                  <span className="fx-away">{f.away_name}</span>
                </button>

                {open && (
                  <div className="match-detail">
                    {f.stats ? (
                      <>
                        <StatsTable
                          stats={f.stats}
                          homeName={f.home_name}
                          awayName={f.away_name}
                        />
                        <PlayerTable
                          rows={f.stats.home_players}
                          title={f.home_name}
                        />
                        <PlayerTable
                          rows={f.stats.away_players}
                          title={f.away_name}
                        />
                      </>
                    ) : (
                      <p className="muted">No statistics recorded.</p>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </section>
      ))}
    </div>
  );
}
