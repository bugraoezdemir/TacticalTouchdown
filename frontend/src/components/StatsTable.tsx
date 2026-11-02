import type { MatchStatsDTO, TeamStatsDTO } from "../types/simulation";

interface Props {
  stats: MatchStatsDTO | undefined;
  homeName: string;
  awayName: string;
}

interface Row {
  label: string;
  home: number;
  away: number;
  /** Render as "n%" rather than a raw count. */
  percent?: boolean;
}

function pct(part: number, whole: number): number {
  return whole > 0 ? Math.round((part / whole) * 100) : 0;
}

function buildRows(h: TeamStatsDTO, a: TeamStatsDTO): Row[] {
  const possTotal = h.possession_ticks + a.possession_ticks;
  return [
    // Possession is stored in ticks, which means nothing to a reader: show the
    // share of the match each side had the ball.
    {
      label: "Possession",
      home: pct(h.possession_ticks, possTotal),
      away: pct(a.possession_ticks, possTotal),
      percent: true,
    },
    { label: "Shots", home: h.shots, away: a.shots },
    { label: "On target", home: h.shots_on_target, away: a.shots_on_target },
    { label: "Saves", home: h.saves, away: a.saves },
    { label: "Passes", home: h.passes, away: a.passes },
    {
      label: "Pass accuracy",
      home: pct(h.passes_completed, h.passes),
      away: pct(a.passes_completed, a.passes),
      percent: true,
    },
    { label: "Crosses", home: h.crosses ?? 0, away: a.crosses ?? 0 },
    {
      label: "Crosses completed",
      home: h.crosses_completed ?? 0,
      away: a.crosses_completed ?? 0,
    },
    { label: "Corners", home: h.corners, away: a.corners },
    { label: "Throw-ins", home: h.throw_ins, away: a.throw_ins },
    { label: "Goal kicks", home: h.goal_kicks, away: a.goal_kicks },
  ];
}

export function StatsTable({ stats, homeName, awayName }: Props) {
  if (!stats) {
    return (
      <div className="stats-table">
        <h2>Match stats</h2>
        <p className="log-empty">No stats yet.</p>
      </div>
    );
  }

  const rows = buildRows(stats.home, stats.away);

  return (
    <div className="stats-table">
      <h2>Match stats</h2>
      <div className="stats-head">
        <span className="stats-team">
          <span className="team-dot home-dot" />
          {homeName}
        </span>
        <span className="stats-team away">
          {awayName}
          <span className="team-dot away-dot" />
        </span>
      </div>

      <ul className="stats-rows">
        {rows.map((r) => {
          const total = r.home + r.away;
          // A bar per row makes the comparison readable at a glance; with no
          // events yet both halves are empty rather than a misleading 50/50.
          const homeShare = total > 0 ? (r.home / total) * 100 : 0;
          const suffix = r.percent ? "%" : "";
          return (
            <li key={r.label}>
              <div className="stats-line">
                <span className="stats-value">
                  {r.home}
                  {suffix}
                </span>
                <span className="stats-label">{r.label}</span>
                <span className="stats-value">
                  {r.away}
                  {suffix}
                </span>
              </div>
              <div className="stats-bar" aria-hidden="true">
                <span className="stats-bar-home" style={{ width: `${homeShare}%` }} />
                <span className="stats-bar-away" style={{ width: `${100 - homeShare}%` }} />
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
