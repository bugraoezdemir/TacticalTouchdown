import { useState } from "react";
import type {
  Tournament,
  TeamTacticsDTO,
  TeamStatsDTO,
} from "../types/simulation";

interface Props {
  tournament: Tournament | null;
  busy: boolean;
  onCreate: (seed: number, userTeamIndex: number) => void;
  onStartMatch: () => void;
  onAdvance: () => void;
  onNew: () => void;
}

/** The eight names are fixed, so setup can offer them before a tournament
 *  exists. Kept in sync with engine/tournament.py CITY_NAMES. */
const CITY_NAMES = [
  "Aachen",
  "Augsburg",
  "Berlin",
  "Bochum",
  "Bremen",
  "Dortmund",
  "Dresden",
  "Essen",
  "Freiburg",
  "Hamburg",
  "Hannover",
  "Kiel",
  "Leipzig",
  "Mainz",
  "Munster",
  "Nurnberg",
  "Rostock",
  "Stuttgart",
];

/** Every numeric slider a team has, in sidebar order. All of them: a partial
 *  profile hides exactly the parameter you are trying to reason about, and the
 *  full header row is abbreviated so it still fits. */
const PROFILE: { key: keyof TeamTacticsDTO; label: string; title: string }[] = [
  { key: "shoot", label: "Sht", title: "Shoot" },
  { key: "pass_", label: "Pas", title: "Pass" },
  { key: "dribble", label: "Drb", title: "Dribble" },
  { key: "direct", label: "Dir", title: "Direct pass" },
  { key: "one_two", label: "1-2", title: "One-two" },
  { key: "attack_call", label: "Atk", title: "Attack call" },
  { key: "roaming", label: "Rom", title: "Roaming" },
  { key: "pass_range", label: "Rng", title: "Pass range" },
  { key: "possession", label: "Pos", title: "Possession" },
  { key: "wing_attacks", label: "Wng", title: "Wing attacks" },
  { key: "crossing", label: "Crs", title: "Crosses" },
  { key: "lofted", label: "Air", title: "Lofted balls" },
  { key: "risk", label: "Rsk", title: "Risk" },
  { key: "tempo", label: "Tmp", title: "Tempo" },
  { key: "press_aggression", label: "PrA", title: "Press aggression" },
  { key: "pressing", label: "Prs", title: "Press numbers" },
  { key: "man_marking", label: "Man", title: "Man marking" },
  { key: "offensiveness", label: "Dep", title: "Depth" },
  { key: "breadth", label: "Brd", title: "Breadth" },
  { key: "compactness", label: "Cmp", title: "Compactness" },
];

/** Match stats, rendered the same way for both sides. Possession is shown as
 *  a share rather than raw ticks, which is meaningless to a reader. */
const STAT_ROWS: {
  key: string;
  label: string;
  render: (s: TeamStatsDTO) => string;
}[] = [
  { key: "shots", label: "Shots", render: (s) => String(s.shots) },
  {
    key: "on_target",
    label: "On target",
    render: (s) => String(s.shots_on_target),
  },
  { key: "passes", label: "Passes", render: (s) => String(s.passes) },
  {
    key: "completion",
    label: "Pass %",
    render: (s) =>
      s.passes > 0
        ? `${Math.round((s.passes_completed / s.passes) * 100)}%`
        : "—",
  },
  { key: "saves", label: "Saves", render: (s) => String(s.saves) },
  { key: "corners", label: "Corners", render: (s) => String(s.corners) },
  { key: "throw_ins", label: "Throw-ins", render: (s) => String(s.throw_ins) },
  {
    key: "goal_kicks",
    label: "Goal kicks",
    render: (s) => String(s.goal_kicks),
  },
];

function tone(value: number): string {
  // Three buckets rather than a gradient: the point is to spot a team's
  // extremes at a glance, not to read the number off the colour.
  if (value >= 1.25) return "high";
  if (value <= 0.75) return "low";
  return "mid";
}

export function TournamentPanel({
  tournament, busy, onCreate, onStartMatch, onAdvance, onNew,
}: Props) {
  const [seed, setSeed] = useState(0);
  const [teamIdx, setTeamIdx] = useState(0);
  // Which result row is showing its stats, as "homeId-awayId".
  const [expanded, setExpanded] = useState<string | null>(null);

  if (!tournament) {
    return (
      <div className="tournament-setup">
        <h2>New tournament</h2>
        <p className="muted">
          Eight teams, home and away — 56 matches over 14 matchdays. Every side
          but yours gets its own tactics, kept for the whole season.
        </p>
        <label>
          Your team
          <select
            value={teamIdx}
            onChange={(e) => setTeamIdx(Number(e.target.value))}
          >
            {CITY_NAMES.map((name, i) => (
              <option key={name} value={i}>{name}</option>
            ))}
          </select>
        </label>
        <label>
          Seed
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
          />
        </label>
        <button
          className="primary"
          disabled={busy}
          onClick={() => onCreate(seed, teamIdx)}
        >
          Create tournament
        </button>
      </div>
    );
  }

  const day = tournament.current_matchday;
  const dayFixtures = tournament.fixtures.filter((f) => f.matchday === day);
  const userFixture = dayFixtures.find((f) => f.involves_user);

  // Every completed matchday, newest first. Without this the user only ever
  // saw the CURRENT day's fixtures, so results vanished the moment the season
  // moved on -- they were stored all along, just never rendered.
  const playedDays = [
    ...new Set(
      tournament.fixtures.filter((f) => f.played).map((f) => f.matchday),
    ),
  ].sort((a, b) => b - a);
  const job = tournament.sim_job;

  return (
    <div className="tournament">
      <div className="tournament-header">
        <h2>
          {tournament.finished
            ? "Season complete"
            : `Matchday ${day} of ${tournament.matchdays}`}
        </h2>
        <div className="tournament-actions">
          <button
            className="primary"
            disabled={busy || tournament.finished || !tournament.user_match_pending}
            onClick={onStartMatch}
          >
            Play my match
          </button>
          <button
            disabled={busy || tournament.finished || tournament.user_match_pending
              || job?.running === true}
            onClick={onAdvance}
            title={
              tournament.user_match_pending
                ? "Play your match first"
                : "Simulate the other fixtures for this matchday"
            }
          >
            {busy ? "Starting…" : "Advance"}
          </button>
          <button disabled={busy} onClick={onNew}>
            New tournament
          </button>
        </div>
      </div>

      {job?.running && (
        <div className="sim-progress">
          <div className="sim-progress-text">
            Simulating matchday {job.matchday} — {job.done} of {job.total} done
          </div>
          <div className="sim-progress-track">
            <div
              className="sim-progress-fill"
              style={{
                width: `${job.total > 0 ? (job.done / job.total) * 100 : 0}%`,
              }}
            />
          </div>
          <div className="sim-progress-note">
            Matches run in parallel, so all three finish at about the same time.
          </div>
        </div>
      )}
      {job?.error && <div className="error">Simulation failed: {job.error}</div>}

      {tournament.user_match_pending && userFixture && (
        <p className="muted">
          Your fixture: <strong>{userFixture.home_name}</strong> v{" "}
          <strong>{userFixture.away_name}</strong>
        </p>
      )}

      <div className="tournament-body">
        <section>
          <h3>Table</h3>
          <table className="league-table">
            <thead>
              <tr>
                <th></th><th>Team</th><th>P</th><th>W</th><th>D</th><th>L</th>
                <th>GF</th><th>GA</th><th>GD</th><th>Pts</th>
              </tr>
            </thead>
            <tbody>
              {tournament.table.map((row) => (
                <tr key={row.team_id} className={row.is_user ? "user-row" : ""}>
                  <td className="muted">{row.position}</td>
                  <td>{row.name}</td>
                  <td>{row.played}</td>
                  <td>{row.won}</td>
                  <td>{row.drawn}</td>
                  <td>{row.lost}</td>
                  <td>{row.goals_for}</td>
                  <td>{row.goals_against}</td>
                  <td>{row.goal_difference > 0 ? `+${row.goal_difference}` : row.goal_difference}</td>
                  <td><strong>{row.points}</strong></td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section>
          <h3>Matchday {day}</h3>
          <ul className="fixture-list">
            {dayFixtures.map((f) => (
              <li
                key={`${f.home_id}-${f.away_id}`}
                className={f.involves_user ? "user-fixture" : ""}
              >
                <span className="fx-home">{f.home_name}</span>
                <span className={f.live ? "fx-score fx-live" : "fx-score"}>
                  {f.played || f.live
                    ? `${f.home_goals} – ${f.away_goals}`
                    : "v"}
                </span>
                <span className="fx-away">{f.away_name}</span>
              </li>
            ))}
          </ul>

          <h3>Results</h3>
          {playedDays.length === 0 ? (
            <p className="muted">No matches played yet.</p>
          ) : (
            <div className="results-history">
              {playedDays.map((md) => (
                <div key={md} className="result-day">
                  <h4>Matchday {md}</h4>
                  <ul className="fixture-list">
                    {tournament.fixtures
                      .filter((f) => f.matchday === md && f.played)
                      .map((f) => {
                        const key = `${f.home_id}-${f.away_id}`;
                        const open = expanded === key;
                        return (
                          <li key={key} className="result-row">
                            <button
                              className={
                                f.involves_user
                                  ? "result-line user-fixture"
                                  : "result-line"
                              }
                              onClick={() => setExpanded(open ? null : key)}
                              title={f.stats ? "Show match stats" : "No stats recorded"}
                            >
                              <span className="fx-home">{f.home_name}</span>
                              <span className="fx-score">
                                {f.home_goals} – {f.away_goals}
                              </span>
                              <span className="fx-away">{f.away_name}</span>
                            </button>
                            {open && f.stats && (
                              <table className="match-stats">
                                <thead>
                                  <tr>
                                    <th>{f.home_name}</th>
                                    <th></th>
                                    <th>{f.away_name}</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {STAT_ROWS.map((row) => (
                                    <tr key={row.key}>
                                      <td>{row.render(f.stats!.home)}</td>
                                      <th>{row.label}</th>
                                      <td>{row.render(f.stats!.away)}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            )}
                            {open && !f.stats && (
                              <p className="muted">No stats recorded.</p>
                            )}
                          </li>
                        );
                      })}
                  </ul>
                </div>
              ))}
            </div>
          )}

          <h3>Tactical profiles</h3>
          <table className="profile-table">
            <thead>
              <tr>
                <th>Team</th>
                {PROFILE.map((p) => (
                  <th key={p.key} title={p.title}>{p.label}</th>
                ))}
                <th>Shape</th>
              </tr>
            </thead>
            <tbody>
              {tournament.teams.map((t) => (
                <tr key={t.team_id} className={t.is_user ? "user-row" : ""}>
                  <td>{t.name}</td>
                  {PROFILE.map((p) => {
                    const v = t.tactics[p.key] as number;
                    return (
                      <td key={p.key} className={`bias-${tone(v)}`}>
                        {v.toFixed(2)}
                      </td>
                    );
                  })}
                  <td className="muted">{t.tactics.formation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
}
