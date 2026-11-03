import { useEffect, useRef, useState } from "react";
import { Pitch } from "./components/Pitch";
import { StatsTable } from "./components/StatsTable";
import { TacticsPanel } from "./components/TacticsPanel";
import { TournamentPanel } from "./components/TournamentPanel";
import { MatchesTab } from "./components/MatchesTab";
import { useSimulation } from "./hooks/useSimulation";
import { useTournament } from "./hooks/useTournament";
import { engineLoaded, engineNotice, IN_BROWSER } from "./api";
import "./App.css";

// Multiples of REAL time: 1x is 30 engine ticks a second, a match minute per
// minute. (These were delays between ticks, added on top of the engine and the
// redraw, so "2x" actually played at about 0.4x.) Max runs as fast as the
// engine can; a speed the engine cannot keep up with plays at that limit.
const SPEEDS = [
  { label: "0.5×", factor: 0.5 },
  { label: "1×",   factor: 1 },
  { label: "1.5×", factor: 1.5 },
  { label: "2×",   factor: 2 },
  { label: "4×",   factor: 4 },
  { label: "Max",  factor: Infinity },
];

type Tab = "match" | "tournament" | "matches";

export default function App() {
  const [tab, setTab] = useState<Tab>("match");
  const [speedIdx, setSpeedIdx] = useState(1);
  const [engineLoading, setEngineLoading] = useState(IN_BROWSER);
  const [notice, setNotice] = useState("");
  useEffect(() => {
    void engineLoaded.finally(() => {
      setEngineLoading(false);
      setNotice(engineNotice());
    });
  }, []);
  const speed = SPEEDS[speedIdx].factor;

  const {
    state, playback, status, homeGoals, awayGoals, error, log, banner,
    start, pause, reset, updateTactics, loadServerMatch,
  } = useSimulation(speed);

  const {
    tournament, busy: tBusy, error: tError,
    create: createTournament, clear: clearTournament, startMatch, finishMatch, advance,
  } = useTournament();

  // A tournament match is over when the ENGINE says so, and the result is
  // recorded automatically rather than on a button: that is what stops a
  // losing position from being abandoned, and it is where the user's tactics
  // are written back to their team for the rest of the season.
  //
  // In an effect, not in render: calling finishMatch() during render would fire
  // on every re-render and post the same result repeatedly.
  const matchFinished = state?.finished === true;
  const awaitingResult = tournament?.user_match_pending === true;
  // Once per match. On the next matchday the last match is still the finished
  // one on show while a new one is pending, which read as "record this result"
  // again and again until the user could press nothing.
  const leagueMatchLive = useRef(false);
  useEffect(() => {
    if (matchFinished && awaitingResult && !tBusy && leagueMatchLive.current) {
      leagueMatchLive.current = false;
      void finishMatch();
    }
  }, [matchFinished, awaitingResult, tBusy, finishMatch]);

  function handleTeamTactic(
    teamId: number,
    action:
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
      | "man_marking"
      | "wing_attacks",
    value: number,
  ) {
    updateTactics({ team_id: teamId, [action]: value });
  }

  // Formation is a string, so it needs its own handler rather than widening the
  // numeric one -- a shared computed-key spread would lose type safety.
  function handleFormationChange(teamId: number, formation: string) {
    updateTactics({ team_id: teamId, formation });
  }

  function handlePlayerTactic(
    teamId: number,
    playerId: number,
    field:
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
      | "bias_man_marking"
      | "bias_wing_attacks",
    value: number | null,
  ) {
    const team = teamId === state?.home_team.team_id
      ? state?.home_team
      : state?.away_team;
    const player = team?.players.find((p) => p.player_id === playerId);
    if (!player) return;
    updateTactics({
      team_id: teamId,
      players: [{
        player_id: playerId,
        bias_shoot: field === "bias_shoot" ? value : player.bias_shoot,
        bias_pass: field === "bias_pass" ? value : player.bias_pass,
        bias_dribble: field === "bias_dribble" ? value : player.bias_dribble,
        bias_pressing:
          field === "bias_pressing" ? value : player.bias_pressing,
        bias_roaming:
          field === "bias_roaming" ? value : player.bias_roaming,
        bias_offensiveness:
          field === "bias_offensiveness" ? value : player.bias_offensiveness,
        bias_direct:
          field === "bias_direct" ? value : player.bias_direct,
        bias_one_two:
          field === "bias_one_two" ? value : player.bias_one_two,
        bias_attack_call:
          field === "bias_attack_call" ? value : player.bias_attack_call,
        // Every override must be sent: the backend sets each field from this
        // entry, so one left out is reset to the team value -- editing one
        // slider used to wipe a player's pass range, breadth and compactness.
        bias_pass_range:
          field === "bias_pass_range" ? value : player.bias_pass_range,
        bias_breadth:
          field === "bias_breadth" ? value : player.bias_breadth,
        bias_compactness:
          field === "bias_compactness" ? value : player.bias_compactness,
        bias_possession:
          field === "bias_possession" ? value : player.bias_possession,
        bias_crossing:
          field === "bias_crossing" ? value : player.bias_crossing,
        bias_lofted:
          field === "bias_lofted" ? value : player.bias_lofted,
        bias_risk:
          field === "bias_risk" ? value : player.bias_risk,
        bias_tempo:
          field === "bias_tempo" ? value : player.bias_tempo,
        bias_press_aggression:
          field === "bias_press_aggression" ? value : player.bias_press_aggression,
        bias_man_marking:
          field === "bias_man_marking" ? value : player.bias_man_marking,
        bias_wing_attacks:
          field === "bias_wing_attacks" ? value : player.bias_wing_attacks,
      }],
    });
  }

  const isRunning = status === "running";

  function handlePlayPause() {
    if (isRunning) pause();
    else start();
  }

  const ball = state?.ball;

  const ownerName = (() => {
    if (!state || ball?.owner_id == null) return "—";
    const all = [...state.home_team.players, ...state.away_team.players];
    const p = all.find((p) => p.player_id === ball.owner_id);
    return p ? `${p.team_id === state.home_team.team_id ? "🔵" : "🔴"} ${p.name}` : "—";
  })();

  return (
    <div className="app">
      <h1>⚽ Football Simulation</h1>

      <div className="tab-bar">
        <button
          className={tab === "match" ? "tab active" : "tab"}
          onClick={() => setTab("match")}
        >
          Match
        </button>
        <button
          className={tab === "tournament" ? "tab active" : "tab"}
          onClick={() => setTab("tournament")}
        >
          Tournament
        </button>
        <button
          className={tab === "matches" ? "tab active" : "tab"}
          onClick={() => setTab("matches")}
        >
          Matches
        </button>
      </div>

      {engineLoading && <div className="engine-loading">Loading the game engine…</div>}
      {notice && <div className="error">{notice}</div>}
      {error && <div className="error">{error}</div>}
      {tError && <div className="error">{tError}</div>}

      {tab === "matches" ? (
        <MatchesTab tournament={tournament} />
      ) : tab === "tournament" ? (
        <TournamentPanel
          tournament={tournament}
          busy={tBusy}
          onCreate={(seed, idx) => void createTournament(seed, idx)}
          onStartMatch={async () => {
            // Load the fixture server-side FIRST, then adopt it, then switch
            // tabs -- otherwise the Match tab renders the previous match for a
            // frame and the user sees the wrong teams.
            const next = await startMatch();
            if (next) {
              await loadServerMatch();
              leagueMatchLive.current = true;
              setTab("match");
            }
          }}
          onAdvance={() => void advance()}
          onNew={() => {
            if (window.confirm("Start a new tournament? This one will be lost.")) {
              leagueMatchLive.current = false;
              void clearTournament().then(reset);
            }
          }}
        />
      ) : (
      <>

      <div className="controls">
        <button
          type="button"
          className={`btn ${isRunning ? "btn-pause" : "btn-play"}`}
          onClick={handlePlayPause}
        >
          {isRunning ? "⏸ Pause" : status === "idle" ? "▶ Start" : "▶ Resume"}
        </button>
        <button type="button" className="btn btn-reset" onClick={reset}>
          ↺ Reset
        </button>

        <div className="speed-group">
          {SPEEDS.map((s, i) => (
            <button
              key={s.label}
              type="button"
              className={`speed-btn ${i === speedIdx ? "active" : ""}`}
              onClick={() => setSpeedIdx(i)}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <div className="scoreboard">
        <div className="score-team home">
          <span className="team-dot home-dot" />
          {state?.home_team.name ?? "Home FC"}
        </div>
        <div className="score-goals">
          <span className="goal-count">{homeGoals}</span>
          <span className="score-sep"> — </span>
          <span className="goal-count">{awayGoals}</span>
        </div>
        <div className="score-team away">
          {state?.away_team.name ?? "Away FC"}
          <span className="team-dot away-dot" />
        </div>
      </div>

      <div className="stage">
        <TacticsPanel
          team={state?.home_team ?? null}
          side="home"
          onTeamChange={handleTeamTactic}
          onFormationChange={handleFormationChange}
          onPlayerChange={handlePlayerTactic}
        />

        {/* Pitch and its caption are one column, so the sidebars cannot
            become extra columns and squeeze the canvas. */}
        <div className="stage-center">
          <div className="pitch-wrap">
            <Pitch state={state} playback={playback} />
            {banner && (
              <div className={`event-banner ${banner.kind === "goal" ? "goal" : ""}`}>
                <div className="event-banner-text">{banner.detail}</div>
                {banner.player_name && (
                  <div className="event-banner-sub">{banner.player_name}</div>
                )}
              </div>
            )}
          </div>

          <div className="info-bar">
            {/* Clock counts within the half, so it restarts at the interval
                the way a real one does. */}
            <span>
              <strong>{state ? (state.half === 1 ? "1st" : "2nd") : "—"}</strong> half
            </span>
            {/* The clock comes from the backend so the tick->minute mapping
                lives in one place and cannot drift. */}
            <span>
              Clock{" "}
              <strong>
                {state
                  ? `${state.clock_minute}'${String(state.clock_second).padStart(2, "0")}`
                  : "—"}
              </strong>
            </span>
            <span>Tick <strong>{state?.tick ?? "—"}</strong></span>
            <span>Ball <strong>{ball?.state ?? "—"}</strong></span>
                <span>Holder <strong>{ownerName}</strong></span>
            {state?.finished && <span className="full-time-flag">Full time</span>}
            {!state?.finished && (state?.restart_pause ?? 0) > 0 && (
              <span className="paused-flag">
                Restart in {((state!.restart_pause / 30)).toFixed(1)}s
              </span>
            )}
          </div>
        </div>

        <TacticsPanel
          team={state?.away_team ?? null}
          side="away"
          onTeamChange={handleTeamTactic}
          onFormationChange={handleFormationChange}
          onPlayerChange={handlePlayerTactic}
        />
      </div>

      {/* Log and stats sit side by side below the pitch. */}
      <div className="summary">
      <div className="match-log">
        <h2>Match log</h2>
        {log.length === 0 ? (
          <p className="log-empty">No events yet.</p>
        ) : (
          <ul>
            {log.map((e) => (
              <li key={`${e.tick}-${e.kind}-${e.detail}`} className={`log-${e.kind}`}>
                <span className="log-time">
                  {Math.floor(e.tick / 30 / 60)}&apos;
                  {String(Math.floor((e.tick / 30) % 60)).padStart(2, "0")}
                </span>
                <span className="log-detail">{e.detail}</span>
                {e.player_name && <span className="log-player">{e.player_name}</span>}
              </li>
            ))}
          </ul>
        )}
      </div>

        <StatsTable
          stats={state?.stats}
          homeName={state?.home_team.name ?? "Home"}
          awayName={state?.away_team.name ?? "Away"}
        />
      </div>

      </>
      )}

      <div className="legend">
        <span><span className="dot home-dot" /> {state?.home_team.name ?? "Home"}</span>
        <span><span className="dot away-dot" /> {state?.away_team.name ?? "Away"}</span>
        <span><span className="dot ball-dot" /> Ball</span>
        <span><span className="dot carrier-ring" /> Ball carrier</span>
      </div>
    </div>
  );
}
