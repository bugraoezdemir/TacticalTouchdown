import { useCallback, useEffect, useRef, useState } from "react";
import type { Tournament } from "../types/simulation";
import { apiFetch, engineLoaded } from "../api";

/** The tournament is server-owned, exactly like the simulation: the client
 *  never computes a table or a fixture, it only renders what it is given. */
export function useTournament() {
  const [tournament, setTournament] = useState<Tournament | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const call = useCallback(
    async (path: string, body?: unknown): Promise<Tournament | null> => {
      setBusy(true);
      setError(null);
      try {
        const res = await apiFetch(`/tournament/${path}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: body === undefined ? undefined : JSON.stringify(body),
        });
        if (!res.ok) {
          // FastAPI puts the reason in `detail`; surfacing it is what makes
          // "play your match first" readable instead of a bare 400.
          const payload = await res.json().catch(() => null);
          throw new Error(payload?.detail ?? `Request failed (${res.status})`);
        }
        const data: Tournament = await res.json();
        setTournament(data);
        return data;
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        return null;
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  /** Read the season without mutating it. Used by the poll loop below. */
  const refresh = useCallback(async (): Promise<Tournament | null> => {
    try {
      const res = await apiFetch("/tournament/state");
      if (!res.ok) return null;
      const data: Tournament = await res.json();
      setTournament(data);
      return data;
    } catch {
      return null;
    }
  }, []);

  // While a matchday is simulating, poll so scores appear one at a time
  // instead of all at the end. Advance returns immediately now, so without
  // this the table would sit stale until the user clicked something.
  const running = tournament?.sim_job?.running === true;
  const refreshRef = useRef(refresh);
  refreshRef.current = refresh;
  // A season already under way: saved in the browser, or held by the server.
  useEffect(() => {
    void engineLoaded.then(() => refreshRef.current());
  }, []);
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => void refreshRef.current(), 1000);
    return () => clearInterval(id);
  }, [running]);

  const create = useCallback(
    (seed: number, userTeamIndex: number) =>
      call("create", { seed, user_team_index: userTeamIndex }),
    [call],
  );
  const clear = useCallback(async () => {
    await apiFetch("/tournament/reset", { method: "POST" });
    setError(null);
    setTournament(null);
  }, []);
  const startMatch = useCallback(() => call("start-match"), [call]);
  const finishMatch = useCallback(() => call("finish-match"), [call]);
  const advance = useCallback(() => call("advance"), [call]);
  const setTeamTactics = useCallback(
    (teamId: number, patch: Record<string, number | string>) =>
      call("tactics", { team_id: teamId, ...patch }),
    [call],
  );

  return {
    tournament, busy, error,
    create, clear, startMatch, finishMatch, advance, setTeamTactics, refresh,
  };
}
