export interface BallDTO {
  pos: [number, number];
  velocity: [number, number];
  state: "stationary" | "in_pass" | "in_shot" | "in_dribble";
  owner_id: number | null;
  last_toucher_team_id: number | null;
  height?: number; // m above the pitch
}

export interface PlayerDTO {
  player_id: number;
  team_id: number;
  name: string;
  position: string;
  tactical: string;
  pos: [number, number];
  velocity: [number, number];
  speed: number;
  strength: number;
  intelligence: number;
  passing: number;
  goalkeeping: number;
  shooting: number;
  dribbling: number;
  bias_shoot: number | null;
  bias_pass: number | null;
  bias_dribble: number | null;
  bias_pressing: number | null;
  bias_roaming: number | null;
  bias_offensiveness: number | null;
  bias_direct: number | null;
  bias_one_two: number | null;
  bias_attack_call: number | null;
  bias_pass_range: number | null;
  bias_breadth: number | null;
  bias_compactness: number | null;
  bias_possession: number | null;
  bias_crossing?: number | null;
  bias_lofted?: number | null;
  bias_risk?: number | null;
  bias_tempo?: number | null;
  bias_press_aggression?: number | null;
  bias_man_marking?: number | null;
  bias_wing_attacks?: number | null;
  assigned_pos: [number, number];
  marking_target_id: number | null;
  has_ball: boolean;
  runway_signal_target: [number, number] | null;
  runway_signal_sender_id: number | null;
}

export interface TeamTacticsDTO {
  shoot: number;
  pass_: number;
  dribble: number;
  pressing: number;
  roaming: number;
  offensiveness: number;
  direct: number;
  one_two: number;
  attack_call: number;
  pass_range: number;
  breadth: number;
  compactness: number;
  possession: number;
  crossing: number;
  lofted: number;
  risk: number;
  tempo: number;
  press_aggression: number;
  man_marking: number;
  wing_attacks: number;
  formation: string;
}

export interface TeamDTO {
  team_id: number;
  name: string;
  players: PlayerDTO[];
  direction: number;
  tactics: TeamTacticsDTO;
}

/** Sliders are a relative bias, not a probability: they need not sum to
 *  anything, and the engine clamps each to [0.5, 2.0]. */
export interface TacticsUpdate {
  team_id: number;
  shoot?: number;
  pass_?: number;
  dribble?: number;
  pressing?: number;
  roaming?: number;
  offensiveness?: number;
  direct?: number;
  one_two?: number;
  attack_call?: number;
  pass_range?: number;
  breadth?: number;
  compactness?: number;
  possession?: number;
  crossing?: number;
  lofted?: number;
  risk?: number;
  tempo?: number;
  press_aggression?: number;
  man_marking?: number;
  wing_attacks?: number;
  formation?: string;
  players?: {
    player_id: number;
    bias_shoot: number | null;
    bias_pass: number | null;
    bias_dribble: number | null;
    bias_pressing: number | null;
  bias_roaming: number | null;
  bias_offensiveness: number | null;
  bias_direct: number | null;
  bias_one_two: number | null;
  bias_attack_call: number | null;
  bias_pass_range: number | null;
  bias_breadth: number | null;
  bias_compactness: number | null;
  bias_possession: number | null;
  bias_crossing?: number | null;
  bias_lofted?: number | null;
  bias_risk?: number | null;
  bias_tempo?: number | null;
  bias_press_aggression?: number | null;
  bias_man_marking?: number | null;
  bias_wing_attacks?: number | null;
  }[];
}

export const FORMATIONS = [
  "4-4-2",
  "4-3-3",
  "4-2-3-1",
  "3-5-2",
  "5-3-2",
] as const;

export interface TeamStatsDTO {
  possession_ticks: number;
  passes: number;
  passes_completed: number;
  shots: number;
  shots_on_target: number;
  goals: number;
  corners: number;
  throw_ins: number;
  goal_kicks: number;
  saves: number;
  crosses?: number;
  crosses_completed?: number;
}

export interface PlayerStatsDTO {
  slot: number;
  name: string;
  position: string;
  passes: number;
  passes_completed: number;
  passes_intercepted: number;
  dribbles: number;
  dribbles_lost: number;
  shots: number;
  shots_on_target: number;
  goals: number;
  touches: number;
}

export interface MatchStatsDTO {
  home: TeamStatsDTO;
  away: TeamStatsDTO;
  home_players: PlayerStatsDTO[];
  away_players: PlayerStatsDTO[];
}

export interface MatchEventDTO {
  kind: "goal" | "throw_in" | "corner" | "goal_kick" | "kickoff";
  tick: number;
  team_id: number | null;
  team_name: string;
  player_name: string | null;
  detail: string;
}

export interface GameStateDTO {
  tick: number;
  ball: BallDTO;
  home_team: TeamDTO;
  away_team: TeamDTO;
  match_id: string;
  half: number;
  finished: boolean;
  clock_minute: number;
  clock_second: number;
  home_score: number;
  away_score: number;
  restart_pause: number;
  events: MatchEventDTO[];
  stats: MatchStatsDTO;
  /** Every tick of a batched /tick request: [ball x, y, height, owner id (-1
   *  for none), home score, away score, then x, y of each player in
   *  `frame_order`]. Empty for other responses. */
  frames?: number[][];
  frame_order?: number[];
}

// -- Tournament ---------------------------------------------------------------

/** One squad slot's stored overrides. Slot, not player_id: ids are rebuilt for
 *  every fixture, so only the formation slot is stable across matches. */
export interface TournamentPlayerBias {
  slot: number;
  bias_shoot: number | null;
  bias_pass: number | null;
  bias_dribble: number | null;
  bias_pressing: number | null;
  bias_roaming: number | null;
  bias_offensiveness: number | null;
  bias_direct: number | null;
  bias_one_two: number | null;
  bias_attack_call: number | null;
  bias_pass_range: number | null;
  bias_breadth: number | null;
  bias_compactness: number | null;
  bias_possession: number | null;
  bias_crossing?: number | null;
  bias_lofted?: number | null;
  bias_risk?: number | null;
  bias_tempo?: number | null;
  bias_press_aggression?: number | null;
  bias_man_marking?: number | null;
  bias_wing_attacks?: number | null;
}

export interface TournamentTeam {
  team_id: number;
  name: string;
  is_user: boolean;
  preset_name: string;
  tactics: TeamTacticsDTO;
  player_biases: TournamentPlayerBias[];
}

export interface Fixture {
  matchday: number;
  home_id: number;
  away_id: number;
  home_name: string;
  away_name: string;
  played: boolean;
  home_goals: number;
  away_goals: number;
  involves_user: boolean;
  /** Present once played. Headless matches keep theirs here too — their
   *  GameState is gone, so this is the only surviving record. */
  stats: MatchStatsDTO | null;
  /** True while this fixture is simulating; its goals are the running score. */
  live: boolean;
}

export interface SimJob {
  running: boolean;
  matchday: number;
  done: number;
  total: number;
  error: string | null;
}

export interface TableRow {
  position: number;
  team_id: number;
  name: string;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  goals_for: number;
  goals_against: number;
  goal_difference: number;
  points: number;
  is_user: boolean;
}

export interface Tournament {
  seed: number;
  user_team_id: number;
  matchdays: number;
  current_matchday: number;
  finished: boolean;
  teams: TournamentTeam[];
  fixtures: Fixture[];
  table: TableRow[];
  user_match_pending: boolean;
  sim_job: SimJob | null;
}
