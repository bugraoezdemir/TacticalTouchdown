"""
Pace with the ball and the Tempo slider (decisions.pace, workplan 2.9).

Every dribble direction is offered at a walk, a run and a sprint; the walk
and the run are planned from his motion. Tempo tilts the pace like Pass range
tilts the length, tilts passes by how far forward they go, and sets how much
waiting is worth. The idle rule is off.
"""
import numpy as np
import pytest

from app.engine import kinematics
from app.engine.decisions import candidates as cands
from app.engine.decisions import on_ball, pace
from app.engine.decisions.candidates import Candidate, _paced_slow
from app.engine.decisions.pace import progress_factor, tempo_factor, wait_gain, wait_value
from app.engine.factory import build_test_state


@pytest.fixture
def paced(monkeypatch):
    monkeypatch.setattr(kinematics, "GRIP", True)
    monkeypatch.setattr(pace, "PACE", True)


def _carrier(state):
    """A centre midfielder alone in the middle, everyone else far away."""
    team = state.home_team
    player = next(p for p in team.players if p.position == "CM")
    player.pos = np.array([40.0, 34.0]); player.velocity = np.zeros(2)
    for o in state.away_team.players:
        o.pos = np.array([100.0, 5.0 + 6.0 * (o.player_id % 10)]); o.velocity = np.zeros(2)
    for m in team.players:
        if m is not player:
            m.pos = np.array([20.0, 5.0 + 6.0 * (m.player_id % 10)]); m.velocity = np.zeros(2)
    return player, team


def test_neutral_tempo_has_no_preference():
    for share in (0.1, 0.3, 0.6, 1.0):
        assert tempo_factor(share, 1.0) == 1.0


def test_tempo_trades_fast_against_slow_reciprocally():
    assert tempo_factor(1.0, 2.0) > 1.0 > tempo_factor(0.3, 2.0)
    assert tempo_factor(1.0, 0.5) < 1.0 < tempo_factor(0.3, 0.5)
    for share in (0.3, 1.0):
        assert tempo_factor(share, 2.0) * tempo_factor(share, 0.5) == pytest.approx(1.0)


def _slow(player, state, band, direction=(1.0, 0.0)):
    """The `band` touch planned in `direction`, from _paced_slow."""
    d = np.array(direction)
    out = [o for o in _paced_slow(player, state.opponents_of(player.team_id)) if o[0] == band]
    return max(out, key=lambda o: float(np.dot(o[1] - player.pos, d)))


def test_from_a_standstill_the_walk_is_shorter_than_the_run(paced):
    state = build_test_state(seed=1)
    player, _ = _carrier(state)
    walk, run = _slow(player, state, "walk"), _slow(player, state, "run")
    assert 0.3 <= np.linalg.norm(walk[1] - player.pos) < np.linalg.norm(run[1] - player.pos)
    assert walk[2] < run[2] and 0.99 < walk[3] <= 1.0   # nobody near: the race is his


def test_a_walk_at_speed_is_a_braking_touch_where_his_momentum_takes_him(paced):
    """At 5 m/s he cannot walk within one touch: the walk is planned from his
    motion, so it goes a long way ahead and does not turn him on the spot."""
    state = build_test_state(seed=1)
    player, _ = _carrier(state)
    still = np.linalg.norm(_slow(player, state, "walk")[1] - player.pos)
    player.velocity = np.array([5.0 / 30.0, 0.0])
    back = _slow(player, state, "walk", direction=(-1.0, 0.0))
    ahead = _slow(player, state, "walk")
    assert np.linalg.norm(ahead[1] - player.pos) > 2.0 * still
    assert back[1][0] > player.pos[0]          # even 'back' ends ahead: he is still braking


def test_pass_tempo_trades_forward_against_back():
    assert progress_factor(25.0, 1.0) == progress_factor(-25.0, 1.0) == 1.0
    assert progress_factor(20.0, 2.0) > progress_factor(0.0, 2.0) == 1.0 > progress_factor(-20.0, 2.0)
    assert progress_factor(20.0, 0.5) < 1.0 < progress_factor(-20.0, 0.5)
    assert progress_factor(40.0, 2.0) == progress_factor(20.0, 2.0)   # clamped at 20 m


def test_patience_counts_waiting_for_more_at_low_tempo(paced):
    state = build_test_state(seed=1)
    player, team = _carrier(state)
    opponents = state.opponents_of(player.team_id)
    mate = next(m for m in team.players if m is not player and m.position != "GK")
    mate.pos = np.array([45.0, 20.0]); mate.velocity = np.array([0.2, 0.0])
    gains = []
    for tempo in (0.5, 1.0, 2.0):
        team.tactics.tempo = tempo
        gains.append(wait_gain(player, state, opponents))
    assert gains[0] >= gains[1] >= gains[2] and gains[0] > gains[2]


def test_the_dribble_options_carry_their_pace(paced):
    state = build_test_state(seed=1)
    player, _ = _carrier(state)
    dribbles = [c for c in cands.build_candidates(player, state) if c.kind == "dribble"]
    assert {c.pace for c in dribbles} == {"walk", "run", "sprint"}
    assert all(c.race is not None for c in dribbles)


def _best_pace(state, player, team, tempo):
    team.tactics.tempo = tempo
    scored = [on_ball.score_candidate(c, player, state)
              for c in cands.build_candidates(player, state) if c.kind == "dribble"]
    return max(scored, key=lambda c: c.utility).pace


def test_tempo_moves_the_chosen_pace(paced):
    state = build_test_state(seed=1)
    player, team = _carrier(state)
    assert _best_pace(state, player, team, 2.0) == "sprint"
    assert _best_pace(state, player, team, 0.5) == "walk"


def test_waiting_is_worth_a_teammates_run_into_space(paced):
    state = build_test_state(seed=1)
    player, team = _carrier(state)
    opponents = state.opponents_of(player.team_id)
    mate = next(m for m in team.players if m is not player and m.position != "GK")
    mate.pos = np.array([45.0, 20.0]); mate.velocity = np.zeros(2)
    standing = wait_value(player, state, opponents)
    mate.velocity = np.array([0.2, 0.0])            # running into the space ahead
    state.tick += 1                                 # a new decision
    assert wait_value(player, state, opponents) > standing


def test_no_idle_walk_under_pace(paced):
    """The owned walk (a 'hold' with a move) is gone: a slow carry is a touch."""
    state = build_test_state(seed=1)
    player, _ = _carrier(state)
    walk = Candidate(kind="dribble", target=player.pos + np.array([1.0, 0.0]), speed=0.0,
                     friction=0.005, pace="walk")
    action = on_ball._action_from(walk, player, state)
    assert action.action_type == "dribble"


def test_a_walking_touch_rolls_ahead_and_is_his_only_at_the_meeting(paced):
    """A short touch is nudged clear of his feet, not 2 m ahead of a 1 m touch,
    and he cannot take it back before the planned meeting -- or he re-decided
    every few ticks and turned on the spot."""
    from app.engine import game_loop as gl
    from app.engine.decisions.candidates import dribble_nudge
    from app.engine.decisions.on_ball import OnBallAction
    from app.engine.constants import CAPTURE_RADIUS, DRIBBLE_NUDGE

    assert CAPTURE_RADIUS < dribble_nudge(0.8) < 0.8 and dribble_nudge(0.4) == 0.4
    assert dribble_nudge(5.0) == DRIBBLE_NUDGE
    state = build_test_state(seed=1)
    player, _ = _carrier(state)
    for p in state.all_players():
        p.has_ball = False
    state.ball.pos = player.pos.copy(); state.ball.owner_id = player.player_id
    player.has_ball = True
    walk = OnBallAction("dribble", np.array([0.02, 0.0]), None, None, touch=0.8)
    monkey = {player.player_id: walk}
    real = gl.on_ball_decision
    gl.on_ball_decision = lambda p, s: monkey.get(p.player_id) or real(p, s)
    try:
        gl.tick(state)
    finally:
        gl.on_ball_decision = real
    assert state.ball.no_capture_id == player.player_id
    assert state.ball.no_capture_until >= state.ball.dribble_meet_tick - 1


def test_a_defender_running_past_is_further_away_than_he_looks(paced):
    """Room and contest read the opponent's motion: standing he counts at his
    distance, running at the spot closer, sprinting past or away much further."""
    from types import SimpleNamespace
    from app.engine.decisions.on_ball import _motion_gaps
    spot = np.array([50.0, 34.0])

    def gap(v):
        o = SimpleNamespace(pos=np.array([47.0, 34.0]), velocity=np.array(v), speed=0.236)
        return float(_motion_gaps([o], spot)[0])
    assert gap((0.0, 0.0)) == pytest.approx(3.0)
    assert gap((0.22, 0.0)) < 3.0 < gap((0.0, 0.22)) < gap((-0.22, 0.0))


def test_a_high_tempo_side_sprints_to_support_sooner(paced):
    """The support-run sprint gap is SUPPORT_SPRINT_GAP / Tempo: an 8 m run in
    his own half is a jog at Tempo 1 and a sprint at 2."""
    from app.engine.decisions.off_ball import _support_sprint
    state = build_test_state(seed=1)
    player, team = _carrier(state)
    player.pos = np.array([30.0, 34.0])
    target = np.array([38.0, 34.0])
    team.tactics.tempo = 1.0
    assert not _support_sprint(player, state, team, target)
    team.tactics.tempo = 2.0
    assert _support_sprint(player, state, team, target)
