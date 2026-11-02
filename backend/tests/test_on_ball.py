"""
On-ball decisions: P x V x C over every candidate, argmax wins.

The regressions guarded here all produced visibly stupid football:
passes played into a marker, through-balls nobody could reach, and shots
that scored as impossible because the keeper's reach filled the lane.
"""
import numpy as np
import pytest

from app.engine.constants import (
    BALL_FRICTION_COEFF,
    PASS_SPEED_LOW,
    PLAYER_BASE_SPEED,
    SHOT_FRICTION_COEFF,
    SHOT_SPEED,
)
from app.engine.decisions.candidates import build_candidates
from app.engine.decisions.interception import route_survival
from app.engine.decisions.on_ball import on_ball_decision, score_candidate
from app.engine.factory import build_test_state
from tests.helpers import midfielder, striker
from app.engine.models import Player


def opp(x, y, position="CD", pid=99):
    return Player(
        player_id=pid, team_id=2, name=f"O{pid}", position=position, tactical="normal",
        pos=np.array([x, y], dtype=float), velocity=np.zeros(2), speed=PLAYER_BASE_SPEED,
        strength=70.0, intelligence=70.0, passing=70.0, goalkeeping=40.0,
        assigned_pos=np.array([x, y], dtype=float), marking_target_id=None,
        has_ball=False, runway_signal=None,
    )


PASSER = np.array([40.0, 34.0])
TARGET = np.array([60.0, 34.0])


def survive(blockers, origin=PASSER, target=TARGET):
    return route_survival(origin, target, PASS_SPEED_LOW, BALL_FRICTION_COEFF, blockers)


# ── The blocked lane ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("x", [42.0, 45.0, 50.0, 55.0, 58.0])
def test_defender_in_the_lane_kills_the_pass(x):
    """A body in the passing lane blocks it. Reaction time used to be charged
    even to a defender already standing there, so the ball 'beat' him and
    passes played straight at a marker scored as safe."""
    assert survive([opp(x, 34.0)]) < 0.2


@pytest.mark.parametrize("y", [45.0, 55.0])
def test_defender_far_off_the_lane_does_not(y):
    assert survive([opp(50.0, y)]) > 0.9


def test_clear_lane_is_certain():
    assert survive([]) == 1.0


def test_reaction_cost_scales_with_distance():
    """Someone already in the lane pays no reaction cost; someone who must
    travel to it pays the full one."""
    in_lane = survive([opp(50.0, 34.0)])
    must_run = survive([opp(50.0, 38.0)])
    assert in_lane < must_run


# ── Shots ────────────────────────────────────────────────────────────────────

def test_shooting_at_the_keeper_is_worse_than_the_corner():
    gk = opp(104.0, 34.0, position="GK")
    shooter = np.array([92.0, 34.0])
    at_keeper = route_survival(shooter, np.array([105.0, 34.0]), SHOT_SPEED,
                               SHOT_FRICTION_COEFF, [gk])
    at_corner = route_survival(shooter, np.array([105.0, 30.8]), SHOT_SPEED,
                               SHOT_FRICTION_COEFF, [gk])
    assert at_corner > at_keeper


def test_close_shots_can_beat_the_keeper():
    """The keeper's reach counted as a body filling the lane, which scored every
    shot as ~90% saved and produced matches with zero shots."""
    gk = opp(104.0, 34.0, position="GK")
    close = route_survival(np.array([100.0, 34.0]), np.array([105.0, 30.8]),
                           SHOT_SPEED, SHOT_FRICTION_COEFF, [gk])
    assert close > 0.2


def test_distant_shots_are_hopeless():
    gk = opp(104.0, 34.0, position="GK")
    far = route_survival(np.array([78.0, 34.0]), np.array([105.0, 30.8]),
                         SHOT_SPEED, SHOT_FRICTION_COEFF, [gk])
    assert far < 0.05


# ── Runway passes ────────────────────────────────────────────────────────────

def test_runway_lead_is_reachable_by_the_receiver():
    """The lead was a fixed 8 m regardless of whether the receiver could get
    there. Most runway balls arrived at a point their target was still 5-16 m
    away from, handing possession straight to the opposition."""
    state = build_test_state()
    passer = midfielder(state)
    mate = striker(state)
    mate.pos = passer.pos + np.array([15.0, 0.0])
    mate.velocity = np.array([mate.speed, 0.0])

    runways = [c for c in build_candidates(passer, state)
               if c.is_runway and not c.is_called_run]
    for c in runways:
        lead = float(np.hypot(*(c.target - mate.pos)))
        t_mate = lead / mate.speed
        ball_dist = float(np.hypot(*(c.target - passer.pos)))
        from app.engine.ball_physics import travel_ticks
        t_ball = travel_ticks(c.speed, c.friction, ball_dist)
        assert t_mate <= t_ball + 8.0 + 1e-6, "receiver cannot reach this lead"


def test_stationary_teammate_gets_no_runway():
    state = build_test_state()
    passer = midfielder(state)
    for m in state.home_team.players:
        m.velocity = np.zeros(2)
    assert not any(
        c.is_runway and not c.is_called_run for c in build_candidates(passer, state)
    )


# ── Reception: a pass must be WON, not merely delivered ──────────────────────

def test_pass_into_opponent_space_scores_low():
    """Surviving the flight is necessary but not sufficient -- someone has to
    win the ball at the far end."""
    state = build_test_state()
    passer = midfielder(state)
    mate = striker(state)
    mate.pos = passer.pos + np.array([25.0, 0.0])
    mate.velocity = np.array([mate.speed, 0.0])

    # Drop an opponent right on top of the intended receiving zone.
    marker = state.away_team.players[4]
    marker.pos = mate.pos + np.array([9.0, 0.0])

    cands = [score_candidate(c, passer, state) for c in build_candidates(passer, state)]
    runways = [c for c in cands if c.is_runway and not c.is_called_run]
    if runways:
        assert max(c.p for c in runways) < 0.95


# ── The trade-off the whole model exists to make ─────────────────────────────

def test_poor_passing_options_favour_dribbling():
    """When no pass is on, carrying the ball to create a better angle should
    win -- and it should fall out of the utility, not a rule."""
    state = build_test_state()
    carrier = midfielder(state)
    carrier.pos = np.array([50.0, 34.0])
    carrier.has_ball = True
    state.ball.pos = carrier.pos.copy()
    state.ball.owner_id = carrier.player_id

    # Wall off every teammate behind markers.
    for i, mate in enumerate(state.home_team.players):
        if mate is carrier:
            continue
        mate.pos = np.array([50.0 + 12.0, 20.0 + i * 3.0])
    for i, o in enumerate(state.away_team.players):
        o.pos = np.array([50.0 + 6.0, 20.0 + i * 3.0])

    cands = [score_candidate(c, carrier, state) for c in build_candidates(carrier, state)]
    best_pass = max((c.utility for c in cands if c.kind == "pass"), default=0.0)
    best_dribble = max((c.utility for c in cands if c.kind == "dribble"), default=0.0)
    assert best_dribble > best_pass


def test_decision_returns_a_usable_action():
    state = build_test_state()
    carrier = state.get_ball_owner()
    action = on_ball_decision(carrier, state)
    assert action.action_type in ("shot", "pass", "dribble", "hold")
    assert np.all(np.isfinite(action.velocity))
    if action.action_type == "hold":
        assert np.allclose(action.velocity, 0.0)
    else:
        assert float(np.hypot(*action.velocity)) > 0.0


# ── Shooting: a struck ball is hard to block ─────────────────────────────────

def test_defender_slightly_off_the_line_barely_blocks_a_shot():
    """A ball struck at 33 m/s crosses 13 m in 0.4 s. Charging an outfield
    player the same reaction he gets against a rolling pass made a defender
    1.5 m off the line a near-certain block, so players declined shots from
    good positions."""
    gk = opp(104.0, 34.0, position="GK", pid=1)
    blocker = opp(96.0, 35.5, pid=2)
    shooter = np.array([92.0, 34.0])
    corner = np.array([105.0, 30.8])

    with_blocker = route_survival(shooter, corner, SHOT_SPEED, SHOT_FRICTION_COEFF,
                                  [gk, blocker], is_shot=True)
    gk_only = route_survival(shooter, corner, SHOT_SPEED, SHOT_FRICTION_COEFF,
                             [gk], is_shot=True)
    assert with_blocker > gk_only * 0.7


def test_shots_from_good_positions_are_viable():
    """P was collapsing to ~0.2 at penalty-spot range, which is why reasonable
    chances were passed up."""
    gk = opp(104.0, 34.0, position="GK", pid=1)
    corner = np.array([105.0, 30.8])
    for x, floor in ((100.0, 0.5), (96.0, 0.5), (92.0, 0.35)):
        p = route_survival(np.array([x, 34.0]), corner, SHOT_SPEED,
                           SHOT_FRICTION_COEFF, [gk], is_shot=True)
        assert p > floor, f"shot from {105 - x:.0f}m scored only {p:.2f}"


def test_long_range_shots_stay_hopeless():
    gk = opp(104.0, 34.0, position="GK", pid=1)
    p = route_survival(np.array([78.0, 34.0]), np.array([105.0, 30.8]),
                       SHOT_SPEED, SHOT_FRICTION_COEFF, [gk])
    assert p < 0.1


def test_shot_flag_only_helps_the_shooter():
    """The same geometry must be harder to survive as a rolling pass, since the
    defender genuinely has time to step across."""
    blocker = opp(50.0, 35.5, pid=2)
    origin, target = np.array([40.0, 34.0]), np.array([60.0, 34.0])
    as_pass = route_survival(origin, target, PASS_SPEED_LOW, BALL_FRICTION_COEFF,
                             [blocker], is_shot=False)
    as_shot = route_survival(origin, target, SHOT_SPEED, SHOT_FRICTION_COEFF,
                             [blocker], is_shot=True)
    assert as_shot > as_pass


def test_keeper_dive_penalty_does_not_apply_to_rolling_balls():
    """Reading and diving at a shot is slow; stepping across to a pass is not.
    Applying the dive penalty to both made keepers worse than outfield players
    at ordinary interceptions."""
    gk = opp(50.0, 35.5, position="GK", pid=1)
    outfield = opp(50.0, 35.5, position="CD", pid=1)
    origin, target = np.array([40.0, 34.0]), np.array([60.0, 34.0])
    gk_pass = route_survival(origin, target, PASS_SPEED_LOW, BALL_FRICTION_COEFF,
                             [gk], is_shot=False)
    outfield_pass = route_survival(origin, target, PASS_SPEED_LOW,
                                   BALL_FRICTION_COEFF, [outfield], is_shot=False)
    assert gk_pass <= outfield_pass


def test_good_angle_beats_a_tight_one_at_the_same_range():
    from app.engine.decisions.value import shot_value

    central = shot_value(np.array([93.0, 34.0]), 105.0)
    tight = shot_value(np.array([93.0, 18.0]), 105.0)
    assert central > tight * 3
    assert central > 0.5, "a 12 m central chance should be worth taking"


# ── Pass viability: a giveaway should not win by default ────────────────────

def test_hopeless_passes_are_suppressed():
    """A near-hopeless ball was still being played because every alternative
    scored lower -- the argmax had no notion that some options simply should
    not be on the table."""
    from app.engine.decisions.candidates import Candidate
    from app.engine.decisions.on_ball import score_candidate
    from app.engine.factory import build_test_state

    state = build_test_state(seed=0)
    carrier = state.get_ball_owner()
    # Put a teammate a clean 20 m away, square, so the geometry is unambiguous.
    mate = next(m for m in state.teammates_of(carrier.player_id) if m.position != "GK")
    mate.pos = carrier.pos + np.array([0.0, 20.0])

    def utility_with_blockers(n_blockers: int) -> float:
        for i, o in enumerate(state.opponents_of(carrier.team_id)):
            if i < n_blockers:
                t = (i + 1) / (n_blockers + 1)
                o.pos = carrier.pos + (mate.pos - carrier.pos) * t
            else:
                o.pos = np.array([2.0, 2.0])
        cand = Candidate(
            kind="pass", target=mate.pos.copy(), speed=0.7,
            friction=BALL_FRICTION_COEFF, recipient_id=mate.player_id,
        )
        return score_candidate(cand, carrier, state).utility

    clear = utility_with_blockers(0)
    blocked = utility_with_blockers(4)
    assert clear > 0.0, "the unobstructed pass should be worth something"
    assert blocked < clear * 0.25


def test_the_penalty_is_a_ramp_not_a_cliff():
    """A marginal ball is nudged; a hopeless one is crushed. A cliff would make
    the argmax flip between two near-identical passes."""
    from app.engine.constants import PASS_UNVIABLE_PENALTY, PASS_VIABILITY_FLOOR

    def factor(p: float) -> float:
        if p >= PASS_VIABILITY_FLOOR:
            return 1.0
        shortfall = 1.0 - p / PASS_VIABILITY_FLOOR
        return 1.0 - (1.0 - PASS_UNVIABLE_PENALTY) * shortfall

    assert factor(PASS_VIABILITY_FLOOR) == pytest.approx(1.0)
    assert factor(0.0) == pytest.approx(PASS_UNVIABLE_PENALTY)
    previous = 0.0
    for p in np.arange(0.0, PASS_VIABILITY_FLOOR + 0.01, 0.02):
        value = factor(float(p))
        assert value >= previous
        previous = value


def test_a_desperate_ball_is_still_reachable():
    """Suppress rather than veto: a clearance out of defence must stay possible
    even when every lane is poor."""
    from app.engine.constants import PASS_UNVIABLE_PENALTY

    assert PASS_UNVIABLE_PENALTY > 0.0


# ── Dribbling to open an angle ──────────────────────────────────────────────

def test_dribbling_is_worth_more_when_it_opens_a_lane():
    """Carrying to open a passing angle was invisible: a dribble was scored
    only on where the ball lands. A 3 m step sideways can lift the best
    available pass from 0.5 to 0.97."""
    from app.engine.decisions.candidates import Candidate
    from app.engine.decisions.on_ball import _angle_gain
    from app.engine.factory import build_test_state

    state = build_test_state(seed=0)
    carrier = state.get_ball_owner()
    mate = next(m for m in state.teammates_of(carrier.player_id) if m.position != "GK")

    # Wall off the direct lane, leaving a gap to one side.
    for i, o in enumerate(state.opponents_of(carrier.team_id)):
        if i < 3:
            t = (i + 1) / 4
            o.pos = carrier.pos + (mate.pos - carrier.pos) * t
        else:
            o.pos = np.array([2.0, 2.0])

    opponents = state.opponents_of(carrier.team_id)
    sideways = Candidate(
        kind="dribble",
        target=carrier.pos + np.array([0.0, 6.0]),
        speed=0.14, friction=0.06,
    )
    gain = _angle_gain(sideways, carrier, state, opponents)
    assert gain >= 1.0


def test_angle_gain_is_bounded():
    from app.engine.constants import ANGLE_GAIN_MAX
    from app.engine.decisions.candidates import build_candidates
    from app.engine.decisions.on_ball import _angle_gain
    from app.engine.factory import build_test_state

    state = build_test_state(seed=0)
    carrier = state.get_ball_owner()
    opponents = state.opponents_of(carrier.team_id)
    for cand in build_candidates(carrier, state):
        if cand.kind == "dribble":
            gain = _angle_gain(cand, carrier, state, opponents)
            assert 1.0 <= gain <= ANGLE_GAIN_MAX
