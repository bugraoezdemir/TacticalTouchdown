"""
The interception primitive is the P in every P x V x C decision, so its
monotonicity properties matter more than any single fixture value.
"""
import numpy as np
import pytest

from app.engine.constants import (
    BALL_FRICTION_COEFF,
    PASS_SPEED_LOW,
    PLAYER_BASE_SPEED,
    REACTION_TICKS,
)
from app.engine.decisions.interception import opponent_risk, route_survival
from app.engine.models import Player


def make_player(x, y, vx=0.0, vy=0.0, speed=PLAYER_BASE_SPEED, position="CD", pid=1):
    return Player(
        player_id=pid, team_id=2, name=f"P{pid}", position=position, tactical="normal",
        pos=np.array([x, y], dtype=float), velocity=np.array([vx, vy], dtype=float),
        speed=speed, strength=70.0, intelligence=70.0, passing=70.0, goalkeeping=40.0,
        assigned_pos=np.array([x, y], dtype=float), marking_target_id=None,
        has_ball=False, runway_signal=None,
    )


ORIGIN = np.array([20.0, 34.0])
TARGET = np.array([40.0, 34.0])


def survival(blockers):
    return route_survival(ORIGIN, TARGET, PASS_SPEED_LOW, BALL_FRICTION_COEFF, blockers)


def test_empty_pitch_is_certain():
    assert survival([]) == 1.0


def test_opponent_far_off_lane_is_harmless():
    assert survival([make_player(30.0, 34.0 + 40.0)]) == pytest.approx(1.0, abs=1e-3)


def test_opponent_on_the_target_is_deadly():
    """The old infinite-line filter dropped this player entirely (risk 0) because
    their projection landed exactly at the segment end. They are in fact the most
    dangerous defender on the pitch.

    """
    assert survival([make_player(*TARGET)]) < 0.2


def test_opponent_standing_in_the_lane_is_deadly():
    assert survival([make_player(30.0, 34.0)]) < 0.2


def test_opponent_behind_the_passer_is_harmless():
    assert survival([make_player(5.0, 34.0)]) == pytest.approx(1.0, abs=1e-3)


def test_survival_decreases_as_opponent_nears_the_lane():
    prev = -1.0
    for offset in (30.0, 20.0, 12.0, 8.0, 4.0, 1.0):
        s = survival([make_player(30.0, 34.0 + offset)])
        assert s <= prev or prev < 0, f"not monotonic at offset {offset}"
        prev = s if prev < 0 else min(prev, s)
    # Closest case must be much worse than the farthest.
    assert survival([make_player(30.0, 35.0)]) < survival([make_player(30.0, 64.0)])


def test_survival_decreases_with_opponent_speed():
    slow = survival([make_player(30.0, 42.0, speed=0.15)])
    fast = survival([make_player(30.0, 42.0, speed=0.60)])
    assert fast < slow


def test_survival_increases_with_ball_speed():
    blockers = [make_player(30.0, 40.0)]
    slow = route_survival(ORIGIN, TARGET, 0.5, BALL_FRICTION_COEFF, blockers)
    fast = route_survival(ORIGIN, TARGET, 1.2, BALL_FRICTION_COEFF, blockers)
    assert fast > slow


def test_velocity_toward_lane_is_more_dangerous():
    """Player.velocity was populated but completely unused by the old scorer."""
    toward = survival([make_player(30.0, 42.0, vx=0.0, vy=-0.30)])
    away = survival([make_player(30.0, 42.0, vx=0.0, vy=+0.30)])
    assert toward < away


def test_reaction_time_helps_the_passer():
    opp = make_player(30.0, 40.0)
    instant = opponent_risk(opp, ORIGIN, TARGET, PASS_SPEED_LOW, BALL_FRICTION_COEFF,
                            reaction_ticks=0.0)
    delayed = opponent_risk(opp, ORIGIN, TARGET, PASS_SPEED_LOW, BALL_FRICTION_COEFF,
                            reaction_ticks=REACTION_TICKS)
    assert delayed < instant


def test_more_blockers_never_helps():
    one = [make_player(30.0, 38.0, pid=1)]
    two = one + [make_player(34.0, 30.0, pid=2)]
    assert survival(two) <= survival(one)


def test_exclude_ids_skips_blockers():
    opp = make_player(30.0, 34.0, pid=7)
    assert route_survival(ORIGIN, TARGET, PASS_SPEED_LOW, BALL_FRICTION_COEFF,
                          [opp], exclude_ids=[7]) == 1.0


def test_gk_has_longer_reach():
    """Keepers use their hands, so they cover slightly more ground than an
    outfield player standing in the same place. The margin is deliberately
    small: an over-long reach let a keeper cover a corner from the centre of
    his goal and scored every shot as saved."""
    from app.engine.decisions.interception import control_radius

    gk = make_player(30.0, 38.0, position="GK", pid=1)
    outfield = make_player(30.0, 38.0, position="CD", pid=1)
    assert control_radius(gk) > control_radius(outfield)
    assert survival([gk]) <= survival([outfield])


def test_always_bounded_randomised():
    rng = np.random.default_rng(20260903)
    for _ in range(3000):
        blockers = [
            make_player(rng.uniform(0, 105), rng.uniform(0, 68),
                        vx=rng.uniform(-0.4, 0.4), vy=rng.uniform(-0.4, 0.4),
                        speed=rng.uniform(0.05, 0.6), pid=i)
            for i in range(rng.integers(0, 6))
        ]
        origin = np.array([rng.uniform(0, 105), rng.uniform(0, 68)])
        target = np.array([rng.uniform(0, 105), rng.uniform(0, 68)])
        s = route_survival(origin, target, rng.uniform(0.2, 1.2),
                           BALL_FRICTION_COEFF, blockers)
        assert 0.0 <= s <= 1.0 and np.isfinite(s)


def test_zero_length_route_is_safe():
    assert route_survival(ORIGIN, ORIGIN.copy(), PASS_SPEED_LOW,
                          BALL_FRICTION_COEFF, [make_player(20.0, 34.0)]) == 1.0


def test_momentum_cannot_carry_a_blocker_out_of_the_lane():
    """The "players pass right into opponents" bug.

    Risk is evaluated after displacing the opponent by velocity x reaction
    time. That is right for someone running TOWARD the lane, but it also let a
    defender already standing in it be projected clear: 76% of the defenders
    whose body genuinely blocked a chosen lane were moved far enough that
    BLOCKED_LANE_RISK stopped applying. A marker 0.08 m off the line, jogging
    across it, scored 0.26 risk instead of 0.90, and the ball was played
    straight into him.

    He is not obliviously vacating the lane -- he is a defender who will react
    to the pass. So momentum may only ever make him more dangerous.
    """
    on_the_line = make_player(30.0, 34.0, vx=-0.31, vy=-0.18)
    assert opponent_risk(on_the_line, ORIGIN, TARGET, PASS_SPEED_LOW,
                         BALL_FRICTION_COEFF) >= 0.85


def test_momentum_still_helps_a_defender_closing_on_the_lane():
    """The other half of the same rule: the displacement must still count when
    it brings someone toward the lane, or a defender sprinting into the line of
    a pass is scored as though he were standing still."""
    closing = make_player(30.0, 40.0, vx=0.0, vy=-0.35)
    static = make_player(30.0, 40.0)
    assert (opponent_risk(closing, ORIGIN, TARGET, PASS_SPEED_LOW, BALL_FRICTION_COEFF)
            > opponent_risk(static, ORIGIN, TARGET, PASS_SPEED_LOW, BALL_FRICTION_COEFF))
