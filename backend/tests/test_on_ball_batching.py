"""
Speed-ups in the on-ball decision must not change a single score: the batched
pass lanes (_batch_lanes) and the batched angle-gain lanes against the
one-at-a-time forms they replaced, under both movement models.
"""
import numpy as np
import pytest

from app.engine import kinematics
from app.engine.factory import build_test_state
import app.engine.game_loop as gl
from app.engine.decisions import on_ball as OB
from app.engine.decisions.candidates import build_candidates
from app.engine.decisions.interception import route_survival


@pytest.mark.parametrize("inertia", [False, True])
def test_batched_pass_lanes_change_no_score(monkeypatch, inertia):
    monkeypatch.setattr(kinematics, "INERTIA", inertia)
    state = build_test_state(seed=5)
    for _ in range(200):
        gl.tick(state)
    carrier = state.get_ball_owner() or state.home_team.players[6]
    batched = build_candidates(carrier, state)
    OB._batch_lanes(batched, carrier, state)
    opponents = state.opponents_of(carrier.team_id)
    checked = 0
    for c in batched:
        if c.lane is not None:
            assert c.lane == pytest.approx(
                route_survival(carrier.pos, OB._lane_end(c, carrier, state), c.speed, c.friction, opponents),
                abs=1e-9)
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("inertia", [False, True])
def test_batched_angle_gain_matches_lane_by_lane(monkeypatch, inertia):
    from app.engine.ball_physics import nominal_pass_speed
    from app.engine.constants import (ANGLE_GAIN_MAX, ANGLE_GAIN_NEED_BELOW, ANGLE_GAIN_WEIGHT,
                                      BALL_FRICTION_COEFF, MIN_PASS_DIST)
    from app.engine.geometry import distance

    monkeypatch.setattr(kinematics, "INERTIA", inertia)
    state = build_test_state(seed=6)
    for _ in range(200):
        gl.tick(state)
    player = state.home_team.players[6]
    opponents = state.opponents_of(player.team_id)
    mates = [m for m in state.teammates_of(player.player_id)
             if m.position != "GK" and distance(m.pos, player.pos) >= MIN_PASS_DIST]

    def best(origin):
        return max(route_survival(origin, m.pos,
                                  nominal_pass_speed(BALL_FRICTION_COEFF, distance(origin, m.pos)),
                                  BALL_FRICTION_COEFF, opponents) for m in mates)

    for c in build_candidates(player, state):
        if c.kind != "dribble":
            continue
        here, there = best(player.pos), best(c.target)
        expect = 1.0 if here >= ANGLE_GAIN_NEED_BELOW else min(
            ANGLE_GAIN_MAX, 1.0 + ANGLE_GAIN_WEIGHT * max(0.0, there - here))
        assert OB._angle_gain(c, player, state, opponents) == pytest.approx(expect, abs=1e-9)
