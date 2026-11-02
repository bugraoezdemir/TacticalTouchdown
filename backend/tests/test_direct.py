"""
Direct pass (on_ball.direct_factor): a pass is scaled by how far the receiver
must run to it, Direct ** (1 - 2 x min(1, run / 20 m)). It replaces Runway, a
flat tilt by kind of pass that flipped every near-tie between a ball to feet and
a ball just ahead of a moving teammate.
"""
import numpy as np
import pytest

from app.engine.decisions.candidates import Candidate
from app.engine.decisions.on_ball import _receiver_run, direct_factor, score_candidate
from app.engine.factory import build_test_state


def test_neutral_leaves_every_pass_alone():
    for run in (0.0, 5.0, 10.0, 20.0, 40.0):
        assert direct_factor(run, 1.0) == 1.0


def test_it_is_graded_by_the_run():
    factors = [direct_factor(run, 2.0) for run in (0.0, 5.0, 10.0, 15.0, 20.0)]
    assert factors == pytest.approx([2.0, 2.0 ** 0.5, 1.0, 2.0 ** -0.5, 0.5])


def test_below_one_favours_the_ball_he_must_chase():
    assert direct_factor(0.0, 0.5) == pytest.approx(0.5)
    assert direct_factor(20.0, 0.5) == pytest.approx(2.0)


def test_nearly_equal_passes_get_nearly_equal_factors():
    """The reason for the grading: a ball to his feet and one 2 m ahead of him
    are nearly the same pass, so the slider must not split them."""
    assert direct_factor(0.0, 2.0) / direct_factor(2.0, 2.0) < 1.15


def test_it_is_capped_at_the_full_run():
    assert direct_factor(35.0, 2.0) == direct_factor(20.0, 2.0)


def test_the_run_is_the_receivers_distance_to_the_target():
    state = build_test_state(seed=1)
    team = state.home_team
    passer, mate = [p for p in team.players if p.position == "CM"]
    c = Candidate(kind="pass", target=mate.pos + np.array([3.0, 4.0]), speed=0.5, friction=0.01,
                  recipient_id=mate.player_id)
    assert _receiver_run(c, passer, state) == pytest.approx(5.0)


def test_it_acts_in_which_pass_not_whether(monkeypatch):
    """In the pass's flavour, so the plain utility that decides whether to pass
    does not change with the slider."""
    state = build_test_state(seed=1)
    team = state.home_team
    passer, mate = [p for p in team.players if p.position == "CM"]

    def plain(direct):
        team.tactics.direct = direct
        c = Candidate(kind="pass", target=mate.pos.copy(), speed=0.5, friction=0.01,
                      recipient_id=mate.player_id)
        return score_candidate(c, passer, state).plain

    assert plain(2.0) == pytest.approx(plain(1.0))
