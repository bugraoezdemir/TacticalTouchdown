"""
Possession as a gradient (on_ball.POSSESSION_GRADUAL): every side counts the
cost of losing the ball, Possession x POSSESSION_SCALE times
(1 - P) x (0.10 + 0.15 x danger where it is lost). Before, only (Possession - 1)
above 1.0, so 0.5 to 1.0 all played like 1.0.
"""
import numpy as np
import pytest

from app.engine.decisions import on_ball
from app.engine.decisions.candidates import Candidate
from app.engine.decisions.on_ball import _turnover_cost
from app.engine.factory import build_test_state


def _cost(possession, gradual=True, monkeypatch=None):
    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.possession = possession
    player = next(p for p in team.players if p.position == "CM")
    c = Candidate(kind="pass", target=np.array([30.0, 34.0]), speed=0.5, friction=0.01)
    c.p = 0.7
    return _turnover_cost(c, player, team)


def test_the_cost_grows_in_proportion_to_the_setting(monkeypatch):
    monkeypatch.setattr(on_ball, "POSSESSION_GRADUAL", True)
    costs = [_cost(s) for s in (0.5, 1.0, 1.5, 2.0)]
    assert all(c > 0.0 for c in costs)
    assert costs == pytest.approx([costs[0] * k for k in (1, 2, 3, 4)])


def test_at_two_it_is_the_old_cost(monkeypatch):
    monkeypatch.setattr(on_ball, "POSSESSION_GRADUAL", True)
    new = _cost(2.0)
    monkeypatch.setattr(on_ball, "POSSESSION_GRADUAL", False)
    assert new == pytest.approx(_cost(2.0))
    assert _cost(0.5) == 0.0 and _cost(1.0) == 0.0     # the old cliff
