"""
The Risk slider (on_ball._set_utility): P ** (1 / risk) for passes and dribbles.

utility = P x V x C is risk-neutral. Risk tilts the trade between a chancy
option and a safe one: above 1.0 the chancy one gains, below it the safe one.
Shots, and the trap of a first touch, are not affected.
"""
import numpy as np
import pytest

from app.engine.decisions.candidates import Candidate
from app.engine.decisions.on_ball import _set_utility
from app.engine.factory import build_test_state


@pytest.fixture(autouse=True)
def _no_turnover_cost(monkeypatch):
    """These pin the utility arithmetic, so the possession cost is kept out."""
    from app.engine.decisions import on_ball
    monkeypatch.setattr(on_ball, "POSSESSION_GRADUAL", False)


def _setup(risk=1.0, player_risk=None):
    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.risk = risk
    player = next(p for p in team.players if p.position == "CM")
    player.bias_risk = player_risk
    return player, team


def _option(kind, p, v, **kw):
    c = Candidate(kind=kind, target=np.array([60.0, 34.0]), speed=0.8, friction=0.01, **kw)
    c.p, c.v, c.context = p, v, 1.0
    return c


def _best(risk, player_risk=None):
    player, team = _setup(risk, player_risk)
    risky = _option("pass", 0.25, 1.0)    # into the box, a 1 in 4 chance
    safe = _option("pass", 0.9, 0.3)      # square, almost certain
    for c in (risky, safe):
        _set_utility(c, player, team)
    return "risky" if risky.utility > safe.utility else "safe"


def test_neutral_is_plain_expected_value():
    player, team = _setup(1.0)
    c = _option("pass", 0.25, 1.0)
    _set_utility(c, player, team)
    assert c.utility == 0.25 * 1.0 * 1.0


def test_a_bold_player_takes_the_chancy_ball_and_a_cautious_one_the_safe_one():
    assert _best(1.0) == "safe"           # 0.25 against 0.27
    assert _best(2.0) == "risky"          # 0.50 against 0.285
    assert _best(0.5) == "safe"


def test_the_players_own_setting_wins_over_the_team():
    assert _best(1.0, player_risk=2.0) == "risky"


@pytest.mark.parametrize("risk", [0.5, 2.0])
def test_dribbles_are_weighed_too(risk):
    player, team = _setup(risk)
    c = _option("dribble", 0.25, 1.0)
    _set_utility(c, player, team)
    assert c.utility == pytest.approx(0.25 ** (1.0 / risk))


@pytest.mark.parametrize("risk", [0.5, 2.0])
def test_shots_and_the_trap_are_not(risk):
    player, team = _setup(risk)
    shot = _option("shot", 0.25, 1.0)
    trap = _option("pass", 0.25, 1.0)
    _set_utility(shot, player, team)
    _set_utility(trap, player, team, risky=False)
    assert shot.utility == 0.25 and trap.utility == 0.25
