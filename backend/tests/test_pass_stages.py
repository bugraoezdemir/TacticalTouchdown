"""
Passing in two steps (on_ball.PASS_STAGES): whether to pass is decided on the
pass's utility WITHOUT the pass-kind tilts (Lofted, Direct, Pass range); which
pass, on the tilted utility. Before, a favoured kind's x2 lifted passing over
dribbling: at a flat tilt of 2.0 or 0.5 a side passed twice as often.
"""
import numpy as np
import pytest

from app.engine.decisions.candidates import Candidate
from app.engine.decisions.on_ball import _set_utility, _two_stage
from app.engine.factory import build_test_state


@pytest.fixture(autouse=True)
def _no_turnover_cost(monkeypatch):
    """These pin the utility arithmetic, so the possession cost is kept out."""
    from app.engine.decisions import on_ball
    monkeypatch.setattr(on_ball, "POSSESSION_GRADUAL", False)


def _setup(direct=1.0):
    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.direct = direct
    return next(p for p in team.players if p.position == "CM"), team


def _option(kind, p, v, flavour=1.0, **kw):
    c = Candidate(kind=kind, target=np.array([60.0, 34.0]), speed=0.5, friction=0.01, **kw)
    c.p, c.v, c.context, c.flavour = p, v, flavour, flavour
    return c


def test_a_tilted_pass_does_not_beat_a_better_dribble():
    player, team = _setup(2.0)
    into_space = _option("pass", 0.8, 0.4, flavour=2.0, is_runway=True)   # 0.64 tilted, 0.32 really
    dribble = _option("dribble", 0.9, 0.4)                                  # 0.36
    for c in (into_space, dribble):
        _set_utility(c, player, team)
    chosen, worth = _two_stage([into_space, dribble])
    assert chosen is dribble and worth == dribble.utility


def test_the_tilt_still_chooses_which_pass():
    player, team = _setup(2.0)
    to_feet = _option("pass", 0.9, 0.5, flavour=0.5)                        # 0.225 tilted, 0.45 really
    into_space = _option("pass", 0.8, 0.4, flavour=2.0, is_runway=True)   # 0.64 tilted, 0.32 really
    dribble = _option("dribble", 0.9, 0.3)                                  # 0.27
    for c in (to_feet, into_space, dribble):
        _set_utility(c, player, team)
    chosen, worth = _two_stage([to_feet, into_space, dribble])
    # The ball into space is the pass he would play; it is really worth 0.32,
    # which beats the dribble. The better ball to feet is not what he plays.
    assert chosen is into_space and abs(worth - 0.32) < 1e-12


def test_with_no_tilt_it_is_one_argmax():
    player, team = _setup(1.0)
    options = [_option("pass", 0.7, 0.5), _option("dribble", 0.9, 0.4), _option("shot", 0.3, 0.9)]
    for c in options:
        _set_utility(c, player, team)
    assert _two_stage(options)[0] is max(options, key=lambda c: c.utility)


def test_a_cross_is_its_own_group():
    """The cross he would play is not hidden by a pass whose tilt is higher:
    each group's tilted best goes on to compete at its plain utility."""
    player, team = _setup(2.0)
    into_space = _option("pass", 0.8, 0.4, flavour=2.0, is_runway=True)   # 0.64 tilted, 0.32 really
    cross = _option("pass", 0.6, 0.6, is_cross=True)                       # 0.36
    for c in (into_space, cross):
        _set_utility(c, player, team)
    assert _two_stage([into_space, cross])[0] is cross
