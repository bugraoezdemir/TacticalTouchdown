"""
Who steps out to the carrier (off_ball._press_set): decided for the whole
team from what each defender would leave behind, not by who is nearest.

The switch (COVER_PRESS) is off by default; these test the selection itself.
"""
import numpy as np

from app.engine.factory import build_test_state
from app.engine.decisions import off_ball as OB


def _scene(defender_spots, attacker_spots, carrier_spot):
    """Home defends the goal at its own end; away attacks it. Everyone not
    placed is parked far upfield, out of the picture."""
    state = build_test_state()
    home, away = state.home_team, state.away_team
    gx = 0.0 if home.direction == 1 else 105.0
    flip = lambda x, y: np.array([abs(gx - x), y])
    for p in state.all_players():
        p.has_ball = False
        p.velocity = np.zeros(2)
    outfield_home = [p for p in home.players if p.position != "GK"]
    outfield_away = [p for p in away.players if p.position != "GK"]
    for p in outfield_home + outfield_away:
        p.pos = flip(95.0, 5.0 + 3.0 * (p.player_id % 20))
    named = {}
    for name, (x, y) in defender_spots.items():
        p = outfield_home.pop(0)
        p.pos = flip(x, y)
        named[name] = p
    for name, (x, y) in attacker_spots.items():
        p = outfield_away.pop(0)
        p.pos = flip(x, y)
        named[name] = p
    carrier = outfield_away.pop(0)
    carrier.pos = flip(*carrier_spot)
    carrier.has_ball = True
    state.ball.pos = carrier.pos.copy()
    state.ball.owner_id = carrier.player_id
    OB._PRESS_SET.clear()
    return state, home, carrier, named, gx


def test_a_centre_back_on_the_striker_stays_and_the_free_man_goes():
    state, home, carrier, n, gx = _scene(
        {"cb": (10.0, 34.0), "mid": (22.0, 40.0)},
        {"striker": (9.0, 34.5)},
        (26.0, 36.0),
    )
    pressers = OB._press_set(state, home, carrier, gx)
    assert n["mid"].player_id in pressers
    assert n["cb"].player_id not in pressers


def test_with_two_on_the_striker_one_of_them_may_go():
    state, home, carrier, n, gx = _scene(
        {"cb1": (10.0, 33.0), "cb2": (10.0, 36.0)},
        {"striker": (9.0, 34.5)},
        (22.0, 35.0),
    )
    pressers = OB._press_set(state, home, carrier, gx)
    assert len(pressers & {n["cb1"].player_id, n["cb2"].player_id}) == 1


def test_somebody_always_presses_the_ball():
    state, home, carrier, n, gx = _scene(
        {"cb": (10.0, 34.0)},
        {"striker": (9.0, 34.5)},
        (30.0, 34.0),
    )
    assert len(OB._press_set(state, home, carrier, gx)) >= 1
