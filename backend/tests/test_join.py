"""
Attack call: the carrier judges his teammates' runs and sends the best
(decisions D-13, D-14; signals.JOIN).

His own setting sets how many, int(2 x Attack call + 0.5). Each teammate's
runs (a few directions and lengths) must be ours to reach first, and are
scored by the worth of the spot, discounted by the time to get there and by
the cover the runner would leave behind -- strongly at 0.5, mildly at 2.0.
"""
import numpy as np
import pytest

from app.engine import signals
from app.engine.decisions import playmaking
from app.engine.decisions.playmaking import _choose_joiners, _risk, _run_spots
from app.engine.factory import build_test_state
from app.engine.signals import AttackCall, expire_attack_calls, set_attack_runs


@pytest.fixture
def joined(monkeypatch):
    monkeypatch.setattr(signals, "JOIN", True)


def _attack(setting=1.0):
    """Home has the ball at the halfway line, attacking toward x = 105."""
    state = build_test_state(seed=1)
    team = state.home_team
    team.tactics.attack_call = setting
    carrier = next(p for p in team.players if p.position == "CM")
    for p in state.all_players():
        p.has_ball = False; p.velocity = np.zeros(2); p.attack_call = None
    carrier.pos = np.array([52.0, 34.0]); carrier.has_ball = True
    state.ball.pos = carrier.pos.copy(); state.ball.owner_id = carrier.player_id
    return state, team, carrier


def _runs(state, carrier):
    plan = _choose_joiners(carrier, state)
    assert plan.runs is not None
    return plan.runs


@pytest.mark.parametrize("setting, wanted", [(0.5, 1), (1.0, 2), (1.5, 3), (2.0, 4)])
def test_his_setting_caps_how_many_he_sends(joined, setting, wanted):
    state, team, carrier = _attack(setting)
    assert len(_runs(state, carrier)) <= wanted


def test_the_carriers_own_setting_decides(joined):
    """A carrier set to 2.0 in a 0.5 team sends what a 2.0 team's carrier
    sends; the team's own value no longer counts for him."""
    state, team, carrier = _attack(2.0)
    as_team = [pid for pid, _ in _runs(state, carrier)]
    team.tactics.attack_call = 0.5
    carrier.bias_attack_call = 2.0
    assert [pid for pid, _ in _runs(state, carrier)] == as_team


def test_runs_are_judged_only_every_few_ticks(joined):
    state, team, carrier = _attack(1.0)
    state.tick = 1
    assert _choose_joiners(carrier, state).runs is None


def test_run_spots_go_forward_onside_and_on_the_pitch(joined):
    state, team, carrier = _attack(1.0)
    from app.engine.offside import onside_limit
    line = onside_limit(state, team)
    mate = next(m for m in team.players if m.position == "LM")
    spots = _run_spots(mate, state, team)
    assert len(spots) >= 5
    for s in spots:
        assert s[0] >= mate.pos[0] + 2.0 - 1e-9 and s[0] <= line + 1e-9
        assert 3.0 <= s[1] <= 65.0


def test_a_spot_an_opponent_reaches_first_is_not_offered(joined, monkeypatch):
    """The run must be ours to win, as for a called run."""
    state, team, carrier = _attack(2.0)
    mate = next(m for m in team.players if m.position == "LM")
    spot = mate.pos + np.array([10.0, 0.0])
    monkeypatch.setattr(playmaking, "_run_spots", lambda m, s, t: [spot.copy()] if m is mate else [])
    for o in state.away_team.players:
        if o.position != "GK":
            o.pos = np.array([90.0, 5.0 + 6.0 * (o.player_id % 10)])
    assert [pid for pid, _ in _runs(state, carrier)] == [mate.player_id]
    defender = next(o for o in state.away_team.players if o.position != "GK")
    defender.pos = spot.copy()                  # standing on it
    assert _runs(state, carrier) == ()


def test_the_deepest_man_is_the_most_critical(joined):
    state, team, carrier = _attack(1.0)
    outfield = [m for m in team.players if m.position != "GK" and m is not carrier]
    staying = {m.player_id for m in outfield}
    deepest = min(outfield, key=lambda m: m.pos[0])
    forward = max(outfield, key=lambda m: m.pos[0])
    deepest.pos = np.array([12.0, 34.0])
    assert _risk(deepest, staying, state, team) == pytest.approx(1.0)
    assert _risk(forward, staying, state, team) == 0.0     # four or more behind him


def test_the_risk_penalty_is_strong_at_low_settings_and_mild_at_high():
    """(1 - risk) ** (2 / setting): a half-critical man keeps 6% of his score
    at 0.5, 25% at 1.0 and 50% at 2.0."""
    from app.engine.constants import JOIN_RISK_EXPONENT
    keep = [(1.0 - 0.5) ** (JOIN_RISK_EXPONENT / s) for s in (0.5, 1.0, 2.0)]
    assert keep == pytest.approx([0.0625, 0.25, 0.5])


def test_the_carrier_drops_runs_he_no_longer_wants_and_moves_kept_ones(joined):
    state, team, carrier = _attack(1.0)
    a, b = [m for m in team.players if m.position in ("LM", "RM")]
    a.attack_call = AttackCall(carrier.player_id, team.team_id, 0, spot=np.array([70.0, 10.0]))
    b.attack_call = AttackCall(carrier.player_id, team.team_id, 0, spot=np.array([70.0, 58.0]))
    set_attack_runs(state, carrier.player_id, ((a.player_id, np.array([75.0, 20.0])),))
    assert b.attack_call is None
    assert np.allclose(a.attack_call.spot, [75.0, 20.0]) and a.attack_call.arrived_tick is None


def test_the_run_ends_when_he_waits_on_the_spot_without_the_ball(joined):
    """The attack has stalled: he has been on his spot 2 s and the ball has
    not come."""
    state, team, carrier = _attack(1.0)
    runner = next(m for m in team.players if m.position[1] == "M" and m is not carrier)
    runner.attack_call = AttackCall(carrier.player_id, team.team_id, state.tick,
                                    spot=np.array([62.0, float(runner.pos[1])]))
    runner.pos = signals.join_spot_now(runner.attack_call, state)
    expire_attack_calls(state)
    assert runner.attack_call is not None and runner.attack_call.arrived_tick == state.tick
    state.tick += 61
    expire_attack_calls(state)
    assert runner.attack_call is None and runner.join_cooldown_until > state.tick


def test_the_run_ends_when_he_cannot_make_it(joined):
    """A man who stops closing on his spot for 2 s gives his place up."""
    state, team, carrier = _attack(1.0)
    runner = next(m for m in team.players if m.position in ("LD", "RD"))
    runner.pos = np.array([20.0, float(runner.pos[1])])
    runner.attack_call = AttackCall(carrier.player_id, team.team_id, state.tick,
                                    spot=np.array([70.0, float(runner.pos[1])]))
    expire_attack_calls(state)
    state.tick += 30
    expire_attack_calls(state)
    assert runner.attack_call is not None
    state.tick += 31
    expire_attack_calls(state)
    assert runner.attack_call is None and runner.join_cooldown_until > state.tick
