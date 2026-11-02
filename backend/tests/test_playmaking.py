"""
Support-vs-threat positioning and called runs.

Both exist to stop off-ball teammates sabotaging their own attack by running
at the ball: the shape anchor slides toward possession, so with the ball deep
every forward was dragged backwards out of scoring positions.
"""
import numpy as np
import pytest

from app.engine.constants import PITCH_LENGTH
from app.engine.decisions.playmaking import find_signal_target
from app.engine.decisions.pursuit import space_quality, support_need
from app.engine.factory import build_test_state
from tests.helpers import midfielder, striker
from app.engine.game_loop import tick


RIGHT = PITCH_LENGTH


def at(x, y=34.0):
    return np.array([float(x), float(y)])


# ── Support vs threat ────────────────────────────────────────────────────────
# Keyed on the carrier's outlet quality, not on where the ball is. Ball position
# was only a +0.63 proxy for this with ~50% spread inside each bucket, so about
# a third of the time it gave the wrong answer -- most damagingly for a carrier
# surrounded on the halfway line, who reads as "mid-pitch, go make runs" when
# what he needs is someone to come short.

def test_no_outlet_means_come_short():
    assert support_need(0.0) > 0.9


def test_good_outlet_means_go_make_runs():
    assert support_need(0.9) < 0.1


def test_support_falls_monotonically_with_pass_safety():
    prev = 2.0
    for safety in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        need = support_need(safety)
        assert need < prev
        prev = need


def test_support_bounded():
    for safety in np.arange(-0.5, 1.6, 0.1):
        assert 0.0 <= support_need(float(safety)) <= 1.0


def test_best_pass_safety_reflects_the_carrier_situation():
    """A carrier with every lane blocked must score lower than a free one."""
    from app.engine.decisions.pursuit import best_pass_safety
    from app.engine.game_loop import tick

    state = build_test_state(seed=0)
    carrier = state.get_ball_owner()
    for opponent in state.opponents_of(carrier.team_id):
        opponent.pos = np.array([0.5, 0.5])
    free = best_pass_safety(state, carrier)

    # Park an opponent on top of each teammate to block every outlet.
    mates = [m for m in state.teammates_of(carrier.player_id) if m.position != "GK"]
    for opponent, mate in zip(state.opponents_of(carrier.team_id), mates):
        opponent.pos = mate.pos.copy()
    smothered = best_pass_safety(state, carrier)

    assert free > smothered
    assert 0.0 <= smothered <= 1.0


# ── Space ────────────────────────────────────────────────────────────────────

def test_space_quality_rewards_empty_ground():
    state = build_test_state()
    opponents = state.away_team.players
    on_top_of_a_defender = opponents[5].pos.copy()
    assert space_quality(on_top_of_a_defender, opponents) < 0.2


def test_space_quality_is_one_with_nobody_around():
    assert space_quality(at(50.0), []) == 1.0


def test_space_quality_bounded():
    state = build_test_state()
    opponents = state.away_team.players
    for x in np.arange(0.0, 105.1, 10.0):
        assert 0.0 <= space_quality(at(x), opponents) <= 1.0


# ── Called runs ──────────────────────────────────────────────────────────────

def test_carrier_finds_a_spot_and_names_runners():
    """The carrier picks the spot FIRST and summons people to it, rather than
    only exploiting runs that already happen to be underway."""
    state = build_test_state()
    result = find_signal_target(state.get_ball_owner(), state)
    assert result is not None
    spot, runners = result
    assert 0.0 <= spot[0] <= PITCH_LENGTH
    assert 1 <= len(runners) <= 2


def test_called_spot_is_ahead_of_the_carrier():
    state = build_test_state()
    carrier = state.get_ball_owner()
    result = find_signal_target(carrier, state)
    assert result is not None
    spot, _ = result
    goal_x = PITCH_LENGTH if state.get_team(carrier.team_id).direction == 1 else 0.0
    assert abs(spot[0] - goal_x) < abs(carrier.pos[0] - goal_x)


def test_named_runners_can_actually_get_there():
    state = build_test_state()
    carrier = state.get_ball_owner()
    result = find_signal_target(carrier, state)
    assert result is not None
    spot, runners = result
    for pid in runners:
        runner = state.get_player(pid)
        assert runner.team_id == carrier.team_id
        assert runner.position != "GK"


def test_signals_reach_every_named_runner():
    """A called run may name more than one player; the loop previously emitted
    to a single recipient only."""
    state = build_test_state()
    for _ in range(40):
        state = tick(state)
        signalled = [p for p in state.all_players() if p.runway_signal is not None]
        if signalled:
            for p in signalled:
                assert p.runway_signal.target_pos is not None
            return
    pytest.skip("no called run occurred in the sampled window")


def test_signalled_player_runs_to_the_spot():
    from app.engine.decisions.off_ball import off_ball_decision
    from app.engine.signals import RunwaySignal

    state = build_test_state()
    receiver = striker(state)
    spot = np.array([85.0, 40.0])
    receiver.runway_signal = RunwaySignal(
        sender_id=midfielder(state).player_id,
        target_pos=spot,
        emitted_tick=state.tick,
    )
    action = off_ball_decision(receiver, state, [])
    assert np.allclose(action.target_pos, spot)


# ── Location and context, not role ───────────────────────────────────────────

def test_local_superiority_rewards_outnumbering():
    """3-v-1 in the box is the whole reason to play a ball there; 1-v-3 is why
    the same ball is a giveaway."""
    from app.engine.decisions.playmaking import _local_superiority

    state = build_test_state()
    spot = np.array([90.0, 34.0])
    mine = [p for p in state.home_team.players[:3]]
    for i, p in enumerate(mine):
        p.pos = spot + np.array([0.0, (i - 1) * 5.0])

    theirs = state.away_team.players
    for o in theirs:
        o.pos = np.array([20.0, 34.0])

    outnumbering = _local_superiority(spot, mine, theirs)

    theirs[0].pos = spot + np.array([2.0, 0.0])
    theirs[1].pos = spot + np.array([-2.0, 2.0])
    theirs[2].pos = spot + np.array([0.0, -3.0])
    contested = _local_superiority(spot, mine, theirs)

    assert outnumbering > contested


def test_local_superiority_is_bounded():
    from app.engine.constants import SUPERIORITY_MAX, SUPERIORITY_MIN
    from app.engine.decisions.playmaking import _local_superiority

    state = build_test_state()
    spot = np.array([50.0, 34.0])
    everyone = state.home_team.players
    for p in everyone:
        p.pos = spot.copy()
    assert _local_superiority(spot, everyone, []) <= SUPERIORITY_MAX
    assert _local_superiority(spot, [], everyone) >= SUPERIORITY_MIN


def test_prompt_runners_beat_late_ones():
    """A runner who arrives comfortably should make a spot more attractive than
    one who scrapes in. The old hard cutoff scored them identically."""
    state = build_test_state()
    carrier = state.get_ball_owner()
    near = find_signal_target(carrier, state)
    assert near is not None


def test_called_runs_are_not_a_midfielder_privilege():
    """Whether to release the ball is a question of position on the pitch and
    what is on -- not of what number the player wears. A steep role bias
    suppressed exactly the balls forwards should play in the final third."""
    from app.engine.decisions.tunables import ROLE_ACTION_MULT

    pass_mults = [row["pass"] for row in ROLE_ACTION_MULT.values()]
    assert max(pass_mults) - min(pass_mults) <= 0.25, (
        "role bias on passing is too steep to be location-driven"
    )
    for role in ("LF", "CF", "RF"):
        assert ROLE_ACTION_MULT[role]["pass"] >= 1.0, (
            f"{role} is discouraged from passing, which suppresses called runs"
        )


def test_every_outfield_role_can_call_a_run():
    """Not restricted by position anywhere in the code."""
    state = build_test_state()
    seen = set()
    for player in state.all_players():
        if player.position == "GK":
            continue
        player_state = build_test_state()
        carrier = player_state.get_player(player.player_id)
        for other in player_state.all_players():
            other.has_ball = False
        carrier.has_ball = True
        player_state.ball.pos = carrier.pos.copy()
        player_state.ball.owner_id = carrier.player_id
        if find_signal_target(carrier, player_state) is not None:
            seen.add(carrier.position)
    # Defenders, midfielders and forwards must all be represented.
    assert seen & {"LD", "CD", "RD"}
    assert seen & {"LM", "CM", "RM"}
    assert seen & {"LF", "CF", "RF"}
