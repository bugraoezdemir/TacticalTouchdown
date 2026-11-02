"""
The attack call: the carrier signals teammates forward, and the run lasts
until the move does.

The signal is a decision, not a mode -- whoever has the ball chooses whether to
call, so there is no global "we are attacking" state to keep in sync.
"""
import numpy as np

from app.engine.constants import ATTACK_CALL_MAX_TICKS
from app.engine.factory import build_test_state
from app.engine.game_loop import tick
from app.engine.signals import AttackCall, emit_attack_call, expire_attack_calls


def call_on(state, player):
    emit_attack_call(state, state.get_ball_owner().player_id, player.player_id)


def test_a_call_ends_when_the_opponent_takes_the_ball():
    """The bug the user reported: "players move when the ball was already lost
    to the opponent and this puts the team unnecessarily into the threat".

    The expiry originally keyed on ball.last_toucher_team_id, which is assigned
    only when a player *kicks* the ball and never when someone *collects* one.
    After an interception it still named the team that played the pass, so runs
    continued for up to 91 ticks with the opponent in possession.
    """
    state = build_test_state(seed=0)
    carrier = state.get_ball_owner()
    mate = next(
        p for p in state.teammates_of(carrier.player_id) if p.position != "GK"
    )
    call_on(state, mate)
    assert mate.attack_call is not None

    # An opponent collects it. last_toucher still names the attacking team,
    # exactly as it does after a real interception.
    opponent = next(iter(state.opponents_of(carrier.team_id)))
    state.ball.owner_id = opponent.player_id
    state.ball.last_toucher_team_id = carrier.team_id

    expire_attack_calls(state)
    assert mate.attack_call is None


def test_a_call_survives_our_own_pass_in_flight():
    """The other half: possession means the MOVE is ours, not that someone is
    standing on the ball. The ball is loose for most of a spell, so expiring on
    a loose ball would end every run at the first pass."""
    state = build_test_state(seed=0)
    carrier = state.get_ball_owner()
    mate = next(
        p for p in state.teammates_of(carrier.player_id) if p.position != "GK"
    )
    call_on(state, mate)

    state.ball.owner_id = None                      # in flight
    state.ball.last_toucher_team_id = carrier.team_id

    expire_attack_calls(state)
    assert mate.attack_call is not None


def test_a_call_expires_on_the_safety_cap():
    """Possession loss is what ends runs in practice -- measured spells are
    median 43 ticks and at most 282 -- but the cap stops a player being
    stranded if possession never resolves cleanly."""
    state = build_test_state(seed=0)
    mate = next(p for p in state.all_players() if p.position != "GK")
    mate.attack_call = AttackCall(
        sender_id=mate.player_id, team_id=mate.team_id, emitted_tick=0
    )
    state.tick = int(ATTACK_CALL_MAX_TICKS) + 1
    state.ball.owner_id = None
    state.ball.last_toucher_team_id = mate.team_id

    expire_attack_calls(state)
    assert mate.attack_call is None


def test_no_decision_ever_sees_a_call_the_opponent_has_invalidated():
    """The invariant that actually matters.

    Expiry runs at the top of COMPUTE, so this asserts the state as decisions
    see it. Sampling at end-of-tick instead is misleading: possession resolves
    mid-tick, after that tick's expiry, so a call looks briefly stale in a
    window where no decision is ever taken.
    """
    state = build_test_state(seed=0)
    for _ in range(3000):
        state = tick(state)
        owner = state.get_ball_owner()
        holder = (
            owner.team_id
            if owner is not None
            else state.ball.last_toucher_team_id
        )
        for player in state.all_players():
            call = player.attack_call
            if call is None:
                continue
            survives = holder is None or holder == call.team_id
            if survives and owner is not None:
                assert owner.team_id == player.team_id, (
                    f"tick {state.tick}: {player.position} would act on an "
                    f"attack call while team {owner.team_id} has the ball"
                )
