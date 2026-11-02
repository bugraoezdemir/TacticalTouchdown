"""
Team shapes.

Formations drive `assigned_pos`, which anchors the off-ball shape leash, so a
broken formation is not a cosmetic problem -- it relocates the whole team.

Position strings repeat within a formation (two "CD", two "CF"). That is safe
only because every position-keyed lookup is a `.get(position, default)` or a
first-match scan; the vocabulary test below is what keeps it safe.
"""
import numpy as np
import pytest

from app.engine.constants import PITCH_LENGTH, PITCH_WIDTH, ROLE_LEASH
from app.engine.decisions.tunables import ROLE_ACTION_MULT
from app.engine.factory import build_test_state
from app.engine.formations import (
    DEFAULT_FORMATION,
    FORMATIONS,
    SQUAD_SIZE,
    apply_formation,
    slot_position,
)
from app.engine.game_loop import tick
from app.engine.models import Position

ALL = sorted(FORMATIONS)
TOUCHLINE_MARGIN = 2.0


# -- Structure ---------------------------------------------------------------

@pytest.mark.parametrize("name", ALL)
def test_eleven_slots(name):
    assert len(FORMATIONS[name]) == SQUAD_SIZE


@pytest.mark.parametrize("name", ALL)
def test_exactly_one_goalkeeper(name):
    assert sum(1 for pos, _, _ in FORMATIONS[name] if pos == "GK") == 1


@pytest.mark.parametrize("name", ALL)
def test_goalkeeper_is_the_deepest(name):
    slots = FORMATIONS[name]
    gk = next(x for pos, x, _ in slots if pos == "GK")
    assert gk == min(x for _, x, _ in slots)


@pytest.mark.parametrize("name", ALL)
def test_slots_are_inside_the_pitch(name):
    for pos, x, y in FORMATIONS[name]:
        assert TOUCHLINE_MARGIN <= x <= PITCH_LENGTH - TOUCHLINE_MARGIN, pos
        assert TOUCHLINE_MARGIN <= y <= PITCH_WIDTH - TOUCHLINE_MARGIN, pos


@pytest.mark.parametrize("name", ALL)
def test_no_two_players_share_a_point(name):
    points = [(x, y) for _, x, y in FORMATIONS[name]]
    assert len(set(points)) == len(points)


def test_default_formation_exists():
    assert DEFAULT_FORMATION in FORMATIONS


# -- Vocabulary: the guardrail that makes duplicate positions safe -----------

def test_every_slot_position_is_a_known_role():
    """A position missing from ROLE_LEASH falls through to a LONGER default
    (15.0 vs a centre-back 12.0), silently unleashing defenders with no error.
    Likewise a missing ROLE_ACTION_MULT row silently neutralises role bias."""
    used = {pos for slots in FORMATIONS.values() for pos, _, _ in slots}
    assert used <= set(Position.__args__)
    assert used <= set(ROLE_LEASH)
    assert used <= set(ROLE_ACTION_MULT)


def test_every_declared_position_is_tuned():
    declared = set(Position.__args__)
    assert declared <= set(ROLE_LEASH)
    assert declared <= set(ROLE_ACTION_MULT)


# -- Mirroring ---------------------------------------------------------------

def test_slot_position_rotates_both_axes():
    """Mirroring x alone leaves a left-back at the same absolute y, which is
    that team's RIGHT touchline."""
    home = slot_position(22.0, 12.0, 1)
    away = slot_position(22.0, 12.0, -1)
    assert np.allclose(home, [22.0, 12.0])
    assert np.allclose(away, [PITCH_LENGTH - 22.0, PITCH_WIDTH - 12.0])


@pytest.mark.parametrize("name", ALL)
def test_teams_are_exact_mirrors(name):
    state = build_test_state(seed=1, formation=name)
    for home, away in zip(state.home_team.players, state.away_team.players):
        assert home.position == away.position
        assert away.assigned_pos[0] == pytest.approx(PITCH_LENGTH - home.assigned_pos[0])
        assert away.assigned_pos[1] == pytest.approx(PITCH_WIDTH - home.assigned_pos[1])


def test_left_players_are_on_their_own_left():
    """For a team attacking leftward (toward x=0), their own left flank is HIGH
    absolute y. This is what the y-mirror fixes."""
    state = build_test_state(seed=1, formation="4-3-3")
    away_left = [p for p in state.away_team.players if p.position in ("LD", "LF")]
    assert away_left
    for p in away_left:
        assert p.assigned_pos[1] > PITCH_WIDTH / 2


# -- Rosters -----------------------------------------------------------------

@pytest.mark.parametrize("name", ALL)
def test_squads_are_eleven(name):
    state = build_test_state(seed=0, formation=name)
    assert len(state.home_team.players) == SQUAD_SIZE
    assert len(state.away_team.players) == SQUAD_SIZE


def test_player_ids_are_disjoint_and_contiguous():
    state = build_test_state(seed=0)
    home = [p.player_id for p in state.home_team.players]
    away = [p.player_id for p in state.away_team.players]
    assert home == list(range(1, SQUAD_SIZE + 1))
    assert away == list(range(SQUAD_SIZE + 1, 2 * SQUAD_SIZE + 1))
    assert not set(home) & set(away)


@pytest.mark.parametrize("name", ALL)
def test_kickoff_taker_exists_for_every_formation(name):
    """A bare next() with no default would raise StopIteration on a formation
    with no literal CF and take the whole match down."""
    state = build_test_state(seed=0, formation=name)
    owner = state.get_ball_owner()
    assert owner is not None
    assert owner.position != "GK"


# -- Changing formation ------------------------------------------------------

def test_apply_formation_moves_players():
    state = build_test_state(seed=0, formation="4-4-2")
    before = {p.player_id: p.assigned_pos.copy() for p in state.home_team.players}
    apply_formation(state.home_team, "4-3-3")

    assert state.home_team.tactics.formation == "4-3-3"
    moved = max(
        float(np.hypot(*(p.assigned_pos - before[p.player_id])))
        for p in state.home_team.players
    )
    assert moved > 5.0


def test_apply_formation_does_not_teleport_players():
    state = build_test_state(seed=0)
    positions = {p.player_id: p.pos.copy() for p in state.home_team.players}
    holder = state.get_ball_owner()
    had_ball = holder.has_ball if holder else None

    apply_formation(state.home_team, "5-3-2")

    for p in state.home_team.players:
        assert np.allclose(p.pos, positions[p.player_id])
    if holder is not None:
        assert holder.has_ball == had_ball


def test_apply_formation_clears_sticky_targets():
    """The sticky bonus favours points near last tick's target; after a shape
    change that target belongs to the old formation and fights the new anchor."""
    state = build_test_state(seed=0)
    # Sticky targets are only written while that team has possession, so set
    # them directly rather than depending on how long the ball takes to arrive.
    for p in state.home_team.players:
        p.last_off_ball_target = p.assigned_pos.copy()

    apply_formation(state.home_team, "4-3-3")
    assert all(p.last_off_ball_target is None for p in state.home_team.players)


def test_apply_formation_leaves_the_other_team_alone():
    state = build_test_state(seed=0)
    before = {p.player_id: p.assigned_pos.copy() for p in state.away_team.players}
    apply_formation(state.home_team, "3-5-2")
    for p in state.away_team.players:
        assert np.allclose(p.assigned_pos, before[p.player_id])
    assert state.away_team.tactics.formation == "4-4-2"


def test_formation_round_trip_restores_positions():
    state = build_test_state(seed=0, formation="4-4-2")
    before = {p.player_id: p.assigned_pos.copy() for p in state.home_team.players}
    apply_formation(state.home_team, "4-3-3")
    apply_formation(state.home_team, "4-4-2")
    for p in state.home_team.players:
        assert np.allclose(p.assigned_pos, before[p.player_id])


@pytest.mark.slow
def test_formation_actually_changes_where_the_team_plays():
    """The test that proves a formation changes PLAY, not just a string."""
    def mean_x(name: str) -> float:
        state = build_test_state(seed=4, formation=name)
        apply_formation(state.away_team, "4-4-2")
        for _ in range(400):
            state = tick(state)
        return float(np.mean([p.pos[0] for p in state.home_team.players]))

    assert mean_x("4-3-3") > mean_x("5-3-2")


# -- Station depth (offensiveness) -------------------------------------------

def test_offensiveness_pushes_stations_upfield():
    deep = build_test_state(seed=0)
    deep.home_team.tactics.offensiveness = 0.6
    apply_formation(deep.home_team)

    high = build_test_state(seed=0)
    high.home_team.tactics.offensiveness = 1.5
    apply_formation(high.home_team)

    def outfield_mean(state):
        return float(np.mean([
            p.assigned_pos[0] for p in state.home_team.players if p.position != "GK"
        ]))

    assert outfield_mean(high) > outfield_mean(deep)


def test_keeper_never_moves():
    """A shifted keeper is an empty net, not a tactic."""
    for offensiveness in (0.5, 1.0, 2.0):
        state = build_test_state(seed=0)
        state.home_team.tactics.offensiveness = offensiveness
        apply_formation(state.home_team)
        gk = next(p for p in state.home_team.players if p.position == "GK")
        assert gk.assigned_pos[0] == pytest.approx(5.0)


def test_depth_shift_is_monotonic_and_bounded():
    from app.engine.formations import _shift_depth

    for x in (22.0, 45.0, 70.0):
        previous = -1.0
        for offensiveness in (0.5, 0.8, 1.0, 1.3, 1.6, 2.0):
            shifted = _shift_depth(x, offensiveness)
            assert shifted > previous
            previous = shifted
            assert 0.0 < shifted < PITCH_LENGTH


def test_neutral_offensiveness_is_the_raw_formation():
    state = build_test_state(seed=0)
    for player, (_, x, _) in zip(state.home_team.players, FORMATIONS["4-4-2"]):
        assert player.assigned_pos[0] == pytest.approx(x)


def test_offensiveness_is_mirrored_for_both_sides():
    """A high line must mean the same thing for a team attacking either way."""
    state = build_test_state(seed=0)
    for team in (state.home_team, state.away_team):
        team.tactics.offensiveness = 1.5
        apply_formation(team)
    for home, away in zip(state.home_team.players, state.away_team.players):
        assert away.assigned_pos[0] == pytest.approx(PITCH_LENGTH - home.assigned_pos[0])


def test_offensiveness_is_clamped():
    state = build_test_state(seed=0)
    state.home_team.tactics.offensiveness = 99.0
    apply_formation(state.home_team)
    for p in state.home_team.players:
        assert 0.0 < p.assigned_pos[0] < PITCH_LENGTH


def test_apply_formation_defaults_to_the_current_shape():
    """Changing depth alone must not require naming the formation again."""
    state = build_test_state(seed=0, formation="4-3-3")
    state.home_team.tactics.offensiveness = 1.4
    apply_formation(state.home_team)
    assert state.home_team.tactics.formation == "4-3-3"


def test_per_player_depth_override():
    """A marauding full-back or a deep-lying forward is a real instruction, so
    depth must be overridable per player, not only per team."""
    state = build_test_state(seed=0)
    team = state.home_team
    baseline = {p.player_id: p.assigned_pos.copy() for p in team.players}

    full_back = next(p for p in team.players if p.position == "LD")
    full_back.bias_offensiveness = 1.8
    apply_formation(team)

    assert full_back.assigned_pos[0] > baseline[full_back.player_id][0]
    for p in team.players:
        if p is full_back:
            continue
        assert np.allclose(p.assigned_pos, baseline[p.player_id])


def test_clearing_a_depth_override_restores_the_team_value():
    state = build_test_state(seed=0)
    team = state.home_team
    baseline = {p.player_id: p.assigned_pos.copy() for p in team.players}

    player = team.players[1]
    player.bias_offensiveness = 1.8
    apply_formation(team)
    assert not np.allclose(player.assigned_pos, baseline[player.player_id])

    player.bias_offensiveness = None
    apply_formation(team)
    assert np.allclose(player.assigned_pos, baseline[player.player_id])


def test_player_depth_override_beats_the_team_value():
    state = build_test_state(seed=0)
    team = state.home_team
    team.tactics.offensiveness = 1.5
    deep_lying = team.players[9]
    deep_lying.bias_offensiveness = 0.6
    apply_formation(team)

    others = [p for p in team.players if p.position == "CF" and p is not deep_lying]
    assert others
    assert deep_lying.assigned_pos[0] < others[0].assigned_pos[0]


def test_player_depth_override_is_clamped():
    state = build_test_state(seed=0)
    team = state.home_team
    team.players[1].bias_offensiveness = 99.0
    apply_formation(team)
    for p in team.players:
        assert 0.0 < p.assigned_pos[0] < PITCH_LENGTH
