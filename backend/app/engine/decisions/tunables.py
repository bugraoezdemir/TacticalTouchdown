"""
Context multipliers, as data.

This is where the design doc's decision tree lives -- demoted from a dispatcher
to a lookup table. "Close to own goal, centre -> pass almost always, choose safe
passes" is one row here, tunable in seconds, rather than a branch buried four
levels deep in an if-tree.

Role tilts are deliberately mild and nearly flat for passing: whether to
release the ball is a question of where you are and what is on, not of what
number you wear. A steep role bias here suppressed exactly the balls forwards
should play in the final third.

The multipliers are a *tilt* on a utility that already means something
(P x V is an expected goal-probability contribution). They are deliberately
kept in a narrow band: if a multiplier has to be 3.0 to get sane behaviour,
the value model is wrong and that is what should be fixed. A test enforces
the band so this table cannot quietly become the load-bearing hack it replaced.
"""
from __future__ import annotations

from typing import Literal

from .zones import Zone

ActionType = Literal["shot", "pass", "dribble", "hold", "move"]

MULTIPLIER_MIN = 0.7
MULTIPLIER_MAX = 1.4

_DEFAULT: dict[str, float] = {"shot": 1.0, "pass": 1.0, "dribble": 1.0, "hold": 1.0, "move": 1.0}

# Where you are on the pitch changes your risk appetite, not your options.
ZONE_ACTION_MULT: dict[Zone, dict[str, float]] = {
    # Dribble was 1.15 here, a hand-set tilt; measured, a final-third pass led
    # to a shot more often than a dribble of the same value, so no tilt.
    Zone.ATTACKING_CENTRAL: {"shot": 1.30, "pass": 1.00, "dribble": 1.00, "hold": 0.80, "move": 1.10},
    Zone.ATTACKING_WING:    {"shot": 0.85, "pass": 1.20, "dribble": 1.15, "hold": 0.85, "move": 1.05},
    Zone.MIDFIELD_CENTRAL:  {"shot": 0.80, "pass": 1.20, "dribble": 0.95, "hold": 1.00, "move": 1.00},
    Zone.MIDFIELD_WING:     {"shot": 0.70, "pass": 1.15, "dribble": 1.05, "hold": 1.00, "move": 1.00},
    Zone.DEFENSIVE_CENTRAL: {"shot": 0.70, "pass": 1.25, "dribble": 0.80, "hold": 0.90, "move": 0.95},
    Zone.DEFENSIVE_WING:    {"shot": 0.70, "pass": 1.20, "dribble": 0.90, "hold": 0.95, "move": 0.95},
}

# Role bias. Replaces the old hard-coded `shot_mult = 1.2 if is_finisher`.
ROLE_ACTION_MULT: dict[str, dict[str, float]] = {
    "GK": {"shot": 0.70, "pass": 1.20, "dribble": 0.70, "hold": 1.10, "move": 1.00},
    "LD": {"shot": 0.80, "pass": 1.05, "dribble": 0.95, "hold": 1.00, "move": 1.00},
    "CD": {"shot": 0.75, "pass": 1.05, "dribble": 0.90, "hold": 1.00, "move": 1.00},
    "RD": {"shot": 0.80, "pass": 1.05, "dribble": 0.95, "hold": 1.00, "move": 1.00},
    "LM": {"shot": 1.00, "pass": 1.05, "dribble": 1.10, "hold": 1.00, "move": 1.00},
    "CM": {"shot": 1.00, "pass": 1.05, "dribble": 1.00, "hold": 1.00, "move": 1.00},
    "RM": {"shot": 1.00, "pass": 1.05, "dribble": 1.10, "hold": 1.00, "move": 1.00},
    "LF": {"shot": 1.20, "pass": 1.05, "dribble": 1.10, "hold": 0.90, "move": 1.05},
    "CF": {"shot": 1.25, "pass": 1.05, "dribble": 1.05, "hold": 0.90, "move": 1.05},
    "RF": {"shot": 1.20, "pass": 1.05, "dribble": 1.10, "hold": 0.90, "move": 1.05},
}

TACTIC_ACTION_MULT: dict[str, dict[str, float]] = {
    "defensive": {"shot": 0.85, "pass": 1.10, "dribble": 0.80, "hold": 1.10, "move": 0.95},
    "normal":    {"shot": 1.00, "pass": 1.00, "dribble": 1.00, "hold": 1.00, "move": 1.00},
    "offensive": {"shot": 1.20, "pass": 0.95, "dribble": 1.20, "hold": 0.85, "move": 1.05},
}


def context_multiplier(zone: Zone, position: str, tactical: str, action: str) -> float:
    """Product of the three context tilts. Never a branch -- always a lookup."""
    return (
        ZONE_ACTION_MULT.get(zone, _DEFAULT).get(action, 1.0)
        * ROLE_ACTION_MULT.get(position, _DEFAULT).get(action, 1.0)
        * TACTIC_ACTION_MULT.get(tactical, _DEFAULT).get(action, 1.0)
    )


from ..constants import TACTIC_BIAS_MAX, TACTIC_BIAS_MIN  # noqa: F401


def tactical_bias(player, team, action: str) -> float:
    """The user's tactical preference for this action, as a multiplier.

    Resolution order: the player's own override, else the team setting. Clamped
    so a slider cannot break the utility scale -- it tilts which situations an
    action wins, it does not override the football.
    """
    override = {
        "shot": player.bias_shoot,
        "pass": player.bias_pass,
        "dribble": player.bias_dribble,
    }.get(action)
    value = override if override is not None else team.tactics.get(action)
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def defensive_bias(player, team) -> float:
    """How hard this player presses, as a multiplier around 1.0.

    Same resolution and clamping as tactical_bias: the player's own override,
    else the team setting. Higher means press the ball; lower means hold the
    interception line and stay patient.
    """
    value = (
        player.bias_pressing
        if player.bias_pressing is not None
        else team.tactics.pressing
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def roaming_bias(player, team) -> float:
    """How far from his own station this player will roam, as a multiplier.

    Scales the off-ball shape leash. Deliberately keyed on distance from the
    player's OWN STATION rather than from the ball: a full-back correctly
    holding width on the far touchline is a long way from the carrier but
    exactly where he should be, and penalising that would drag the whole team
    ball-side and collapse the shape.
    """
    value = (
        player.bias_roaming
        if player.bias_roaming is not None
        else team.tactics.roaming
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def offensiveness_bias(player, team) -> float:
    """How far up the pitch this player is stationed, as a multiplier.

    Unlike the other sliders this one does not enter a per-tick utility: it is
    consumed by formations.apply_formation when stations are (re)computed, so a
    change only takes effect once that runs.
    """
    value = (
        player.bias_offensiveness
        if player.bias_offensiveness is not None
        else team.tactics.offensiveness
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def direct_bias(player, team) -> float:
    """Preference for a pass that does not make the receiver run, around 1.0.

    Distinct from the Pass slider, which decides whether to release the ball at
    all. This one only chooses BETWEEN passes (on_ball.direct_factor).
    """
    value = (
        player.bias_direct
        if player.bias_direct is not None
        else team.tactics.direct
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def one_two_bias(player, team) -> float:
    """How readily this player moves on after passing, around 1.0.

    Higher means push forward for a return ball; lower means hold position and
    let the move develop without him. Governs how strongly a passer is pushed
    off the ball he has just played -- 18% of passes were being collected by the
    passer himself, which is a wasted touch that stalls the move.
    """
    value = (
        player.bias_one_two
        if player.bias_one_two is not None
        else team.tactics.one_two
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def attack_call_bias(player, team) -> float:
    """How hard this player breaks forward when the ball is moving upfield fast.

    Only applies while the ball is genuinely progressing -- a deep pass or a
    carry at pace. It is a support instinct, not a permanent shift in station:
    that is what Depth and Roaming are for.
    """
    value = (
        player.bias_attack_call
        if player.bias_attack_call is not None
        else team.tactics.attack_call
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))

def pass_range_bias(player, team) -> float:
    """Which LENGTH of pass this player prefers, around 1.0.

    Above 1.0 favours longer balls, below it favours shorter ones, and exactly
    1.0 expresses no preference. Like direct_bias this is a FLAVOUR control: it
    is applied as a reciprocal pair in on_ball scoring so it trades short
    against long without changing how much the team passes overall, which is
    what the Pass slider governs.
    """
    value = (
        player.bias_pass_range
        if player.bias_pass_range is not None
        else team.tactics.pass_range
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def breadth_bias(player, team) -> float:
    """How far from the centre line this player's station sits, around 1.0.

    Above 1.0 the shape spreads toward the touchlines; below it the team packs
    centrally, shortening the gaps between teammates. Read by
    formations.apply_formation, not by any decision -- breadth changes WHERE
    players stand, and every consequence for passing follows from that.
    """
    value = (
        player.bias_breadth
        if player.bias_breadth is not None
        else team.tactics.breadth
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def compactness_bias(player, team) -> float:
    """How close this player keeps to his teammates, around 1.0.

    Read in three places, all about the distance between teammates: the
    station contraction in formations.apply_formation, the width of the shape
    leash, and the radius inside which a teammate counts as crowding him.
    """
    value = (
        player.bias_compactness
        if player.bias_compactness is not None
        else team.tactics.compactness
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def possession_bias(player, team) -> float:
    """How much this player values keeping the ball, around 1.0.

    Read by on_ball._turnover_cost: above 1.0 a pass, dribble or shot that may
    lose the ball pays a cost by where it would be lost. At or below 1.0 there
    is none -- a 'direct' side below it is not defined yet.
    """
    value = (
        player.bias_possession
        if player.bias_possession is not None
        else team.tactics.possession
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def crossing_bias(player, team) -> float:
    """How readily this player crosses from the flank, around 1.0.

    A plain multiplier on a cross's utility: crossing competes with every
    other option a wide carrier has, not with other kinds of pass only.
    """
    value = (
        player.bias_crossing
        if player.bias_crossing is not None
        else team.tactics.crossing
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def lofted_bias(player, team) -> float:
    """Air against ground, around 1.0. A FLAVOUR control like direct_bias:
    applied as a reciprocal pair, so it changes how the ball travels, not how
    often it is passed or crossed."""
    value = (
        player.bias_lofted
        if player.bias_lofted is not None
        else team.tactics.lofted
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def risk_bias(player, team) -> float:
    """Appetite for risk, around 1.0: the exponent 1 / risk on P.

    utility = P x V x C is expected value, so it is risk-neutral -- a chancy
    forward ball and a safe square ball with the same P x V are worth the same.
    Read by on_ball._set_utility for passes and dribbles: above 1.0 low-P
    options gain on safe ones (P 0.25 counts as 0.5 at 2.0), below it they lose
    (0.0625 at 0.5). Neutral is exactly the old scoring.
    """
    value = (
        player.bias_risk
        if player.bias_risk is not None
        else team.tactics.risk
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def press_aggression_bias(player, team) -> float:
    """How directly he goes at the carrier: at 2.0 straight at him, at 0.5 fully
    onto the lane of his likeliest pass (off_ball.press_lane_share)."""
    value = (
        player.bias_press_aggression
        if player.bias_press_aggression is not None
        else team.tactics.press_aggression
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def man_marking_bias(player, team) -> float:
    """How he shares himself between zone and man, around 1.0 (off_ball
    ._zonal_mark_detail): above 1.0 nearer his man and following him further,
    below it nearer his zone. Neutral is the old share."""
    value = (
        player.bias_man_marking
        if player.bias_man_marking is not None
        else team.tactics.man_marking
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def wing_attacks_bias(player, team) -> float:
    """How much he favours a pass to the flank, around 1.0 (on_ball._wing_tilt).
    Neutral treats flank and middle alike."""
    value = (
        player.bias_wing_attacks
        if player.bias_wing_attacks is not None
        else team.tactics.wing_attacks
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))


def tempo_bias(player, team) -> float:
    """How fast he runs with the ball, around 1.0 (pace.tempo_factor): above
    1.0 the run and the sprint gain on the walk, below it the walk gains.
    Neutral expresses no preference."""
    value = (
        player.bias_tempo
        if player.bias_tempo is not None
        else team.tactics.tempo
    )
    return float(min(TACTIC_BIAS_MAX, max(TACTIC_BIAS_MIN, value)))
