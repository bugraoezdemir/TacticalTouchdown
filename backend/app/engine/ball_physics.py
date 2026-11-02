"""
Closed-form ball kinematics under constant deceleration.

The discrete simulation in physics.compute_ball_update does, each tick:

    v <- max(0, v - friction)      # friction is m/tick^2, NOT a fraction
    pos <- pos + v
    if |v| < BALL_STOP_THRESHOLD: v <- 0   (ball is stationary)

Constant deceleration is what a rolling ball actually does: rolling resistance
is near-constant, so the ball sheds a fixed amount of speed per tick rather
than a fixed fraction of what it has. Under the old geometric decay the ball
bled speed fastest when fastest, so travel time grew explosively with distance
(a 30 m pass took 3.4 s) and short-pass realism and long-pass reach could not
hold at once -- tuning one always broke the other. Here time grows very nearly
linearly with distance, so both hold with a single friction value.

These helpers invert that analytically so route scoring can ask "how long
does the ball take to travel d metres?" without stepping a loop.

Everything here is plain floats (no numpy) and everything is in TICKS, not
seconds -- consistent with the rest of the engine, where speeds are m/tick.

The stop threshold matters. Without it the closed form assumes the ball
decays forever and overestimates reach by ~33% at realistic friction, which
would make the decision layer believe in passes that physically die short.
"""
from __future__ import annotations

import math
from functools import lru_cache

from .constants import BALL_STOP_THRESHOLD

_EPS = 1e-9


@lru_cache(maxsize=4096)
def carry_distance(
    speed: float,
    friction: float,
    stop_threshold: float = BALL_STOP_THRESHOLD,
) -> float:
    """
    Total distance the ball covers before its speed drops below
    `stop_threshold` and it is declared stationary.

    Cached: called ~370k times per 200 ticks from a handful of distinct
    (speed, friction) pairs, and pure in its arguments.

    With v decreasing by `friction` each tick and the position updated AFTER
    the decay, the ball covers sum_{k=1..n} (speed - k*friction), where n is
    the last tick with speed still at or above the stop threshold.
    """
    if speed < _EPS or speed <= stop_threshold:
        return 0.0
    if friction <= _EPS:
        return float("inf")  # frictionless: never stops

    # The sim decays, MOVES, and only then checks the threshold -- so the tick
    # that takes the speed below it still contributes its displacement. n is
    # therefore one more than the count of ticks that stay at or above it.
    # One tick of slack at the boundary is deliberate. The sim decays
    # ITERATIVELY (v = v - f), so after thousands of ticks its v has drifted
    # from `speed - k*f` by float rounding, and whether the tick landing on the
    # threshold stops it differs case by case: 1.8 - 4000*0.00044 evaluates
    # just ABOVE 0.04 while 0.14 - 25*0.004 lands just BELOW. No closed form
    # reproduces both, and replaying the tail does not help because the drift
    # is spread over the whole flight, not the last few ticks.
    #
    # It does not matter. Under real rolling resistance the ball carries
    # 200-2000 m on a 105 m pitch, so carry_distance now only ever answers
    # "yes, reachable", and travel_ticks -- the quantity the engine actually
    # consumes -- is unaffected to 3 decimal places at any real pass range.
    n = math.floor((speed - stop_threshold) / friction) + 1
    n = min(n, math.floor(speed / friction))
    if n <= 0:
        return 0.0
    return n * speed - friction * n * (n + 1) / 2.0


def travel_ticks(speed: float, friction: float, dist: float) -> float:
    """
    Ticks for a decelerating ball to cover `dist` metres.

    Returns inf if the ball stops first -- callers treat that as "this route is
    not physically possible", which is why the stop threshold (via
    carry_distance) matters rather than the idealised kinematics.

    Position after t ticks is  speed*t - friction*t*(t+1)/2, matching the sim's
    decay-then-move order. Solving for t gives the positive root of
        (friction/2) t^2 + (friction/2 - speed) t + dist = 0
    """
    if dist <= 0.0:
        return 0.0
    if speed < _EPS:
        return float("inf")
    if dist > carry_distance(speed, friction):
        return float("inf")
    if friction <= _EPS:
        return dist / speed

    a = friction / 2.0
    b = friction / 2.0 - speed
    c = dist
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return float("inf")
    return (-b - math.sqrt(disc)) / (2.0 * a)


def travel_distance(speed: float, friction: float, ticks: float) -> float:
    """Distance covered after `ticks` ticks, capped at the carry distance."""
    if ticks <= 0.0 or speed < _EPS:
        return 0.0
    if friction <= _EPS:
        return speed * ticks
    reach = speed * ticks - friction * ticks * (ticks + 1.0) / 2.0
    return min(max(0.0, reach), carry_distance(speed, friction))


def weighted_speed(
    friction: float,
    dist: float,
    overrun: float,
    max_speed: float,
    stop_threshold: float = BALL_STOP_THRESHOLD,
) -> float:
    """The speed to strike a ball so it dies `overrun` metres past `dist`.

    A pass struck at a fixed speed carries the same distance whatever it was
    aimed at, so a 10 m ball overshot by 22 m and ran through the receiver to
    whoever stood behind them. Real players weight a pass to arrive.

    Inverts carry_distance. Under constant deceleration carry is ~speed^2/2f,
    but the stop threshold makes n a floor() of speed, so the exact inverse is
    still awkward -- bisection, monotone in speed, exact to well under a
    centimetre in 24 iterations.
    """
    target = dist + max(0.0, overrun)
    if target <= 0.0 or friction <= _EPS:
        return max_speed
    if carry_distance(max_speed, friction, stop_threshold) <= target:
        return max_speed  # cannot be hit hard enough; strike it as hard as we can

    low, high = stop_threshold, max_speed
    for _ in range(24):
        mid = 0.5 * (low + high)
        if carry_distance(mid, friction, stop_threshold) < target:
            low = mid
        else:
            high = mid
    return high


def speed_for_arrival(friction: float, dist: float, ticks: float) -> float:
    """Strike speed that delivers the ball `dist` metres in `ticks` ticks.

    The counterpart to weighted_speed, for the model where the ball stops
    because someone CONTROLS it rather than because it ran out of pace. The
    passer chooses when the ball should arrive; what it would have done
    afterwards is irrelevant, since capture zeroes its velocity.

    Position after t ticks is speed*t - friction*t*(t+1)/2, so
        speed = dist/t + friction*(t+1)/2
    """
    if ticks <= 0.0 or dist <= 0.0:
        return 0.0
    return dist / ticks + friction * (ticks + 1.0) / 2.0


def nominal_pass_speed(friction: float, dist: float) -> float:
    """The pace an ordinary (undriven) pass over `dist` is actually struck at.

    Every model that asks "could this pass be intercepted" must use the speed
    the passer would really use. PASS_SPEED_LOW/HIGH are legacy weights from
    the geometric-decay era and are roughly TWICE the real pace now, which made
    lanes look safe, runway leads land behind the runner, and receivers arrive
    before the ball.
    """
    from .constants import PASS_TICKS_BASE, PASS_TICKS_PER_M, PASS_SPEED_MIN

    if dist <= 0.0:
        return PASS_SPEED_MIN
    ticks = PASS_TICKS_BASE + PASS_TICKS_PER_M * dist
    return max(PASS_SPEED_MIN, speed_for_arrival(friction, dist, ticks))
