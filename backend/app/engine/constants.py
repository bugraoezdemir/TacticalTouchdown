import numpy as np

# ── Pitch ─────────────────────────────────────────────────────────────────────
# Coordinate system: (0,0) = top-left, x→right, y→down
PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0
PITCH_CENTER = np.array([52.5, 34.0])

GOAL_WIDTH  = 7.32
GOAL_HALF   = GOAL_WIDTH / 2          # 3.66
LEFT_GOAL_X  = 0.0
RIGHT_GOAL_X = 105.0
GOAL_CENTER_Y  = 34.0
GOAL_TOP_Y     = GOAL_CENTER_Y - GOAL_HALF   # 30.34
GOAL_BOTTOM_Y  = GOAL_CENTER_Y + GOAL_HALF   # 37.66

LEFT_GOAL_TOP    = np.array([LEFT_GOAL_X,  GOAL_TOP_Y])
LEFT_GOAL_BOTTOM = np.array([LEFT_GOAL_X,  GOAL_BOTTOM_Y])
RIGHT_GOAL_TOP    = np.array([RIGHT_GOAL_X, GOAL_TOP_Y])
RIGHT_GOAL_BOTTOM = np.array([RIGHT_GOAL_X, GOAL_BOTTOM_Y])

# ── Timing ────────────────────────────────────────────────────────────────────
TICK_RATE = 30       # Hz
TICK_DT   = 1.0 / TICK_RATE

# ── Ball speeds (m/tick at 30 Hz) ─────────────────────────────────────────────
# Verified against real football: a 20 m ground pass should arrive in ~1.2-1.5 s,
# a 20 m shot in ~0.7 s. See tests/test_ball_physics.py for the regression guards.
SHOT_SPEED       = 1.10   # 33 m/s   – 20 m in 0.73 s, carries 69.6 m
PASS_SPEED_HIGH  = 1.80   # firm ball; 20 m in 0.51 s, carries 42.3 m
# Rescaled with the friction change above: these are used as REPRESENTATIVE
# pass weights (best_pass_safety, _angle_gain, off-ball receivability), so they
# must keep carrying what they always carried. At friction 0.040, 0.70 reaches
# only 15.9 m -- leaving it there would have scored every pass beyond 16 m as
# physically unreachable, in exactly the outlet-finding code this change is
# meant to help.
PASS_SPEED_LOW   = 1.35   # gentle ball; 20 m in 0.78 s, carries 31.5 m
# The ceiling weighted_speed may strike a ball at. Distinct from PASS_SPEED_HIGH,
# which is a representative weight: this is the hardest a player can hit it, and
# it is what keeps 40 m ground passes possible under the raised friction.
# 0.80 m/tick = 24 m/s, the hardest a ground pass is realistically struck --
# and deliberately BELOW SHOT_SPEED (33 m/s), because a pass that outpaces a
# shot is not a pass. This was 2.00 (60 m/s!), sized when friction was 0.040
# and the cap had to be enormous to reach 47 m of carry. Under real rolling
# resistance the ball carries hundreds of metres, so the cap stopped binding on
# distance and instead let the driven branch compress a short pass until it was
# faster than a shot.
PASS_SPEED_MAX   = 0.80   # 24 m/s; hardest realistic ground pass

DRIBBLE_SPEED    = 0.14   # ~4 m/s   – tap away from feet, carries 1.6 m

# ── Friction (velocity *= 1 - coeff each tick) ────────────────────────────────
# These were 0.001, which let a pass carry 470 m on a 105 m pitch. Every travel
# time fed to the interception model was correspondingly meaningless.
# Raised from 0.020. The weighting solves BACKWARDS from carry distance -- a
# pass is struck so it dies `overrun` metres past the receiver -- so with a
# near-frictionless ball the only way to stop it in time was to strike it
# feebly: a 10 m settling pass went out at 0.305 m/tick (~9 m/s) and took 1.83 s
# to arrive, which is what made short balls so easy to intercept. Real rolling
# resistance is far higher, so a firm pass decelerates noticeably and must be
# hit hard to arrive at all. At 0.040 the same 10 m pass is struck at 0.58 and
# lands in 1.03 s, and it crosses the presser's danger zone in half the time.
#
# PASS_SPEED_HIGH must rise with it: it is the CAP fed to weighted_speed, and
# higher friction needs more power for the same carry. At 0.040/2.00 the longest
# ground pass is 47 m, better than the 42 m it was -- long range is preserved.
# 4.5 m/s^2. Checked empirically, not from a remembered "rolling resistance"
# figure: a firm 20 m/s ground pass travels 40-60 m before stopping, which is
# v^2/2d = 4-5 m/s^2. The first value here was 0.4 m/s^2 -- ten times too low --
# and it meant a 16 m/s pass carried 342 m on a 105 m pitch. The ball could
# never stop in play, so it always reached a touchline and NO ball was ever
# collected while still moving.
BALL_FRICTION_COEFF    = 0.00500  # 4.5 m/s^2; a 20 m/s pass carries ~44 m
SHOT_FRICTION_COEFF    = 0.00889  # 8.0 m/s^2; a 33 m/s shot carries ~67 m
# Same surface as a pass, because it IS the same surface -- a dribble touch is
# just a gentle roll. At 4.5 m/s^2 a 4.2 m/s touch dies after 1.8 m, which is a
# player knocking it ahead of himself.
DRIBBLE_FRICTION_COEFF = 0.00500  # 4.5 m/s^2; a touch carries ~1.8 m
BALL_STOP_THRESHOLD    = 0.04    # m/tick below which ball is stationary

# ── Penalty area (real dimensions: 16.5 m deep, 40.32 m wide) ─────────────────
PENALTY_AREA_DEPTH  = 16.5
PENALTY_AREA_WIDTH  = 40.32
PENALTY_AREA_TOP    = GOAL_CENTER_Y - PENALTY_AREA_WIDTH / 2.0   # 13.84
PENALTY_AREA_BOTTOM = GOAL_CENTER_Y + PENALTY_AREA_WIDTH / 2.0   # 54.16
# How far outside the area opponents are put when the keeper has the ball. A
# little clearance so they are not standing on the line and straight back in.
BOX_CLEARANCE = 1.5

# ── Goalkeeper handling ───────────────────────────────────────────────────────
# A keeper gathers a ball he can actually catch, rather than only touching it
# like an outfield player. Without this, attackers standing on him at a goal
# kick simply took it off him and scored.
# 0.35 m/tick (10.5 m/s) was set when a pass left the foot at 0.30-0.90 under
# geometric decay. Under the real ball model EVERY pass arrives above it --
# 0.40 for a gentle 10 m ball, 0.68 for a driven 40 m one -- so a keeper could
# not catch a single pass, only parry it, however gently it was played.
#
# The separation that matters is pass versus SHOT: passes arrive at up to
# 0.68 m/tick (20.5 m/s), a struck shot at ~1.09 (33 m/s). 0.85 sits between
# them, so he claims any pass with his hands -- which is routine -- while a
# genuine shot still beats him.
GK_GATHER_SPEED    = 0.85   # m/tick; faster than this and he can only parry
# Where he stands with the ball in play: this far out from the middle of his
# goal, on the line toward the ball (off_ball._gk_spot) -- the angle cut
# equally to both posts.
GK_ADVANCE         = 2.0    # m
# A parry: the ball comes back out at this fraction of its pace, within this
# many radians of straight out -- a loose rebound, not possession.
GK_PARRY_RESTITUTION = 0.30
GK_PARRY_SPREAD      = 1.0
GK_HOLD_TICKS      = 45     # 1.5 s holding it while the box clears
# How long after distributing a keeper is barred from gathering that same ball.
# Long enough for it to clear his control radius at a pass's launch speed.
GK_RERECEIVE_LOCKOUT = 20   # ticks
# Distribution: the one moment a keeper is never under pressure, so he does not
# gamble. A giveaway from his own hands is a shot on his own goal.
GK_SAFE_PASS_FLOOR = 0.50   # never distribute into a pass less safe than this
GK_SAFETY_EXPONENT = 0.25   # value still breaks ties, but safety dominates
# ── Possession ────────────────────────────────────────────────────────────────
CAPTURE_RADIUS = 0.51   # metres – increased so players can actually catch the ball
# A loose ball is WON within a player's full reach (interception.control_radius:
# OUTFIELD_CONTROL_RADIUS, a keeper's hands), the radius every decision plans
# with; CAPTURE_RADIUS above is now only how close a dribbler keeps it. The
# kicker cannot take back his own ball for KICK_LOCKOUT_TICKS.
KICK_LOCKOUT_TICKS = 4
# For STRIKE_REACT_TICKS after a kick an OPPONENT has had no time to react: he
# cannot reach out for it, only block it -- the ball has to come within
# BLOCK_RADIUS of him, travelling toward him (at least BLOCK_MIN_AHEAD past the
# kicker's foot in his direction). With the full 1 m reach, a ball struck from
# the feet dropped straight into the reach of an opponent beside the kicker, and
# the two traded it every tick. Kept short: over the full lockout, defenders
# standing in the lane lost their reach too (interceptions 18 -> 7 a match).
STRIKE_REACT_TICKS = 1   # teammates other than the receiver do not touch it
BLOCK_RADIUS = 0.51
BLOCK_MIN_AHEAD = 0.25
# A struck ball (shot or pass) that meets a body in that window DEFLECTS: it
# comes off him loose at this fraction of its pace, reflected off his body and
# turned by up to BLOCK_SPREAD radians, and he cannot touch it again for
# BLOCK_LOCKOUT_TICKS. Taking it to his feet put it inside the kicker's reach,
# and the two blocked each other's kicks every tick: 36 of 213 shots in 12
# halves were such repeats, one burst of 16.
BLOCK_RESTITUTION = 0.35
BLOCK_SPREAD = 0.6
BLOCK_LOCKOUT_TICKS = 2

# ── Decision thresholds [0, 1] ────────────────────────────────────────────────
SHOT_SCORE_THRESHOLD    = 0.35
PASS_SCORE_THRESHOLD    = 0.25
DRIBBLE_SCORE_THRESHOLD = 0.20

# ── Look-ahead / geometry ─────────────────────────────────────────────────────
MAX_INTERCEPT_TICKS    = 120   # 4 s look-ahead
RUNWAY_DIST            = 8.0   # m; legacy fixed lead, kept for tests
# The lead is now SOLVED so the receiver and the ball arrive together (see
# _runway_leads); these only bound the search. A fixed lead could not work:
# it ignored how far the ball had to travel, so the runner beat it to the spot
# a median 6 ticks and stood waiting while a defender closed.
# A runway ball is weighted to the RUNNER's arrival, not to its lane danger.
# Slightly under 1.0 so it is still moving into his path as he gets there
# rather than sitting and waiting to be caught up with.
RUNWAY_ARRIVE_MARGIN   = 0.92
# ...for a ball travelling ALONG his run. A ball cutting square across it must
# not arrive first -- it would pass through the meeting point and away from
# him -- so it is weighted to reach the spot slightly after he does. Blended by
# sin(angle between the ball's path and his run).
RUNWAY_CROSS_MARGIN    = 1.08
# How far a receiver can reach without really moving -- a step and a stretch.
# Inside this a pass may be struck as hard as the lane demands, because he only
# has to control it. Beyond it the ball must wait for him: pace is governed by
# the receiver's travel, not the ball's.
PASS_RECEIVER_FREE_REACH = 2.5   # m
RUNWAY_MIN_LEAD        = 2.0   # m; shorter than this is a ball to feet
RUNWAY_MAX_LEAD        = 25.0  # m; beyond this it is a through ball, not a lead
DRIBBLE_LOOKAHEAD_DIST = 3.0
BACK_DRIBBLE_PENALTY   = 0.5
CLOSE_SHOT_DISTANCE    = 14.0  # bonus range for shots

# ── Player movement (m/tick) ──────────────────────────────────────────────────
# 0.30 m/tick was 9.0 m/s SUSTAINED -- near Usain Bolt's peak, held by every
# player for the whole match. It is why a physically correct ball looked slow:
# the pass was racing sprinters. A real outfielder covers ground at 2-4 m/s and
# tops out near 9.5 m/s in short bursts, so a top-quality player here peaks at
# ~6.9 m/s of continuous movement, which is a fast sustained run.
PLAYER_BASE_SPEED  = 0.20    # 6.0 m/s at quality 50
PLAYER_SPEED_SCALE = 0.0018  # per quality point above 50
GK_SPEED           = 0.19    # GK overall movement speed

# ── Player inertia (kinematics.py) ────────────────────────────────────────────
# Velocity used to be set fresh every tick: full speed in any direction, from
# any previous motion, instantly. So a marker who simply aims at his man each
# tick could never be shaken. Real players lose markers through the cost of
# starting, stopping and turning, which falls on whoever is REACTING.
#
# One law (rebuilt 2026-10-26): each tick a player's velocity closes the gap
# to the velocity he wants by a fixed share, i.e. it relaxes toward it with
# time constant PLAYER_TAU. From rest he reaches 63% of his top speed in one
# tau and 95% in three; a full reversal at speed stops him after 0.69 tau and
# has him back to half speed the other way after 1.1 tau. The previous model
# (5 m/s^2 fading to zero at top speed, i.e. tau = top/accel ~ 1.2-1.7 s, and a
# 6 m/s^2 sideways limit) made players so sluggish that a match held half as
# many on-ball decisions and passes spent 40% longer rolling.
# Under this law "how soon can he get there" is exact (kinematics.reach_ticks),
# so decisions and movement cannot disagree.
# Switched by FB_INERTIA (off by default).
PLAYER_TAU          = 0.6 * TICK_RATE   # ticks
# Stopping on a spot: aim at (gap / ARRIVE_TAUS x tau), capped -- with this
# damping he settles on it without circling it (overshoot ~4% at 2.0).
ARRIVE_TAUS         = 2.0
# Grip (FB_GRIP, kinematics.GRIP). Under the law above a sprinter stops and
# cuts as briskly, relative to his speed, as a jogger: his stopping distance
# and turning circle grow only WITH speed. A real one's grow with speed
# SQUARED, because grip limits how hard he can brake or push sideways. So
# braking and turning are quick in themselves -- they close their part of the
# gap at BRAKE_TAU -- but held to his grip: per second, GRIP_TOP_PER_S of his
# top speed (8 m/s^2 for a 7 m/s player). A jogger stops and cuts sharply; a
# sprinter is committed. Pushing on in his running direction stays the law
# above, which matches real sprint starts.
BRAKE_TAU           = 0.3 * TICK_RATE   # ticks
GRIP_TOP_PER_S      = 1.13
# Running at a point off to one side, he eases off to the speed at which his
# grip can still bend his run onto it (kinematics.corner_speed) -- but never
# below this share of his top speed.
CORNER_MIN_SHARE    = 0.3
# Choosing to run slower (under grip: speed costs agility).
# The receiver of a pass is told WHERE (the pass point) and decides how fast:
# when no opponent can contest the point, he runs at the slowest pace that
# still has him there TIMED_RUN_EARLY_TICKS before the ball -- arriving
# balanced, not flat out -- but never slower than TIMED_RUN_MIN_SHARE of top.
TIMED_RUN_EARLY_TICKS = 6
TIMED_RUN_MIN_SHARE   = 0.25
# ...but the point is not all he watches. If an opponent could reach the ball's
# path within COME_TO_BALL_MARGIN of the ball's own time to that spot, on any
# part of the path before the ball gets to him, he stops waiting and goes to
# meet it (off_ball._comes_to_ball). A share, not ticks: near the kicker's foot
# a tick or two is all that separates a man the ball is leaving behind from one
# it runs into. The first COME_TO_BALL_FROM_TICKS after the strike do not
# count: the man pressing the passer stands on that part of every path.
COME_TO_BALL_MARGIN     = 0.3
COME_TO_BALL_FROM_TICKS = 6
COME_TO_BALL_HORIZON    = 120   # ticks ahead watched once it is past its point
# The passer plays a moving teammate where his run takes him (to-feet and
# lofted balls), if he is running faster than LEAD_MIN_SPEED_SHARE of his top
# speed; at most LEAD_MAX_TICKS of his run ahead.
LEAD_MIN_SPEED_SHARE  = 0.3
LEAD_MAX_TICKS        = 60
# Support runs in possession jog unless the spot is more than SUPPORT_SPRINT_GAP
# away or is in behind the defence (within SUPPORT_BEHIND_MARGIN of the
# offside line, or beyond it).
SUPPORT_SPRINT_GAP    = 12.0  # m
SUPPORT_BEHIND_MARGIN = 3.0   # m
# A carrier with nobody clear to pass to and nobody near him idles: he walks
# the ball at IDLE_PACE_SHARE of his top speed, ball at his feet, while the
# others move -- instead of dribbling at full pace into nothing. Only while
# no opponent can reach him within IDLE_FREE_TICKS, his best dribble gains
# less than IDLE_GAIN_MAX in position value, and for IDLE_MAX_TICKS at most.
IDLE_PACE_SHARE       = 0.3
IDLE_FREE_TICKS       = 20
IDLE_GAIN_MAX         = 0.30
IDLE_MAX_TICKS        = 90
IDLE_MIN_GAP          = 6.0   # m; and nobody this close, whatever his motion
# A target that jumps further than this in one tick (x his top speed) was
# re-chosen, not moving: its jump is not a velocity to match.
TARGET_MOTION_MAX   = 1.5

# ── Interception model ────────────────────────────────────────────────────────
REACTION_TICKS          = 6.0    # ~0.2 s before a player reacts and accelerates
RISK_TAU                = 4.0    # ticks; smoothness of the interception sigmoid
OUTFIELD_CONTROL_RADIUS = 1.0    # m a player can reach to touch the ball
GK_CONTROL_RADIUS       = 1.3    # m; keepers use hands, scaled by goalkeeping

# ── Value model ───────────────────────────────────────────────────────────────
SHOT_VALUE_D0           = 14.5   # m; logistic midpoint
SHOT_VALUE_K            = 3.5    # m; logistic width
SHOT_VALUE_ANGLE_POWER  = 1.2    # how hard a tight angle is punished
VALUE_W_SHOT            = 0.75   # weight of shot value vs raw progression
VALUE_PROGRESSION_POWER = 2.0    # >1 makes value accelerate toward the goal

# ── Zones ─────────────────────────────────────────────────────────────────────
ZONE_CENTRAL_HALF_WIDTH = 20.15  # m from the centre line; the penalty-area half-width

# ── Decision scale ────────────────────────────────────────────────────────────
MIN_PASS_DIST             = 6.0   # m; don't "pass" to someone at your feet
HOLD_VALUE_DISCOUNT       = 0.50  # holding keeps your position but makes no progress
# A dribble leaves the ball contestable, and in a block the next touches are
# just as contested: measured in the final third (2026-10-15, 16 x 7500), for
# the SAME engine value a pass was followed by a shot within 5 s 1.4x as
# often as a dribble against a neutral side and 2.1x against a press, while
# carriers dribbled 70-78% of the time. 0.75 -> 0.60, deliberately short of
# the full gap: at 0.45 (before penetration_value) dribbling vanished.
DRIBBLE_CONTINUATION_DISCOUNT = 0.60
# Was 0.45, tuned before penetration_value existed. Once passes could earn a
# 2.2x line-breaking bonus that a 3 m touch almost never earns, the compounded
# gap reached 5-6x and dribbling stopped happening at all.
PASS_CONTINUATION_DISCOUNT = 0.85 # a shot's value is terminal; a pass's is not

# ── Execution error (skill applies to delivery, not to the decision) ──────────
# Was 0.20: a skill-70 passer then missed by a median 0.65 m (1.2 m spread at
# 20 m), 32% of passes landed more than a metre off and those reached the
# receiver 58-65% of the time against 79% for accurate ones -- and under
# inertia a receiver cannot simply step across to fix it. Halved (2026-10-26).
PASS_ERROR_MAX_RAD = 0.10   # rad of angular error at skill 0; 0 at skill 100
SHOT_ERROR_MAX_RAD = 0.16
# A ground shot is also offered this far inside each post (and 0.5 m, and the
# middle): with his miss in the plan, the shooter picks the margin.
SHOT_AIM_INSIDE_M  = 1.2   # m
DRIBBLE_TOUCH_MIN  = 1.2    # m; a great dribbler keeps it this close
DRIBBLE_TOUCH_MAX  = 3.0    # m; a poor one knocks it this far

# ── Touch-length dribbling ────────────────────────────────────────────────────
# A dribble is a touch and a chase, and the touch LENGTH is the speed: knock it
# far and sprint onto it, keep it close and walk. For each of 16 directions the
# touch is SOLVED: the length that maximises P(he gets to his own ball first) x
# V(the spot it leaves him in). Near goal V rises steeply, so a longer touch
# into a shooting position wins; at the back V barely changes, so safety wins
# and the touch stays short. The shortest touch away from pressure IS
# shielding, so outfield players no longer hold. Skill sets how far he can
# knock it and still recover it.
DRIBBLE_TOUCH_MODEL     = True
DRIBBLE_DIRECTIONS      = 16
DRIBBLE_MIN_TOUCH       = 0.8   # m; just beyond his feet (capture radius 0.51)
DRIBBLE_MAX_TOUCH_POOR  = 3.0   # m; the longest touch a skill-0 dribbler recovers
DRIBBLE_MAX_TOUCH_GREAT = 6.0   # m; ... and a skill-100 dribbler
DRIBBLE_TOUCH_SAMPLES   = 8     # lengths tried per direction
DRIBBLE_RACE_TAU        = 4.0   # ticks; smoothness of the race to his own ball
DRIBBLE_NUDGE           = 2   # m; the ball leaves his feet at once, then rolls
# Touch cadence: a dribbler touches the ball about every 0.6 s, so his pace is
# touch length / cadence (capped by his top speed) and he MEETS the ball at the
# touch point while it is still rolling. Timing the ball to STOP there instead
# capped every dribble near 3.5 m/s: a ball rolling to a halt averages half its
# launch speed, so no touch, however long, produced a fast dribble.
DRIBBLE_TOUCH_INTERVAL  = 18.0  # ticks
# Carrying the ball costs speed: even at full pace a dribbler is slower than he
# is without it. Without this a pressed player knocked it 5 m past the presser
# and simply outran him at his own sprint speed.
DRIBBLE_PACE_POOR       = 0.81  # of top speed, skill 0
DRIBBLE_PACE_GREAT      = 0.96  # of top speed, skill 100

# ── Pace with the ball: Tempo (workplan 2.9, decision D-12 (a)) ──────────────
# Under decisions.pace.PACE every dribble direction is offered at three paces,
# as a share of his top pace with the ball: a walk, a run and a sprint, and the
# scorer picks the pace by the same argmax as everything else. The race to his
# own ball is part of the option's P, so a long, fast touch near a defender is
# the exposed one. Idling is the walk winning: there is no separate idle rule.
#
# The walk and the run are planned FROM HIS MOTION: the touch goes where he
# will be after one touch cadence of easing toward that pace under grip. A man
# at 5 m/s cannot walk within 0.6 s; planned from a standstill he overran his
# own ball and 29% of walking touches were lost. The sprint keeps the solved
# knock-on (the length from PACE_SPRINT_FROM of his top pace to his longest
# recoverable touch that maximises race x position value).
PACE_SLOW = (("walk", 0.30), ("run", 0.60))   # (band, target pace share)
PACE_SPRINT_FROM = 0.75
PACE_TOUCH_SAMPLES = 4      # lengths tried for the sprint
PACE_MIN_TOUCH = 0.3        # m; a slower touch is not offered
# Tempo tilts the pace like Pass range tilts the length: a pace faster than
# the pivot scales with tempo, a slower one with its reciprocal; 1.0 for all
# at tempo 1.0.
TEMPO_PIVOT_SHARE = 0.6
TEMPO_EXPONENT    = 0.5
# What walking buys: the best pass a second from now, from where teammates'
# runs (and the opponents tracking them) will be, discounted because a later
# pass is worth less than the same pass now: PACE_WAIT_DISCOUNT ** Tempo, so a
# patient side (low Tempo) counts waiting for more.
PACE_WAIT_TICKS    = 30
PACE_WAIT_DISCOUNT = 0.6
# Tempo in pass choices: how fast a pass moves the ball toward goal. Forward
# metres dx scale the pass by Tempo ** (TEMPO_PASS_EXPONENT x clamp(dx /
# TEMPO_PROGRESS_M, -1, 1)): a forward ball gains with Tempo, a back pass by
# the reciprocal, a square one not at all; 1.0 for every pass at Tempo 1.
TEMPO_PROGRESS_M    = 20.0
TEMPO_PASS_EXPONENT = 1.0
# Tempo off the ball, our ball: midfielders' and forwards' anchors move
# TEMPO_RUN_M upfield per doubling of Tempo (and back per halving), so a
# high-tempo side runs ahead of the ball and a low-tempo one comes short.
# Support runs sprint beyond SUPPORT_SPRINT_GAP / Tempo instead of the gap.
# Defenders keep their stations. Nothing changes at Tempo 1.0.
TEMPO_RUN_M = 8.0

# ── Off-ball pursuit (speed vs safety) ────────────────────────────────────────
LAMBDA_CONFIDENCE_HORIZON = 45.0  # ticks (1.5 s); beyond this a prediction decays
LAMBDA_COVER_FLOOR        = 0.40  # lambda scale when no teammate is covering
LAMBDA_OUTLET_WEIGHT      = 0.50  # how much open passing options suppress lambda
MARGINAL_ENGAGE_THRESHOLD = 0.05  # below this added value, hold shape instead

# ── Off-ball shape and sampling ───────────────────────────────────────────────
SHAPE_BLOCK_SLIDE_X = 0.45   # how far the block slides with the ball, lengthwise
SHAPE_BLOCK_SLIDE_Y = 0.35   # ... and across
STICKY_BONUS        = 0.15   # hysteresis toward last tick's target
OFFBALL_RING_RADII  = (6.0, 12.0)
OFFBALL_RING_POINTS = 8
# The leash: the only place role enters off-ball movement.
ROLE_LEASH = {
    "GK": 8.0, "LD": 14.0, "CD": 12.0, "RD": 14.0,
    "LM": 20.0, "CM": 18.0, "RM": 20.0,
    "LF": 30.0, "CF": 32.0, "RF": 30.0,
}
TACTICAL_LEASH = {"defensive": 0.8, "normal": 1.0, "offensive": 1.3}
CLAIM_MARGIN_TICKS = 6.0   # ticks; how much sooner you must arrive to claim a loose ball outright
# ── Danger-zone closing down ──────────────────────────────────────────────────
# Defending is position-dependent: out wide or at halfway keep the zone, near
# our goal close the carrier down so he cannot shoot comfortably. Measured
# before this existed: at the moment of a shot the nearest defender was only
# 1.0-1.2 m away, but 75% of shots from 12-25 m and 100% inside 12 m had nobody
# in the shot's path -- defenders closed on the ball, not on the shot.
DANGER_FAR          = 35.0  # m from goal; beyond this, no change to defending
# Recovery runs: out of possession, a player ahead of the ball is pulled back to
# a line RECOVER_DEPTH goal-side of it -- partly always (RECOVER_BASE), fully
# as the ball nears our goal -- and sprints once he is RECOVER_SPRINT_GAP away.
RECOVER_DEPTH      = 3.0   # m goal-side of the ball
RECOVER_BASE       = 0.4   # recovery strength with the ball far from our goal
RECOVER_SPRINT_GAP = 3.0   # m; recovering players sprint beyond this
# ...and "goal-side" is along the ball's line to our goal: a recovering player
# is drawn into the triangle from the ball to our posts, this far either side
# of it, where he cuts the shooting angle (off_ball._zone_centre).
RECOVER_TRIANGLE_MARGIN = 2.0   # m
# No deeper than needed (off_ball._not_too_deep): a station is never nearer
# our goal than LINE_SHARE of the ball's distance from it, so the gap behind
# the ball shrinks as it comes closer (ball 30 m out -> line at 18 m, the
# edge of the box; 20 m -> 12 m; 12 m -> 7 m). Fully within DANGER_FAR, fading
# out by LINE_FADE_FAR, where dropping back is legitimate. The back four's
# zone centres had sat at 0.41-0.50 of the ball's distance.
LINE_SHARE       = 0.60
LINE_FADE_FAR    = 55.0   # m
STATION_MIN_FROM_GOAL = 2.0   # m; never stationed in the goal mouth itself
# The back caller's shape call (decisions.shape_call). Made every SHAPE_CALL_TICKS,
# sooner if the ball has moved SHAPE_CALL_BALL_M, and heard SHAPE_CALL_HEAR_TICKS
# later. Stations re-decided every tick changed a defender's job 31.6 times a
# minute, 35% of them undone within half a second.
SHAPE_CALL_TICKS     = 30      # ticks
SHAPE_CALL_MIN_TICKS = 10      # ticks
SHAPE_CALL_BALL_M    = 8.0     # m
SHAPE_CALL_HEAR_TICKS = 6      # ticks
SHAPE_CALL_EASE_TICKS = 24     # ticks for the team to move onto a call it has heard
SHAPE_CALL_OUTLET_KEEP_M = 5.0 # m; the outlet stays his while within this of the highest man
# Its shape. The deepest slot is never nearer our goal line than BLOCK_LINE_MIN.
# The outfielder of rank BLOCK_FRONT_RANK (0 = deepest) fits in behind the ball,
# so only the men in front of him stay ahead of it, but the formation keeps at
# least BLOCK_LENGTH_MIN of its length. Width narrows to BLOCK_WIDTH_MIN from
# BLOCK_SQUEEZE_FAR in to DANGER_FAR, which takes centre-backs 14 m apart to 8 m
# and a back four 37 m wide to 28. Only BLOCK_ACROSS_NEAR of the sideways slide
# applies with the ball at our line.
# Were 0.72 and 0.5 (2026-11-03): with the ball within 35 m the back four stood
# 32 m wide, a 14 m hole between a centre-back and his full-back 72% of the
# time, and 85% of goals came with one there at the final ball, the scorer
# shooting from 22 m. At 0.55 and 0.8 (scratch/wide_goals.py, 16 matches):
# holes 42% and 33% at goals, shots 209 -> 148, goals 37 -> 30, from 14 m.
BLOCK_LINE_MIN    = 6.0        # m
BLOCK_FRONT_RANK  = 7
BLOCK_LENGTH_MIN  = 0.35
BLOCK_WIDTH_MIN   = 0.55
BLOCK_SQUEEZE_FAR = 70.0       # m
BLOCK_ACROSS_NEAR = 0.8
# Closing down the carrier under inertia: the arrive time constant, in taus.
# 1.0 brakes late (a little overshoot is contact, not a fault); the 2.0 used to
# settle on a fixed spot reached him at walking pace.
PRESS_ARRIVE_TAUS = 1.0
DANGER_NEAR         = 12.0  # m from goal; full urgency
DANGER_PRESS_GAIN   = 1.0   # pressing multiplied by up to (1 + this) at full danger
# ── Zonal marking ─────────────────────────────────────────────────────────────
# Out of possession, whoever is not pressing defends his zone: the attackers in
# it are shared out one-to-one among the defenders whose zone centres are
# nearest, and each takes up a goal-side, ball-side position on his man.
ZONE_RADIUS          = 14.0  # m from a zone centre; attackers beyond are someone else's
ZONE_FULL            = 7.0   # m; inside this an attacker is squarely in the zone
ZONE_POS_WEIGHT      = 0.5   # zone ground = this much where he stands, the rest his zone centre
ZONE_THREAT_FAR      = 60.0  # m from our goal; attackers beyond are not marked
ZONE_THREAT_NEAR     = 25.0  # m from our goal; marked at full strength inside this
ZONE_MARK_FLOOR      = 0.4   # marking strength at ZONE_THREAT_FAR
MARK_STANDOFF_FAR    = 3.5   # m goal-side of a marked attacker, far from goal
MARK_STANDOFF_NEAR   = 1.0   # m goal-side of him from ZONE_THREAT_NEAR in (was 1.5)
# A player who is not sprinting moves at this share of his top speed. A marker
# whose man is faster than that sprints, whatever the gap (off_ball._mark_action).
JOG_SHARE            = 0.6
MARK_BALL_SIDE       = 0.6   # weight of "toward the ball" against "toward goal"
MARK_SPRINT_GAP      = 3.0   # m; a marker further than this from his spot sprints
# ── Goal funnel: the block draws in toward goal as the danger rises ──────────
# Out of possession every zone centre is contracted toward the middle of our
# goal -- deeper AND narrower together -- by
#   f = FUNNEL_MAX * (1 - exp(-(level / FUNNEL_SCALE) ** FUNNEL_POWER))
# where `level` = threat(ball) + sum over attackers of threat(him) x how
# reachable he is from the ball (value.threat; lane from the ball, floored).
# Continuous in the positions of the ball and both teams -- no distance
# threshold switches it on. Withdrawing stations straight down the pitch had
# drifted wide defenders to the corner flags.
# Tried first and REJECTED (2026-10-15, scratch/defend_shape.py): placing each
# defender at a weighted average of his station and the cover points of the
# threats near him. Averaging left him between threats covering none
# (attackers near goal marked 33% -> 18-27%, shots conceded up), and the threat
# surface rates the corners high (its progress term is x-only), pulling
# defenders out to the corner flags. Zonal one-to-one marking is kept intact.
# The curve is S-shaped (the power), and that matters: the danger level only
# roughly doubles from midfield (median 0.31) to our defensive third (0.55),
# so the first, gentle form (1 - exp(-level), max 0.40) pulled 11% in midfield
# and only 17% near goal -- a general sag, invisible where it was needed.
# Now ~8% in midfield, ~32% typical in our third, up to ~55% under heavy threat.
FUNNEL_MAX         = 0.60   # largest contraction toward goal
FUNNEL_SCALE       = 0.6    # danger level at which ~63% of FUNNEL_MAX applies
FUNNEL_POWER       = 3.0    # steepness of the rise
FUNNEL_REACH_FLOOR = 0.3    # an attacker the ball cannot reach now still counts this much
# Who leaves his zone to go at the ball. One presser always; pressing above
# neutral buys extras, so at the default only one man leaves.
PRESS_EXTRA_PER_UNIT = 2.0   # extra pressers per unit of pressing above 1
# Press numbers is graded, not stepped. The whole places in that count go to
# whoever may take them; the part of a place left over (0.5 at Pressing 1.25) goes
# only to a man who can reach the ball within that share of PRESS_EXTRA_WINDOW.
# Below 1.0 even the first man goes only if he can get there in time: within
# PRESS_FIRST_TICKS x s / (1 - s) ticks, s = Pressing x the urgency near our
# goal (1 s at 0.5, 3 s at 0.75, no limit from 1.0 up). A loose ball is always
# gone for.
PRESS_EXTRA_WINDOW = 60.0     # ticks
PRESS_FIRST_TICKS  = 30.0     # ticks
def rust_constants() -> dict[str, float]:
    """The values fb_core needs, by name (fb_core/src/lib.rs, Tuning)."""
    import math

    names = (
        "REACTION_TICKS", "RISK_TAU", "BLOCKING_BODY_RADIUS", "BLOCKING_MIN_ALONG",
        "BLOCKED_LANE_RISK", "PRESSER_BLOCK_RANGE", "PRESSER_SIDE_RISK",
        "BALL_STOP_THRESHOLD", "PITCH_LENGTH", "GOAL_CENTER_Y", "GOAL_TOP_Y",
        "GOAL_BOTTOM_Y", "SHOT_VALUE_D0", "SHOT_VALUE_K", "SHOT_VALUE_ANGLE_POWER",
        "VALUE_W_SHOT", "VALUE_PROGRESSION_POWER", "RECEIVABILITY_FLOOR",
        "SPACE_RADIUS", "COMPACT_REACH", "SUPPORT_RANGE", "SUPPORT_MAX_BONUS",
        "TEAMMATE_SPACING_FLOOR", "SPACING_SOFTNESS", "ONE_TWO_STANDOFF",
        "ONE_TWO_CROWD_PENALTY", "ONE_TWO_SUPPORT_RANGE", "ONE_TWO_PULL_BASE",
        "STICKY_BONUS", "OFFSIDE_FADE_DISTANCE", "OFFSIDE_MIN_FACTOR",
        "ZONE_CENTRAL_HALF_WIDTH", "PENALTY_AREA_DEPTH", "PENALTY_AREA_TOP",
        "PENALTY_AREA_BOTTOM", "PASS_TICKS_BASE", "PASS_TICKS_PER_M", "PASS_SPEED_MIN",
    )
    values = {name: float(globals()[name]) for name in names}
    values["PRESSER_BLOCK_COS"] = math.cos(math.radians(PRESSER_BLOCK_ANGLE_DEG))
    values["REFERENCE_ANGLE"] = 0.6424      # value._REFERENCE_ANGLE
    return values


# The man in front of a carrier running at our goal (off_ball._running_at_goal).
# The one place at a carrier goes to whoever reaches the ball first, which beyond
# DANGER_FAR was a man chasing from behind on 6 ticks in 10, and the goal-side
# rule stopped at DANGER_FAR, so the men in front held their zones and he ran on.
# Out there a goal-side man may step up too, if he can reach the ball within
# FRONT_MAN_TICKS x Press numbers.
FRONT_MAN_RUN   = 2.0         # m/s toward our goal
FRONT_MAN_TICKS = 60.0        # ticks
# Press aggression: how far the man going at the carrier moves off the ball and
# onto the lane to the carrier's best outlet (the pass with the highest lane x
# position value), as he approaches. He can do both at once, so the slider sets
# the share: (1 - log2(Press aggression)) / 2, all of it at 0.5, half at 1.0,
# none at 2.0. Cutting the carrier's RUN instead changed almost nothing: forced
# to the extreme it left him further from the carrier and spells 1 s longer,
# because the carrier passes on most decisions and few dribbles are lost.
PRESS_LANE_MIN_M = 2.0        # m out from the carrier: the lane is cut, not his feet
# ── Who steps out to the carrier: cover-aware (off_ball._press_set) ──────────
# Decided once per tick for the whole team. Stepping out GAINS the carrier's
# threat (now, or PRESS_LOOKAHEAD_TICKS on along his run, whichever is worse)
# times the chance of getting goal-side of him in time -- a sigmoid in how
# much sooner than PRESS_REACH_TICKS he can be there, the window widening
# with pressing. It COSTS, for every other attacker, his danger (threat where
# he stands x how reachable he is from the carrier) times the share of the
# cover on him that only this defender provides: a man two defenders stand
# over costs nothing to leave, a man only he covers costs all of it. One
# presser always goes -- the best net, not the nearest -- and up to
# PRESS_MAX in all, each only while his gain (discounted by the chance the
# ones already going win it) still exceeds what he leaves behind.
PRESS_REACH_TICKS     = 30.0   # 1 s
PRESS_TAU             = 6.0    # ticks
PRESS_LOOKAHEAD_TICKS = 15.0
PRESS_MAX             = 3
MARK_KEEP_CLOSENESS  = 0.5   # a defender with a mark at least this dangerous stays on it
GOAL_SIDE_STANDOFF  = 1.2   # m goal-side of the carrier: in his shot, not on his toes
# In shooting range the man in front going at a carrier stands in his shot
# fully, not blended with the run he would cut off: full from SHOT_LINE_NEAR m
# from our goal, fading out by SHOT_LINE_FAR. With the blend alone (danger:
# about half at 24 m) the nearest goal-side man stood 3-4 m off the line,
# moving mostly across the pitch beside the carrier, and 79 of 96 shots from
# 12-30 m had an open line (all 12 goals). 16 matches (scratch/shot_diag.py,
# shotline_diag.py): goals 26 to 20, goal hidden from a carrier 12-24 m out
# 15% to 16-18% of the time. A man stepping across into the line at 2-5 m
# instead of going at him was tried and dropped: 36 goals.
SHOT_LINE_NEAR      = 25.0  # m
SHOT_LINE_FAR       = 32.0  # m
# How far ahead of the carrier (toward our goal) a defender must be to count as
# cover. A little positive, so a man exactly level with him -- who has no body
# in the shot -- does not stand a goal-side teammate down.
GOAL_SIDE_MIN_AHEAD = 0.5   # m
CONTEST_SAME_POINT_RADIUS = 8.0  # m; teammates aiming further apart than this are not competing
RECEIVABILITY_FLOOR = 0.25  # a run into space is worth making even if the ball cannot arrive yet
REACTION_RAMP_DISTANCE = 3.0  # m; below this, reaction cost scales down (already in the lane)
BLOCKED_LANE_RISK = 0.90  # a defender whose body already occupies the lane blocks it
HOLD_PRESSURE_DISTANCE = 8.0  # m; a carrier with an opponent closer than this is under pressure
# Shielding the ball: lost to the combined strength of opponents close enough
# to contest it. Calibrated by forcing carriers to hold under pressure: one
# presser of equal strength within 3 m leaves the ball ours ~80% of the time.
HOLD_CONTEST_RADIUS   = 3.0   # m
HOLD_LOSS_PER_PRESSER = 0.25  # 1 / (1 + 0.25) = 0.80 against one equal presser
# Holding is a short wait, not a resting state. Its probability decays with
# how long he has already held (e-fold per HOLD_PATIENCE_TICKS), and past
# HOLD_MAX_TICKS it is not offered at all -- a carrier can never freeze.
HOLD_PATIENCE_TICKS   = 30.0  # 1 s
HOLD_MAX_TICKS        = 60    # 2 s
# Holding is SHIELDING, not standing still: the carrier drifts away from the
# pressers at a fraction of his speed, steered off the lines, with the ball on
# the far side of his body.
SHIELD_SPEED_FRACTION = 0.35  # of his top speed
SHIELD_AWARENESS      = 5.0   # m; opponents this close shape the direction
SHIELD_EDGE_MARGIN    = 4.0   # m from a line at which he starts steering in
SHIELD_BALL_OFFSET    = 0.3   # m; ball kept this far on his far side (< capture radius)
BLOCKING_BODY_RADIUS = 0.6  # m; a torso actually occupying the lane, not a reaching arm
# How far ALONG a lane a body must be before it counts as blocking it.
#
# Every lane radiates from the carrier's feet, so a presser 0.9 m away has a
# tiny perpendicular gap to almost ALL of them and would otherwise trip the
# floor on every option at once. Excluding that one body lifts the median best
# available pass from 0.215 to 0.738 and viable (P>=0.5) passes from 2% to 60%
# of pressed decisions.
#
# THE DISCONTINUITY IS LOAD-BEARING. Below this distance the floor does not
# apply AT ALL, so a carrier with a man at his feet can genuinely turn and
# play. Replacing the hard cut with a smooth convex ramp (authority
# (d/2)**2, giving the same presser 0.182 instead of 0.000) was tried on
# 2026-09-15: it improved every offline metric, passed the suite, and the user
# playing it said "this is a huge regression. The team under pressure is
# completely overwhelmed." Free-space decisions collapsed 96 -> 3 -- that count
# was the one honest warning and it was tuned past rather than heeded.
#
# So: a cliff here, deliberately. Smooth it and the pressed side dies.
BLOCKING_MIN_ALONG = 1.5  # m
# The one exception to the cliff above: a presser within this range of the
# passer who stands within this angle of the lane's direction DOES block it.
# He blocks what is in front of him, not every lane at once -- measured, passes
# played within 30 deg of a presser at the feet kept the ball 38% of the time
# (52% taken by that presser), against 67-77% in every other direction.
PRESSER_BLOCK_RANGE     = 2.0   # m
PRESSER_BLOCK_ANGLE_DEG = 30.0  # either side of the lane
# ...and OUTSIDE that cone he is capped here, the same idea carried into the
# interception race. Within reach of the passer his nearest point on EVERY lane
# is the passer's own foot: ball there at t=0, he there at t=0, a dead heat, and
# the sigmoid's midpoint charged ~0.5 on every pass -- even one played straight
# away from him. Measured (2026-10-14, pressed side vs a 1.6 press): on 28% of
# its passes the riskiest man was one at the feet outside the cone, rated 0.52,
# and 89% of those passes were kept. So he costs what was measured, ~0.10:
# passing under pressure IS a little harder, but he is not in every lane.
PRESSER_SIDE_RISK       = 0.10
# Re-tuned after the ball physics was made realistic. These were set when shots
# lost pace quickly; with a real 33 m/s shot that holds its speed, 0.37 s of
# reaction plus a 3 m/s dive meant he needed 0.93 s to reach a corner -- beaten
# from anywhere inside 25 m. Measured: 47% of ALL shots scored and 20-36% from
# beyond 18 m, against ~10% and ~3% in real football. Real keepers read a shot
# in ~0.2-0.3 s and cover ~3 m in about half a second.
GK_REACTION_TICKS = 8.0   # ~0.27 s to read a shot, pick a side and dive
# A diving keeper is not a player running. He commits to one side and covers
# ground far slower than a sprint, which is precisely why corners are scoreable
# and why he cannot guard both. Modelling his dive at running speed made keeper
# risk RISE with distance -- 0.356 at 8 m to 0.977 at 25 m for the same corner,
# backwards -- and left the mean shot at P=0.125 needing a 3.0x multiplier to
# win the argmax when the Shoot slider caps at 2.0x.
GK_DIVE_SPEED = 0.20  # m/tick; ~6 m/s of lateral extension once committed
SHOT_BLOCK_REACTION_TICKS = 12.0  # outfield players barely react to a struck ball
RUNWAY_LATE_TOLERANCE = 8.0  # ticks the receiver may arrive after the ball and still get it
RECEPTION_GRACE_TICKS = 6.0  # a receiver arriving slightly late still controls a slowing ball
RECEPTION_TAU         = 6.0  # ticks; smoothness of the receiver-vs-opponent race
# The reception race follows the ball along its real path, including past the
# target, so PACE decides who collects it. It starts part-way down the path --
# the flight before that is route_survival's -- and runs until the ball stops
# or leaves the pitch, up to the horizon.
RECEPTION_FROM_FRACTION = 0.6
# A runway ball landing this far beyond every outfield opponent is a through
# ball: its lane stops where the race starts (on_ball._lane_end).
THROUGH_BEYOND_M = 1.0   # m
# A through ball is worth the shot it sets up (on_ball._through_chance): the
# runner carries it on toward goal, at PACE_SPRINT_FROM of his top pace, as
# long as he stays ahead of the defenders, and shoots by THROUGH_SHOT_FROM_M
# from the middle of the goal.
THROUGH_SHOT_FROM_M = 11.0   # m
RECEPTION_HORIZON_TICKS = 150.0   # 5 s
# Each to-feet pass is offered at these multiples of the rule-based pace, and
# the reception race picks. One pace from a fixed rule could not see who stood
# beyond the receiver.
PASS_PACE_FACTORS = (0.8, 1.0, 1.25)

# ── Off-ball support vs threat ────────────────────────────────────────────────
SUPPORT_FALLOFF = 1.5   # >1 makes the switch from support to threat happen later
SPACE_RADIUS    = 12.0  # m; distance to the nearest opponent that counts as "free"
SUPPORT_MAX_BONUS = 1.0 # how strongly deep possession pulls teammates into support
SUPPORT_RANGE = 25.0  # m; within this of the ball you count as a supporting outlet
# ── Compactness in possession ─────────────────────────────────────────────────
# In possession, stations are pulled toward a ring around the ball so the
# carrier has short options. Measured before: under pressure he had 1.0
# teammate within 10 m against 2.6 opponents, and the team's spacing did not
# contract when he was pressed.
COMPACT_SUPPORT_DISTANCE = 10.0  # m; the ring supporters are drawn toward
COMPACT_FULL             = 25.0  # m; stations within this take full part in supporting
COMPACT_REACH            = 40.0  # m; stations beyond this from the ball keep their width
# OFF (0.0). Measured on the pressing spread, home pressing 0.6 vs 1.6 against
# a neutral side, 4 seeds of half matches: with the pull OFF the spread was
# +0.75 GD -- the pressing bias had already largely gone -- and with it ON it
# grew to +3.75, the high-pressing side winning all four seeds. Drawing
# supporters in toward a pressed ball also draws them into the presser's reach,
# so one swarm covers the carrier AND his short options. Kept, not deleted, so
# it can be revisited: e.g. pulling only toward points with a clear lane.
COMPACT_BASE_PULL        = 0.0   # was 0.45
COMPACT_PRESSURE_PULL    = 0.0   # was 0.50
COMPACT_PRESSURE_FAR     = 6.0   # m; nearest presser at which pressure starts to count

# ── Playmaking signals ────────────────────────────────────────────────────────
SIGNAL_RADII          = (12.0, 20.0, 28.0)  # m ahead of the carrier to look for space
SIGNAL_ARC_COUNT      = 7      # samples across the forward arc
SIGNAL_MAX_RECIPIENTS = 2      # how many teammates are sent to the same spot
SIGNAL_MIN_UTILITY    = 0.02   # below this, do not bother calling a run
TOUCHLINE_MARGIN = 3.0  # m; keep pass targets this far inside the lines, since the ball overruns
BALL_OVERRUN_DIST  = 6.0   # m the ball may run past its target before someone gets a touch
OUT_OF_PLAY_FLOOR  = 0.05  # residual value of a ball that will run out of play
ARRIVAL_TOLERANCE  = 1.35  # runner may take this multiple of the ball's flight time
PROMPTNESS_TAU     = 12.0  # ticks; how sharply a late runner devalues the spot
SUPERIORITY_RADIUS = 14.0  # m; the neighbourhood that counts for a numerical advantage
SUPERIORITY_WEIGHT = 0.18  # per extra body; a tilt, not a decider
SUPERIORITY_MIN    = 0.6
SUPERIORITY_MAX    = 1.6

# ── Penetration: a ball that takes defenders out of the game ─────────────────
PENETRATION_WEIGHT = 0.22   # per defender left behind the ball
PENETRATION_MAX    = 2.2

# ── Dribble direction: run into space, not into people ───────────────────────
DRIBBLE_ROOM_RADIUS   = 6.0   # m to the nearest opponent that counts as free
DRIBBLE_LOOKAHEAD_FACTOR = 2.5  # also probe this many touch-lengths ahead
DRIBBLE_ROOM_FLOOR    = 0.15  # even a crowded dribble is not worth exactly zero

# ── Goal-side accounting: who is between the ball and the goal ───────────────
# One-sided: a penalty for putting defenders back in front of the ball, never
# a bonus. Rewarding forward play is penetration_value's job.
GOALSIDE_WEIGHT   = 0.15   # per defender newly between the ball and the goal
GOALSIDE_MIN      = 0.55
CLEAR_PATH_BONUS  = 1.90   # carrying with nobody in front is worth far more
CLEAR_PATH_BLOCKER_LIMIT = 2.5   # weighted blockers at which the bonus is gone
CLEAR_PATH_LANE_WIDTH    = 14.0  # m; how far off the running lane still counts
DRIBBLE_CONTEST_RADIUS = 5.0   # m; an opponent this close can step in
DRIBBLE_CONTEST_WEIGHT = 0.55  # how much each realistic challenger hurts
DRIBBLE_CONTEST_FLOOR  = 0.25

# ── Shooting ─────────────────────────────────────────────────────────────────
# A shot ends the move: no interception risk afterwards, no chance to be closed
# down, and a goal is worth strictly more than a good position. Passes and
# dribbles both carry continuation discounts and dribbles earn a clear-path
# bonus, so without this a carry beat a 15 m chance with P=0.67.
SHOT_DECISIVENESS = 1.45

# ── Defensive tactics ─────────────────────────────────────────────────────────
# Pressing scales the pursuit lambda: >1 pulls it toward 0 (go at the ball),
# <1 pushes it toward aligning with the ball's trajectory and waiting.
PRESSING_LAMBDA_WEIGHT = 0.80
HOLD_BALL_PULL     = 0.35  # per unit of pressing above 1, how far a holding
HOLD_BALL_PULL_MAX = 0.45  # player drifts from his anchor toward the ball
MIN_SUPPORT_PASS_DIST = 6.0  # m; a ball to someone at your feet is not an outlet

# ── Station depth (offensiveness) ────────────────────────────────────────────
GK_DEPTH_LIMIT       = 8.0   # m; stations at or behind this are the keeper's
STATION_DEPTH_MARGIN = 6.0   # m; never station anyone on the opponent goal line
# Shared clamp for every tactical slider, so they cannot break the utility scale.
TACTIC_BIAS_MIN = 0.5
TACTIC_BIAS_MAX = 2.0
# ── Ball control: pace has a price at the receiving end ───────────────────────
# Capture used to stop the ball dead at the receiver's feet at ANY speed, so
# a 24 m/s driven pass was as easy to take as a gentle one and the passer had
# no reason not to drive everything (faster always survives the lane better).
# Now a ball arriving faster than the easy speed (relative to the receiver's
# own motion) is not simply trapped: he plays it FIRST TIME -- a pass or shot,
# which also degrades with pace -- or tries to CONTROL it, which fails more
# often the faster it comes; a failed touch squirms loose. The passer's P
# includes the receiver's chance of controlling it, so he weights the ball.
# Speeds in m/tick (x30 = m/s).
CONTROL_EASY_SPEED    = 0.50   # 15 m/s: anything slower is a clean first touch
CONTROL_HARD_SPEED    = 1.00   # 30 m/s
CONTROL_MIN_P         = 0.25   # chance of a clean touch at the hard speed
CONTROL_SKILL_SPAN    = 0.30   # easy speed x0.85 (dribbling 0) .. x1.15 (100)
FIRST_TIME_MIN_Q      = 0.60   # first-time pass/shot P multiplier at the hard speed
FIRST_TIME_ERROR_GAIN = 1.5    # execution error x(1 + this) at the hard speed
MISCONTROL_RESTITUTION = 0.35  # share of the incoming pace a bad touch keeps
MISCONTROL_SPREAD      = 1.0   # rad either side of the incoming line
MISCONTROL_LOCKOUT     = 8     # ticks before the same player can touch it again
# The judged pace offered for every to-feet pass besides the lane rule's: the
# firmest ball that still arrives at this share of the receiver's easy speed.
# The lane rule only asked how fast the ball must go to survive, never how fast
# it may go to be taken, so it overhit balls the race would have let be softer.
CONTROL_COMFORT        = 0.90
# ── The ball in the air ───────────────────────────────────────────────────────
# A lofted ball flies a parabola: no rolling friction, gravity on its height,
# horizontal pace kept (drag ignored). It bounces when it lands and rolls once
# a bounce is too small to matter. Height in metres, vertical speed in m/tick.
BALL_GRAVITY           = 9.81 / TICK_RATE ** 2   # m/tick^2
BOUNCE_RESTITUTION     = 0.50   # share of the falling speed a bounce sends back up
BOUNCE_KEEP            = 0.80   # share of the horizontal pace a bounce keeps
BOUNCE_STOP_VZ         = 0.06   # m/tick (1.8 m/s): a smaller bounce is a roll
# Who can reach it. Below the chest he can bring it down (a harder first touch
# than a rolling ball); up to a jumping header he can only play it first time;
# above that it is over everyone except a keeper using his hands in his area.
CONTROL_HEIGHT         = 1.50   # m
HEADER_HEIGHT          = 2.50   # m
# ...but only for a ball he could see coming. One still RISING has just been
# struck, nobody can time a jump to it, and a standing player reaches about
# head height. With 2.5 m for it too, any defender within a few metres of a
# crosser "headed" the ball as it left the foot, and flank carriers saw every
# cross blocked (lane 0.10 with an opponent 2-8 m away) and dribbled instead.
RISING_REACH           = 2.00   # m
GK_REACH_HEIGHT        = 2.80   # m
CROSSBAR_HEIGHT        = 2.44   # m; over it is not a goal
# A ball dropping out of the sky is harder to take than one along the ground
# at the same speed: control_probability is multiplied by this.
AERIAL_CONTROL_FACTOR  = 0.85
# Lofted passes: offered to teammates at least this far away, at these launch
# angles, and struck no harder than LOFT_SPEED_MAX (3-D, m/tick).
LOFT_MIN_DIST          = 14.0   # m
LOFT_ANGLES_DEG        = (22.0, 38.0)
LOFT_SPEED_MAX         = 1.00   # 30 m/s
# A lofted ball into a runner's path is timed to his run, which fixes its
# angle; flatter than this it clears nobody, steeper it is not a pass.
LOFT_RUNWAY_MIN_DEG    = 15.0
LOFT_RUNWAY_MAX_DEG    = 55.0
# A lofted ball is harder to hit accurately than one along the ground: its
# angular error is this multiple of a ground pass's, and its length is off by
# a further LOFT_RANGE_ERROR_MAX (fraction, at skill 0; 0 at skill 100).
LOFT_ANGLE_ERROR_GAIN  = 1.5
LOFT_RANGE_ERROR_MAX   = 0.08   # was 0.12: lofted balls missed by a median 2.7 m
# Lofted shots, offered from LOFT_SHOT_MIN_DIST to LOFT_SHOT_MAX_DIST from the
# aim point. The top-corner shot is struck at SHOT_SPEED for the upper corners,
# to cross the line at TOP_CORNER_HEIGHT, the hardest part of the goal to
# reach. The chip goes up at CHIP_ANGLE_DEG and drops under the bar, over a
# keeper who has come out.
LOFT_SHOT_MIN_DIST     = 10.0   # m
LOFT_SHOT_MAX_DIST     = 40.0   # m
CHIP_MAX_DIST          = 28.0   # m
TOP_CORNER_HEIGHT      = 2.1    # m at the goal line, 0.34 m under the bar
CHIP_HEIGHT            = 1.7    # m at the goal line
CHIP_ANGLE_DEG         = 38.0
# Its lift is off by up to this share (at skill 0; 0 at skill 100), and the
# further the shot goes the more metres that is: from range many go over.
SHOT_LIFT_ERROR_MAX    = 0.5
# A lofted shot's V is its chance of hitting the frame, from his skill, times
# this. The keeper and the defenders are in its P.
LOFT_SHOT_VALUE        = 0.5
# The bent shot, curled into a top corner from either side, from BEND_MIN_DIST
# to BEND_MAX_DIST from it. It turns BEND_CURL radians a tick in the air, so the
# further out the more it bends: 20 degrees in all from 25 m, which is 4.4 m off
# the line it left his foot on. Curl costs power: BEND_PACE of SHOT_SPEED,
# 28 m/s. Its spin is off by up to SHOT_CURL_ERROR_MAX (share, at skill 0; 0 at
# skill 100).
BEND_MIN_DIST          = 12.0   # m
BEND_MAX_DIST          = 26.0   # m; at 35, 81 of 139 shots came from 25-32 m
BEND_CURL              = 0.013  # rad/tick
BEND_PACE              = 0.85
SHOT_CURL_ERROR_MAX    = 0.25
# A keeper's reach with his feet where they are. Up to GK_HANDS_HEIGHT it is
# his whole control radius, along the ground as ever; above it, it narrows like
# an arm swung up, to nothing at GK_REACH_HEIGHT straight over his head.
GK_HANDS_HEIGHT        = 1.2    # m
# A shot crossing above GK_HIGH_SHOT cannot be run under and taken: he reads it
# (GK_REACTION_TICKS), then leaps at GK_DIVE_SPEED. Below it he goes as ever.
# These are the numbers the shooter has always planned on, so it is why a top
# corner is out of his reach from close and within it from far.
GK_HIGH_SHOT           = 1.5    # m
# He stays on his feet, adjusting, until the ball is this far from the line,
# then leaves them for good: under half a second, whatever the range. A ball
# flying straight is where he read it; only one that bends has moved. Measured
# (scratch/bent_probe.py, true strikes 20-32 m): at 12 a bend hardly scores, at
# 14 about a third, at 16 nearly all.
GK_LEAP_TICKS          = 14.0
# A keeper leaves a shot going this far over the bar, or more.
GK_LEAVE_OVER_BAR      = 0.3    # m
# Further than this off his line a ball over his head has to be run back for.
GK_CAUGHT_OUT          = 3.0    # m
# A marker keeps his man out to this multiple of his reach (off_ball.STICKY_MARKS).
MARK_KEEP              = 1.3
# ── Crosses ──────────────────────────────────────────────────────────────────
# From the FLANK in the final third the carrier is offered balls into the box
# ROLES (BOX_SLOTS), each for the teammate filling that role: along the
# ground (timed to his run, or driven at PASS_SPEED_MAX) and lofted (timed to
# his run, whipped in at CROSS_WHIP_DEG or floated at CROSS_FLOAT_DEG). Each is a runway ball, so its
# receiver is signalled to attack the spot.
# The flank is outside the width of the penalty area. It was 12 m off the
# centre line, inside the box's width, so half-space carriers "crossed" too
# (8% of all passes). A ball into the box from there is an ordinary lofted
# pass or ball over the top, valued like any other pass.
CROSS_FROM             = 70.0   # m from our own goal line
CROSS_WIDE             = PENALTY_AREA_WIDTH / 2.0   # 20.16 m from the centre line (y)
CROSS_WHIP_DEG         = 18.0   # driven, below head height: quick, but blockable
CROSS_FLOAT_DEG        = 32.0   # floated over the man in front of him
# A cross is valued by the first-time finish it sets up (see on_ball). That
# finish -- a header, a volley, a stab at a ball across goal -- converts far
# worse than a shot with time on the ball from the same spot, so its value
# is this share of one. Calibrated, not measured from football: at 1.0 flank
# carriers crossed on 74% of visits at slider 1.0 and 80% at 2.0 -- the
# slider had nothing left to raise. At 0.5 (scratch/cross_curve.py, 151
# visits): 34% at slider 0.5, 66% at 1.0, 74% at 2.0. The ceiling near 80% is
# real: on the remaining visits a presser stands within 2 m in the line, and
# forced trials (scratch/loft_block_forced.py) showed he takes ANY ball there,
# ground or lofted, 81-82% of the time.
CROSS_FINISH_FACTOR    = 0.5
# Under grip a cross also weighs how free the finisher is when it arrives:
# the nearest defender's arrival at the spot after the ball, dt ticks, gives
# floor + (1 - floor) * (1 - exp(-dt / CROSS_FREE_TAU)). A man alone has time
# to finish; one a defender reaches with the ball gets a contested touch.
CROSS_FREE_TAU         = 8.0
CROSS_FREE_FLOOR       = 0.25
# ── Clearing headers ─────────────────────────────────────────────────────────
# A defender heading the ball in his own half may also put it out: into touch
# a little upfield, or behind his own goal line for a corner when close to it.
# Only in that decision, every option is charged for where the ball ends up:
#   a header that loses it:  (1 - P) x CLEAR_DANGER x threat(where he is)
#   a clearance out of play:  P x CLEAR_DANGER x CLEAR_RESTART_SHARE x threat(restart)
# so a safe header to a teammate still beats kicking it out, and a hopeful
# one near his own goal does not.
CLEAR_DANGER           = 0.30
CLEAR_RESTART_SHARE    = 0.25
CLEAR_CORNER_RANGE     = 25.0   # m from own goal line within which he may concede a corner
# ── Possession: the turnover cost ─────────────────────────────────────────────
# Above possession 1.0 a pass or dribble pays
#   (1 - P) x (possession - 1) x (BASE + THREAT x threat(where it is lost))
# P x V x C only rewards success; nothing charged for failure, so a safe ball
# backwards (V ~ 0) lost to almost any forward gamble. Subtracted, never
# multiplied in, so P still vetoes giveaways (additive SCORING was measured to
# pick 3.3x more of them). BASE: losing the ball anywhere costs the possession;
# THREAT: losing it near your own goal costs more. Starting values against a
# chosen-option utility of ~0.08-0.11 -- to be set by match outcomes.
POSSESSION_LOSS_BASE   = 0.10
POSSESSION_LOSS_THREAT = 0.15
# Possession as a gradient (on_ball.POSSESSION_GRADUAL, OFF: rejected after
# watching, 2026-10-30): every side counts the cost of losing the ball, Possession x POSSESSION_SCALE times the
# cost above, so 0.5 is half the care, 1.0 the normal care and 2.0 double --
# at 2.0 the same cost as the old 2.0. The old rule charged only (Possession - 1),
# so every value from 0.5 to 1.0 played exactly like 1.0.
POSSESSION_SCALE = 0.5
# ── Attacking the box ─────────────────────────────────────────────────────────
# With our ball in the final third, fewer than one attacker on average stood
# in the box (0.79; 0.58 against a press) -- the wide carrier had nobody to
# cross or cut back to and dribbled into the block, where 56-65% of
# final-third attacks died. So box ROLES are filled, each by the nearest
# eligible attacker (not a defender, not the carrier), and his anchor moves
# toward his role by the attack level: a smoothstep of how far up the pitch
# the ball is -- 0 at BOX_ATTACK_FROM, 1 by BOX_ATTACK_FULL, continuous.
# Inside the box the crowding penalty fades by the same level: there, being
# among defenders is the job.
BOX_ATTACK_FROM = 55.0   # m from our own goal line
BOX_ATTACK_FULL = 85.0
# (role, m from our goal line, m across toward the BALL's side of the pitch)
BOX_SLOTS = (
    ("penalty spot", 94.0, 0.0),
    ("near post",    99.0, 4.0),
    ("far post",     99.0, -5.0),
    ("cut-back",     87.0, 7.0),
)
# ── Lookahead: a pass is worth what the receiver can do next ──────────────────
# One-step scoring values a pass by where the ball lands. The plays that beat
# a compact press are two- and three-step plays -- a switch to the free side,
# a lay-off to a free man who plays forward, a third-man ball -- and their
# first pass lands at the same depth or behind, so it scored like a square
# ball and the pressed carrier dribbled into the press instead (77% of his
# decisions in the presser's half were dribble touches).
# So the most likely passes are EXPANDED: at the moment the ball would arrive,
# with the defenders closed in toward the receiving point, the receiver's own
# best follow-up is scored -- a pass on, a shot, or carrying on into the
# space he has -- and the pass's value becomes
#   V' = (1 - WEIGHT) * V + WEIGHT * best_follow_up
# The follow-up's own value (position value) stands in for the third step.
LOOKAHEAD_WEIGHT    = 0.5
LOOKAHEAD_CLOSING   = 0.5   # share of their run toward the receiver defenders make while the ball travels
LOOKAHEAD_EXPAND    = 8     # receivers expanded per decision, most likely pass first
STATION_BREADTH_GAIN = 0.55  # damping on the lateral spread, so a wide shape is not on the touchlines
# ── Compactness ───────────────────────────────────────────────────────────────
# How close teammates stay to one another, at all times, attacking and
# defending. One slider, four effects, all about the distance between
# teammates rather than distance to the ball:
STATION_COMPACT_GAIN = 0.6   # station contraction: block x compactness^-0.6 (2.0 -> 0.66)
COMPACT_LEASH_GAIN   = 0.5   # leash sigma x compactness^-0.5: stray less from station
COMPACT_SLIDE_GAIN   = 0.8   # block follows the ball x compactness^0.8: move as one unit
# The follow at neutral. It used to be a cap on the follow itself (min(0.95,
# compactness^0.8)), which neutral already reached, so no compactness above
# ~0.94 made the block follow the ball any closer. Now a scale, so neutral
# plays exactly as before and the slider works on both sides; the slide itself
# stays below 0.95 of the ball's offset (the block tracks it, never overtakes).
COMPACT_SLIDE_NEUTRAL = 0.95
# (the teammate-crowding radius is simply divided by compactness)
STATION_BREADTH_MARGIN = 3.0  # m; never station anyone on the touchline
# Pass range: the distance a "neutral" pass is measured against. A candidate
# longer than this is favoured as the slider rises and penalised as it falls,
# so the slider trades short against long WITHOUT changing how much a team
# passes -- the reciprocal keeps the average context multiplier near 1.
PASS_RANGE_PIVOT_M = 18.0
PASS_RANGE_EXPONENT = 0.6   # how sharply preference ramps away from the pivot
# Direct pass: a pass is scaled by how far the receiver must run to it,
#     Direct ** (1 - 2 x min(1, run / DIRECT_RUN_M))
# run = his distance now to the ball's target. So at Direct 2.0: x2 to his feet,
# x1 at 10 m, x0.5 from 20 m on; below 1.0 the reciprocal, favouring passes that
# make him run; 1.0 for every pass at Direct 1.0. Graded by distance rather than
# by kind of pass, because the to-feet ball and the ball just ahead of a moving
# teammate are nearly equal options: a flat tilt between them flipped every such
# near-tie (97-99% one way even at x1.23), while here they get nearly the same
# factor and the tilt separates short balls from long ones.
DIRECT_RUN_M = 20.0
# Wing attacks: a pass to the flank is scaled by
#     wing ** (WING_TILT_GAIN x flank),  flank = 0 within WING_INNER_M of the
#     middle of the pitch's width, 1 from WING_OUTER_M out, linear between.
# One-sided: the middle keeps its utility, so at 2.0 a flank pass gains at most
# x1.23 and wins only when the two are close -- an obvious chance through the
# middle is never given up for it. Within the passes only (PASS_STAGES), so it
# picks which pass, not whether to pass rather than shoot.
WING_INNER_M   = 9.0
WING_OUTER_M   = 20.0
WING_TILT_GAIN = 0.3
STATION_DEPTH_GAIN = 0.45  # damping on the depth shift, so a high line is not goal-hanging

# ── Restart pauses (ticks at 30 Hz) ──────────────────────────────────────────
# A beat for the viewer to read what happened, and time for players to walk
# back into shape before play resumes.
KICKOFF_PAUSE_TICKS   = 90   # 3.0 s after a goal, all told:
# the first GOAL_PAUSE_TICKS with the ball resting in the net, where the viewer
# can see it went in (the kick-off used to be set up on the very tick it
# crossed the line, so the ball was never seen in the goal and the banner
# seemed to come first); the rest with both teams reformed for the kick-off.
GOAL_PAUSE_TICKS      = 45   # 1.5 s
# Every other stoppage -- throw-in, corner, goal kick, offside -- the same
# way: the restart is decided and announced at the whistle, but the ball is
# left where play stopped and nobody moves for DEAD_BALL_TICKS, so the viewer
# sees it go out before the taker appears on the spot. Before, ball and
# taker were moved on the tick it crossed the line and the message seemed to
# come first. Added on top of each restart's own pause.
DEAD_BALL_TICKS       = 20   # 0.67 s
# Through the beat the ball keeps its motion -- rolling or flying on, falling,
# stopped by the net or DEAD_BALL_LIMIT_M beyond the line -- and nobody can
# touch it. Stopped dead on the line (and dropped to the grass at once if it
# was in the air) it was never SEEN to go out: it just appeared there.
DEAD_BALL_LIMIT_M     = 4.0
DEAD_BALL_FRICTION    = 0.02  # m/tick^2 on the grass beyond the line
GOAL_NET_DEPTH        = 2.0  # m behind the line the net stops it

# ── Match length ──────────────────────────────────────────────────────────────
# 5000 ticks per half at 30 Hz = 2m47s of simulated play per half. Not real
# time: a full 45 minutes would be 81000 ticks and nothing in the model needs
# it. Teams change ends at the interval, so a first-half advantage from the
# formation or the pitch cannot persist for the whole match.
HALF_LENGTH_TICKS = 5000
HALF_TIME_PAUSE_TICKS = 90   # 3.0 s to read the interval
# The match clock is DISPLAY ONLY: 5000 ticks are shown as 45 minutes, so a
# half reads 0'00-45'00 the way football does. The simulation itself is
# unscaled -- every speed, friction and reaction constant stays in m/tick at
# 30 Hz, and nothing in the engine reads this. A literal 45 minutes would be
# 81000 ticks, roughly 10 minutes of wall clock per half at current speed.
DISPLAY_HALF_MINUTES = 45.0
GOAL_KICK_PAUSE_TICKS = 60   # 2.0 s
CORNER_PAUSE_TICKS    = 45   # 1.5 s
THROW_IN_PAUSE_TICKS  = 30   # 1.0 s

# ── Pass viability and dribbling for an angle ────────────────────────────────
PASS_VIABILITY_FLOOR  = 0.25  # below this a pass is a giveaway, not an option
PASS_UNVIABLE_PENALTY = 0.10  # floor of the ramp; a hopeless ball is crushed, not vetoed
ANGLE_GAIN_WEIGHT     = 1.6   # how much a dribble is worth for the lanes it opens
ANGLE_GAIN_MAX        = 2.2
# How far past its target a pass is weighted to run.
#
# Not a fixed distance: a constant 20 m overrun kept the ball quick but WAS the
# miss distance, and 74 of 94 lost passes overshot by more than 8 m. The limit
# is what the receiver can chase down while the ball rolls on.
PASS_CHASE_TICKS  = 20.0   # ticks (~0.7 s) a receiver has to run onto the ball
PASS_OVERRUN_MIN  = 3.0    # m; still used to judge a receiver's chase
PASS_OVERRUN_MAX  = 9.0    # m; beyond this the ball belongs to nobody
ANGLE_GAIN_NEED_BELOW = 0.45  # only dribble for an angle when you lack one already
LANE_SAFE_ENOUGH    = 0.75  # a gently weighted ball this safe needs no extra pace
# ── Pass pace: chosen as a FLIGHT TIME, not a stopping point ─────────────────
# The ball stops when someone controls it -- _give_possession zeroes its
# velocity on capture -- so a pass never needed weighting to die at the
# receiver's feet. That rule was compensating for a near-frictionless ball, and
# it forced every settling pass to be struck feebly: 10 m left the foot at
# 9 m/s and took 1.83 s, which is precisely what made short balls so easy to
# intercept. Under real rolling resistance it degenerates completely (a ball
# weighted to stop at 13 m must be struck at 3 m/s).
#
# So the passer picks how long the ball should take to ARRIVE. Real football:
# a firm ground pass covers 10 m in ~0.7 s and 25 m in ~1.5 s, i.e. roughly
# 0.06 s per metre plus a fixed strike overhead.
PASS_TICKS_BASE     = 6.0   # ticks of overhead on any pass
# 2.4 ticks/m puts 10 m at 1.07 s, 20 m at 1.87 s, 30 m at 2.67 s -- real
# ground-pass pace. At 1.8 the ball was genuinely too quick (20 m in 1.4 s, and
# 0.77 s once driven, i.e. 26 m/s): receivers could not react to a meet-point
# that moved faster than they could turn, so they looked frozen and twitching
# instead of running onto the ball.
PASS_TICKS_PER_M    = 1.9   # ticks per metre -- 20 m in ~1.5 s
# 0.70, not 0.55. A player under pressure hits it harder, he does not halve the
# flight time -- at 0.55 a threatened 20 m pass arrived in 0.77 s at 26 m/s.
PASS_TICKS_DRIVEN   = 0.70  # threatened lanes: this fraction of the flight time
PASS_SPEED_MIN      = 0.25  # m/tick; below this it is a roll, not a pass
PASS_DRIVEN_OVERRUN = 22.0  # m; the overrun accepted on a genuinely threatened lane

# ── Keeping shape: teammates must not occupy the same ground ─────────────────
# Two players in one spot act as a single player, wasting a body. Same-role
# teammates share a leash and score the same points, so they converge without
# an explicit term -- two centre-forwards were measured 0.42 m apart.
TEAMMATE_SPACING_RADIUS = 9.0   # m; inside this a teammate's ground is crowded
TEAMMATE_SPACING_FLOOR  = 0.15  # crowding is sometimes right, just never free
TEAMMATE_SPREAD_STRENGTH = 0.45  # how firmly an out-of-possession target is pushed off a teammate
SPACING_SOFTNESS = 0.35  # exponent on the in-possession spacing penalty
CONTEST_MARGIN_TICKS = 6.0  # ticks; an opponent arriving within this of you makes it a race
WAIT_SAFETY_TICKS  = 25.0  # ticks of clearance needed before letting a ball run on
WAIT_VALUE_MARGIN  = 0.25  # waiting must be worth at least this much more

# ── One-two: what the passer does after releasing ────────────────────────────
ONE_TWO_WINDOW_TICKS = 45.0  # ticks (1.5 s) a pass still counts as "just played"
# Below this survival on the ball's remaining path, the passer chases his own
# pass instead of standing off: an opponent is about to cut it out and it was
# never a safe ball. Replaces an 8-tick arrival margin that held on only 12% of
# in-flight ticks, which left the one-two active for 11% of its own window.
ONE_TWO_SAFE_ROUTE = 0.50
ONE_TWO_SAFETY_TICKS = 8.0   # a teammate must be this much closer to the ball
ONE_TWO_PULL_BASE = 0.35  # how far a passer's anchor moves toward the ball at neutral
ONE_TWO_SUPPORT_RANGE = 22.0  # m; within this of the ball counts as supporting the move
ONE_TWO_STANDOFF      = 8.0   # m; closer than this to your own pass is hovering, not supporting
ONE_TWO_CROWD_PENALTY = 0.25  # how hard hovering over your own pass is discouraged

# ── Offside ───────────────────────────────────────────────────────────────────
# A point beyond the offside line is discounted, fading with how far beyond it
# is rather than snapping at the line. A hard cliff made forwards oscillate
# across the threshold; the gradient lets a striker hold the last defender's
# shoulder, which is what the position actually is.
OFFSIDE_FADE_DISTANCE = 8.0   # m beyond the line at which the discount bottoms out
OFFSIDE_MIN_FACTOR    = 0.05  # what a hopelessly offside point is worth
OFFSIDE_PAUSE_TICKS   = 60    # 2 s; the flag goes up and play stops

# ── Forward runs: supporting a break ─────────────────────────────────────────
# Ordinary play advances the ball ~0.16 m/tick (median); the 95th percentile is
# 0.86. The threshold sits well above the median so this fires on deep passes
# and carries at pace, not on shuffling.
ATTACK_CALL_MIN_RATE  = 0.45   # m/tick of advance before a break counts
ATTACK_CALL_FULL_RATE = 0.90   # m/tick at which it is fully on
ATTACK_RUN_RANGE     = 20.0   # m of upfield gain that earns the full reward
# ── Attack call: the carrier summons teammates forward ────────────────────────
# The parameter is the CARRIER's willingness to call runs, not a runner's own
# eagerness. The old per-player "forward run" reward could never work: it only
# reweighted candidate points inside the Gaussian shape leash, so at maximum
# setting the net score peaked at 6 m of gain and a genuine 20 m run scored
# WORSE than standing still (0.70 vs 1.00). Measured effect on where players
# actually stood: teammates ahead of the ball during a break 2.21 vs 2.23
# otherwise -- literally nothing.
ATTACK_CALL_MAX_TICKS   = 300.0  # 10 s safety valve; possession loss is the real end
# Same idea for the called run: the runner keeps going while the ball is loose
# and stops the moment anyone owns it, so this is only a safety valve for a
# ball that never resolves cleanly. Shorter than the attack call because it
# covers one pass in flight (12-28 m), not a whole attacking move.
RUNWAY_SIGNAL_MAX_TICKS = 120.0  # 4 s; ball ownership is the real end
# A called spot must be OURS to reach. Nothing in find_signal_target compared
# our runner against the opponents: promptness raced him against the BALL's
# flight time, and space_quality only asked how far the nearest opponent stood
# from the spot right now. So a spot their defender reached in 8 ticks and our
# runner in 20 scored exactly like one we owned outright, and the runner was
# sent into a race he had already lost -- arrival 19-30%, and once signals
# persisted and he actually went, called runs kept the ball 12-19% against 45-53%
# for ordinary passes. This margin is how much sooner our man must get there;
# beating them by a single tick still means controlling it under contact.
SIGNAL_RACE_MARGIN_TICKS = 6.0
# How close to the called spot counts as "arrived", after which the runner
# stops aiming at the point and starts tracking the ball itself. Generous
# rather than tight: the spot is where the SPACE is, and the pass only has to
# land near it, so insisting on the exact coordinate is what left him standing
# still with the ball 3.6 m away.
RUNWAY_MEET_RADIUS = 6.0
# How far the meeting point must move before a runner re-aims. Below this he
# keeps running at the point he committed to: re-solving every tick against a
# decelerating ball swung the target a mean 28 m per tick and left him
# shivering in place instead of covering ground.
MEET_TARGET_HYSTERESIS = 2.0
ATTACK_CALL_RADIUS      = 30.0   # m; how far the carrier's voice carries
ATTACK_CALL_BASE_RUNNERS = 2.0   # runners called at a neutral 1.0x setting
ATTACK_CALL_MIN_ROOM    = 6.0    # m of upfield room before a run is worth making
# A carrier who already has good outlets does not need to call anyone; he may
# still do so to put more bodies in the box, which is what the slider buys.
ATTACK_CALL_SATED_SAFETY = 0.75  # best pass safety above which calls are damped
# ── Attack call: the carrier judges the runs (decisions D-13, D-14; signals.JOIN)
# Every JOIN_REJUDGE_TICKS while he owns the ball, the carrier judges each
# outfield teammate's possible runs -- JOIN_RUN_ANGLES (degrees off straight
# at their goal) x JOIN_RUN_LENGTHS m, plus a run in behind to the offside
# line -- by the worth of the spot, if it beats where he is now:
#     worth = lane from the ball now x position value x space
#     gain  = worth(spot) x P(he gets there first) - worth(here)
# and sends the best: up to int(JOIN_PER_SETTING x his Attack call + 0.5).
#     score = worth(spot) x P(first) x (1 - risk) ** (JOIN_RISK_EXPONENT / Attack call)
#             / (1 + ticks to arrive / JOIN_REACH_TAU),   only if gain > JOIN_MIN_GAIN
#     risk  = danger x shortage: danger = min(1, threat where he stands / threat
#             JOIN_DANGER_REF_M from our goal); shortage = the larger of
#             1 - teammates staying deeper than him / JOIN_REST_DEFENCE
#             and, in our half, 1 - those of them within JOIN_COVER_RADIUS of him / JOIN_LOCAL_COVER.
#             Deeper than HIM, not goal-side of the ball: judged against the
#             ball, a centre-back beside the carrier in the build-up counted as
#             no cover; judged by the whole back line, a full-back with nobody
#             on his flank counted as no risk and took about 90% of the runs.
# A current run keeps (1 + JOIN_STICKY) on its score and its spot, so runs do
# not flicker. A run ends when he gets the ball; after JOIN_WAIT_TICKS within
# JOIN_ARRIVE_M of his spot without it (the attack has stalled); when his gap to
# the spot has not shrunk by JOIN_CLOSE_MIN_M for JOIN_CLOSE_TICKS (he cannot
# make it); when the carrier drops him; or when we lose the ball. After the
# first three he may not be called for JOIN_COOLDOWN_TICKS.
JOIN_PER_SETTING = 2.0
JOIN_REJUDGE_TICKS = 6
JOIN_RUN_ANGLES = (0.0, 30.0, -30.0, 60.0, -60.0)
JOIN_RUN_LENGTHS = (6.0, 10.0, 18.0)
# The race to the spot as a chance, not a gate: P(he gets there first) =
# 1 / (1 + exp((his ticks - the first opponent's ticks) / JOIN_RACE_TAU)),
# 0.5 for an even race; runs below JOIN_MIN_RACE are not offered. As a hard
# gate (6 ticks clear of every defender) a 10-18 m run against a marker almost
# never qualified: 1.1 runs at 2.0 against a cap of 4.
JOIN_RACE_TAU = 8.0
JOIN_MIN_RACE = 0.05
JOIN_MIN_GAIN = 0.005
JOIN_REACH_TAU = 60.0
JOIN_RISK_EXPONENT = 2.0
JOIN_DANGER_REF_M = 30.0
JOIN_REST_DEFENCE = 4
JOIN_COVER_RADIUS = 20.0
JOIN_LOCAL_COVER = 2
JOIN_STICKY = 0.25
JOIN_ARRIVE_M = 3.0
JOIN_WAIT_TICKS = 60
JOIN_COOLDOWN_TICKS = 45
JOIN_CLOSE_TICKS = 60
JOIN_CLOSE_MIN_M = 0.5
ATTACK_RUN_BASE      = 0.40   # how strongly a neutral setting pushes players on
WAIT_STRIDE_TICKS  = 5    # sample the ball trajectory every N ticks, not every one
WAIT_HORIZON_TICKS = 45   # ticks past the earliest touch worth considering
