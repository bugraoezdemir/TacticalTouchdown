import numpy as np
import math
import random
from enum import Enum
from dataclasses import dataclass
from typing import Optional

FIELD_WIDTH = 100.0
FIELD_HEIGHT = 100.0

# Goal dimensions (y coordinates for goal posts)
GOAL_TOP = 40.0
GOAL_BOTTOM = 60.0

# Physics constants
PLAYER_SPEED = 0.5
PLAYER_SPRINT_SPEED = 0.7
GK_SPEED = 0.6  # Goalkeepers move at 2/3 speed
GK_SPRINT_SPEED = 0.8  # GK sprint at 2/3 speed
GK_DIVE_REACH = 10.0  # How far GK can dive to save
BALL_PASS_SPEED = 2.8  # Faster passes for quicker play
BALL_SHOOT_SPEED = 4.5  # At least 2x faster than passes
BALL_DRIBBLE_SPEED = 0.6
TACKLE_DISTANCE = 3.5  # Increased to catch fast-moving balls (ball speed 2.8/tick)
BALL_FRICTION = 0.95

# Probabilistic decision parameters
DECISION_RANDOMNESS = 0.15  # How much randomness in decisions (0-1)
PASS_SELECTION_RANDOMNESS = 0.25  # Randomness in pass target selection

# Decision weights - forward-focused passing
PASS_SAFETY_WEIGHT = 0.35      # Reduced - balance with progress
PASS_GOAL_PROGRESS_WEIGHT = 0.45 # Increased - prioritize forward movement
PASS_DISTANCE_WEIGHT = 0.2
SPACE_PASS_BONUS = 0.15  # Re-enabled for space passes
FORWARD_PASS_BONUS = 0.2  # Bonus for passes that significantly advance the ball

DRIBBLE_CLEARANCE_WEIGHT = 0.35  # Reduced - dribble less
DRIBBLE_GOAL_PROGRESS_WEIGHT = 0.35  # Reduced - favor passing

SHOOT_DISTANCE_THRESHOLD = 35.0  # Increased from 25 - shoot from further
SHOOT_ANGLE_THRESHOLD = 30.0

# Action thresholds - balanced for good passing frequency
SHOOT_SCORE_THRESHOLD = 0.10  # Very low threshold - shoot frequently when chance arises
PASS_SCORE_THRESHOLD = 0.18   # Balanced - safe passes but not too restrictive

# Pass evaluation - no artificial bonus, safety-first approach
PASS_BONUS = 0.0  # Removed - let natural safety scoring decide

# Home position attraction weight - lower for more dynamic positioning
HOME_POSITION_WEIGHT = 0.15  # 15% home bias, 85% tactical - players help attack/defense

# Dribble touch system
DRIBBLE_TOUCH_INTERVAL = 5  # Every N ticks, ball moves ahead
DRIBBLE_TOUCH_DISTANCE = 2.5  # How far ball moves ahead (must be > TACKLE_DISTANCE to allow interception)

# Vision system constants
VISION_ANGLE = 140.0  # Degrees - realistic peripheral vision
VISION_DISTANCE = 50.0  # Can see teammates this far ahead
BACK_VISION_DISTANCE = 25.0  # Limited rear vision

# TACTICAL SYSTEM

# Formation templates - defines player positions for each formation
# Format: formation_name -> list of (role, lateral_role, longitudinal_role, home_x, home_y, jersey_number)
FORMATIONS = {
    '4-4-2': [
        ('GK', 'center', 'goalkeeper', 5, 50, 1),
        ('DEF', 'left', 'back', 18, 15, 2),
        ('DEF', 'center_left', 'back', 18, 38, 4),
        ('DEF', 'center_right', 'back', 18, 62, 5),
        ('DEF', 'right', 'back', 18, 85, 3),
        ('MID', 'left', 'midfielder', 40, 15, 11),
        ('MID', 'center_left', 'midfielder', 42, 38, 8),
        ('MID', 'center_right', 'midfielder', 42, 62, 6),
        ('MID', 'right', 'midfielder', 40, 85, 7),
        ('FWD', 'left', 'forward', 65, 35, 9),
        ('FWD', 'right', 'forward', 65, 65, 10),
    ],
    '4-3-3': [
        ('GK', 'center', 'goalkeeper', 5, 50, 1),
        ('DEF', 'left', 'back', 18, 15, 2),
        ('DEF', 'center_left', 'back', 18, 38, 4),
        ('DEF', 'center_right', 'back', 18, 62, 5),
        ('DEF', 'right', 'back', 18, 85, 3),
        ('MID', 'left', 'midfielder', 42, 25, 8),
        ('MID', 'center', 'midfielder', 45, 50, 6),
        ('MID', 'right', 'midfielder', 42, 75, 7),
        ('FWD', 'left', 'forward', 65, 20, 11),
        ('FWD', 'center', 'forward', 70, 50, 9),
        ('FWD', 'right', 'forward', 65, 80, 10),
    ],
    '3-5-2': [
        ('GK', 'center', 'goalkeeper', 5, 50, 1),
        ('DEF', 'left', 'back', 18, 25, 4),
        ('DEF', 'center', 'back', 15, 50, 5),
        ('DEF', 'right', 'back', 18, 75, 2),
        ('MID', 'left', 'midfielder', 35, 10, 11),
        ('MID', 'center_left', 'midfielder', 42, 35, 8),
        ('MID', 'center', 'midfielder', 45, 50, 6),
        ('MID', 'center_right', 'midfielder', 42, 65, 7),
        ('MID', 'right', 'midfielder', 35, 90, 3),
        ('FWD', 'left', 'forward', 65, 40, 9),
        ('FWD', 'right', 'forward', 65, 60, 10),
    ],
    '5-3-2': [
        ('GK', 'center', 'goalkeeper', 5, 50, 1),
        ('DEF', 'left', 'back', 20, 10, 11),
        ('DEF', 'center_left', 'back', 18, 30, 4),
        ('DEF', 'center', 'back', 15, 50, 5),
        ('DEF', 'center_right', 'back', 18, 70, 2),
        ('DEF', 'right', 'back', 20, 90, 3),
        ('MID', 'left', 'midfielder', 42, 30, 8),
        ('MID', 'center', 'midfielder', 45, 50, 6),
        ('MID', 'right', 'midfielder', 42, 70, 7),
        ('FWD', 'left', 'forward', 65, 40, 9),
        ('FWD', 'right', 'forward', 65, 60, 10),
    ],
}

# Mentality offsets - how much to shift player positions
# Positive = more attacking (shift toward opponent goal)
# Negative = more defensive (shift toward own goal)
MENTALITY_OFFSETS = {
    'defensive': -10,
    'normal': 0,
    'offensive': 10,
}


class GameState(Enum):
    PLAYING = "playing"
    GOAL_SCORED = "goal_scored"
    CORNER_KICK = "corner_kick"
    GOAL_KICK = "goal_kick"
    THROW_IN = "throw_in"
    KICKOFF = "kickoff"


@dataclass
class LastTouch:
    team: str
    player_id: int


# Home positions for each role combination (11v11 4-4-2 formation)
# Format: (lateral_role, longitudinal_role, team) -> (x, y)
ROLE_HOME_POSITIONS = {
    # Home team (attacking right toward x=100) - 4-4-2 formation
    ('center', 'goalkeeper', 'home'): (5, 50),
    ('left', 'back', 'home'): (18, 15),
    ('center_left', 'back', 'home'): (18, 38),
    ('center_right', 'back', 'home'): (18, 62),
    ('right', 'back', 'home'): (18, 85),
    ('left', 'midfielder', 'home'): (40, 15),
    ('center_left', 'midfielder', 'home'): (42, 38),
    ('center_right', 'midfielder', 'home'): (42, 62),
    ('right', 'midfielder', 'home'): (40, 85),
    ('left', 'forward', 'home'): (65, 35),
    ('right', 'forward', 'home'): (65, 65),
    
    # Away team (attacking left toward x=0) - 4-4-2 formation
    ('center', 'goalkeeper', 'away'): (95, 50),
    ('left', 'back', 'away'): (82, 85),
    ('center_left', 'back', 'away'): (82, 62),
    ('center_right', 'back', 'away'): (82, 38),
    ('right', 'back', 'away'): (82, 15),
    ('left', 'midfielder', 'away'): (60, 85),
    ('center_left', 'midfielder', 'away'): (58, 62),
    ('center_right', 'midfielder', 'away'): (58, 38),
    ('right', 'midfielder', 'away'): (60, 15),
    ('left', 'forward', 'away'): (35, 65),
    ('right', 'forward', 'away'): (35, 35),
}

# Zone bounds for each role - (x_min, x_max, y_min, y_max)
# Players should stay mostly within their zone (11v11 4-4-2)
ROLE_ZONE_BOUNDS = {
    # Home team zones (attacking toward x=100)
    ('center', 'goalkeeper', 'home'): (0, 20, 30, 70),
    ('left', 'back', 'home'): (5, 45, 0, 30),
    ('center_left', 'back', 'home'): (5, 45, 20, 50),
    ('center_right', 'back', 'home'): (5, 45, 50, 80),
    ('right', 'back', 'home'): (5, 45, 70, 100),
    ('left', 'midfielder', 'home'): (25, 75, 0, 30),
    ('center_left', 'midfielder', 'home'): (25, 75, 25, 55),
    ('center_right', 'midfielder', 'home'): (25, 75, 45, 75),
    ('right', 'midfielder', 'home'): (25, 75, 70, 100),
    ('left', 'forward', 'home'): (40, 100, 15, 55),
    ('right', 'forward', 'home'): (40, 100, 45, 85),
    
    # Away team zones (attacking toward x=0)
    ('center', 'goalkeeper', 'away'): (80, 100, 30, 70),
    ('left', 'back', 'away'): (55, 95, 70, 100),
    ('center_left', 'back', 'away'): (55, 95, 50, 80),
    ('center_right', 'back', 'away'): (55, 95, 20, 50),
    ('right', 'back', 'away'): (55, 95, 0, 30),
    ('left', 'midfielder', 'away'): (25, 75, 70, 100),
    ('center_left', 'midfielder', 'away'): (25, 75, 45, 75),
    ('center_right', 'midfielder', 'away'): (25, 75, 25, 55),
    ('right', 'midfielder', 'away'): (25, 75, 0, 30),
    ('left', 'forward', 'away'): (0, 60, 45, 85),
    ('right', 'forward', 'away'): (0, 60, 15, 55),
}

# Zone weight by role - how strongly to enforce staying in zone
ZONE_WEIGHT_BY_ROLE = {
    'GK': 1.0,   # GK must stay in zone
    'DEF': 0.7,  # Defenders fairly strict
    'MID': 0.4,  # Midfielders more flexible
    'FWD': 0.3,  # Forwards most free to roam
}


def normalize(v):
    """Normalize a vector, return zero vector if magnitude is zero."""
    mag = np.linalg.norm(v)
    if mag < 1e-6:
        return np.zeros_like(v)
    return v / mag

def is_in_vision_cone(observer_pos, observer_facing, target_pos, 
                      max_angle=VISION_ANGLE, max_distance=VISION_DISTANCE):
    """Check if target is within observer's vision cone."""
    to_target = target_pos - observer_pos
    dist = np.linalg.norm(to_target)
    
    if dist < 1e-6 or dist > max_distance:
        return False, float(dist), 180.0
    
    to_target_normalized = to_target / dist
    angle = math.degrees(math.acos(np.clip(
        np.dot(observer_facing, to_target_normalized), -1.0, 1.0)))
    
    return angle <= max_angle / 2.0, float(dist), float(angle)

def calculate_passing_lane_quality(passer_pos, target_pos, opponents):
    """Calculate quality of passing lane - 0 (blocked) to 1 (clear)."""
    corridor_width = 3.0
    blocking_opponents = 0
    
    for opp_pos in opponents:
        dist_to_line = distance_point_to_segment(opp_pos, passer_pos, target_pos)
        to_opp = opp_pos - passer_pos
        pass_vec = target_pos - passer_pos
        proj = np.dot(to_opp, pass_vec) / max(np.linalg.norm(pass_vec), 1e-6)
        
        if 0 < proj < np.linalg.norm(pass_vec) and dist_to_line < corridor_width:
            blocking_opponents += 1
    
    if blocking_opponents == 0:
        return 1.0
    return max(0.0, 1.0 / (1.0 + blocking_opponents * 0.5))

def project_point_to_segment(point, seg_start, seg_end):
    """Project a point onto a line segment, return closest point on segment."""
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    if seg_len_sq < 1e-6:
        return seg_start.copy()
    t = np.clip(np.dot(point - seg_start, seg_vec) / seg_len_sq, 0, 1)
    return seg_start + t * seg_vec

def distance_point_to_segment(point, seg_start, seg_end):
    """Calculate minimum distance from point to line segment."""
    closest = project_point_to_segment(point, seg_start, seg_end)
    return np.linalg.norm(point - closest)

def time_to_intercept(interceptor_pos, interceptor_speed, ball_start, ball_end, ball_speed):
    """
    Calculate if an interceptor can reach the ball path before the ball arrives.
    Returns (can_intercept: bool, time_margin: float)
    Negative time_margin means interceptor arrives before ball (dangerous).
    """
    ball_vec = ball_end - ball_start
    ball_dist = np.linalg.norm(ball_vec)
    if ball_dist < 1e-6:
        return False, float('inf')
    
    ball_travel_time = ball_dist / ball_speed
    
    closest_point = project_point_to_segment(interceptor_pos, ball_start, ball_end)
    intercept_dist = np.linalg.norm(closest_point - interceptor_pos)
    intercept_time = intercept_dist / interceptor_speed
    
    ball_to_closest = np.linalg.norm(closest_point - ball_start)
    ball_time_to_closest = ball_to_closest / ball_speed
    
    time_margin = ball_time_to_closest - intercept_time
    can_intercept = time_margin < 0.5
    
    return can_intercept, time_margin

def clearance_in_direction(pos, direction, opponents, max_distance=20.0):
    """
    Calculate how far a player can move in a direction before hitting an opponent.
    Returns clearance distance (capped at max_distance).
    """
    if len(opponents) == 0:
        return max_distance
    
    dir_normalized = normalize(direction)
    if np.linalg.norm(dir_normalized) < 1e-6:
        return 0.0
    
    min_clearance = max_distance
    
    for opp_pos in opponents:
        to_opp = opp_pos - pos
        proj_dist = np.dot(to_opp, dir_normalized)
        
        if proj_dist <= 0:
            continue
        
        perp_dist = np.linalg.norm(to_opp - proj_dist * dir_normalized)
        
        if perp_dist < 3.0:
            effective_clearance = proj_dist - (3.0 - perp_dist)
            min_clearance = min(min_clearance, max(0.0, float(effective_clearance)))
    
    return min_clearance

def angle_to_goal(pos, team):
    """Calculate angle to the center of the goal."""
    goal_x = 100.0 if team == 'home' else 0.0
    goal_center = np.array([goal_x, 50.0])
    to_goal = goal_center - pos
    return math.atan2(to_goal[1], to_goal[0])

def distance_to_goal(pos, team):
    """Calculate distance to goal."""
    goal_x = 100.0 if team == 'home' else 0.0
    return abs(goal_x - pos[0])


class DecisionContext:
    """Caches useful information for decision-making."""
    
    def __init__(self, player, game):
        self.player = player
        self.game = game
        self.team = player.team
        
        self.teammates = []
        self.teammate_positions = []
        self.opponents = []
        self.opponent_positions = []
        
        for p in game.players:
            if p.id == player.id:
                continue
            if p.team == player.team:
                self.teammates.append(p)
                self.teammate_positions.append(p.pos.copy())
            else:
                self.opponents.append(p)
                self.opponent_positions.append(p.pos.copy())
        
        self.teammate_positions = np.array(self.teammate_positions) if self.teammate_positions else np.zeros((0, 2))
        self.opponent_positions = np.array(self.opponent_positions) if self.opponent_positions else np.zeros((0, 2))
        
        self.goal_x = 100.0 if self.team == 'home' else 0.0
        self.goal_center = np.array([self.goal_x, 50.0])
        self.dist_to_goal = distance_to_goal(player.pos, self.team)
        
        # Calculate facing direction for vision system
        if np.linalg.norm(player.vel) > 0.1:
            self.facing = normalize(player.vel)
        else:
            self.facing = normalize(self.goal_center - player.pos)


class Player:
    def __init__(self, id, team, role, x, y, number, lateral_role='center', longitudinal_role='midfielder'):
        self.id = id
        self.team = team
        self.role = role
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([0.0, 0.0])
        self.has_ball = False
        self.number = number
        self.lateral_role = lateral_role
        self.longitudinal_role = longitudinal_role
        
        # Set home position from role mapping
        role_key = (lateral_role, longitudinal_role, team)
        if role_key in ROLE_HOME_POSITIONS:
            home = ROLE_HOME_POSITIONS[role_key]
            self.home_pos = np.array([float(home[0]), float(home[1])])
        else:
            self.home_pos = self.pos.copy()
        
        # Set zone bounds
        if role_key in ROLE_ZONE_BOUNDS:
            self.zone_bounds = ROLE_ZONE_BOUNDS[role_key]
        else:
            self.zone_bounds = (0, 100, 0, 100)  # Full field as fallback
        
        # Zone weight based on role
        self.zone_weight = ZONE_WEIGHT_BY_ROLE.get(role, 0.5)
        
        # Dribble touch cooldown - prevents instant reacquisition after dribbling
        self.touch_cooldown_until = 0.0
        
        # Track who passed to this player (to avoid immediate return passes)
        self.received_from_player_id: Optional[int] = None
        
        # Track if player is currently in a dribble sequence (for re-evaluation)
        self.is_dribbling = False
        self.dribble_touches = 0  # Count touches in current dribble sequence
        self.last_dribble_dir = None  # Last dribble direction for continuity
    
    def to_dict(self):
        return {
            "id": self.id,
            "team": self.team,
            "role": self.role,
            "x": float(self.pos[0]),
            "y": float(self.pos[1]),
            "vx": float(self.vel[0]),
            "vy": float(self.vel[1]),
            "hasBall": self.has_ball,
            "number": self.number,
            "lateralRole": self.lateral_role,
            "longitudinalRole": self.longitudinal_role
        }

    def make_decision(self, game):
        """Main decision-making method using the improved AI system."""
        ctx = DecisionContext(self, game)
        
        if self.has_ball:
            self._make_ball_decision(ctx, game)
        else:
            self._make_off_ball_decision(ctx, game)
        
        self.pos += self.vel
        self.pos[0] = np.clip(self.pos[0], 0, FIELD_WIDTH)
        self.pos[1] = np.clip(self.pos[1], 0, FIELD_HEIGHT)

    def _make_ball_decision(self, ctx, game):
        """Decision making when player has the ball."""
        
        # DRIBBLE RE-EVALUATION: Smart decision making during dribble sequence
        if self.is_dribbling and self.dribble_touches > 0:
            # Evaluate current situation
            shoot_score = self._evaluate_shoot(ctx)
            pass_option, pass_target, pass_score = self._evaluate_pass_while_dribbling(ctx)
            new_dribble_dir, dribble_score = self._evaluate_dribble(ctx)
            
            # Apply tactical multipliers
            if self.team == 'home':
                shoot_score *= game.home_shoot_frequency
                dribble_score *= game.home_dribble_frequency
            
            # Calculate clearance for the NEW direction (always available)
            new_clearance = clearance_in_direction(self.pos, new_dribble_dir, ctx.opponent_positions)
            
            # Check current trajectory clearance (use new direction if no last direction)
            if self.last_dribble_dir is not None:
                current_clearance = clearance_in_direction(self.pos, self.last_dribble_dir, ctx.opponent_positions)
            else:
                current_clearance = new_clearance  # First touch - use new direction
            
            # Pressure check - how close is nearest opponent?
            min_opp_dist = float('inf')
            for opp_pos in ctx.opponent_positions:
                d = np.linalg.norm(opp_pos - self.pos)
                min_opp_dist = min(min_opp_dist, float(d))
            
            # DECISION PRIORITY during dribble:
            # 1. SHOOT if good opportunity (lower threshold when dribbling)
            if shoot_score > 0.10 and ctx.dist_to_goal < 30.0:
                self._clear_dribble_state()
                self._execute_shoot(ctx, game)
                return
            
            # 2. PASS if good target found AND (under pressure OR trajectory blocked OR 2+ touches)
            should_pass = False
            if pass_option is not None and pass_score > 0.15:
                if min_opp_dist < 10.0:  # Under pressure
                    should_pass = True
                elif current_clearance < 8.0 and new_clearance < 10.0:  # No good path
                    should_pass = True
                elif self.dribble_touches >= 2:  # Don't hog the ball
                    should_pass = True
            
            if should_pass:
                self._clear_dribble_state()
                self._execute_pass(ctx, game, pass_option, pass_target)
                return
            
            # 3. CHANGE DIRECTION if current path is bad but new path is better
            if self.last_dribble_dir is not None and current_clearance < 12.0 and new_clearance > current_clearance + 2.0:
                # Better direction found - take it
                self._execute_dribble(ctx, game, new_dribble_dir)
                return
            
            # 4. FORCE RELEASE after 3 touches - no more dribbling
            if self.dribble_touches >= 3:
                self._clear_dribble_state()
                if pass_option is not None:
                    self._execute_pass(ctx, game, pass_option, pass_target)
                elif shoot_score > 0.05 and ctx.dist_to_goal < 40.0:
                    self._execute_shoot(ctx, game)
                else:
                    self._execute_clearance(ctx, game)
                return
            
            # 5. CONTINUE DRIBBLING if good path available (use new_clearance for first touch)
            if dribble_score > 0.3 and new_clearance > 8.0:
                self._execute_dribble(ctx, game, new_dribble_dir)
                return
            
            # 6. FALLBACK: pass or shoot
            if pass_option is not None:
                self._clear_dribble_state()
                self._execute_pass(ctx, game, pass_option, pass_target)
                return
            elif shoot_score > 0.05:
                self._clear_dribble_state()
                self._execute_shoot(ctx, game)
                return
        
        # GK behavior: pass when safe, throw to corner when under pressure
        if self.role == 'GK':
            # Check pressure level
            nearby_opponents = sum(1 for opp_pos in ctx.opponent_positions 
                                   if np.linalg.norm(opp_pos - self.pos) < 20.0)
            under_pressure = nearby_opponents >= 1
            
            pass_option, pass_target, pass_score = self._evaluate_pass(ctx)
            
            if not under_pressure and pass_score >= 0.15 and pass_option is not None:
                self._execute_pass(ctx, game, pass_option, pass_target)
            else:
                # Under pressure or no safe pass - throw to corner
                self._execute_gk_throw_to_corner(ctx, game)
            return
        
        # Evaluate all options
        shoot_score = self._evaluate_shoot(ctx)
        pass_option, pass_target, pass_score = self._evaluate_pass(ctx)
        dribble_dir, dribble_score = self._evaluate_dribble(ctx)
        
        # Apply tactical frequency multipliers for home team
        if self.team == 'home':
            shoot_score *= game.home_shoot_frequency
            dribble_score *= game.home_dribble_frequency
        
        # DISTANCE-BASED BEHAVIOR: closer to goal = more aggressive
        # Zone 1: Danger zone (< 20 units) - shoot first, dribble second
        # Zone 2: Attack zone (20-40 units) - balanced, slight dribble preference
        # Zone 3: Midfield (40-60 units) - balanced passing and movement
        # Zone 4: Build-up (> 60 units) - pass-focused, safe play
        
        dist = ctx.dist_to_goal
        
        # Calculate aggression multiplier based on distance (closer = higher)
        if dist < 20:
            shoot_boost = 1.5
            dribble_boost = 1.3
            pass_reduction = 0.7
        elif dist < 40:
            shoot_boost = 1.2
            dribble_boost = 1.1
            pass_reduction = 0.85
        elif dist < 60:
            shoot_boost = 1.0
            dribble_boost = 0.8
            pass_reduction = 1.0
        else:
            shoot_boost = 0.5
            dribble_boost = 0.5
            pass_reduction = 1.2
        
        # Apply distance-based modifiers
        shoot_score *= shoot_boost
        dribble_score *= dribble_boost
        pass_score *= pass_reduction
        
        # Check pressure level for clearance decision
        min_opp_dist = float('inf')
        for opp_pos in ctx.opponent_positions:
            d = np.linalg.norm(opp_pos - self.pos)
            min_opp_dist = min(min_opp_dist, float(d))
        
        # Dribbling allowed in attacking half (< 50 units from goal)
        # OR for midfielders/forwards with empty space ahead (and not too far from goal)
        is_attacking_player = self.role in ['MID', 'FWD']
        goal_dir = ctx.goal_center - self.pos
        goal_dir_norm = np.linalg.norm(goal_dir)
        if goal_dir_norm > 0.1:
            clearance_ahead = clearance_in_direction(self.pos, goal_dir / goal_dir_norm, ctx.opponent_positions)
        else:
            clearance_ahead = 0.0
        has_space_ahead = clearance_ahead > 15.0 and dist < 70.0  # Must have good space and not too far back
        
        # All attacking players can dribble when they have space
        # But AVOID dribbling when no space is available (would run into opponents)
        # Also avoid dribbling when opponents are too close - pass instead!
        under_pressure = min_opp_dist < 8.0  # Opponent within 8 units = under pressure
        
        if under_pressure:
            can_dribble = False  # Under pressure - must pass, don't dribble
        elif is_attacking_player:
            if has_space_ahead:
                can_dribble = True
            elif dist < 50.0 and clearance_ahead > 8.0:
                # In attacking half with some space
                can_dribble = True
            else:
                can_dribble = False  # No space - don't dribble into opponents
        else:
            can_dribble = dist < 50.0  # Defenders/GK use simple distance check
        
        # Check for hot teammate (someone with better shooting chance)
        hot_teammate, hot_opportunity = self._find_hot_teammate(ctx)
        
        # Evaluate runway (lead) passes for MID and FWD
        runway_target, runway_score = None, 0.0
        runway_teammate = None
        if self.role in ['MID', 'FWD']:
            for teammate in ctx.teammates:
                target, score = self._evaluate_runway_pass(ctx, teammate)
                if score > runway_score:
                    runway_score = score
                    runway_target = target
                    runway_teammate = teammate
        
        # PROBABILISTIC DECISION-MAKING: Add randomness to thresholds
        rand_factor = random.uniform(1.0 - DECISION_RANDOMNESS, 1.0 + DECISION_RANDOMNESS)
        
        # Randomize scores slightly for varied behavior
        shoot_score_adj = shoot_score * random.uniform(0.85, 1.15)
        dribble_score_adj = dribble_score * random.uniform(0.85, 1.15)
        pass_score_adj = pass_score * random.uniform(0.85, 1.15)
        runway_score_adj = runway_score * random.uniform(0.9, 1.1)
        
        # DANGER ZONE: Shoot first, then pass to HOT teammate only, dribble last
        if dist < 20:
            if shoot_score_adj > 0.12 * rand_factor:
                self._execute_shoot(ctx, game)
            elif hot_teammate is not None and hot_opportunity > 0.3:
                # Pass ONLY to hot teammate (near goal with scoring chance)
                self._execute_pass(ctx, game, hot_teammate, hot_teammate.pos)
            elif dribble_score_adj > 0.15 * rand_factor and can_dribble:
                self._execute_dribble(ctx, game, dribble_dir)
            else:
                self._execute_shoot(ctx, game)  # Force a shot when close
        # ATTACK ZONE: Shoot > pass to HOT teammate > winger cross > dribble
        elif dist < 40:
            # WINGER SPECIAL: Near sideline, consider dribbling down line or crossing
            is_winger = self.role in ['MID', 'FWD'] and self.lateral_role in ['left', 'right']
            near_sideline = (self.lateral_role == 'left' and self.pos[1] < 15) or \
                           (self.lateral_role == 'right' and self.pos[1] > 85)
            in_final_third = (ctx.team == 'home' and self.pos[0] > 70) or \
                            (ctx.team == 'away' and self.pos[0] < 30)
            
            # Check for clearance along the byline before winger special
            byline_dir = np.array([1.0 if ctx.team == 'home' else -1.0, 0.0])
            byline_clearance = clearance_in_direction(self.pos, byline_dir, ctx.opponent_positions)
            has_teammates_in_box = any(
                (ctx.team == 'home' and t.pos[0] > 75 and 35 < t.pos[1] < 65) or
                (ctx.team == 'away' and t.pos[0] < 25 and 35 < t.pos[1] < 65)
                for t in ctx.teammates if t.role != 'GK'
            )
            
            winger_action_taken = False
            if is_winger and near_sideline and in_final_third and (byline_clearance > 8.0 or has_teammates_in_box):
                # Winger near byline with space or teammates to cross to
                if byline_clearance > 10.0 and random.random() < 0.5:
                    # Dribble along the sideline toward goal line
                    self._execute_dribble(ctx, game, byline_dir)
                    winger_action_taken = True
                elif has_teammates_in_box:
                    # Cross into the box
                    self._execute_cross(ctx, game)
                    winger_action_taken = True
            
            if winger_action_taken:
                return
            elif shoot_score_adj > 0.18 * rand_factor:
                self._execute_shoot(ctx, game)
            elif runway_score_adj > 0.30 * rand_factor and runway_target is not None:
                # Execute through ball to create scoring opportunity
                self._execute_runway_pass(ctx, game, runway_teammate, runway_target)
            elif hot_teammate is not None and hot_opportunity > 0.3:
                # Pass ONLY to hot teammate (near goal with scoring chance)
                self._execute_pass(ctx, game, hot_teammate, hot_teammate.pos)
            elif dribble_score_adj > 0.20 * rand_factor and can_dribble:
                # Dribble if no hot teammate available
                self._execute_dribble(ctx, game, dribble_dir)
            elif pass_option is not None:
                # Fall back to normal pass only if can't dribble
                self._execute_pass(ctx, game, pass_option, pass_target)
            else:
                self._execute_dribble(ctx, game, dribble_dir)
        # MIDFIELD/BUILD-UP: Pass-focused with runway option, but allow dribble with space
        else:
            if runway_score_adj > 0.28 * rand_factor and runway_target is not None and self.role in ['MID', 'FWD']:
                # Execute through ball to start attack
                self._execute_runway_pass(ctx, game, runway_teammate, runway_target)
            elif is_attacking_player and has_space_ahead and dribble_score_adj > 0.25 * rand_factor and can_dribble:
                # Midfielders/forwards can dribble forward into empty space
                self._execute_dribble(ctx, game, dribble_dir)
            elif pass_score_adj >= 0.08 and pass_option is not None:
                self._execute_pass(ctx, game, pass_option, pass_target)
            elif min_opp_dist < 5.0 and pass_score < 0.1:
                self._execute_clearance(ctx, game)
            elif pass_option is not None:
                self._execute_pass(ctx, game, pass_option, pass_target)
            else:
                self._execute_clearance(ctx, game)

    def _evaluate_shoot(self, ctx):
        """Evaluate shooting option. Returns score 0-1."""
        dist = ctx.dist_to_goal
        
        if dist > SHOOT_DISTANCE_THRESHOLD:
            return 0.0
        
        # Check how many opponents block the shot
        blocking_count = 0
        for opp_pos in ctx.opponent_positions:
            if distance_point_to_segment(opp_pos, self.pos, ctx.goal_center) < 4.0:
                blocking_count += 1
        
        # Distance factor - closer is better, but also reward medium range shots
        if dist < 15.0:
            distance_factor = 1.0  # Very close - great chance
        elif dist < 25.0:
            distance_factor = 0.8  # Good shooting range
        else:
            distance_factor = 0.5 * (1.0 - (dist - 25.0) / (SHOOT_DISTANCE_THRESHOLD - 25.0))
        
        # Block factor - no blockers = clear shot
        if blocking_count == 0:
            block_factor = 1.2  # Bonus for completely clear shot
        elif blocking_count == 1:
            block_factor = 0.6  # Still worth trying
        else:
            block_factor = 0.2  # Multiple blockers - bad idea
        
        # Role bonus - forwards should shoot more
        role_bonus = 0.2 if self.role == 'FWD' else (0.1 if self.role == 'MID' else 0.0)
        
        return distance_factor * block_factor + role_bonus

    def _evaluate_pass(self, ctx):
        """Evaluate passing options with vision checks and lead passes. Returns (best_teammate, target_pos, score)."""
        if len(ctx.teammates) == 0:
            return None, None, 0.0
        
        best_teammate = None
        best_target = None
        best_score = 0.0
        
        # DEFENSIVE ZONE DETECTION: Check if we're near own goal
        own_goal_x = 0.0 if ctx.team == 'home' else 100.0
        dist_to_own_goal = abs(self.pos[0] - own_goal_x)
        in_defensive_third = dist_to_own_goal < 35.0  # Within 35 units of own goal
        
        # Generate candidate pass positions in 360 degrees at various distances
        num_directions = 16  # Check 16 directions around player
        # In defensive zone: prefer longer passes
        if in_defensive_third:
            pass_distances = [15.0, 25.0, 35.0, 45.0]  # Longer distances in defense
        else:
            pass_distances = [8.0, 15.0, 25.0]  # Short, medium, long
        
        candidate_positions = []
        
        for i in range(num_directions):
            angle = (2 * np.pi * i) / num_directions
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            for dist in pass_distances:
                target = self.pos + direction * dist
                # Keep within field bounds
                if 2.0 < target[0] < 98.0 and 2.0 < target[1] < 98.0:
                    candidate_positions.append(target)
        
        # Also add direct teammate positions AND lead pass positions for moving teammates
        for teammate in ctx.teammates:
            if teammate.role != 'GK' or ctx.dist_to_goal >= 50:
                candidate_positions.append(teammate.pos.copy())
                # Add lead pass position if teammate is moving
                if np.linalg.norm(teammate.vel) > 0.1:
                    lead_target = teammate.pos + teammate.vel * 5.0  # Lead by 5 time steps
                    if 2.0 < lead_target[0] < 98.0 and 2.0 < lead_target[1] < 98.0:
                        candidate_positions.append(lead_target)
        
        # Evaluate each candidate position
        for target_pos in candidate_positions:
            # VISION CHECK - prefer targets we can see
            in_vision, vision_dist, vision_angle = is_in_vision_cone(
                self.pos, ctx.facing, target_pos)
            
            # Blind spot penalty - behind player gets reduced score
            vision_penalty = 0.0
            if not in_vision and vision_angle > 90:
                vision_penalty = 0.3  # Significant penalty for passes behind us
            
            # Find which teammate can best reach this position
            best_reach_teammate = None
            best_reach_time = float('inf')
            
            for teammate in ctx.teammates:
                if teammate.role == 'GK' and ctx.dist_to_goal < 50:
                    continue
                
                # Time for teammate to reach target
                dist_to_target = np.linalg.norm(teammate.pos - target_pos)
                reach_time = dist_to_target / PLAYER_SPRINT_SPEED
                
                if reach_time < best_reach_time:
                    best_reach_time = reach_time
                    best_reach_teammate = teammate
            
            if best_reach_teammate is None:
                continue
            
            # Calculate ball travel time - no minimum distance restriction
            pass_dist = np.linalg.norm(target_pos - self.pos)
            if pass_dist > 45.0:
                continue
            if pass_dist < 0.5:
                continue  # Only skip if literally on top of target
            ball_time = pass_dist / BALL_PASS_SPEED
            
            # Check if teammate can reach target before/when ball arrives
            if best_reach_time > ball_time + 1.5:
                continue  # Teammate can't reach in time
            
            # Check passing lane quality - allow more passes through
            lane_quality = calculate_passing_lane_quality(
                self.pos, target_pos, ctx.opponent_positions)
            if lane_quality < 0.2:
                continue  # Lane is too blocked
            
            # Evaluate interception risk - check all opponents
            # Margin = how much sooner ball arrives than opponent (positive = safe)
            min_intercept_margin = float('inf')
            for opp_pos in ctx.opponent_positions:
                closest_on_path = project_point_to_segment(opp_pos, self.pos, target_pos)
                dist_to_path = np.linalg.norm(opp_pos - closest_on_path)
                dist_along_path = np.linalg.norm(closest_on_path - self.pos)
                
                time_ball_at_point = dist_along_path / BALL_PASS_SPEED
                time_opp_at_point = dist_to_path / PLAYER_SPRINT_SPEED
                
                # FIX: margin = opponent_time - ball_time (positive means ball arrives first = SAFE)
                margin = time_opp_at_point - time_ball_at_point
                min_intercept_margin = min(min_intercept_margin, float(margin))
            
            # Skip only if opponent can intercept (margin < 0 means opponent arrives first)
            if min_intercept_margin < -0.5:
                continue
            
            # Score components - higher margin = safer (margin > 0 means ball arrives first)
            safety_score = min(1.0, max(0.0, (min_intercept_margin + 1.0) / 3.0))
            
            # Space around target with larger radius
            space_score = 1.0
            for opp_pos in ctx.opponent_positions:
                opp_dist = np.linalg.norm(opp_pos - target_pos)
                if opp_dist < 12.0:  # Larger check radius
                    space_score -= (1.0 - opp_dist / 12.0) * 0.25
            space_score = max(0.0, space_score)
            
            # Progress toward goal - strongly reward forward passes
            my_goal_dist = ctx.dist_to_goal
            target_goal_dist = distance_to_goal(target_pos, ctx.team)
            goal_progress = my_goal_dist - target_goal_dist  # Positive = forward
            is_back_pass = target_goal_dist > my_goal_dist + 5.0
            is_forward_pass = goal_progress > 8.0  # Significant forward movement
            
            # Calculate progress score
            if is_back_pass:
                progress_score = 0.05  # Very low score for backward passes
            else:
                progress_score = 0.5 + 0.5 * goal_progress / max(my_goal_dist, 1)
            progress_score = float(np.clip(progress_score, 0, 1))
            
            # Forward pass bonus
            forward_bonus = FORWARD_PASS_BONUS if is_forward_pass else 0.0
            
            # Teammate accessibility bonus
            access_score = 1.0 - min(best_reach_time / 3.0, 1.0)
            
            # Combined score with lane quality and vision
            combined_safety = safety_score * lane_quality
            
            score = (
                PASS_SAFETY_WEIGHT * combined_safety +
                0.15 * space_score +
                PASS_GOAL_PROGRESS_WEIGHT * progress_score +
                0.05 * access_score +
                forward_bonus -
                vision_penalty
            )
            
            # Bonus for very safe forward options
            if combined_safety > 0.5 and is_forward_pass:
                score += SPACE_PASS_BONUS
            
            # DEFENSIVE ZONE PENALTY: Strongly penalize short passes near own goal
            if in_defensive_third:
                if pass_dist < 15.0 and goal_progress < 12.0:
                    score -= 0.6  # Very strong penalty for short lateral passes in defense
                    # Skip short risky passes entirely if score goes negative
                    if score < 0.1:
                        continue
                elif pass_dist > 30.0 and goal_progress > 5.0:
                    score += 0.35  # Strong bonus for longer forward clearances
            
            # AVOID RETURN PASS: Penalize passing back to who just passed to us
            if self.received_from_player_id is not None and best_reach_teammate.id == self.received_from_player_id:
                score -= 0.5  # Strong penalty for return pass
            
            # Collect all viable options instead of just best
            if score > 0.1:  # Minimum viable score
                # Add randomness to score for probabilistic selection
                randomized_score = score + random.uniform(-PASS_SELECTION_RANDOMNESS, PASS_SELECTION_RANDOMNESS) * score
                if randomized_score > best_score:
                    best_score = score  # Store actual score, not randomized
                    best_teammate = best_reach_teammate
                    best_target = target_pos
        
        return best_teammate, best_target, best_score
    
    def _evaluate_pass_to_target(self, ctx, target_pos, teammate):
        """Evaluate a pass to a specific target position with strict interception analysis."""
        pass_vec = target_pos - self.pos
        pass_dist = np.linalg.norm(pass_vec)
        
        # No minimum distance - allow all short passes
        if pass_dist < 0.5 or pass_dist > 50.0:
            return 0.0
        
        # Check if this is a back pass (toward own goal)
        my_goal_dist = ctx.dist_to_goal
        target_goal_dist = distance_to_goal(target_pos, ctx.team)
        is_back_pass = target_goal_dist > my_goal_dist + 3.0
        
        # VERY STRONG back pass penalty
        back_pass_penalty = 0.4 if is_back_pass else 0.0
        
        # Check passing lane quality
        lane_quality = calculate_passing_lane_quality(
            self.pos, target_pos, ctx.opponent_positions)
        
        # GEOMETRIC INTERCEPTION CHECK
        ball_travel_time = pass_dist / BALL_PASS_SPEED
        
        # Check each opponent's ability to intercept
        interception_risk = 0.0
        can_be_intercepted = False
        
        for opp_pos in ctx.opponent_positions:
            closest_on_path = project_point_to_segment(opp_pos, self.pos, target_pos)
            dist_to_path = np.linalg.norm(opp_pos - closest_on_path)
            
            dist_along_path = np.linalg.norm(closest_on_path - self.pos)
            time_ball_reaches_point = dist_along_path / BALL_PASS_SPEED
            
            time_opp_reaches_point = dist_to_path / PLAYER_SPRINT_SPEED
            
            # FIX: margin = opponent_time - ball_time (positive = ball arrives first = SAFE)
            time_margin = time_opp_reaches_point - time_ball_reaches_point
            
            # Negative margin means opponent arrives before ball = dangerous
            if time_margin < 0.5:  # Opponent arrives before or close to ball
                can_be_intercepted = True
                # Risk increases as margin becomes more negative
                risk_factor = min(1.0, max(0.0, 0.5 - time_margin))
                interception_risk = max(interception_risk, float(risk_factor))
        
        # CHECK RECEIVER SAFETY - LARGER radius (10 units instead of 8)
        receiver_safety = 1.0
        nearby_opponent_count = 0
        for opp_pos in ctx.opponent_positions:
            dist_opp_to_receiver = np.linalg.norm(opp_pos - target_pos)
            if dist_opp_to_receiver < 10.0:  # Larger radius
                nearby_opponent_count += 1
                proximity_penalty = 1.0 - (dist_opp_to_receiver / 10.0)
                receiver_safety -= proximity_penalty * 0.35
        
        # Extra penalty for crowded areas
        if nearby_opponent_count >= 2:
            receiver_safety -= 0.25 * (nearby_opponent_count - 1)
        
        receiver_safety = max(0.0, float(receiver_safety))
        
        # Calculate combined safety - all three must be good
        path_safety = max(0.0, 1.0 - interception_risk) if can_be_intercepted else 1.0
        combined_safety = path_safety * receiver_safety * lane_quality
        
        # REJECT only if very unsafe
        if combined_safety < 0.2:
            return 0.0
        
        # ROLE-BASED RISK TOLERANCE
        if self.role == 'DEF' and (can_be_intercepted or receiver_safety < 0.5):
            return 0.0
        if self.role == 'GK' and (can_be_intercepted or receiver_safety < 0.7):
            return 0.0
        
        # Progress factor
        progress_factor = 0.5 + 0.5 * (my_goal_dist - target_goal_dist) / max(my_goal_dist, 1)
        progress_factor = float(np.clip(progress_factor, 0, 1))
        
        # Distance factor
        if pass_dist < 5.0:
            dist_factor = 0.8
        elif pass_dist < 25.0:
            dist_factor = 1.0
        else:
            dist_factor = max(0.3, 1.0 - (pass_dist - 25.0) / 25.0)
        
        # SAFETY-FIRST SCORING
        score = (
            PASS_SAFETY_WEIGHT * combined_safety +
            PASS_GOAL_PROGRESS_WEIGHT * progress_factor +
            0.10 * dist_factor -
            back_pass_penalty
        )
        
        # Minimum score only for very safe passes
        if combined_safety > 0.5:
            score = max(score, 0.20)
        
        return max(0.0, float(score))

    def _evaluate_dribble(self, ctx):
        """Evaluate dribbling options with opponent avoidance. Returns (best_direction, score)."""
        
        num_directions = 16  # More directions for finer control
        angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
        
        best_dir = normalize(ctx.goal_center - self.pos)
        best_score = 0.0
        
        # Calculate avoidance vector - sum of repulsion from nearby opponents
        avoidance_vec = np.zeros(2)
        for opp_pos in ctx.opponent_positions:
            to_me = self.pos - opp_pos
            dist = np.linalg.norm(to_me)
            if dist < 15.0 and dist > 0.1:  # Nearby opponents
                # Stronger repulsion for closer opponents
                repulsion_strength = (15.0 - dist) / 15.0
                avoidance_vec += normalize(to_me) * repulsion_strength
        
        # Normalize avoidance if significant
        avoidance_strength = np.linalg.norm(avoidance_vec)
        if avoidance_strength > 0.1:
            avoidance_vec = normalize(avoidance_vec)
        else:
            avoidance_vec = np.zeros(2)
        
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            clearance = clearance_in_direction(self.pos, direction, ctx.opponent_positions)
            clearance_factor = clearance / 20.0
            
            goal_dir = normalize(ctx.goal_center - self.pos)
            alignment = np.dot(direction, goal_dir)
            goal_factor = (alignment + 1) / 2
            
            # OPPONENT AVOIDANCE: Bonus for directions that align with avoidance, penalty for moving into pressure
            avoidance_bonus = 0.0
            if avoidance_strength > 0.1:
                avoidance_alignment = np.dot(direction, avoidance_vec)
                if avoidance_alignment > 0:
                    avoidance_bonus = avoidance_alignment * 0.5  # Bonus for moving away
                else:
                    avoidance_bonus = avoidance_alignment * 0.6  # Stronger penalty for moving into pressure
            
            if ctx.team == 'home' and direction[0] < -0.5:
                goal_factor *= 0.3
            elif ctx.team == 'away' and direction[0] > 0.5:
                goal_factor *= 0.3
            
            future_pos = self.pos + direction * 10
            if future_pos[1] < 10 or future_pos[1] > 90:
                clearance_factor *= 0.5
            
            score = (
                DRIBBLE_CLEARANCE_WEIGHT * clearance_factor +
                DRIBBLE_GOAL_PROGRESS_WEIGHT * goal_factor +
                avoidance_bonus
            )
            
            if score > best_score:
                best_score = score
                best_dir = direction
        
        return best_dir, best_score

    def _clear_dribble_state(self):
        """Clear dribble sequence tracking."""
        self.is_dribbling = False
        self.dribble_touches = 0
        self.last_dribble_dir = None
    
    def _evaluate_pass_while_dribbling(self, ctx):
        """Evaluate pass options while dribbling - stricter criteria for good passes only."""
        best_teammate = None
        best_target = None
        best_score = 0.0
        
        for teammate in ctx.teammates:
            if teammate.role == 'GK' and ctx.dist_to_goal < 60:
                continue  # Don't pass to GK when attacking
            
            # Skip teammates behind us (backward passes during dribble are usually bad)
            to_teammate = teammate.pos - self.pos
            to_goal = ctx.goal_center - self.pos
            
            # Check if pass is generally forward
            is_forward = np.dot(to_teammate, to_goal) > 0
            
            # Calculate pass distance
            pass_dist = np.linalg.norm(to_teammate)
            if pass_dist < 5.0 or pass_dist > 40.0:
                continue  # Too close or too far
            
            # Check receiver pressure - is teammate marked?
            receiver_pressure = float('inf')
            for opp_pos in ctx.opponent_positions:
                d = np.linalg.norm(opp_pos - teammate.pos)
                receiver_pressure = min(receiver_pressure, float(d))
            
            if receiver_pressure < 5.0:
                continue  # Teammate is heavily marked
            
            # Check passing lane quality
            lane_quality = calculate_passing_lane_quality(
                self.pos, teammate.pos, ctx.opponent_positions)
            
            if lane_quality < 0.4:
                continue  # Lane is blocked
            
            # Calculate time margins for interception
            ball_time = pass_dist / BALL_PASS_SPEED
            min_intercept_margin = float('inf')
            for opp_pos in ctx.opponent_positions:
                closest_on_path = project_point_to_segment(opp_pos, self.pos, teammate.pos)
                dist_to_path = np.linalg.norm(opp_pos - closest_on_path)
                dist_along_path = np.linalg.norm(closest_on_path - self.pos)
                
                time_ball_at_point = dist_along_path / BALL_PASS_SPEED
                time_opp_at_point = dist_to_path / PLAYER_SPRINT_SPEED
                margin = time_opp_at_point - time_ball_at_point
                min_intercept_margin = min(min_intercept_margin, float(margin))
            
            if min_intercept_margin < 0.3:
                continue  # High interception risk
            
            # Score the pass
            safety_score = min(1.0, max(0.0, min_intercept_margin / 2.0))
            space_score = min(1.0, receiver_pressure / 15.0)
            forward_bonus = 0.2 if is_forward else 0.0
            
            # Goal progress
            my_goal_dist = ctx.dist_to_goal
            teammate_goal_dist = distance_to_goal(teammate.pos, ctx.team)
            progress_score = (my_goal_dist - teammate_goal_dist) / 50.0 if teammate_goal_dist < my_goal_dist else 0.0
            
            score = (
                0.3 * safety_score +
                0.25 * space_score +
                0.25 * lane_quality +
                0.2 * progress_score +
                forward_bonus
            )
            
            # Avoid return passes
            if self.received_from_player_id is not None and teammate.id == self.received_from_player_id:
                score -= 0.3
            
            if score > best_score:
                best_score = score
                best_teammate = teammate
                # Lead the pass slightly
                best_target = teammate.pos + teammate.vel * 2.0
        
        return best_teammate, best_target, best_score

    def _execute_shoot(self, ctx, game):
        """Execute a shot on goal - target corners away from goalkeeper."""
        self._clear_dribble_state()
        self.has_ball = False
        game.ball.owner_id = None
        
        # Target goal corners (GOAL_TOP=40, GOAL_BOTTOM=60) instead of center
        # Randomly pick upper or lower corner, with slight variation
        if random.random() < 0.5:
            # Upper corner - aim just inside top post
            target_y = GOAL_TOP + random.uniform(1.0, 3.0)
        else:
            # Lower corner - aim just inside bottom post
            target_y = GOAL_BOTTOM - random.uniform(1.0, 3.0)
        
        target = np.array([ctx.goal_x, target_y])
        
        shoot_dir = normalize(target - self.pos)
        game.ball.vel = shoot_dir * BALL_SHOOT_SPEED
        
        # Track last touch
        game.last_touch = LastTouch(team=self.team, player_id=self.id)

    def _execute_pass(self, ctx, game, target_player, target_pos=None):
        """Execute a pass to target position (or teammate if no target specified)."""
        self._clear_dribble_state()
        passer_id = self.id  # Remember who is passing
        self.has_ball = False
        self.received_from_player_id = None  # Clear on pass
        game.ball.owner_id = None
        
        # Use provided target position (for space passes) or teammate position
        if target_pos is None:
            # Pass directly to teammate with small lead
            target_pos = target_player.pos + target_player.vel * 3.0
        
        # Track pass target for recipient movement
        game.ball.target_player_id = target_player.id
        game.ball.target_pos = target_pos.copy() if isinstance(target_pos, np.ndarray) else np.array(target_pos)
        game.ball.passer_id = passer_id  # Track who made this pass
        
        # ALWAYS pass toward the target - never redirect to goal
        pass_vec = target_pos - self.pos
        pass_dist = np.linalg.norm(pass_vec)
        
        # If target is very close, pass directly to teammate
        if pass_dist < 2.0:
            pass_vec = target_player.pos - self.pos
            pass_dist = np.linalg.norm(pass_vec)
        
        # Safety: if still too close, use direction to teammate
        if pass_dist < 0.5:
            pass_dir = normalize(target_player.pos - self.pos)
        else:
            pass_dir = normalize(pass_vec)
        
        game.ball.vel = pass_dir * BALL_PASS_SPEED
        
        # Track last touch
        game.last_touch = LastTouch(team=self.team, player_id=self.id)

    def _execute_dribble(self, ctx, game, direction):
        """Execute dribbling as short kicks - ball is released ahead, player chases it."""
        # Release ball ownership - this is a kick, not attached ball
        self.has_ball = False
        game.ball.owner_id = None
        game.ball.clear_pass_target()  # Clear any stale pass targeting
        
        # Set touch cooldown - cannot reacquire ball for DRIBBLE_TOUCH_INTERVAL ticks
        # Each tick is 0.1 time units, so multiply by 0.1
        self.touch_cooldown_until = game.time + (DRIBBLE_TOUCH_INTERVAL * 0.1)
        
        # Mark as dribbling for re-evaluation on next touch
        self.is_dribbling = True
        self.dribble_touches += 1
        self.last_dribble_dir = direction.copy()  # Track direction for trajectory checking
        
        # Kick ball ahead by DRIBBLE_TOUCH_DISTANCE in the movement direction
        kick_distance = DRIBBLE_TOUCH_DISTANCE
        game.ball.pos = self.pos + direction * kick_distance
        
        # Give ball velocity faster than player to stay ahead between touches
        game.ball.vel = direction * PLAYER_SPRINT_SPEED * 1.2
        
        # Player sprints to chase the ball
        self.vel = direction * PLAYER_SPRINT_SPEED
        
        # Track last touch for out-of-bounds decisions
        game.last_touch = LastTouch(team=self.team, player_id=self.id)
    
    def _execute_gk_throw_to_corner(self, ctx, game):
        """Goalkeeper throws ball to corner/wing - evaluates multiple safe targets."""
        self.has_ball = False
        game.ball.owner_id = None
        
        # Generate multiple wide/corner targets and pick safest
        candidates = []
        
        # Forward direction based on team
        if self.team == 'home':
            forward_distances = [25.0, 35.0, 45.0]
            get_target_x = lambda d: min(self.pos[0] + d, 90.0)
        else:
            forward_distances = [25.0, 35.0, 45.0]
            get_target_x = lambda d: max(self.pos[0] - d, 10.0)
        
        # Wide positions - top and bottom wings
        wing_y_positions = [10.0, 15.0, 20.0, 80.0, 85.0, 90.0]
        
        for dist in forward_distances:
            target_x = get_target_x(dist)
            for target_y in wing_y_positions:
                candidates.append(np.array([target_x, target_y]))
        
        # Also add touchline targets for safety clearance
        for dist in [20.0, 30.0]:
            target_x = get_target_x(dist)
            candidates.append(np.array([target_x, 2.0]))   # Near bottom touchline
            candidates.append(np.array([target_x, 98.0]))  # Near top touchline
        
        # Score each candidate based on safety
        best_target = None
        best_score = -999.0
        
        for target in candidates:
            throw_vec = target - self.pos
            throw_dist = np.linalg.norm(throw_vec)
            if throw_dist < 10.0:
                continue
            
            # Check lane quality and opponent interception
            lane_quality = calculate_passing_lane_quality(self.pos, target, ctx.opponent_positions)
            
            # Calculate time margins
            min_margin = float('inf')
            for opp_pos in ctx.opponent_positions:
                closest = project_point_to_segment(opp_pos, self.pos, target)
                dist_opp = np.linalg.norm(opp_pos - closest)
                dist_ball = np.linalg.norm(closest - self.pos)
                
                time_ball = dist_ball / (BALL_SHOOT_SPEED * 0.8)
                time_opp = dist_opp / PLAYER_SPRINT_SPEED
                margin = time_opp - time_ball
                min_margin = min(min_margin, float(margin))
            
            # Safety score
            safety = lane_quality * max(0.0, min(1.0, (min_margin + 1.0) / 2.0))
            
            # Prefer wide positions (far from center)
            width_bonus = abs(target[1] - 50.0) / 50.0 * 0.3
            
            score = safety + width_bonus
            
            if score > best_score:
                best_score = score
                best_target = target
        
        # If no safe target found, kick to touchline for throw-in
        if best_target is None or best_score < 0.2:
            if self.team == 'home':
                best_target = np.array([self.pos[0] + 20.0, 0.0 if self.pos[1] < 50 else 100.0])
            else:
                best_target = np.array([self.pos[0] - 20.0, 0.0 if self.pos[1] < 50 else 100.0])
        
        throw_dir = normalize(best_target - self.pos)
        game.ball.vel = throw_dir * BALL_SHOOT_SPEED * 0.8
        
        game.last_touch = LastTouch(team=self.team, player_id=self.id)

    def _execute_clearance(self, ctx, game):
        """Clear the ball upfield away from danger - evaluate safe directions."""
        self.has_ball = False
        game.ball.owner_id = None
        
        # Count nearby opponents to determine pressure level
        nearby_opponents = 0
        for opp_pos in ctx.opponent_positions:
            if np.linalg.norm(opp_pos - self.pos) < 15.0:
                nearby_opponents += 1
        under_heavy_pressure = nearby_opponents >= 2
        
        # Generate multiple clearance target candidates and pick safest
        candidates = []
        goal_x = ctx.goal_x
        
        # Wide range of target Y positions - forward clearances
        for target_y in [20.0, 35.0, 50.0, 65.0, 80.0]:
            target = np.array([goal_x, target_y])
            candidates.append(('forward', target))
            # Also add shorter clearances (safer, less distance)
            mid_x = self.pos[0] + (goal_x - self.pos[0]) * 0.5
            candidates.append(('forward', np.array([mid_x, target_y])))
        
        # Add TOUCHLINE targets (corner/throw-in) - safe when under pressure
        # Direction depends on team - always push toward opponent goal
        x_direction = 1.0 if self.team == 'home' else -1.0
        for x_offset in [10.0, 20.0, 30.0]:
            target_x = self.pos[0] + x_offset * x_direction
            target_x = np.clip(target_x, 5.0, 95.0)
            candidates.append(('touchline', np.array([target_x, 0.0])))   # Bottom touchline (throw-in/corner)
            candidates.append(('touchline', np.array([target_x, 100.0]))) # Top touchline (throw-in/corner)
        
        # Score each candidate based on opponent proximity to ball path
        best_target = np.array([goal_x, 50.0])
        best_score = -999.0
        
        for clear_type, target in candidates:
            clear_vec = target - self.pos
            clear_dist = np.linalg.norm(clear_vec)
            if clear_dist < 8.0:
                continue
            
            # Check safety - how close can opponents get to the ball path?
            min_time_margin = float('inf')
            for opp_pos in ctx.opponent_positions:
                closest_on_path = project_point_to_segment(opp_pos, self.pos, target)
                dist_to_path = np.linalg.norm(opp_pos - closest_on_path)
                dist_along_path = np.linalg.norm(closest_on_path - self.pos)
                
                time_ball_reaches = dist_along_path / BALL_SHOOT_SPEED
                time_opp_reaches = dist_to_path / PLAYER_SPRINT_SPEED
                time_margin = time_ball_reaches - time_opp_reaches
                min_time_margin = min(min_time_margin, time_margin)
            
            # Score: higher time margin = safer
            safety_score = min(float(min_time_margin), 2.0)  # Cap at 2.0
            progress_score = (goal_x - self.pos[0]) / 100.0 if self.team == 'home' else (self.pos[0] - goal_x) / 100.0
            
            score = safety_score * 2.0 + progress_score + clear_dist / 50.0
            
            # UNDER HEAVY PRESSURE: Strongly prefer touchline clearances
            if under_heavy_pressure and clear_type == 'touchline':
                score += 3.0  # Big bonus for touchline when crowded
            
            if score > best_score:
                best_score = score
                best_target = target
        
        clear_dir = normalize(best_target - self.pos)
        game.ball.vel = clear_dir * BALL_SHOOT_SPEED
        
        # Track last touch
        game.last_touch = LastTouch(team=self.team, player_id=self.id)

    def _make_off_ball_decision(self, ctx, game):
        """Decision making when player doesn't have the ball."""
        
        # Special GK behavior - stay near goal, only move laterally
        if self.role == 'GK':
            self._make_gk_decision(ctx, game)
            return
        
        # PASS RECIPIENT MOVEMENT: When pass is in flight, move TOWARD ball trajectory to receive
        # ONLY activate if there's an actual pass target (not dribble or shot)
        ball_speed = np.linalg.norm(game.ball.vel)
        has_pass_target = game.ball.target_player_id is not None
        if ball_speed > 0.5 and game.ball.owner_id is None and has_pass_target:
            # Check if I am the intended recipient OR near the ball's path
            am_target = game.ball.target_player_id == self.id
            
            # Check if I'm near the ball trajectory
            if ball_speed > 0.1:
                ball_dir = normalize(game.ball.vel)
                ball_future = game.ball.pos + ball_dir * 50.0
                closest_on_path = project_point_to_segment(self.pos, game.ball.pos, ball_future)
                dist_to_path = np.linalg.norm(self.pos - closest_on_path)
                dist_ball_to_me = np.linalg.norm(game.ball.pos - self.pos)
                
                # Near the ball's path - even from long distance for attackers
                near_path = dist_to_path < 12.0 and dist_ball_to_me < 50.0
                
                if am_target or near_path:
                    # PRIMARY: Move TOWARD the ball trajectory intercept point
                    # Calculate where ball will be when we can reach it
                    time_to_closest = dist_to_path / PLAYER_SPRINT_SPEED
                    ball_intercept = game.ball.pos + ball_dir * (ball_speed * time_to_closest)
                    
                    # Clamp intercept to reasonable range
                    if np.linalg.norm(ball_intercept - game.ball.pos) > 40.0:
                        ball_intercept = game.ball.pos + ball_dir * 40.0
                    
                    # Move toward intercept point, with slight perpendicular offset for space
                    perp = np.array([-ball_dir[1], ball_dir[0]])
                    
                    # Choose perpendicular side with more space
                    best_offset = 0.0
                    if len(ctx.opponent_positions) > 0:
                        for sign in [1.0, -1.0]:
                            test_pos = closest_on_path + perp * sign * 3.0
                            nearest_opp_dist = float(min(
                                float(np.linalg.norm(test_pos - opp_pos))
                                for opp_pos in ctx.opponent_positions
                            ))
                            if abs(best_offset) < 0.1 or nearest_opp_dist > abs(best_offset):
                                best_offset = sign * 2.0  # Small offset for receiving angle
                    
                    # Target is on the ball path with small perpendicular offset
                    move_target = closest_on_path + perp * best_offset
                    move_target = self._clamp_to_zone(move_target)
                    
                    to_target = move_target - self.pos
                    if np.linalg.norm(to_target) > 0.5:
                        # Sprint to intercept the pass
                        self.vel = normalize(to_target) * PLAYER_SPRINT_SPEED
                        return
        
        ball_owner = None
        if game.ball.owner_id is not None:
            for p in game.players:
                if p.id == game.ball.owner_id:
                    ball_owner = p
                    break
        
        # When ball is loose, determine if we should chase or hold position
        if ball_owner is None:
            to_ball = game.ball.pos - self.pos
            my_dist = np.linalg.norm(to_ball)
            
            # Check if ball is in our zone
            ball_in_zone = self._is_in_zone(game.ball.pos)
            
            # Check if ball is in our defensive half (defenders should be aggressive)
            own_goal_x = 0.0 if self.team == 'home' else 100.0
            ball_dist_to_own_goal = abs(game.ball.pos[0] - own_goal_x)
            ball_near_own_goal = ball_dist_to_own_goal < 30.0
            
            # NEW: Check if ball is in our defensive half (first 50 units from our goal)
            ball_in_defensive_half = ball_dist_to_own_goal < 50.0
            
            # Check if we're one of the closest teammates
            closer_teammates = sum(1 for t in ctx.teammates 
                                   if np.linalg.norm(t.pos - game.ball.pos) < my_dist - 3.0)
            
            # Check opponent proximity to ball
            closest_opp_dist = 100.0
            for opp_pos in ctx.opponent_positions:
                d = np.linalg.norm(opp_pos - game.ball.pos)
                closest_opp_dist = min(float(closest_opp_dist), float(d))
            
            # Check if I'm the absolute closest on my team
            am_closest_on_team = closer_teammates == 0
            
            # Decide whether to chase based on multiple factors
            should_chase = False
            if my_dist < 15.0:  # Very close - always run to the ball
                should_chase = True
            elif am_closest_on_team and my_dist < 35.0:  # Closest on team and within reasonable range
                should_chase = True
            elif am_closest_on_team and my_dist < closest_opp_dist:  # Closest and can beat opponent
                should_chase = True
            elif self.role == 'DEF' and ball_in_defensive_half and my_dist < 35.0:
                # DEFENDERS ARE AGGRESSIVE: Chase when ball enters their half
                should_chase = True
            elif ball_near_own_goal and self.role == 'DEF' and my_dist < 25.0:
                # Defenders MUST chase when ball is near own goal
                should_chase = True
            elif ball_in_zone and am_closest_on_team:  # In our zone and closest
                should_chase = True
            
            if should_chase:
                if my_dist > 1.0:
                    # Defenders chase directly in their defensive half - no zone clamping
                    if self.role == 'DEF' and ball_in_defensive_half:
                        chase_target = game.ball.pos  # Chase directly
                    elif ball_near_own_goal and self.role == 'DEF':
                        chase_target = game.ball.pos  # Chase directly
                    else:
                        chase_target = self._clamp_to_zone(game.ball.pos)
                    to_target = chase_target - self.pos
                    self.vel = normalize(to_target) * PLAYER_SPRINT_SPEED
                else:
                    self.vel = np.zeros(2)
            else:
                # Not chasing directly - check if we can intercept a moving ball
                ball_speed = np.linalg.norm(game.ball.vel)
                
                if ball_speed > 0.5 and my_dist < 40.0:
                    # Ball is moving - calculate intercept point
                    ball_dir = normalize(game.ball.vel)
                    
                    # Project player position onto ball's path to find closest intercept point
                    ball_future = game.ball.pos + ball_dir * 50.0  # Extended ball path
                    intercept_point = project_point_to_segment(self.pos, game.ball.pos, ball_future)
                    
                    # Check if intercept point is reachable before ball arrives
                    dist_to_intercept = np.linalg.norm(intercept_point - self.pos)
                    ball_dist_to_intercept = np.linalg.norm(intercept_point - game.ball.pos)
                    
                    time_player = dist_to_intercept / PLAYER_SPRINT_SPEED
                    time_ball = ball_dist_to_intercept / ball_speed
                    
                    # Move to intercept if we can get there in time
                    if time_player < time_ball + 1.5 and dist_to_intercept > 2.0:
                        # Move perpendicular toward intercept point
                        to_intercept = intercept_point - self.pos
                        self.vel = normalize(to_intercept) * PLAYER_SPRINT_SPEED
                        return
                
                # Default: Return toward home position
                to_home = self.home_pos - self.pos
                if np.linalg.norm(to_home) > 2.0:
                    self.vel = normalize(to_home) * PLAYER_SPEED
                else:
                    self.vel = np.zeros(2)
            return
        
        # Calculate tactical target when ball is owned
        skip_zone_blend = False  # Flag to skip home position blending for runs
        
        # Check team possession - based on last touch when ball is loose, or ball owner
        team_has_possession = False
        if ball_owner is not None:
            team_has_possession = (ball_owner.team == self.team)
        elif game.last_touch is not None:
            team_has_possession = (game.last_touch.team == self.team)
        
        if team_has_possession:
            # Determine if player should make forward runs
            # MIDFIELDERS and WINGERS (FWD on left/right) make forward runs
            is_midfielder = self.role == 'MID'
            is_winger = self.role == 'FWD' and self.lateral_role in ['left', 'right']
            should_make_runs = (is_midfielder or is_winger) and (ball_owner is None or ball_owner.id != self.id)
            
            if should_make_runs:
                # Check if ball is past own defensive third (more permissive)
                if self.team == 'home':
                    can_run_forward = game.ball.pos[0] > 25.0  # Past own third
                else:
                    can_run_forward = game.ball.pos[0] < 75.0
                
                if can_run_forward:
                    # Make continuous forward run toward goal - skip zone blending
                    tactical_target = self._make_forward_run(ctx, ball_owner, game)
                    skip_zone_blend = True
                else:
                    tactical_target = self._find_support_spot(ctx, ball_owner, game) if ball_owner else self.home_pos.copy()
            else:
                tactical_target = self._find_support_spot(ctx, ball_owner, game) if ball_owner else self.home_pos.copy()
        else:
            tactical_target = self._get_defend_target(ctx, ball_owner) if ball_owner else self.home_pos.copy()
        
        # Skip clamping for forward runs, otherwise clamp to zone
        if skip_zone_blend:
            clamped_target = tactical_target  # Use raw target for forward runs
        else:
            clamped_target = self._clamp_to_zone(tactical_target)
        
        # DYNAMIC POSITIONING: Shift based on ball position (attack vs defense)
        # When team is attacking (ball in opponent half), push forward
        # When team is defending (ball in own half), drop back
        own_goal_x = 0.0 if self.team == 'home' else 100.0
        opp_goal_x = 100.0 if self.team == 'home' else 0.0
        ball_x = game.ball.pos[0]
        
        # Calculate attack/defense shift (-1 = full defense, +1 = full attack)
        if self.team == 'home':
            attack_shift = (ball_x - 50.0) / 50.0  # Positive when ball is in opponent half
        else:
            attack_shift = (50.0 - ball_x) / 50.0
        attack_shift = float(np.clip(attack_shift, -1.0, 1.0))
        
        # Apply shift to home position (move forward when attacking, back when defending)
        shift_amount = attack_shift * 15.0  # Max 15 units shift
        if self.team == 'home':
            shifted_home = np.array([self.home_pos[0] + shift_amount, self.home_pos[1]])
        else:
            shifted_home = np.array([self.home_pos[0] - shift_amount, self.home_pos[1]])
        
        # Skip home blending for forward runs to maintain run direction
        if skip_zone_blend:
            blended_target = clamped_target
        else:
            # Blend with shifted home position - much lower weight for more tactical movement
            home_blend = HOME_POSITION_WEIGHT * self.zone_weight * 0.5  # Even lower blend
            blended_target = (1 - home_blend) * clamped_target + home_blend * shifted_home
        
        # Move towards blended target - sprint during forward runs
        to_target = blended_target - self.pos
        dist = np.linalg.norm(to_target)
        
        if dist > 2.0:
            # Sprint during forward runs, normal speed otherwise
            move_speed = PLAYER_SPRINT_SPEED if skip_zone_blend else PLAYER_SPEED
            self.vel = normalize(to_target) * move_speed
        else:
            self.vel = np.zeros(2)
    
    def _is_in_zone(self, pos):
        """Check if a position is within this player's zone."""
        x_min, x_max, y_min, y_max = self.zone_bounds
        return x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max
    
    def _clamp_to_zone(self, pos):
        """Clamp a position to be within zone bounds."""
        x_min, x_max, y_min, y_max = self.zone_bounds
        return np.array([
            np.clip(pos[0], x_min, x_max),
            np.clip(pos[1], y_min, y_max)
        ])

    def _make_forward_run(self, ctx, ball_owner, game):
        """Midfielder/winger makes aggressive forward run into attacking/scoring areas."""
        goal_center = ctx.goal_center
        
        # WINGER-SPECIFIC RUNS: Wide crossing or diagonal cuts
        is_winger = self.role == 'FWD' and self.lateral_role in ['left', 'right']
        
        if is_winger:
            # Wingers have two run types: wide byline run or diagonal cut
            run_type = random.choice(['wide', 'diagonal', 'diagonal'])  # Bias toward diagonal
            
            if self.team == 'home':
                goal_x = 100.0
                if run_type == 'wide':
                    # Run to byline for crossing
                    target_x = 90.0 + random.uniform(0, 8)
                    if self.lateral_role == 'left':
                        target_y = 8.0 + random.uniform(0, 12)  # Near top touchline
                    else:
                        target_y = 80.0 + random.uniform(0, 12)  # Near bottom touchline
                else:
                    # Diagonal cut toward goal
                    target_x = 85.0 + random.uniform(0, 10)
                    if self.lateral_role == 'left':
                        target_y = 30.0 + random.uniform(0, 15)  # Cut inside from left
                    else:
                        target_y = 55.0 + random.uniform(0, 15)  # Cut inside from right
            else:
                goal_x = 0.0
                if run_type == 'wide':
                    target_x = 2.0 + random.uniform(0, 8)
                    if self.lateral_role == 'left':
                        target_y = 80.0 + random.uniform(0, 12)
                    else:
                        target_y = 8.0 + random.uniform(0, 12)
                else:
                    target_x = 5.0 + random.uniform(0, 10)
                    if self.lateral_role == 'left':
                        target_y = 55.0 + random.uniform(0, 15)
                    else:
                        target_y = 30.0 + random.uniform(0, 15)
            
            run_target = np.array([target_x, target_y])
        else:
            # MIDFIELDER RUNS: Into half-spaces and scoring zones
            my_to_goal = goal_center - self.pos
            goal_dir = normalize(my_to_goal)
            my_dist_to_goal = np.linalg.norm(my_to_goal)
            
            # Aggressive run distances
            if my_dist_to_goal > 35:
                run_distance = 28.0 + random.uniform(0, 12)
            elif my_dist_to_goal > 20:
                run_distance = 20.0 + random.uniform(0, 10)
            else:
                run_distance = 10.0 + random.uniform(0, 8)
            
            # Half-space positioning
            perp = np.array([-goal_dir[1], goal_dir[0]])
            side = 1 if self.pos[1] > 50 else -1
            
            if my_dist_to_goal < 30:
                lateral_offset = side * (6.0 + random.uniform(0, 8))
            else:
                lateral_offset = side * (10.0 + random.uniform(0, 12))
            
            run_target = self.pos + goal_dir * run_distance + perp * lateral_offset
        
        # Clip to field - allow wingers to reach touchline
        run_target[0] = float(np.clip(run_target[0], 3.0, 97.0))
        run_target[1] = float(np.clip(run_target[1], 2.0, 98.0))
        
        return run_target

    def _find_hot_teammate(self, ctx):
        """Find teammate with good scoring opportunity using actual shooting evaluation."""
        best_teammate = None
        best_opportunity = 0.0
        
        for teammate in ctx.teammates:
            if teammate.role == 'GK':
                continue
            
            # Calculate teammate's distance to goal
            teammate_dist = distance_to_goal(teammate.pos, ctx.team)
            
            # Must be within shooting range
            if teammate_dist > SHOOT_DISTANCE_THRESHOLD:
                continue
            
            # Create a context for the teammate to evaluate their shot
            teammate_goal_center = np.array([100.0 if ctx.team == 'home' else 0.0, 50.0])
            
            # Simple shot evaluation for teammate
            # Check blocking opponents
            blocking = 0
            for opp_pos in ctx.opponent_positions:
                if distance_point_to_segment(opp_pos, teammate.pos, teammate_goal_center) < 4.0:
                    blocking += 1
            
            # Distance score
            if teammate_dist < 15.0:
                dist_score = 1.0
            elif teammate_dist < 25.0:
                dist_score = 0.8
            elif teammate_dist < 35.0:
                dist_score = 0.5
            else:
                dist_score = 0.3
            
            # Block penalty - but more lenient
            if blocking == 0:
                block_score = 1.0
            elif blocking == 1:
                block_score = 0.7
            else:
                block_score = 0.4
            
            # Combined opportunity score
            opportunity = dist_score * block_score
            
            # Check if pass lane is safe (relaxed threshold)
            lane_quality = calculate_passing_lane_quality(self.pos, teammate.pos, ctx.opponent_positions)
            
            # Hot teammate: decent opportunity (>=0.25) AND passable lane (>=0.2)
            if opportunity >= 0.25 and lane_quality >= 0.2 and opportunity > best_opportunity:
                best_opportunity = opportunity
                best_teammate = teammate
        
        return best_teammate, best_opportunity

    def _evaluate_runway_pass(self, ctx, teammate):
        """Evaluate a lead pass (through ball) to where teammate is running."""
        if teammate.role == 'GK':
            return None, 0.0
        
        # Get teammate's velocity/direction
        teammate_vel = teammate.vel
        teammate_speed = np.linalg.norm(teammate_vel)
        
        # Consider teammates moving OR in good positions ahead
        teammate_dist_to_goal = distance_to_goal(teammate.pos, ctx.team)
        my_goal_dist = ctx.dist_to_goal
        
        # If teammate is moving forward
        if teammate_speed > 0.2:
            teammate_dir = normalize(teammate_vel)
            
            # Longer lead distance for through balls (18-25 units)
            lead_distance = 18.0 + teammate_speed * 8.0
            lead_target = teammate.pos + teammate_dir * lead_distance
        else:
            # Stationary teammate in good position - pass to space ahead of them
            if teammate_dist_to_goal >= my_goal_dist - 5:
                return None, 0.0  # Not ahead of us
            
            goal_dir = normalize(ctx.goal_center - teammate.pos)
            lead_target = teammate.pos + goal_dir * 12.0
        
        # Clamp to field bounds
        lead_target[0] = float(np.clip(lead_target[0], 3.0, 97.0))
        lead_target[1] = float(np.clip(lead_target[1], 3.0, 97.0))
        
        # Check forward progress
        target_goal_dist = distance_to_goal(lead_target, ctx.team)
        goal_progress = my_goal_dist - target_goal_dist
        
        if goal_progress < 5.0:
            return None, 0.0  # Not enough forward progress
        
        # Check lane safety - relaxed threshold
        lane_quality = calculate_passing_lane_quality(self.pos, lead_target, ctx.opponent_positions)
        
        if lane_quality < 0.25:
            return None, 0.0  # Too risky
        
        # Check if teammate can reach the target before opponents
        teammate_reach_time = np.linalg.norm(lead_target - teammate.pos) / PLAYER_SPRINT_SPEED
        
        min_opp_time = float('inf')
        for opp_pos in ctx.opponent_positions:
            opp_time = np.linalg.norm(lead_target - opp_pos) / PLAYER_SPRINT_SPEED
            min_opp_time = min(min_opp_time, opp_time)
        
        time_advantage = min_opp_time - teammate_reach_time
        if time_advantage < 0.3:
            return None, 0.0  # Opponent would get there first
        
        # Score with generous base for forward progress
        progress_bonus = 0.25 if goal_progress > 8 else 0.15
        space_score = min(time_advantage / 2.0, 0.3)
        
        score = progress_bonus + 0.3 * lane_quality + space_score
        
        # Bonus if lead target is in shooting/crossing range
        if target_goal_dist < 25:
            score += 0.2
        
        # Bonus for passes to wingers making runs
        if teammate.role == 'FWD' and teammate.lateral_role in ['left', 'right']:
            score += 0.1
        
        return lead_target, score

    def _execute_cross(self, ctx, game):
        """Winger crosses ball into the penalty area toward teammates."""
        self.has_ball = False
        game.ball.owner_id = None
        
        # Find best target in the box
        goal_x = 100.0 if ctx.team == 'home' else 0.0
        penalty_x = 85.0 if ctx.team == 'home' else 15.0
        
        best_target = None
        best_score = 0.0
        
        for teammate in ctx.teammates:
            if teammate.role == 'GK':
                continue
            
            # Check if teammate is in/near the box
            in_box = (ctx.team == 'home' and teammate.pos[0] > penalty_x - 10) or \
                    (ctx.team == 'away' and teammate.pos[0] < penalty_x + 10)
            
            if not in_box:
                continue
            
            # Score based on position - prefer central, near goal
            dist_to_goal = abs(teammate.pos[0] - goal_x)
            centrality = 1.0 - abs(teammate.pos[1] - 50.0) / 50.0
            
            # Check for defenders near target
            nearby_defenders = 0
            for opp_pos in ctx.opponent_positions:
                if np.linalg.norm(opp_pos - teammate.pos) < 5.0:
                    nearby_defenders += 1
            
            score = centrality * (1.0 - dist_to_goal / 30.0) * (1.0 - nearby_defenders * 0.3)
            
            if score > best_score:
                best_score = score
                best_target = teammate.pos.copy()
        
        # Default cross target: near post or far post
        if best_target is None:
            if self.lateral_role == 'left':
                best_target = np.array([penalty_x, 55.0])  # Far post
            else:
                best_target = np.array([penalty_x, 45.0])  # Near post
        
        # Add some randomness to cross target
        best_target[1] += random.uniform(-3.0, 3.0)
        
        # Execute the cross - higher arc (slightly slower, more lofted)
        cross_vec = best_target - self.pos
        cross_dir = normalize(cross_vec)
        
        # Set ball position to player's position (like other pass executions)
        game.ball.pos = self.pos.copy()
        game.ball.vel = cross_dir * BALL_PASS_SPEED * 0.9  # Slightly slower for "loft"
        
        game.last_touch = LastTouch(team=self.team, player_id=self.id)

    def _execute_runway_pass(self, ctx, game, teammate, lead_target):
        """Execute a lead pass to where teammate is running."""
        self.has_ball = False
        game.ball.owner_id = None
        
        pass_vec = lead_target - self.pos
        pass_dir = normalize(pass_vec)
        
        # Slightly faster than normal pass for through balls
        game.ball.vel = pass_dir * BALL_PASS_SPEED * 1.1
        
        game.last_touch = LastTouch(team=self.team, player_id=self.id)

    def _find_support_spot(self, ctx, ball_owner, game):
        """Find best support position - safe for receiving passes, minimizing interception risk."""
        # Generate candidate positions (unclamped first)
        candidates = []
        
        # 1. Home position (already in zone)
        candidates.append(self.home_pos.copy())
        
        # 2. Position toward attacking goal from current spot
        goal_dir = normalize(ctx.goal_center - self.pos)
        candidates.append(self.pos + goal_dir * 10.0)
        candidates.append(self.pos + goal_dir * 15.0)
        
        # 3. Lateral positions for width
        perp = np.array([-goal_dir[1], goal_dir[0]])
        candidates.append(self.pos + perp * 8.0)
        candidates.append(self.pos - perp * 8.0)
        candidates.append(self.pos + goal_dir * 8.0 + perp * 6.0)
        
        # 4. PASS RECIPIENT MOVEMENT: If ahead of ball, find open passing lanes
        if ball_owner is not None:
            my_dist_to_goal = distance_to_goal(self.pos, self.team)
            ball_dist_to_goal = distance_to_goal(ball_owner.pos, self.team)
            
            # Am I a potential pass recipient? (ahead of ball but not too far)
            is_ahead = my_dist_to_goal < ball_dist_to_goal - 5.0
            not_too_far = my_dist_to_goal > ball_dist_to_goal - 30.0
            
            if is_ahead and not_too_far and self.role in ['MID', 'FWD']:
                # Find positions with best passing lane quality from ball carrier
                for offset_angle in [-45, -30, -15, 0, 15, 30, 45]:
                    angle_rad = np.radians(offset_angle)
                    # Direction from ball carrier to goal
                    to_goal = normalize(ctx.goal_center - ball_owner.pos)
                    # Rotate to create offset
                    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                    rotated = np.array([
                        to_goal[0] * cos_a - to_goal[1] * sin_a,
                        to_goal[0] * sin_a + to_goal[1] * cos_a
                    ])
                    # Candidate position ahead of ball carrier
                    for dist in [12.0, 18.0, 25.0]:
                        cand_pos = ball_owner.pos + rotated * dist
                        # Keep within bounds
                        cand_pos[0] = float(np.clip(cand_pos[0], 5.0, 95.0))
                        cand_pos[1] = float(np.clip(cand_pos[1], 5.0, 95.0))
                        candidates.append(cand_pos)
        
        candidates.append(self.pos + goal_dir * 8.0 - perp * 6.0)
        
        # 5. Support positions relative to ball carrier
        ball_to_goal = ctx.goal_center - ball_owner.pos
        ball_dir = normalize(ball_to_goal)
        ball_perp = np.array([-ball_dir[1], ball_dir[0]])
        
        # Side based on current position
        side = 1 if np.dot(self.pos - ball_owner.pos, ball_perp) > 0 else -1
        
        # Various distances ahead and to the side of ball carrier
        for dist in [12.0, 18.0, 25.0]:
            for offset in [8.0, 15.0]:
                candidates.append(ball_owner.pos + ball_dir * dist + ball_perp * side * offset)
        
        # Score each candidate BEFORE clamping
        best_spot = self.home_pos.copy()
        best_score = -999.0
        
        for cand in candidates:
            # Check distance to owner on ORIGINAL candidate
            dist_to_owner = np.linalg.norm(cand - ball_owner.pos)
            if dist_to_owner < 8.0 or dist_to_owner > 45.0:
                continue
            
            score = 0.0
            
            # Clamp candidate to zone first, then check if still valid
            clamped_cand = self._clamp_to_zone(cand)
            clamped_dist_to_goal = distance_to_goal(clamped_cand, self.team)
            ball_dist_to_goal = distance_to_goal(ball_owner.pos, self.team)
            
            # Skip if clamped position is no longer ahead of ball by at least 5 units
            if clamped_dist_to_goal > ball_dist_to_goal - 5.0:
                continue
            
            # Use clamped candidate for scoring
            cand = clamped_cand
            
            # 1. PASSING LANE QUALITY - important but balanced
            lane_quality = calculate_passing_lane_quality(ball_owner.pos, cand, ctx.opponent_positions)
            score += lane_quality * 0.35
            
            # 2. Pass safety - can ball reach here without interception?
            pass_safety = 1.0
            for opp_pos in ctx.opponent_positions:
                can_intercept, time_margin = time_to_intercept(
                    opp_pos, PLAYER_SPRINT_SPEED,
                    ball_owner.pos, cand, BALL_PASS_SPEED
                )
                if can_intercept:
                    pass_safety *= max(0.1, float(0.5 + time_margin))
            score += pass_safety * 0.25
            
            # 3. Space from opponents
            min_opp_dist = 100.0
            for opp_pos in ctx.opponent_positions:
                d = np.linalg.norm(cand - opp_pos)
                min_opp_dist = min(float(min_opp_dist), float(d))
            space_score = min(min_opp_dist / 15.0, 1.0)
            score += space_score * 0.3
            
            # 3. Goal progress - closer to attacking goal is better (for attackers)
            my_goal_dist = distance_to_goal(self.pos, ctx.team)
            cand_goal_dist = distance_to_goal(cand, ctx.team)
            if my_goal_dist > 5:
                progress = (my_goal_dist - cand_goal_dist) / my_goal_dist
                score += max(0, progress) * 0.2
            
            # 4. Zone adherence bonus (larger bonus for in-zone)
            if self._is_in_zone(cand):
                score += 0.15
            else:
                # Penalty based on how far outside zone
                clamped = self._clamp_to_zone(cand)
                zone_dist = np.linalg.norm(cand - clamped)
                score -= min(zone_dist / 30.0, 0.2) * self.zone_weight
            
            if score > best_score:
                best_score = score
                # Clamp the winning spot to zone at the end
                best_spot = self._clamp_to_zone(cand)
        
        return best_spot

    def _make_gk_decision(self, ctx, game):
        """Special decision making for goalkeepers - stay CENTERED in goal, only move when ball is close."""
        # Define GK zone boundaries
        if self.team == 'home':
            goal_x = 5.0
            min_x = 3.0
            max_x = 15.0
            penalty_x = 20.0
            own_goal_x = 0.0
        else:
            goal_x = 95.0
            min_x = 85.0
            max_x = 97.0
            penalty_x = 80.0
            own_goal_x = 100.0
        
        ball_pos = game.ball.pos
        ball_vel = game.ball.vel
        ball_speed = np.linalg.norm(ball_vel)
        goal_center_y = 50.0  # Center of goal
        
        # Calculate ball distance from goal
        ball_dist_to_goal = abs(ball_pos[0] - own_goal_x)
        ball_close_to_goal = ball_dist_to_goal < 25.0  # Ball within 25 units of goal
        
        # Check if ball is being shot at goal (fast ball moving toward our goal)
        ball_toward_goal = False
        if self.team == 'home' and ball_vel[0] < -0.5:
            ball_toward_goal = True
        elif self.team == 'away' and ball_vel[0] > 0.5:
            ball_toward_goal = True
        
        shot_incoming = ball_speed > 1.5 and ball_toward_goal and ball_dist_to_goal < 40.0
        
        # Check if ball is in GK's penalty area (and loose and NOT a shot)
        ball_in_penalty = (self.team == 'home' and ball_pos[0] < penalty_x) or \
                          (self.team == 'away' and ball_pos[0] > penalty_x)
        
        # SHOT INCOMING: Only move laterally (stay on goal line)
        if shot_incoming:
            # Project where ball will cross goal line using proper trajectory math
            x_vel = ball_vel[0]
            y_vel = ball_vel[1]
            
            # Calculate time for ball to reach goal line
            if self.team == 'home':
                # Ball moving toward x=0
                if x_vel < -0.1:
                    time_to_goal = (own_goal_x - ball_pos[0]) / x_vel
                else:
                    time_to_goal = 10.0
            else:
                # Ball moving toward x=100
                if x_vel > 0.1:
                    time_to_goal = (own_goal_x - ball_pos[0]) / x_vel
                else:
                    time_to_goal = 10.0
            
            # Ensure positive time
            time_to_goal = max(0.0, time_to_goal)
            
            # Project Y position at goal line
            predicted_y = ball_pos[1] + y_vel * time_to_goal
            
            # Clamp to goal posts
            predicted_y = float(np.clip(predicted_y, GOAL_TOP, GOAL_BOTTOM))
            
            # Move laterally toward predicted position - proportional control
            lateral_diff = predicted_y - self.pos[1]
            
            # Move at speed proportional to distance, capped at sprint speed
            if abs(lateral_diff) > 0.3:
                move_speed = min(abs(lateral_diff) * 0.5, GK_SPRINT_SPEED)
                self.vel = np.array([0.0, np.sign(lateral_diff) * move_speed])
            else:
                self.vel = np.zeros(2)
            
            # Also ensure GK stays on goal line
            if abs(self.pos[0] - goal_x) > 1.0:
                self.vel[0] = np.sign(goal_x - self.pos[0]) * GK_SPEED
            
            return
        
        # If ball is loose and in penalty area, GK can chase it - FAST
        if game.ball.owner_id is None and ball_in_penalty and ball_speed < 1.5:
            target_x = np.clip(ball_pos[0], min_x, max_x)
            target_y = np.clip(ball_pos[1], 35.0, 65.0)
            target = np.array([target_x, target_y])
            
            to_target = target - self.pos
            dist = np.linalg.norm(to_target)
            if dist > 1.0:
                self.vel = normalize(to_target) * GK_SPRINT_SPEED  # GK sprints faster
            else:
                self.vel = np.zeros(2)
            return
        
        # STAY CENTERED unless ball is close to goal
        target_x = goal_x
        if ball_close_to_goal:
            # Ball is close - track it laterally more aggressively
            target_y = goal_center_y + (ball_pos[1] - goal_center_y) * 0.8  # More aggressive tracking
            target_y = np.clip(target_y, 38.0, 62.0)  # Wider range within goal
        else:
            # Ball is far - stay in center of goal
            target_y = goal_center_y
        
        target = np.array([target_x, target_y])
        to_target = target - self.pos
        dist = np.linalg.norm(to_target)
        
        if dist > 1.0:
            self.vel = normalize(to_target) * GK_SPEED  # GK moves faster
        else:
            self.vel = np.zeros(2)

    def _get_defend_target(self, ctx, ball_owner):
        """Calculate defensive position when opponent has ball."""
        goal_center = np.array([0.0 if self.team == 'home' else 100.0, 50.0])
        ball_to_goal = goal_center - ball_owner.pos
        
        if self.role == 'DEF':
            defense_point = goal_center + normalize(ball_owner.pos - goal_center) * 25.0
        elif self.role == 'MID':
            defense_point = ball_owner.pos + normalize(ball_to_goal) * 5.0
        else:  # FWD
            defense_point = ball_owner.pos
        
        return defense_point


class Ball:
    def __init__(self):
        self.pos = np.array([50.0, 50.0])
        self.vel = np.array([0.0, 0.0])
        self.owner_id: Optional[int] = None
        self.target_player_id: Optional[int] = None  # Track intended pass recipient
        self.target_pos: Optional[np.ndarray] = None  # Track pass target position
        self.passer_id: Optional[int] = None  # Track who made the current pass

    def clear_pass_target(self):
        """Clear pass target info when ball is controlled or goes out."""
        self.target_player_id = None
        self.target_pos = None
        self.passer_id = None

    def to_dict(self):
        return {
            "x": float(self.pos[0]),
            "y": float(self.pos[1]),
            "vx": float(self.vel[0]),
            "vy": float(self.vel[1]),
            "ownerId": self.owner_id
        }


class Game:
    def __init__(self):
        self.players = []
        self.ball = Ball()
        self.score = {"home": 0, "away": 0}
        self.time = 0.0
        self.state = GameState.KICKOFF
        self.last_touch: Optional[LastTouch] = None
        self.state_timer = 0.0
        self.restart_pos = np.array([50.0, 50.0])
        self.restart_team = 'home'
        self.last_ball_pos = np.array([50.0, 50.0])  # Track last in-bounds position
        self.last_restart_type = None  # Track last restart type (for throw-in goal prevention)
        self.throw_in_touched = False  # Has ball been touched after throw-in?
        
        # Tactical settings for home team (user controlled)
        self.home_formation = '4-4-2'
        self.home_mentality = 'normal'
        self.home_dribble_frequency = 1.0  # 0.5 to 2.0 multiplier
        self.home_shoot_frequency = 1.0    # 0.5 to 2.0 multiplier
        
        # Away team uses fixed tactics
        self.away_formation = '4-4-2'
        self.away_mentality = 'normal'
        
        self.init_players()

    def init_players(self):
        self.players = []
        
        # Create home team from formation template
        home_formation = FORMATIONS.get(self.home_formation, FORMATIONS['4-4-2'])
        home_offset = MENTALITY_OFFSETS.get(self.home_mentality, 0)
        
        for i, (role, lat, lon, base_x, y, number) in enumerate(home_formation):
            # Apply mentality offset (except for GK)
            x = base_x if role == 'GK' else np.clip(base_x + home_offset, 5, 95)
            player = Player(i + 1, 'home', role, x, y, number, lat, lon)
            # Update home position with mentality
            if role != 'GK':
                player.home_pos[0] = x
            self.players.append(player)
        
        # Create away team (mirror formation)
        away_formation = FORMATIONS.get(self.away_formation, FORMATIONS['4-4-2'])
        away_offset = MENTALITY_OFFSETS.get(self.away_mentality, 0)
        
        for i, (role, lat, lon, base_x, y, number) in enumerate(away_formation):
            # Mirror x position and apply mentality (inverted for away team)
            mirror_x = 100 - base_x
            x = mirror_x if role == 'GK' else np.clip(mirror_x - away_offset, 5, 95)
            # Mirror y position for lateral roles
            mirror_y = 100 - y
            player = Player(i + 21, 'away', role, x, mirror_y, number, lat, lon)
            if role != 'GK':
                player.home_pos[0] = x
            self.players.append(player)
        
        # Give ball to Home FWD for kickoff
        self.ball.owner_id = 10
        self.ball.pos = np.array([50.0, 50.0])
        self.players[9].has_ball = True  # Index 9 is the first home forward
        self.players[9].pos = np.array([50.0, 50.0])
        self.state = GameState.PLAYING
        self.last_touch = LastTouch(team='home', player_id=10)
    
    def set_tactics(self, formation=None, mentality=None, dribble_frequency=None, shoot_frequency=None):
        """Set tactical options for home team. Called by API."""
        if formation and formation in FORMATIONS:
            self.home_formation = formation
        if mentality and mentality in MENTALITY_OFFSETS:
            self.home_mentality = mentality
        if dribble_frequency is not None:
            self.home_dribble_frequency = max(0.5, min(2.0, dribble_frequency))
        if shoot_frequency is not None:
            self.home_shoot_frequency = max(0.5, min(2.0, shoot_frequency))
        
        # Update player positions based on new tactics
        self._apply_formation_to_team('home')
    
    def _apply_formation_to_team(self, team):
        """Apply current formation and mentality to a team's players."""
        if team == 'home':
            formation = FORMATIONS.get(self.home_formation, FORMATIONS['4-4-2'])
            offset = MENTALITY_OFFSETS.get(self.home_mentality, 0)
            start_id = 1
        else:
            formation = FORMATIONS.get(self.away_formation, FORMATIONS['4-4-2'])
            offset = -MENTALITY_OFFSETS.get(self.away_mentality, 0)  # Inverted for away
            start_id = 21
        
        for i, (role, lat, lon, base_x, y, number) in enumerate(formation):
            player_id = start_id + i
            player = next((p for p in self.players if p.id == player_id), None)
            if not player:
                continue
            
            if team == 'away':
                # Mirror positions for away team
                base_x = 100 - base_x
                y = 100 - y
            
            # Apply mentality offset (not for GK)
            new_x = base_x if role == 'GK' else np.clip(base_x + offset, 5, 95)
            
            # Update player properties
            player.role = role
            player.lateral_role = lat
            player.longitudinal_role = lon
            player.number = number
            player.home_pos = np.array([float(new_x), float(y)])
            player.zone_weight = ZONE_WEIGHT_BY_ROLE.get(role, 0.5)
            
            # Immediately move player toward new home position (smooth transition)
            # Players without ball teleport instantly for better responsiveness
            if not player.has_ball:
                player.pos = player.home_pos.copy()
            
            # Update zone bounds
            role_key = (lat, lon, team)
            if role_key in ROLE_ZONE_BOUNDS:
                player.zone_bounds = ROLE_ZONE_BOUNDS[role_key]
    
    def get_tactics(self):
        """Return current tactical settings for home team."""
        return {
            'formation': self.home_formation,
            'mentality': self.home_mentality,
            'dribbleFrequency': self.home_dribble_frequency,
            'shootFrequency': self.home_shoot_frequency,
            'availableFormations': list(FORMATIONS.keys()),
            'availableMentalities': list(MENTALITY_OFFSETS.keys()),
        }

    def iterate(self):
        self.time += 0.1
        
        # Handle non-playing states
        if self.state != GameState.PLAYING:
            self.state_timer -= 0.1
            if self.state_timer <= 0:
                self._execute_restart()
            return
        
        # Store last in-bounds position before moving
        if 0 < self.ball.pos[0] < 100 and 0 < self.ball.pos[1] < 100:
            self.last_ball_pos = self.ball.pos.copy()
        
        # Ball physics - always apply velocity and friction
        self.ball.pos += self.ball.vel
        self.ball.vel *= BALL_FRICTION
        if np.linalg.norm(self.ball.vel) < 0.05:
            self.ball.vel = np.zeros(2)
        
        # Check for goalkeeper saves before detecting goals
        self._check_goalkeeper_save()
        
        # Detect ball events (out of bounds, goals)
        event = self._detect_ball_event()
        if event is not None:
            self._transition_state(event)
            return
        
        # Determine ball controller by proximity
        # Find all players and their distances to the ball
        player_distances = []
        for p in self.players:
            dist = np.linalg.norm(p.pos - self.ball.pos)
            player_distances.append((p, dist))
        
        # Find minimum distance
        min_dist = min(d for _, d in player_distances)
        
        # Find all players at that minimum distance (within small tolerance)
        closest_players = [p for p, d in player_distances if abs(d - min_dist) < 0.5]
        
        # Clear old ownership
        for p in self.players:
            p.has_ball = False
        self.ball.owner_id = None
        
        # Only assign control if someone is close enough AND not on touch cooldown
        if min_dist < TACKLE_DISTANCE and closest_players:
            # Filter out players still on cooldown (either dribble or tackle stall)
            eligible_players = [p for p in closest_players if p.touch_cooldown_until <= self.time]
            
            if eligible_players:
                # Check if this is a tackle (defender winning ball from opponent)
                previous_owner_team = self.last_touch.team if self.last_touch else None
                previous_owner_id = self.last_touch.player_id if self.last_touch else None
                
                # Random tiebreaker if multiple players equally close
                controller = random.choice(eligible_players)
                controller.has_ball = True
                controller.touch_cooldown_until = 0.0  # Reset cooldown on ball acquisition
                self.ball.owner_id = controller.id
                
                # Track who passed to this player (for avoiding return passes)
                # Only track if this was a same-team pass (not interception/tackle)
                is_same_team_pass = (previous_owner_team == controller.team) and self.ball.passer_id is not None
                if is_same_team_pass:
                    controller.received_from_player_id = self.ball.passer_id
                else:
                    controller.received_from_player_id = None  # Clear on interception/tackle
                
                self.ball.clear_pass_target()  # Clear pass tracking on control
                self.last_touch = LastTouch(team=controller.team, player_id=controller.id)
                
                # Mark throw-in as touched (for goal prevention rule)
                if self.last_restart_type == 'throw_in':
                    self.throw_in_touched = True
                
                # TACKLE STALL: The player who lost the ball gets a stall cooldown
                is_tackle = previous_owner_team is not None and previous_owner_team != controller.team
                if is_tackle and previous_owner_id is not None:
                    # Find the player who lost the ball and stall them
                    for p in self.players:
                        if p.id == previous_owner_id:
                            p.touch_cooldown_until = self.time + 1.0  # 1 second stall after being tackled
                            break
                
                # TACKLE CLEARANCE: If defender wins ball from opponent, immediately clear it
                if is_tackle and controller.role == 'DEF':
                    self._execute_tackle_clearance(controller)
        
        # Update Players - each player makes a decision
        for p in self.players:
            p.make_decision(self)

    def _detect_ball_event(self):
        """Detect if ball went out of play. Returns event type or None."""
        x, y = self.ball.pos[0], self.ball.pos[1]
        
        # Ball crossed end line (x < 0 or x > 100)
        if x <= 0:
            if GOAL_TOP < y < GOAL_BOTTOM:
                # Cannot score directly from throw-in
                if self.last_restart_type == 'throw_in' and not self.throw_in_touched:
                    return 'goal_kick_home'  # Award goal kick instead
                return 'goal_away'  # Away team scored (ball in home goal)
            else:
                # Corner or goal kick based on last touch
                if self.last_touch and self.last_touch.team == 'home':
                    return 'corner_away'  # Away gets corner
                else:
                    return 'goal_kick_home'  # Home gets goal kick
        
        if x >= 100:
            if GOAL_TOP < y < GOAL_BOTTOM:
                # Cannot score directly from throw-in
                if self.last_restart_type == 'throw_in' and not self.throw_in_touched:
                    return 'goal_kick_away'  # Award goal kick instead
                return 'goal_home'  # Home team scored
            else:
                if self.last_touch and self.last_touch.team == 'away':
                    return 'corner_home'  # Home gets corner
                else:
                    return 'goal_kick_away'  # Away gets goal kick
        
        # Ball crossed sideline (y < 0 or y > 100)
        if y <= 0 or y >= 100:
            if self.last_touch:
                if self.last_touch.team == 'home':
                    return 'throw_in_away'
                else:
                    return 'throw_in_home'
            return 'throw_in_home'  # Default
        
        return None

    def _check_goalkeeper_save(self):
        """Check if a goalkeeper can save an incoming shot."""
        ball_speed = np.linalg.norm(self.ball.vel)
        
        # Check for any moving ball (lowered threshold for friction-affected balls)
        if ball_speed < 0.8:
            return
        
        # Don't attempt saves if ball already crossed goal line
        if self.ball.pos[0] <= 0 or self.ball.pos[0] >= 100:
            return
        
        # Check each goalkeeper
        for p in self.players:
            if p.role != 'GK':
                continue
            
            # Calculate distance to ball
            dist_to_ball = np.linalg.norm(p.pos - self.ball.pos)
            
            # GK can save if ball is within reach (increased from 5 to 8)
            if dist_to_ball > 8.0:
                continue
            
            # Check if ball is heading toward GK's goal
            if p.team == 'home':
                if self.ball.vel[0] > 0.1:  # Ball going away from home goal
                    continue
                # Ball must be in goal area (x < 20)
                if self.ball.pos[0] > 20:
                    continue
            else:
                if self.ball.vel[0] < -0.1:  # Ball going away from away goal
                    continue
                # Ball must be in goal area (x > 80)
                if self.ball.pos[0] < 80:
                    continue
            
            # Calculate save difficulty based on:
            # 1. Ball speed (faster = harder to save)
            # 2. Distance from GK (closer = easier)
            # 3. Shot placement (corners harder to save)
            
            # Speed difficulty: faster shots are harder
            speed_difficulty = min(ball_speed / 3.0, 1.0)
            
            # Distance factor: closer is easier to save (updated for 8 unit range)
            distance_factor = 1.0 - (dist_to_ball / 8.0)
            
            # Corner difficulty: shots toward goal corners are harder
            goal_y_center = 50.0
            ball_y_offset = abs(self.ball.pos[1] - goal_y_center)
            corner_difficulty = ball_y_offset / 10.0  # Max 1.0 at edges
            
            # Calculate save probability
            base_save_chance = 0.7  # GK is generally good
            save_chance = base_save_chance * distance_factor * (1.0 - speed_difficulty * 0.5) * (1.0 - corner_difficulty * 0.4)
            save_chance = max(0.1, min(0.9, save_chance))  # Clamp between 10% and 90%
            
            # Roll for save
            if random.random() < save_chance:
                # SAVE! Determine if catch or deflect
                if ball_speed < 2.0 and distance_factor > 0.7:
                    # Catch the ball - GK gets possession
                    self.ball.vel = np.zeros(2)
                    self.ball.pos = p.pos.copy()
                    self.ball.owner_id = p.id
                    p.has_ball = True
                    self.last_touch = LastTouch(team=p.team, player_id=p.id)
                else:
                    # Deflect - reflect ball velocity off goal plane with safety margins
                    if p.team == 'home':
                        # Home GK: reflect x-velocity to positive (away from x=0)
                        deflect_x = abs(self.ball.vel[0]) if self.ball.vel[0] != 0 else 1.0
                        safe_x = 8.0  # Safe distance from goal line
                    else:
                        # Away GK: reflect x-velocity to negative (away from x=100)
                        deflect_x = -abs(self.ball.vel[0]) if self.ball.vel[0] != 0 else -1.0
                        safe_x = 92.0  # Safe distance from goal line
                    
                    # Add randomness to y-component for varied deflection angles
                    deflect_y = self.ball.vel[1] * 0.5 + (random.random() - 0.5) * 1.5
                    deflect_dir = normalize(np.array([deflect_x, deflect_y]))
                    
                    # Set new velocity (slower after deflection)
                    self.ball.vel = deflect_dir * ball_speed * 0.4
                    
                    # Force ball position to safe zone away from goal
                    new_pos = np.array([safe_x, p.pos[1] + deflect_y * 2.0])
                    new_pos[1] = np.clip(new_pos[1], 10.0, 90.0)  # Keep well in bounds
                    self.ball.pos = new_pos
                    
                    self.last_touch = LastTouch(team=p.team, player_id=p.id)
                break  # Only one GK can save

    def _execute_tackle_clearance(self, defender):
        """Defender immediately clears the ball after winning a tackle."""
        defender.has_ball = False
        self.ball.owner_id = None
        
        # Find teammates and opponent positions
        teammates = [p for p in self.players if p.team == defender.team and p.id != defender.id]
        opponents = [p for p in self.players if p.team != defender.team]
        opponent_positions = [p.pos for p in opponents]
        
        # Try to find a safe teammate to pass to
        best_target = None
        best_safety = 0.0
        
        for teammate in teammates:
            if teammate.role == 'GK':
                continue
            
            dist = np.linalg.norm(teammate.pos - defender.pos)
            if dist < 10.0 or dist > 35.0:
                continue
            
            # Check lane quality
            lane_quality = calculate_passing_lane_quality(defender.pos, teammate.pos, opponent_positions)
            
            # Prefer forward passes (away from own goal)
            if defender.team == 'home':
                forward_bonus = 0.2 if teammate.pos[0] > defender.pos[0] else 0.0
            else:
                forward_bonus = 0.2 if teammate.pos[0] < defender.pos[0] else 0.0
            
            safety = lane_quality + forward_bonus
            
            if safety > best_safety and lane_quality > 0.5:
                best_safety = safety
                best_target = teammate.pos
        
        # If safe teammate found, pass to them
        if best_target is not None:
            pass_dir = normalize(best_target - defender.pos)
            self.ball.vel = pass_dir * BALL_PASS_SPEED
        else:
            # No safe teammate - clear to touchline
            # Pick the nearest touchline (top or bottom)
            if defender.pos[1] < 50:
                target_y = 0.0  # Top touchline
            else:
                target_y = 100.0  # Bottom touchline
            
            # Aim slightly forward too
            if defender.team == 'home':
                target_x = defender.pos[0] + 20.0
            else:
                target_x = defender.pos[0] - 20.0
            target_x = float(np.clip(target_x, 10.0, 90.0))
            
            clear_target = np.array([target_x, target_y])
            clear_dir = normalize(clear_target - defender.pos)
            self.ball.vel = clear_dir * BALL_SHOOT_SPEED * 0.7  # Strong clearance
        
        self.last_touch = LastTouch(team=defender.team, player_id=defender.id)

    def _transition_state(self, event):
        """Transition game state based on event."""
        # Clear ball ownership
        for p in self.players:
            p.has_ball = False
        self.ball.owner_id = None
        self.ball.vel = np.zeros(2)
        
        if event == 'goal_home':
            self.score['home'] += 1
            self.state = GameState.GOAL_SCORED
            self.state_timer = 2.0
            self.restart_team = 'away'
            self.restart_pos = np.array([50.0, 50.0])
        elif event == 'goal_away':
            self.score['away'] += 1
            self.state = GameState.GOAL_SCORED
            self.state_timer = 2.0
            self.restart_team = 'home'
            self.restart_pos = np.array([50.0, 50.0])
        elif event == 'corner_home':
            self.state = GameState.CORNER_KICK
            self.state_timer = 1.0
            self.restart_team = 'home'
            # Use last in-bounds position to determine corner side
            corner_y = 2.0 if self.last_ball_pos[1] < 50 else 98.0
            self.restart_pos = np.array([98.0, corner_y])
        elif event == 'corner_away':
            self.state = GameState.CORNER_KICK
            self.state_timer = 1.0
            self.restart_team = 'away'
            # Use last in-bounds position to determine corner side
            corner_y = 2.0 if self.last_ball_pos[1] < 50 else 98.0
            self.restart_pos = np.array([2.0, corner_y])
        elif event == 'goal_kick_home':
            self.state = GameState.GOAL_KICK
            self.state_timer = 1.0
            self.restart_team = 'home'
            self.restart_pos = np.array([10.0, 50.0])
        elif event == 'goal_kick_away':
            self.state = GameState.GOAL_KICK
            self.state_timer = 1.0
            self.restart_team = 'away'
            self.restart_pos = np.array([90.0, 50.0])
        elif event == 'throw_in_home':
            self.state = GameState.THROW_IN
            self.state_timer = 0.5
            self.restart_team = 'home'
            self.last_restart_type = 'throw_in'
            self.throw_in_touched = False
            # Use last in-bounds position for x, clamp y inside field
            y_pos = 2.0 if self.last_ball_pos[1] < 50 else 98.0
            x_pos = float(np.clip(self.last_ball_pos[0], 5, 95))
            self.restart_pos = np.array([x_pos, y_pos])
        elif event == 'throw_in_away':
            self.state = GameState.THROW_IN
            self.state_timer = 0.5
            self.restart_team = 'away'
            self.last_restart_type = 'throw_in'
            self.throw_in_touched = False
            # Use last in-bounds position for x, clamp y inside field
            y_pos = 2.0 if self.last_ball_pos[1] < 50 else 98.0
            x_pos = float(np.clip(self.last_ball_pos[0], 5, 95))
            self.restart_pos = np.array([x_pos, y_pos])
        
        # Move ball to restart position
        self.ball.pos = self.restart_pos.copy()

    def _execute_restart(self):
        """Execute the restart and return to playing state."""
        if self.state == GameState.GOAL_SCORED:
            # Reset all players to starting positions
            self._reset_player_positions()
            # Transition to kickoff state
            self.state = GameState.KICKOFF
            self.state_timer = 1.0
            self.restart_pos = np.array([50.0, 50.0])
            self.ball.pos = self.restart_pos.copy()
            return
        
        # Find nearest player from restart team to take the restart
        restart_player = None
        min_dist = float('inf')
        for p in self.players:
            if p.team == self.restart_team and p.role != 'GK':
                dist = np.linalg.norm(p.pos - self.restart_pos)
                if dist < min_dist:
                    min_dist = dist
                    restart_player = p
        
        if restart_player:
            restart_player.pos = self.restart_pos.copy()
            restart_player.has_ball = True
            self.ball.owner_id = restart_player.id
            self.ball.pos = restart_player.pos.copy()
            self.last_touch = LastTouch(team=restart_player.team, player_id=restart_player.id)
        
        self.state = GameState.PLAYING

    def _reset_player_positions(self):
        """Reset all players to their home positions."""
        for p in self.players:
            p.pos = p.home_pos.copy()
            p.vel = np.zeros(2)
            p.has_ball = False

    def reset_positions(self):
        self.init_players()

    def to_dict(self):
        return {
            "players": [p.to_dict() for p in self.players],
            "ball": self.ball.to_dict(),
            "score": self.score,
            "time": self.time,
            "state": self.state.value,
            "lastTouch": {"team": self.last_touch.team, "playerId": self.last_touch.player_id} if self.last_touch else None,
            "restartTeam": self.restart_team,
            "tactics": self.get_tactics()
        }
