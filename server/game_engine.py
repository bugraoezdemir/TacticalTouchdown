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
BALL_PASS_SPEED = 2.0
BALL_SHOOT_SPEED = 3.0
BALL_DRIBBLE_SPEED = 0.6
TACKLE_DISTANCE = 2.0
BALL_FRICTION = 0.95

# Decision weights - safety-first passing with stricter thresholds
PASS_SAFETY_WEIGHT = 0.6       # Was 0.5 - value safe passes more
PASS_GOAL_PROGRESS_WEIGHT = 0.3 # Was 0.4 - balance with safety
PASS_DISTANCE_WEIGHT = 0.3
SPACE_PASS_BONUS = 0.15  # Re-enabled for space passes

DRIBBLE_CLEARANCE_WEIGHT = 0.35  # Reduced - dribble less
DRIBBLE_GOAL_PROGRESS_WEIGHT = 0.35  # Reduced - favor passing

SHOOT_DISTANCE_THRESHOLD = 35.0  # Increased from 25 - shoot from further
SHOOT_ANGLE_THRESHOLD = 30.0

# Action thresholds - balanced for good passing frequency
SHOOT_SCORE_THRESHOLD = 0.10  # Very low threshold - shoot frequently when chance arises
PASS_SCORE_THRESHOLD = 0.18   # Balanced - safe passes but not too restrictive

# Pass evaluation - no artificial bonus, safety-first approach
PASS_BONUS = 0.0  # Removed - let natural safety scoring decide

# Home position attraction weight
HOME_POSITION_WEIGHT = 0.3  # 30% home bias, 70% tactical

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
        
        # GK should pass to defenders, only clear under heavy pressure
        if self.role == 'GK':
            pass_option, pass_target, pass_score = self._evaluate_pass(ctx)
            if pass_score >= 0.2 and pass_option is not None:
                self._execute_pass(ctx, game, pass_option, pass_target)
            else:
                self._execute_clearance(ctx, game)
            return
        
        # Evaluate all options
        shoot_score = self._evaluate_shoot(ctx)
        pass_option, pass_target, pass_score = self._evaluate_pass(ctx)
        dribble_dir, dribble_score = self._evaluate_dribble(ctx)
        
        # Apply tactical frequency multipliers for home team
        if self.team == 'home':
            shoot_score *= game.home_shoot_frequency
            dribble_score *= game.home_dribble_frequency
        
        # DRIBBLING ONLY ALLOWED IN FINAL THIRD (near opponent goal)
        in_final_third = ctx.dist_to_goal < 33.0  # Last 1/3 of pitch
        can_dribble = in_final_third
        
        # Check pressure level for clearance decision
        min_opp_dist = float('inf')
        for opp_pos in ctx.opponent_positions:
            d = np.linalg.norm(opp_pos - self.pos)
            min_opp_dist = min(min_opp_dist, float(d))
        
        # IN FINAL THIRD: Can shoot, pass, or dribble
        if in_final_third:
            if shoot_score > 0.25:
                self._execute_shoot(ctx, game)
            elif pass_score >= 0.15 and pass_option is not None:
                self._execute_pass(ctx, game, pass_option, pass_target)
            elif dribble_score > 0.2:
                self._execute_dribble(ctx, game, dribble_dir)
            elif pass_option is not None:
                self._execute_pass(ctx, game, pass_option, pass_target)
            else:
                self._execute_dribble(ctx, game, dribble_dir)
        else:
            # OUTSIDE FINAL THIRD: Short passing only, no dribbling
            # Only clear under EXTREME pressure with no pass option
            if pass_score >= 0.1 and pass_option is not None:
                self._execute_pass(ctx, game, pass_option, pass_target)
            elif min_opp_dist < 5.0 and pass_score < 0.1:
                # Under extreme pressure with no pass - clear
                self._execute_clearance(ctx, game)
            elif pass_option is not None:
                # Any pass is better than clearance
                self._execute_pass(ctx, game, pass_option, pass_target)
            else:
                # No pass option at all - must clear
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
        
        # Generate candidate pass positions in 360 degrees at various distances
        num_directions = 16  # Check 16 directions around player
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
            min_intercept_margin = float('inf')
            for opp_pos in ctx.opponent_positions:
                closest_on_path = project_point_to_segment(opp_pos, self.pos, target_pos)
                dist_to_path = np.linalg.norm(opp_pos - closest_on_path)
                dist_along_path = np.linalg.norm(closest_on_path - self.pos)
                
                time_ball_at_point = dist_along_path / BALL_PASS_SPEED
                time_opp_at_point = dist_to_path / PLAYER_SPRINT_SPEED
                
                margin = time_ball_at_point - time_opp_at_point
                min_intercept_margin = min(min_intercept_margin, float(margin))
            
            # Skip if easily interceptable - STRICTER threshold
            if min_intercept_margin < 0.5:
                continue
            
            # Score components
            safety_score = min(1.0, min_intercept_margin / 2.0)
            
            # Space around target with larger radius
            space_score = 1.0
            for opp_pos in ctx.opponent_positions:
                opp_dist = np.linalg.norm(opp_pos - target_pos)
                if opp_dist < 12.0:  # Larger check radius
                    space_score -= (1.0 - opp_dist / 12.0) * 0.25
            space_score = max(0.0, space_score)
            
            # Progress toward goal
            my_goal_dist = ctx.dist_to_goal
            target_goal_dist = distance_to_goal(target_pos, ctx.team)
            progress_score = 0.5 + 0.5 * (my_goal_dist - target_goal_dist) / max(my_goal_dist, 1)
            progress_score = float(np.clip(progress_score, 0, 1))
            
            # Teammate accessibility bonus
            access_score = 1.0 - min(best_reach_time / 3.0, 1.0)
            
            # Combined score with lane quality and vision
            combined_safety = safety_score * lane_quality
            
            score = (
                0.45 * combined_safety +
                0.20 * space_score +
                0.20 * progress_score +
                0.15 * access_score -
                vision_penalty
            )
            
            # Bonus for very safe options in clear lanes
            if combined_safety > 0.6 and space_score > 0.6:
                score += SPACE_PASS_BONUS
            
            if score > best_score:
                best_score = score
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
        
        # STRONGER back pass penalty
        back_pass_penalty = 0.25 if is_back_pass else 0.0
        
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
            
            # STRICTER margin check: 1.0 instead of 0.5
            time_margin = time_ball_reaches_point - time_opp_reaches_point
            
            if time_margin < 1.0:  # Opponent arrives within 1.0 time units of ball
                can_be_intercepted = True
                risk_factor = min(1.0, max(0.0, 1.0 - time_margin))
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
        
        # REJECT if combined safety is too low
        if combined_safety < 0.4:
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
        """Evaluate dribbling options. Returns (best_direction, score)."""
        
        num_directions = 12
        angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
        
        best_dir = normalize(ctx.goal_center - self.pos)
        best_score = 0.0
        
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            clearance = clearance_in_direction(self.pos, direction, ctx.opponent_positions)
            clearance_factor = clearance / 20.0
            
            goal_dir = normalize(ctx.goal_center - self.pos)
            alignment = np.dot(direction, goal_dir)
            goal_factor = (alignment + 1) / 2
            
            if ctx.team == 'home' and direction[0] < -0.5:
                goal_factor *= 0.3
            elif ctx.team == 'away' and direction[0] > 0.5:
                goal_factor *= 0.3
            
            future_pos = self.pos + direction * 10
            if future_pos[1] < 10 or future_pos[1] > 90:
                clearance_factor *= 0.5
            
            score = (
                DRIBBLE_CLEARANCE_WEIGHT * clearance_factor +
                DRIBBLE_GOAL_PROGRESS_WEIGHT * goal_factor
            )
            
            if score > best_score:
                best_score = score
                best_dir = direction
        
        return best_dir, best_score

    def _execute_shoot(self, ctx, game):
        """Execute a shot on goal - target corners away from goalkeeper."""
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
        self.has_ball = False
        game.ball.owner_id = None
        
        # Use provided target position (for space passes) or teammate position
        if target_pos is None:
            # Pass directly to teammate with small lead
            target_pos = target_player.pos + target_player.vel * 3.0
        
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
        
        # Set touch cooldown - cannot reacquire ball for DRIBBLE_TOUCH_INTERVAL ticks
        # Each tick is 0.1 time units, so multiply by 0.1
        self.touch_cooldown_until = game.time + (DRIBBLE_TOUCH_INTERVAL * 0.1)
        
        # Kick ball ahead by DRIBBLE_TOUCH_DISTANCE in the movement direction
        kick_distance = DRIBBLE_TOUCH_DISTANCE
        game.ball.pos = self.pos + direction * kick_distance
        
        # Give ball velocity faster than player to stay ahead between touches
        game.ball.vel = direction * PLAYER_SPRINT_SPEED * 1.2
        
        # Player sprints to chase the ball
        self.vel = direction * PLAYER_SPRINT_SPEED
        
        # Track last touch for out-of-bounds decisions
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
            
            # Decide whether to chase based on multiple factors
            should_chase = False
            if my_dist < 8.0:  # Very close - always chase
                should_chase = True
            elif self.role == 'DEF' and ball_in_defensive_half and my_dist < 35.0:
                # DEFENDERS ARE AGGRESSIVE: Chase when ball enters their half
                should_chase = True
            elif ball_near_own_goal and self.role == 'DEF' and my_dist < 25.0:
                # Defenders MUST chase when ball is near own goal
                should_chase = True
            elif ball_in_zone and closer_teammates == 0:  # In our zone and we're closest teammate
                should_chase = True
            elif my_dist < closest_opp_dist and closer_teammates == 0:  # We can get there before opponent
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
        if ball_owner.team == self.team:
            tactical_target = self._find_support_spot(ctx, ball_owner, game)
        else:
            tactical_target = self._get_defend_target(ctx, ball_owner)
        
        # Clamp target to zone with role-based flexibility
        clamped_target = self._clamp_to_zone(tactical_target)
        
        # Blend with home position - use HOME_POSITION_WEIGHT scaled by role's zone_weight
        # Defenders blend more toward home, attackers follow tactical target more
        home_blend = HOME_POSITION_WEIGHT * self.zone_weight
        blended_target = (1 - home_blend) * clamped_target + home_blend * self.home_pos
        
        # Move towards blended target
        to_target = blended_target - self.pos
        dist = np.linalg.norm(to_target)
        
        if dist > 2.0:
            self.vel = normalize(to_target) * PLAYER_SPEED
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

    def _find_support_spot(self, ctx, ball_owner, game):
        """Find best support position - safe for receiving passes, within zone."""
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
        candidates.append(self.pos + goal_dir * 8.0 - perp * 6.0)
        
        # 4. Support positions relative to ball carrier
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
            
            # 1. Pass safety - can ball reach here without interception?
            pass_safety = 1.0
            for opp_pos in ctx.opponent_positions:
                can_intercept, time_margin = time_to_intercept(
                    opp_pos, PLAYER_SPRINT_SPEED,
                    ball_owner.pos, cand, BALL_PASS_SPEED
                )
                if can_intercept:
                    pass_safety *= max(0.1, float(0.5 + time_margin))
            score += pass_safety * 0.4
            
            # 2. Space from opponents
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
        goal_center_y = 50.0  # Center of goal
        
        # Calculate ball distance from goal
        ball_dist_to_goal = abs(ball_pos[0] - own_goal_x)
        ball_close_to_goal = ball_dist_to_goal < 25.0  # Ball within 25 units of goal
        
        # Check if ball is in GK's penalty area (and loose)
        ball_in_penalty = (self.team == 'home' and ball_pos[0] < penalty_x) or \
                          (self.team == 'away' and ball_pos[0] > penalty_x)
        
        # If ball is loose and in penalty area, GK can chase it
        if game.ball.owner_id is None and ball_in_penalty:
            target_x = np.clip(ball_pos[0], min_x, max_x)
            target_y = np.clip(ball_pos[1], 35.0, 65.0)
            target = np.array([target_x, target_y])
            
            to_target = target - self.pos
            dist = np.linalg.norm(to_target)
            if dist > 1.0:
                self.vel = normalize(to_target) * PLAYER_SPRINT_SPEED
            else:
                self.vel = np.zeros(2)
            return
        
        # STAY CENTERED unless ball is close to goal
        target_x = goal_x
        if ball_close_to_goal:
            # Ball is close - track it laterally but not too aggressively
            # Ease toward ball y with damping - don't fully commit
            target_y = goal_center_y + (ball_pos[1] - goal_center_y) * 0.6
            target_y = np.clip(target_y, 40.0, 60.0)  # Stay within goal posts
        else:
            # Ball is far - stay in center of goal
            target_y = goal_center_y
        
        target = np.array([target_x, target_y])
        to_target = target - self.pos
        dist = np.linalg.norm(to_target)
        
        if dist > 1.0:
            self.vel = normalize(to_target) * PLAYER_SPEED * 0.7  # Move slowly to center
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
            # Filter out players still on dribble cooldown
            eligible_players = [p for p in closest_players if p.touch_cooldown_until <= self.time]
            
            if eligible_players:
                # Random tiebreaker if multiple players equally close
                controller = random.choice(eligible_players)
                controller.has_ball = True
                controller.touch_cooldown_until = 0.0  # Reset cooldown on ball acquisition
                self.ball.owner_id = controller.id
                self.last_touch = LastTouch(team=controller.team, player_id=controller.id)
        
        # Update Players - each player makes a decision
        for p in self.players:
            p.make_decision(self)

    def _detect_ball_event(self):
        """Detect if ball went out of play. Returns event type or None."""
        x, y = self.ball.pos[0], self.ball.pos[1]
        
        # Ball crossed end line (x < 0 or x > 100)
        if x <= 0:
            if GOAL_TOP < y < GOAL_BOTTOM:
                return 'goal_away'  # Away team scored (ball in home goal)
            else:
                # Corner or goal kick based on last touch
                if self.last_touch and self.last_touch.team == 'home':
                    return 'corner_away'  # Away gets corner
                else:
                    return 'goal_kick_home'  # Home gets goal kick
        
        if x >= 100:
            if GOAL_TOP < y < GOAL_BOTTOM:
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
            # Use last in-bounds position for x, clamp y inside field
            y_pos = 2.0 if self.last_ball_pos[1] < 50 else 98.0
            x_pos = float(np.clip(self.last_ball_pos[0], 5, 95))
            self.restart_pos = np.array([x_pos, y_pos])
        elif event == 'throw_in_away':
            self.state = GameState.THROW_IN
            self.state_timer = 0.5
            self.restart_team = 'away'
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
