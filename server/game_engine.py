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

# Decision weights - increased pass weights so players pass more often
PASS_SAFETY_WEIGHT = 0.5       # Was 0.4 - value safe passes more
PASS_GOAL_PROGRESS_WEIGHT = 0.4 # Was 0.3 - value forward progress more
PASS_DISTANCE_WEIGHT = 0.3
SPACE_PASS_BONUS = 0.2  # Bonus for passing to space ahead of running teammates

DRIBBLE_CLEARANCE_WEIGHT = 0.35  # Reduced - dribble less
DRIBBLE_GOAL_PROGRESS_WEIGHT = 0.35  # Reduced - favor passing

SHOOT_DISTANCE_THRESHOLD = 35.0  # Increased from 25 - shoot from further
SHOOT_ANGLE_THRESHOLD = 30.0

# Lowered thresholds for actions
SHOOT_SCORE_THRESHOLD = 0.20  # Low threshold - shoot when opportunity arises
PASS_SCORE_THRESHOLD = 0.15   # Only pass if it's a decent option (no back passes)

# Increased pass appeal, reduced dribble appeal
PASS_BONUS = 0.15  # Added bonus to pass scores

# Home position attraction weight
HOME_POSITION_WEIGHT = 0.3  # 30% home bias, 70% tactical

# Dribble touch system
DRIBBLE_TOUCH_INTERVAL = 5  # Every N ticks, ball moves ahead
DRIBBLE_TOUCH_DISTANCE = 2.5  # How far ball moves ahead (must be > TACKLE_DISTANCE to allow interception)

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
        
        # GK always clears the ball immediately when they have it
        if self.role == 'GK':
            self._execute_clearance(ctx, game)
            return
        
        # Check if we need a defensive clearance (ball near own goal) - not for GK
        own_goal_x = 0.0 if self.team == 'home' else 100.0
        dist_to_own_goal = abs(self.pos[0] - own_goal_x)
        
        # If very close to own goal, clear the ball upfield
        if dist_to_own_goal < 20.0 and self.role != 'FWD':
            self._execute_clearance(ctx, game)
            return
        
        shoot_score = self._evaluate_shoot(ctx)
        pass_option, pass_target, pass_score = self._evaluate_pass(ctx)
        dribble_dir, dribble_score = self._evaluate_dribble(ctx)
        
        # Apply tactical frequency multipliers for home team
        if self.team == 'home':
            shoot_score *= game.home_shoot_frequency
            dribble_score *= game.home_dribble_frequency
        
        # DEFENDER SAFETY CHECK: Clear if pass is risky or under pressure
        if self.role == 'DEF':
            min_opp_dist = float('inf')
            for opp_pos in ctx.opponent_positions:
                d = np.linalg.norm(opp_pos - self.pos)
                min_opp_dist = min(min_opp_dist, float(d))
            
            # Clear if: pass score is low OR opponent is close
            if pass_score < 0.5 or min_opp_dist < 12.0:
                self._execute_clearance(ctx, game)
                return
        
        # Attackers in shooting range should prefer dribble/shoot unless pass is very safe
        in_shooting_range = ctx.dist_to_goal < 30.0
        is_attacker = self.role in ['FWD', 'MID']
        
        if in_shooting_range and is_attacker:
            # In attacking zone: shoot > dribble > very safe pass
            if shoot_score > SHOOT_SCORE_THRESHOLD:
                self._execute_shoot(ctx, game)
            elif dribble_score > 0.3:
                self._execute_dribble(ctx, game, dribble_dir)
            elif pass_score > 0.6 and pass_option is not None:  # Only very safe passes
                self._execute_pass(ctx, game, pass_option, pass_target)
            elif shoot_score > 0.1:
                self._execute_shoot(ctx, game)
            else:
                self._execute_dribble(ctx, game, dribble_dir)
        else:
            # Normal decision making outside shooting range
            if shoot_score > SHOOT_SCORE_THRESHOLD and shoot_score >= pass_score * 0.8:
                self._execute_shoot(ctx, game)
            elif pass_score > PASS_SCORE_THRESHOLD and pass_score > dribble_score and pass_option is not None:
                self._execute_pass(ctx, game, pass_option, pass_target)
            elif dribble_score > 0.2:
                self._execute_dribble(ctx, game, dribble_dir)
            elif shoot_score > 0.1:
                self._execute_shoot(ctx, game)
            else:
                self._execute_dribble(ctx, game, dribble_dir)

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
        """Evaluate passing options. Returns (best_teammate, target_pos, score)."""
        if len(ctx.teammates) == 0:
            return None, None, 0.0
        
        best_teammate = None
        best_target = None
        best_score = 0.0
        
        for teammate in ctx.teammates:
            if teammate.role == 'GK' and ctx.dist_to_goal < 50:
                continue
            
            # Evaluate direct pass to teammate
            direct_score = self._evaluate_pass_to_target(ctx, teammate.pos, teammate)
            direct_target = teammate.pos.copy()
            
            # Evaluate "space pass" - passing ahead of running teammates
            space_score = 0.0
            space_target = None
            if np.linalg.norm(teammate.vel) > 0.2:
                # Calculate lead position 10-15 units ahead of teammate's run
                teammate_dir = normalize(teammate.vel)
                lead_pos = teammate.pos + teammate_dir * 12.0  # 12 units ahead
                
                # Make sure lead position is on field and toward goal
                lead_pos[0] = np.clip(lead_pos[0], 5, 95)
                lead_pos[1] = np.clip(lead_pos[1], 5, 95)
                
                # Check if space pass makes progress toward goal
                lead_goal_dist = distance_to_goal(lead_pos, ctx.team)
                if lead_goal_dist < distance_to_goal(teammate.pos, ctx.team):
                    space_score = self._evaluate_pass_to_target(ctx, lead_pos, teammate) + SPACE_PASS_BONUS
                    space_target = lead_pos
            
            # Use the better of direct pass or space pass
            if space_score > direct_score and space_target is not None:
                score = space_score
                target = space_target
            else:
                score = direct_score
                target = direct_target
            
            if score > best_score:
                best_score = score
                best_teammate = teammate
                best_target = target
        
        return best_teammate, best_target, best_score
    
    def _evaluate_pass_to_target(self, ctx, target_pos, teammate):
        """Evaluate a pass to a specific target position with strict interception analysis."""
        pass_vec = target_pos - self.pos
        pass_dist = np.linalg.norm(pass_vec)
        
        if pass_dist < 5.0 or pass_dist > 50.0:
            return 0.0
        
        # Check if this is a back pass (toward own goal)
        my_goal_dist = ctx.dist_to_goal
        target_goal_dist = distance_to_goal(target_pos, ctx.team)
        is_back_pass = target_goal_dist > my_goal_dist + 3.0
        
        if is_back_pass:
            return 0.05
        
        # GEOMETRIC INTERCEPTION CHECK
        # Calculate ball travel time
        ball_travel_time = pass_dist / BALL_PASS_SPEED
        
        # Check each opponent's ability to intercept
        interception_risk = 0.0
        can_be_intercepted = False
        
        for opp_pos in ctx.opponent_positions:
            # Find closest point on pass path to opponent
            closest_on_path = project_point_to_segment(opp_pos, self.pos, target_pos)
            dist_to_path = np.linalg.norm(opp_pos - closest_on_path)
            
            # Calculate how far along the path this intercept point is
            dist_along_path = np.linalg.norm(closest_on_path - self.pos)
            time_ball_reaches_point = dist_along_path / BALL_PASS_SPEED
            
            # Time for opponent to reach intercept point
            time_opp_reaches_point = dist_to_path / PLAYER_SPRINT_SPEED
            
            # If opponent can reach the ball path before ball arrives - DANGEROUS
            time_margin = time_ball_reaches_point - time_opp_reaches_point
            
            if time_margin < 0.5:  # Opponent arrives within 0.5 time units of ball
                can_be_intercepted = True
                # Risk increases as margin decreases
                risk_factor = max(0, 1.0 - time_margin)
                interception_risk = max(interception_risk, risk_factor)
        
        # Calculate base safety score
        if can_be_intercepted:
            safety_score = max(0.05, 1.0 - interception_risk)
        else:
            safety_score = 1.0
        
        # ROLE-BASED RISK TOLERANCE - HARD GATES for defenders
        # Defenders and GK: if ANY interception risk, return 0 to force clearance
        if self.role == 'DEF' and can_be_intercepted:
            return 0.0  # Defenders NEVER make interceptable passes
        if self.role == 'GK' and (can_be_intercepted or interception_risk > 0.1):
            return 0.0  # GK NEVER makes risky passes
        
        # For other roles, apply penalties
        if can_be_intercepted:
            safety_score *= 0.3  # Significant penalty but still possible
        
        # Progress factor
        progress_factor = 0.5 + 0.5 * (my_goal_dist - target_goal_dist) / max(my_goal_dist, 1)
        progress_factor = float(np.clip(progress_factor, 0, 1))
        
        optimal_dist = 20.0
        dist_factor = 1.0 - abs(pass_dist - optimal_dist) / 50.0
        dist_factor = max(0.0, float(dist_factor))
        
        score = (
            PASS_SAFETY_WEIGHT * safety_score +
            PASS_GOAL_PROGRESS_WEIGHT * progress_factor +
            PASS_DISTANCE_WEIGHT * dist_factor +
            PASS_BONUS
        )
        
        if np.linalg.norm(teammate.vel) > 0.3:
            score += 0.1
        
        return score

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
        """Execute a shot on goal."""
        self.has_ball = False
        game.ball.owner_id = None
        
        target_y = 50.0 + (random.random() - 0.5) * 8
        target = np.array([ctx.goal_x, target_y])
        
        shoot_dir = normalize(target - self.pos)
        game.ball.vel = shoot_dir * BALL_SHOOT_SPEED
        
        # Track last touch
        game.last_touch = LastTouch(team=self.team, player_id=self.id)

    def _execute_pass(self, ctx, game, target_player, target_pos=None):
        """Execute a pass to target position (or teammate if no target specified)."""
        self.has_ball = False
        game.ball.owner_id = None
        
        # Use provided target position (for space passes) or calculate from teammate
        if target_pos is None:
            lead_factor = 0.3
            target_pos = target_player.pos + target_player.vel * lead_factor * 10
        
        # Prevent zero-length passes that cause freeze - ensure minimum distance
        pass_vec = target_pos - self.pos
        pass_dist = np.linalg.norm(pass_vec)
        if pass_dist < 3.0:
            # Fall back to teammate position if target is too close
            target_pos = target_player.pos.copy()
            pass_vec = target_pos - self.pos
            pass_dist = np.linalg.norm(pass_vec)
            if pass_dist < 1.0:
                # Emergency: just kick toward attacking goal
                target_pos = ctx.goal_center
                pass_vec = target_pos - self.pos
        
        pass_dir = normalize(pass_vec)
        game.ball.vel = pass_dir * BALL_PASS_SPEED
        
        # Track last touch
        game.last_touch = LastTouch(team=self.team, player_id=self.id)

    def _execute_dribble(self, ctx, game, direction):
        """Execute dribbling - give ball a small kick in movement direction."""
        self.vel = direction * BALL_DRIBBLE_SPEED
        # Dribble is a small kick - ball gets velocity in the dribble direction
        game.ball.vel = direction * BALL_DRIBBLE_SPEED * 1.5  # Ball moves slightly faster than player
    
    def _execute_clearance(self, ctx, game):
        """Clear the ball upfield away from danger."""
        self.has_ball = False
        game.ball.owner_id = None
        
        # Kick toward attacking goal with some randomness
        target_x = ctx.goal_x
        target_y = 50.0 + (random.random() - 0.5) * 40  # Random y between 30-70
        target = np.array([target_x, target_y])
        
        clear_dir = normalize(target - self.pos)
        game.ball.vel = clear_dir * BALL_SHOOT_SPEED  # Use shoot speed for power
        
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
            
            # Check if ball is near our own goal (defenders must chase)
            own_goal_x = 0.0 if self.team == 'home' else 100.0
            ball_dist_to_own_goal = abs(game.ball.pos[0] - own_goal_x)
            ball_near_own_goal = ball_dist_to_own_goal < 30.0
            
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
            elif ball_near_own_goal and self.role == 'DEF' and my_dist < 25.0:
                # Defenders MUST chase when ball is near own goal
                should_chase = True
            elif ball_in_zone and closer_teammates == 0:  # In our zone and we're closest teammate
                should_chase = True
            elif my_dist < closest_opp_dist and closer_teammates == 0:  # We can get there before opponent
                should_chase = True
            
            if should_chase:
                if my_dist > 1.0:
                    # Defenders near own goal chase directly, others respect zone
                    if ball_near_own_goal and self.role == 'DEF':
                        chase_target = game.ball.pos  # Chase directly
                    else:
                        chase_target = self._clamp_to_zone(game.ball.pos)
                    to_target = chase_target - self.pos
                    self.vel = normalize(to_target) * PLAYER_SPRINT_SPEED
                else:
                    self.vel = np.zeros(2)
            else:
                # Return toward home position
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
        """Special decision making for goalkeepers - stay near goal, track ball laterally."""
        # Define GK zone boundaries
        if self.team == 'home':
            goal_x = 5.0
            min_x = 3.0
            max_x = 15.0  # Stay within 10 units of goal line
            penalty_x = 20.0  # Penalty area boundary
        else:
            goal_x = 95.0
            min_x = 85.0
            max_x = 97.0
            penalty_x = 80.0
        
        ball_pos = game.ball.pos
        
        # Check if ball is in GK's penalty area (and loose)
        ball_in_penalty = (self.team == 'home' and ball_pos[0] < penalty_x) or \
                          (self.team == 'away' and ball_pos[0] > penalty_x)
        
        # If ball is loose and in penalty area, GK can chase it (but still respect x limits)
        if game.ball.owner_id is None and ball_in_penalty:
            # Clamp target to the GK zone
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
        
        # Otherwise, stay on goal line and track ball laterally
        target_x = goal_x
        target_y = np.clip(ball_pos[1], 35.0, 65.0)  # Stay within goal area (slightly wider than posts)
        
        target = np.array([target_x, target_y])
        to_target = target - self.pos
        dist = np.linalg.norm(to_target)
        
        if dist > 1.0:
            self.vel = normalize(to_target) * PLAYER_SPEED
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
        
        # Only assign control if someone is close enough
        if min_dist < TACKLE_DISTANCE and closest_players:
            # Random tiebreaker if multiple players equally close
            controller = random.choice(closest_players)
            controller.has_ball = True
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
