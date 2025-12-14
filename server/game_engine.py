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
SHOOT_SCORE_THRESHOLD = 0.25  # Was 0.5 - shoot more often
PASS_SCORE_THRESHOLD = 0.10   # Lower threshold - pass more often

# Increased pass appeal, reduced dribble appeal
PASS_BONUS = 0.15  # Added bonus to pass scores

# Home position attraction weight
HOME_POSITION_WEIGHT = 0.3  # 30% home bias, 70% tactical

# Dribble touch system
DRIBBLE_TOUCH_INTERVAL = 5  # Every N ticks, ball moves ahead
DRIBBLE_TOUCH_DISTANCE = 2.5  # How far ball moves ahead (must be > TACKLE_DISTANCE to allow interception)


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


# Home positions for each role combination
# Format: (lateral_role, longitudinal_role, team) -> (x, y)
ROLE_HOME_POSITIONS = {
    # Home team (attacking right toward x=100)
    ('center', 'goalkeeper', 'home'): (5, 50),
    ('left', 'back', 'home'): (20, 20),
    ('center', 'back', 'home'): (20, 50),
    ('right', 'back', 'home'): (20, 80),
    ('left', 'midfielder', 'home'): (45, 25),
    ('right', 'midfielder', 'home'): (45, 75),
    ('center', 'forward', 'home'): (65, 50),
    
    # Away team (attacking left toward x=0)
    ('center', 'goalkeeper', 'away'): (95, 50),
    ('left', 'back', 'away'): (80, 80),
    ('center', 'back', 'away'): (80, 50),
    ('right', 'back', 'away'): (80, 20),
    ('left', 'midfielder', 'away'): (55, 75),
    ('right', 'midfielder', 'away'): (55, 25),
    ('center', 'forward', 'away'): (35, 50),
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
        
        # GK holds ball briefly then clears - use a timer based decision
        if self.role == 'GK':
            # GK waits a bit then clears (decision is made each tick, so just clear)
            # The holding happens in iterate() - GK keeps ball securely
            # After ~15 ticks of holding, GK will clear (controlled by gk_hold_timer)
            if game.gk_hold_timer >= 15:
                game.gk_hold_timer = 0
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
        
        # Choose best action with lowered thresholds
        if shoot_score > pass_score and shoot_score > dribble_score and shoot_score > SHOOT_SCORE_THRESHOLD:
            self._execute_shoot(ctx, game)
        elif pass_score > dribble_score and pass_option is not None and pass_score > PASS_SCORE_THRESHOLD:
            self._execute_pass(ctx, game, pass_option, pass_target)
        else:
            self._execute_dribble(ctx, game, dribble_dir)

    def _evaluate_shoot(self, ctx):
        """Evaluate shooting option. Returns score 0-1."""
        dist = ctx.dist_to_goal
        
        if dist > SHOOT_DISTANCE_THRESHOLD:
            return 0.0
        
        blocked = False
        for opp_pos in ctx.opponent_positions:
            if distance_point_to_segment(opp_pos, self.pos, ctx.goal_center) < 3.0:
                blocked = True
                break
        
        distance_factor = 1.0 - (dist / SHOOT_DISTANCE_THRESHOLD)
        block_factor = 0.3 if blocked else 1.0
        
        return distance_factor * block_factor

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
        """Evaluate a pass to a specific target position."""
        pass_vec = target_pos - self.pos
        pass_dist = np.linalg.norm(pass_vec)
        
        if pass_dist < 5.0 or pass_dist > 50.0:
            return 0.0
        
        safety_score = 1.0
        for opp_pos in ctx.opponent_positions:
            can_intercept, time_margin = time_to_intercept(
                opp_pos, PLAYER_SPRINT_SPEED,
                self.pos, target_pos, BALL_PASS_SPEED
            )
            if can_intercept:
                safety_score *= max(0.1, float(0.5 + time_margin))
        
        my_goal_dist = ctx.dist_to_goal
        target_goal_dist = distance_to_goal(target_pos, ctx.team)
        progress_factor = 0.5 + 0.5 * (my_goal_dist - target_goal_dist) / max(my_goal_dist, 1)
        progress_factor = float(np.clip(progress_factor, 0, 1))
        
        optimal_dist = 20.0
        dist_factor = 1.0 - abs(pass_dist - optimal_dist) / 50.0
        dist_factor = max(0.0, float(dist_factor))
        
        score = (
            PASS_SAFETY_WEIGHT * safety_score +
            PASS_GOAL_PROGRESS_WEIGHT * progress_factor +
            PASS_DISTANCE_WEIGHT * dist_factor +
            PASS_BONUS  # Added bonus to encourage passing
        )
        
        # Extra bonus for passing to moving teammates
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
        """Execute dribbling in the given direction."""
        self.vel = direction * BALL_DRIBBLE_SPEED
    
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
        
        # When ball is loose, chase it directly (no blending)
        if ball_owner is None:
            to_ball = game.ball.pos - self.pos
            dist = np.linalg.norm(to_ball)
            if dist > 1.0:
                self.vel = normalize(to_ball) * PLAYER_SPRINT_SPEED
            else:
                self.vel = np.zeros(2)
            return
        
        # Calculate tactical target when ball is owned
        tactical_target = None
        if ball_owner.team == self.team:
            tactical_target = self._get_support_target(ctx, ball_owner)
        else:
            tactical_target = self._get_defend_target(ctx, ball_owner)
        
        # Blend tactical target with home position (70% tactical, 30% home)
        if tactical_target is not None:
            blended_target = (1 - HOME_POSITION_WEIGHT) * tactical_target + HOME_POSITION_WEIGHT * self.home_pos
        else:
            blended_target = self.home_pos
        
        # Move towards blended target
        to_target = blended_target - self.pos
        dist = np.linalg.norm(to_target)
        
        if dist > 2.0:
            self.vel = normalize(to_target) * PLAYER_SPEED
        else:
            self.vel = np.zeros(2)

    def _get_support_target(self, ctx, ball_owner):
        """Calculate support position when teammate has ball."""
        ball_to_goal = ctx.goal_center - ball_owner.pos
        ball_to_goal_normalized = normalize(ball_to_goal)
        
        perp = np.array([-ball_to_goal_normalized[1], ball_to_goal_normalized[0]])
        
        my_offset = self.pos - ball_owner.pos
        side = 1 if np.dot(my_offset, perp) > 0 else -1
        
        support_distance = 15.0
        support_offset = 10.0
        target = ball_owner.pos + ball_to_goal_normalized * support_distance + perp * side * support_offset
        
        return target

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
        self.dribble_tick = 0  # Counter for dribble touch system
        self.steal_cooldown = 0  # Prevent immediate re-steals
        self.last_dribbler_id = None  # Track who just released ball to prevent instant re-pickup
        self.gk_hold_timer = 0  # Timer for GK holding the ball before clearing
        self.init_players()

    def init_players(self):
        self.players = []
        # HOME TEAM (4-2-1 formation for 7 players)
        self.players.append(Player(1, 'home', 'GK', 5, 50, 1, 'center', 'goalkeeper'))
        self.players.append(Player(2, 'home', 'DEF', 20, 20, 4, 'left', 'back'))
        self.players.append(Player(3, 'home', 'DEF', 20, 50, 5, 'center', 'back'))
        self.players.append(Player(4, 'home', 'DEF', 20, 80, 3, 'right', 'back'))
        self.players.append(Player(5, 'home', 'MID', 40, 30, 8, 'left', 'midfielder'))
        self.players.append(Player(6, 'home', 'MID', 40, 70, 6, 'right', 'midfielder'))
        self.players.append(Player(7, 'home', 'FWD', 60, 50, 9, 'center', 'forward'))
        
        # AWAY TEAM
        self.players.append(Player(11, 'away', 'GK', 95, 50, 1, 'center', 'goalkeeper'))
        self.players.append(Player(12, 'away', 'DEF', 80, 80, 4, 'left', 'back'))
        self.players.append(Player(13, 'away', 'DEF', 80, 50, 5, 'center', 'back'))
        self.players.append(Player(14, 'away', 'DEF', 80, 20, 3, 'right', 'back'))
        self.players.append(Player(15, 'away', 'MID', 60, 75, 8, 'left', 'midfielder'))
        self.players.append(Player(16, 'away', 'MID', 60, 25, 6, 'right', 'midfielder'))
        self.players.append(Player(17, 'away', 'FWD', 40, 50, 9, 'center', 'forward'))

        # Give ball to Home FWD for kickoff
        self.ball.owner_id = 7
        self.ball.pos = np.array([50.0, 50.0])
        self.players[6].has_ball = True
        self.players[6].pos = np.array([50.0, 50.0])
        self.state = GameState.PLAYING
        self.last_touch = LastTouch(team='home', player_id=7)

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
        
        # Move Ball
        if self.ball.owner_id is None:
            self.ball.pos += self.ball.vel
            self.ball.vel *= BALL_FRICTION
            
            if np.linalg.norm(self.ball.vel) < 0.1:
                self.ball.vel = np.zeros(2)
        else:
            owner = next((p for p in self.players if p.id == self.ball.owner_id), None)
            if owner:
                # GK holds the ball securely - no tackle steal allowed
                if owner.role == 'GK':
                    # GK keeps ball until they decide to clear it
                    self.ball.pos = owner.pos.copy()
                    self.gk_hold_timer += 1  # Increment hold timer
                else:
                    # Check if opponent is pressuring the ball carrier (tackle attempt)
                    # Cooldown prevents ping-pong steals
                    if self.steal_cooldown > 0:
                        self.steal_cooldown -= 1
                    else:
                        for p in self.players:
                            if p.team != owner.team:
                                dist = np.linalg.norm(p.pos - owner.pos)
                                if dist <= TACKLE_DISTANCE:
                                    # 30% chance to steal when pressing
                                    if random.random() < 0.30:
                                        owner.has_ball = False
                                        self.ball.owner_id = p.id
                                        p.has_ball = True
                                        self.ball.pos = p.pos.copy()
                                        self.last_touch = LastTouch(team=p.team, player_id=p.id)
                                        self.dribble_tick = 0
                                        self.steal_cooldown = 10  # 10 ticks cooldown before next steal possible
                                        break
                    
                    # If still has ball, apply dribble touch system (not for GK)
                    if self.ball.owner_id == owner.id:
                        self.dribble_tick += 1
                        if self.dribble_tick >= DRIBBLE_TOUCH_INTERVAL:
                            self.dribble_tick = 0
                            # Push ball ahead in dribble direction
                            if np.linalg.norm(owner.vel) > 0.1:
                                dribble_dir = normalize(owner.vel)
                                self.ball.pos = owner.pos + dribble_dir * DRIBBLE_TOUCH_DISTANCE
                                # Ball becomes loose - clear ownership
                                self.last_dribbler_id = owner.id  # Remember who released it
                                owner.has_ball = False
                                self.ball.owner_id = None
                                self.ball.vel = dribble_dir * 0.5  # Give ball more momentum
                            else:
                                # Standing still - ball stays with player
                                self.ball.pos = owner.pos.copy()
                        else:
                            # Between touches - ball follows player closely
                            self.ball.pos = owner.pos.copy()
        
        # Detect ball events
        event = self._detect_ball_event()
        if event is not None:
            self._transition_state(event)
            return

        # Update Players
        for p in self.players:
            p.make_decision(self)
        
        # Proximity-based ball ownership: closest player gets the loose ball
        if self.ball.owner_id is None:
            closest_player = None
            min_dist = float('inf')
            for p in self.players:
                # Skip the player who just released the ball (dribble touch)
                if p.id == self.last_dribbler_id:
                    continue
                dist = np.linalg.norm(p.pos - self.ball.pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_player = p
            
            # Only pick up if within tackle distance
            if closest_player and min_dist < TACKLE_DISTANCE:
                closest_player.has_ball = True
                self.ball.owner_id = closest_player.id
                self.ball.vel = np.zeros(2)
                self.last_touch = LastTouch(team=closest_player.team, player_id=closest_player.id)
                self.last_dribbler_id = None  # Clear after someone else picks up
            elif min_dist >= TACKLE_DISTANCE:
                # No one close enough - dribbler can get it back next tick
                self.last_dribbler_id = None

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
            "restartTeam": self.restart_team
        }
