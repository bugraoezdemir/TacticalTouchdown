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

# Decision weights - lowered thresholds so players pass/shoot more often
PASS_SAFETY_WEIGHT = 0.4
PASS_GOAL_PROGRESS_WEIGHT = 0.3
PASS_DISTANCE_WEIGHT = 0.3

DRIBBLE_CLEARANCE_WEIGHT = 0.5
DRIBBLE_GOAL_PROGRESS_WEIGHT = 0.5

SHOOT_DISTANCE_THRESHOLD = 35.0  # Increased from 25 - shoot from further
SHOOT_ANGLE_THRESHOLD = 30.0

# Lowered thresholds for actions
SHOOT_SCORE_THRESHOLD = 0.25  # Was 0.5 - shoot more often
PASS_SCORE_THRESHOLD = 0.15   # Was 0.3 - pass more often

# Home position attraction weight
HOME_POSITION_WEIGHT = 0.3  # 30% home bias, 70% tactical


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
        
        shoot_score = self._evaluate_shoot(ctx)
        pass_option, pass_score = self._evaluate_pass(ctx)
        dribble_dir, dribble_score = self._evaluate_dribble(ctx)
        
        # Choose best action with lowered thresholds
        if shoot_score > pass_score and shoot_score > dribble_score and shoot_score > SHOOT_SCORE_THRESHOLD:
            self._execute_shoot(ctx, game)
        elif pass_score > dribble_score and pass_option is not None and pass_score > PASS_SCORE_THRESHOLD:
            self._execute_pass(ctx, game, pass_option)
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
        """Evaluate passing options. Returns (best_teammate, score)."""
        if len(ctx.teammates) == 0:
            return None, 0.0
        
        best_teammate = None
        best_score = 0.0
        
        for teammate in ctx.teammates:
            if teammate.role == 'GK' and ctx.dist_to_goal < 50:
                continue
            
            pass_vec = teammate.pos - self.pos
            pass_dist = np.linalg.norm(pass_vec)
            
            if pass_dist < 5.0 or pass_dist > 50.0:
                continue
            
            safety_score = 1.0
            for opp_pos in ctx.opponent_positions:
                can_intercept, time_margin = time_to_intercept(
                    opp_pos, PLAYER_SPRINT_SPEED,
                    self.pos, teammate.pos, BALL_PASS_SPEED
                )
                if can_intercept:
                    safety_score *= max(0.1, float(0.5 + time_margin))
            
            my_goal_dist = ctx.dist_to_goal
            teammate_goal_dist = distance_to_goal(teammate.pos, ctx.team)
            progress_factor = 0.5 + 0.5 * (my_goal_dist - teammate_goal_dist) / max(my_goal_dist, 1)
            progress_factor = float(np.clip(progress_factor, 0, 1))
            
            optimal_dist = 20.0
            dist_factor = 1.0 - abs(pass_dist - optimal_dist) / 50.0
            dist_factor = max(0.0, float(dist_factor))
            
            score = (
                PASS_SAFETY_WEIGHT * safety_score +
                PASS_GOAL_PROGRESS_WEIGHT * progress_factor +
                PASS_DISTANCE_WEIGHT * dist_factor
            )
            
            if score > best_score:
                best_score = score
                best_teammate = teammate
        
        return best_teammate, best_score

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

    def _execute_pass(self, ctx, game, target_player):
        """Execute a pass to target teammate."""
        self.has_ball = False
        game.ball.owner_id = None
        
        lead_factor = 0.3
        target_pos = target_player.pos + target_player.vel * lead_factor * 10
        
        pass_dir = normalize(target_pos - self.pos)
        game.ball.vel = pass_dir * BALL_PASS_SPEED
        
        # Track last touch
        game.last_touch = LastTouch(team=self.team, player_id=self.id)

    def _execute_dribble(self, ctx, game, direction):
        """Execute dribbling in the given direction."""
        self.vel = direction * BALL_DRIBBLE_SPEED

    def _make_off_ball_decision(self, ctx, game):
        """Decision making when player doesn't have the ball."""
        
        ball_owner = None
        if game.ball.owner_id is not None:
            for p in game.players:
                if p.id == game.ball.owner_id:
                    ball_owner = p
                    break
        
        # Calculate tactical target
        tactical_target = None
        if ball_owner is not None and ball_owner.team == self.team:
            tactical_target = self._get_support_target(ctx, ball_owner)
        elif ball_owner is not None and ball_owner.team != self.team:
            tactical_target = self._get_defend_target(ctx, ball_owner)
        else:
            tactical_target = game.ball.pos.copy()
        
        # Blend tactical target with home position (70% tactical, 30% home)
        if tactical_target is not None:
            blended_target = (1 - HOME_POSITION_WEIGHT) * tactical_target + HOME_POSITION_WEIGHT * self.home_pos
        else:
            blended_target = self.home_pos
        
        # Move towards blended target
        to_target = blended_target - self.pos
        dist = np.linalg.norm(to_target)
        
        if dist > 2.0:
            speed = PLAYER_SPRINT_SPEED if ball_owner is None else PLAYER_SPEED
            self.vel = normalize(to_target) * speed
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

    def _get_defend_target(self, ctx, ball_owner):
        """Calculate defensive position when opponent has ball."""
        if self.role == 'GK':
            goal_x = 5.0 if self.team == 'home' else 95.0
            target_y = np.clip(ctx.game.ball.pos[1], 40, 60)
            return np.array([goal_x, target_y])
        
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
                self.ball.pos = owner.pos.copy()
        
        # Detect ball events
        event = self._detect_ball_event()
        if event is not None:
            self._transition_state(event)
            return

        # Update Players
        for p in self.players:
            p.make_decision(self)
            
            # Ball pickup/steal logic
            if self.ball.owner_id is None:
                dist = np.linalg.norm(p.pos - self.ball.pos)
                if dist < TACKLE_DISTANCE:
                    self.ball.owner_id = p.id
                    p.has_ball = True
                    self.ball.vel = np.zeros(2)
                    self.last_touch = LastTouch(team=p.team, player_id=p.id)
            elif self.ball.owner_id != p.id:
                dist = np.linalg.norm(p.pos - self.ball.pos)
                owner = next((pl for pl in self.players if pl.id == self.ball.owner_id), None)
                if owner and owner.team != p.team and dist < TACKLE_DISTANCE:
                    if random.random() < 0.15:
                        owner.has_ball = False
                        self.ball.owner_id = p.id
                        p.has_ball = True
                        self.last_touch = LastTouch(team=p.team, player_id=p.id)

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
