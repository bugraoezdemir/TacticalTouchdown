import numpy as np
import math
import random

FIELD_WIDTH = 100.0
FIELD_HEIGHT = 100.0

# Physics constants
PLAYER_SPEED = 0.5
PLAYER_SPRINT_SPEED = 0.7
BALL_PASS_SPEED = 2.0
BALL_SHOOT_SPEED = 3.0
BALL_DRIBBLE_SPEED = 0.6
TACKLE_DISTANCE = 2.0
BALL_FRICTION = 0.95

# Decision weights
PASS_SAFETY_WEIGHT = 0.4
PASS_GOAL_PROGRESS_WEIGHT = 0.3
PASS_DISTANCE_WEIGHT = 0.3

DRIBBLE_CLEARANCE_WEIGHT = 0.5
DRIBBLE_GOAL_PROGRESS_WEIGHT = 0.5

SHOOT_DISTANCE_THRESHOLD = 25.0
SHOOT_ANGLE_THRESHOLD = 30.0

# Helper functions for geometric calculations
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
    
    # Find closest point on ball path to interceptor
    closest_point = project_point_to_segment(interceptor_pos, ball_start, ball_end)
    
    # Distance interceptor needs to travel
    intercept_dist = np.linalg.norm(closest_point - interceptor_pos)
    intercept_time = intercept_dist / interceptor_speed
    
    # Time for ball to reach closest point
    ball_to_closest = np.linalg.norm(closest_point - ball_start)
    ball_time_to_closest = ball_to_closest / ball_speed
    
    # If interceptor can reach the point before the ball
    time_margin = ball_time_to_closest - intercept_time
    can_intercept = time_margin < 0.5  # Within 0.5 time units = risky
    
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
        # Vector from player to opponent
        to_opp = opp_pos - pos
        
        # Project opponent onto the direction vector
        proj_dist = np.dot(to_opp, dir_normalized)
        
        # Only consider opponents ahead
        if proj_dist <= 0:
            continue
        
        # Perpendicular distance from opponent to movement line
        perp_dist = np.linalg.norm(to_opp - proj_dist * dir_normalized)
        
        # If opponent is close to the path, reduce clearance
        if perp_dist < 3.0:  # Opponent radius of influence
            effective_clearance = proj_dist - (3.0 - perp_dist)
            min_clearance = min(min_clearance, max(0, effective_clearance))
    
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
        
        # Separate teammates and opponents
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
        
        # Goal information
        self.goal_x = 100.0 if self.team == 'home' else 0.0
        self.goal_center = np.array([self.goal_x, 50.0])
        self.dist_to_goal = distance_to_goal(player.pos, self.team)


class Player:
    def __init__(self, id, team, role, x, y, number):
        self.id = id
        self.team = team
        self.role = role
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([0.0, 0.0])
        self.has_ball = False
        self.number = number
    
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
            "number": self.number
        }

    def make_decision(self, game):
        """Main decision-making method using the improved AI system."""
        ctx = DecisionContext(self, game)
        
        if self.has_ball:
            self._make_ball_decision(ctx, game)
        else:
            self._make_off_ball_decision(ctx, game)
        
        # Apply velocity and bounds
        self.pos += self.vel
        self.pos[0] = np.clip(self.pos[0], 0, FIELD_WIDTH)
        self.pos[1] = np.clip(self.pos[1], 0, FIELD_HEIGHT)

    def _make_ball_decision(self, ctx, game):
        """Decision making when player has the ball."""
        
        # Evaluate all options
        shoot_score = self._evaluate_shoot(ctx)
        pass_option, pass_score = self._evaluate_pass(ctx)
        dribble_dir, dribble_score = self._evaluate_dribble(ctx)
        
        # Choose best action
        if shoot_score > pass_score and shoot_score > dribble_score and shoot_score > 0.5:
            self._execute_shoot(ctx, game)
        elif pass_score > dribble_score and pass_option is not None and pass_score > 0.3:
            self._execute_pass(ctx, game, pass_option)
        else:
            self._execute_dribble(ctx, game, dribble_dir)

    def _evaluate_shoot(self, ctx):
        """Evaluate shooting option. Returns score 0-1."""
        dist = ctx.dist_to_goal
        
        # Too far to shoot effectively
        if dist > SHOOT_DISTANCE_THRESHOLD:
            return 0.0
        
        # Check angle to goal
        goal_top = np.array([ctx.goal_x, 45.0])
        goal_bottom = np.array([ctx.goal_x, 55.0])
        
        # Check if path to goal is blocked
        blocked = False
        for opp_pos in ctx.opponent_positions:
            if distance_point_to_segment(opp_pos, self.pos, ctx.goal_center) < 3.0:
                blocked = True
                break
        
        # Score based on distance and blockage
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
            # Skip goalkeeper for forward passes usually
            if teammate.role == 'GK' and ctx.dist_to_goal < 50:
                continue
            
            pass_vec = teammate.pos - self.pos
            pass_dist = np.linalg.norm(pass_vec)
            
            # Skip very short or very long passes
            if pass_dist < 5.0 or pass_dist > 50.0:
                continue
            
            # Calculate pass safety
            safety_score = 1.0
            for opp_pos in ctx.opponent_positions:
                can_intercept, time_margin = time_to_intercept(
                    opp_pos, PLAYER_SPRINT_SPEED,
                    self.pos, teammate.pos, BALL_PASS_SPEED
                )
                if can_intercept:
                    # Reduce safety based on how easily they can intercept
                    safety_score *= max(0.1, 0.5 + time_margin)
            
            # Goal progress factor
            my_goal_dist = ctx.dist_to_goal
            teammate_goal_dist = distance_to_goal(teammate.pos, ctx.team)
            progress_factor = 0.5 + 0.5 * (my_goal_dist - teammate_goal_dist) / max(my_goal_dist, 1)
            progress_factor = np.clip(progress_factor, 0, 1)
            
            # Distance factor (prefer medium-range passes)
            optimal_dist = 20.0
            dist_factor = 1.0 - abs(pass_dist - optimal_dist) / 50.0
            dist_factor = max(0, dist_factor)
            
            # Combined score
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
        
        # Sample 12 directions
        num_directions = 12
        angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
        
        # Bias towards goal
        goal_angle = angle_to_goal(self.pos, ctx.team)
        
        best_dir = normalize(ctx.goal_center - self.pos)
        best_score = 0.0
        
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Calculate clearance in this direction
            clearance = clearance_in_direction(self.pos, direction, ctx.opponent_positions)
            clearance_factor = clearance / 20.0  # Normalize
            
            # Goal progress factor
            goal_dir = normalize(ctx.goal_center - self.pos)
            alignment = np.dot(direction, goal_dir)
            goal_factor = (alignment + 1) / 2  # Normalize to 0-1
            
            # Avoid going backwards too much
            if ctx.team == 'home' and direction[0] < -0.5:
                goal_factor *= 0.3
            elif ctx.team == 'away' and direction[0] > 0.5:
                goal_factor *= 0.3
            
            # Avoid sidelines
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
        
        # Aim for goal with some variation
        target_y = 50.0 + (random.random() - 0.5) * 8  # Slight randomness
        target = np.array([ctx.goal_x, target_y])
        
        shoot_dir = normalize(target - self.pos)
        game.ball.vel = shoot_dir * BALL_SHOOT_SPEED

    def _execute_pass(self, ctx, game, target_player):
        """Execute a pass to target teammate."""
        self.has_ball = False
        game.ball.owner_id = None
        
        # Lead the pass slightly
        lead_factor = 0.3
        target_pos = target_player.pos + target_player.vel * lead_factor * 10
        
        pass_dir = normalize(target_pos - self.pos)
        game.ball.vel = pass_dir * BALL_PASS_SPEED

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
        
        if ball_owner is not None and ball_owner.team == self.team:
            # Teammate has ball - support
            self._support_teammate(ctx, ball_owner)
        elif ball_owner is not None and ball_owner.team != self.team:
            # Opponent has ball - defend
            self._defend_against(ctx, ball_owner)
        else:
            # Loose ball - chase
            self._chase_ball(ctx, game)

    def _support_teammate(self, ctx, ball_owner):
        """Move to support a teammate with the ball."""
        
        # Find good supporting position
        # Move towards goal but offset from ball carrier
        ball_to_goal = ctx.goal_center - ball_owner.pos
        ball_to_goal_normalized = normalize(ball_to_goal)
        
        # Perpendicular offset based on player's current position
        perp = np.array([-ball_to_goal_normalized[1], ball_to_goal_normalized[0]])
        
        # Decide which side to support from
        my_offset = self.pos - ball_owner.pos
        side = 1 if np.dot(my_offset, perp) > 0 else -1
        
        # Target position: ahead of ball carrier, offset to the side
        support_distance = 15.0
        support_offset = 10.0
        target = ball_owner.pos + ball_to_goal_normalized * support_distance + perp * side * support_offset
        
        # Move towards target
        to_target = target - self.pos
        dist = np.linalg.norm(to_target)
        
        if dist > 2.0:
            self.vel = normalize(to_target) * PLAYER_SPEED
        else:
            self.vel = np.zeros(2)

    def _defend_against(self, ctx, ball_owner):
        """Defensive positioning against opponent with ball."""
        
        if self.role == 'GK':
            # Goalkeeper stays in goal area, tracks ball
            goal_x = 5.0 if self.team == 'home' else 95.0
            target_y = np.clip(ctx.game.ball.pos[1], 40, 60)
            target = np.array([goal_x, target_y])
        else:
            # Field players: position between ball and goal
            goal_center = np.array([0.0 if self.team == 'home' else 100.0, 50.0])
            
            # Defensive position is between ball and goal
            ball_to_goal = goal_center - ball_owner.pos
            defense_point = ball_owner.pos + normalize(ball_to_goal) * 8.0
            
            # Adjust based on role
            if self.role == 'DEF':
                # Defenders stay deeper
                defense_point = goal_center + normalize(ball_owner.pos - goal_center) * 25.0
            elif self.role == 'MID':
                # Midfielders press more
                defense_point = ball_owner.pos + normalize(ball_to_goal) * 5.0
            elif self.role == 'FWD':
                # Forwards press high
                defense_point = ball_owner.pos
            
            target = defense_point
        
        # Move towards defensive position
        to_target = target - self.pos
        dist = np.linalg.norm(to_target)
        
        if dist > 1.0:
            self.vel = normalize(to_target) * PLAYER_SPEED
        else:
            self.vel = np.zeros(2)

    def _chase_ball(self, ctx, game):
        """Chase a loose ball."""
        to_ball = game.ball.pos - self.pos
        dist = np.linalg.norm(to_ball)
        
        if dist > 1.0:
            self.vel = normalize(to_ball) * PLAYER_SPRINT_SPEED
        else:
            self.vel = np.zeros(2)


class Ball:
    def __init__(self):
        self.pos = np.array([50.0, 50.0])
        self.vel = np.array([0.0, 0.0])
        self.owner_id = None

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
        self.init_players()

    def init_players(self):
        self.players = []
        # HOME TEAM (4-2-1 formation for 7 players)
        self.players.append(Player(1, 'home', 'GK', 5, 50, 1))
        self.players.append(Player(2, 'home', 'DEF', 20, 20, 4))
        self.players.append(Player(3, 'home', 'DEF', 20, 50, 5))
        self.players.append(Player(4, 'home', 'DEF', 20, 80, 3))
        self.players.append(Player(5, 'home', 'MID', 40, 30, 8))
        self.players.append(Player(6, 'home', 'MID', 40, 70, 6))
        self.players.append(Player(7, 'home', 'FWD', 60, 50, 9))
        
        # AWAY TEAM
        self.players.append(Player(11, 'away', 'GK', 95, 50, 1))
        self.players.append(Player(12, 'away', 'DEF', 80, 20, 4))
        self.players.append(Player(13, 'away', 'DEF', 80, 50, 5))
        self.players.append(Player(14, 'away', 'DEF', 80, 80, 3))
        self.players.append(Player(15, 'away', 'MID', 60, 30, 8))
        self.players.append(Player(16, 'away', 'MID', 60, 70, 6))
        self.players.append(Player(17, 'away', 'FWD', 40, 50, 9))

        # Give ball to Home FWD
        self.ball.owner_id = 7
        self.ball.pos = np.array([60.0, 50.0])
        self.players[6].has_ball = True

    def iterate(self):
        self.time += 0.1
        
        # Move Ball
        if self.ball.owner_id is None:
            self.ball.pos += self.ball.vel
            self.ball.vel *= BALL_FRICTION
            
            # Stop ball if very slow
            if np.linalg.norm(self.ball.vel) < 0.1:
                self.ball.vel = np.zeros(2)
        else:
            owner = next((p for p in self.players if p.id == self.ball.owner_id), None)
            if owner:
                self.ball.pos = owner.pos.copy()
        
        # Check Goals
        goal_scored = False
        if self.ball.pos[0] < 0 and 40 < self.ball.pos[1] < 60:
            self.score['away'] += 1
            goal_scored = True
        elif self.ball.pos[0] > 100 and 40 < self.ball.pos[1] < 60:
            self.score['home'] += 1
            goal_scored = True
        
        # Ball out of bounds
        if self.ball.pos[0] < 0 or self.ball.pos[0] > 100:
            goal_scored = True  # Reset
        if self.ball.pos[1] < 0 or self.ball.pos[1] > 100:
            self.ball.pos[1] = np.clip(self.ball.pos[1], 0, 100)
            self.ball.vel[1] *= -0.5  # Bounce off sidelines
        
        if goal_scored:
            self.reset_positions()
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
            elif self.ball.owner_id != p.id:
                dist = np.linalg.norm(p.pos - self.ball.pos)
                owner = next((pl for pl in self.players if pl.id == self.ball.owner_id), None)
                if owner and owner.team != p.team and dist < TACKLE_DISTANCE:
                    # Tackle attempt
                    if random.random() < 0.15:  # 15% tackle success
                        owner.has_ball = False
                        self.ball.owner_id = p.id
                        p.has_ball = True

    def reset_positions(self):
        self.init_players()

    def to_dict(self):
        return {
            "players": [p.to_dict() for p in self.players],
            "ball": self.ball.to_dict(),
            "score": self.score,
            "time": self.time
        }
