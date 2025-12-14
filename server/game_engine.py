import numpy as np
import math
import random

FIELD_WIDTH = 100.0
FIELD_HEIGHT = 100.0

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
        # Simplified decision logic
        if self.has_ball:
            # Attack logic
            target_x = 100.0 if self.team == 'home' else 0.0
            target_y = 50.0
            
            # Shoot?
            dist_to_goal = abs(target_x - self.pos[0])
            if dist_to_goal < 20 and random.random() < 0.05:
                # Shoot!
                self.has_ball = False
                game.ball.owner_id = None
                shoot_vel_x = (1.0 if self.team == 'home' else -1.0) * 2.0
                shoot_vel_y = (50.0 - self.pos[1]) * 0.05
                game.ball.vel = np.array([shoot_vel_x, shoot_vel_y])
                return

            # Dribble
            dx = target_x - self.pos[0]
            dy = target_y - self.pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0:
                self.vel = np.array([dx/dist, dy/dist]) * 0.5
            
        else:
            # Move towards ball or position
            ball_pos = game.ball.pos
            if game.ball.owner_id is None:
                # Chase ball
                target = ball_pos
            else:
                # Support or Defend
                target = ball_pos # Simplified: everyone chases ball a bit
            
            dx = target[0] - self.pos[0]
            dy = target[1] - self.pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 1.0:
                self.vel = np.array([dx/dist, dy/dist]) * 0.4
            else:
                self.vel = np.array([0.0, 0.0])

        # Apply velocity
        self.pos += self.vel
        # Bounds
        self.pos[0] = np.clip(self.pos[0], 0, FIELD_WIDTH)
        self.pos[1] = np.clip(self.pos[1], 0, FIELD_HEIGHT)

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
        # HOME TEAM
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
        self.players[6].has_ball = True

    def iterate(self):
        self.time += 0.1
        
        # Move Ball
        if self.ball.owner_id is None:
            self.ball.pos += self.ball.vel
            self.ball.vel *= 0.95 # Friction
        else:
            owner = next((p for p in self.players if p.id == self.ball.owner_id), None)
            if owner:
                self.ball.pos = owner.pos.copy()
                self.ball.pos[0] += 1.0 # Slightly ahead
        
        # Check Goals
        if self.ball.pos[0] < 0:
            self.score['away'] += 1
            self.reset_positions()
        elif self.ball.pos[0] > 100:
            self.score['home'] += 1
            self.reset_positions()

        # Update Players
        for p in self.players:
            p.make_decision(self)
            
            # Ball Stealing/Pickup logic
            if self.ball.owner_id is None:
                dist = np.linalg.norm(p.pos - self.ball.pos)
                if dist < 2.0:
                    self.ball.owner_id = p.id
                    p.has_ball = True
            elif self.ball.owner_id != p.id:
                dist = np.linalg.norm(p.pos - self.ball.pos)
                if dist < 1.5 and random.random() < 0.1:
                    # Steal
                    current_owner = next((pl for pl in self.players if pl.id == self.ball.owner_id), None)
                    if current_owner:
                        current_owner.has_ball = False
                    self.ball.owner_id = p.id
                    p.has_ball = True

    def reset_positions(self):
        self.players = []
        self.init_players()
        self.ball = Ball()
        self.ball.owner_id = 7 # Give to home FWD
        self.players[6].has_ball = True

    def to_dict(self):
        return {
            "players": [p.to_dict() for p in self.players],
            "ball": self.ball.to_dict(),
            "score": self.score,
            "time": self.time
        }
