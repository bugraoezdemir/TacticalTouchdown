"""Headless debug runner - prints game events tick by tick."""
import sys
import numpy as np

sys.path.insert(0, ".")
from app.engine.factory import build_test_state
from app.engine.game_loop import tick


def fmt_state(state):
    b = state.ball
    owner = state.get_ball_owner()
    owner_str = owner.name if owner else "LOOSE"
    near = []
    for p in state.all_players():
        d = float(np.linalg.norm(p.pos - b.pos))
        if d < 8:
            near.append(f"{p.name}({d:.1f}m)")
    near_str = ", ".join(near) if near else "nobody"
    return (
        f"tick={state.tick:4d}  ball=[{b.pos[0]:.1f},{b.pos[1]:.1f}]  "
        f"bvel=[{b.velocity[0]:.3f},{b.velocity[1]:.3f}]  "
        f"state={b.state:12s}  owner={owner_str:20s}  near={near_str}"
    )


state = build_test_state()
print("=== INITIAL STATE ===")
print(fmt_state(state))
print()

prev_owner_id = state.ball.owner_id
prev_ball_state = state.ball.state
prev_ball_pos = state.ball.pos.copy()

for i in range(600):
    state = tick(state)

    cur_owner_id = state.ball.owner_id
    cur_ball_state = state.ball.state

    possession_change = cur_owner_id != prev_owner_id
    state_change = cur_ball_state != prev_ball_state
    ball_moved = float(np.linalg.norm(state.ball.pos - prev_ball_pos)) > 0.05

    if possession_change or state_change or state.tick % 60 == 0:
        print(fmt_state(state))
        if possession_change:
            prev_name = "LOOSE"
            if prev_owner_id is not None:
                try:
                    prev_name = state.get_player(prev_owner_id).name
                except KeyError:
                    prev_name = f"id={prev_owner_id}"
            cur_name = state.get_ball_owner().name if cur_owner_id is not None else "LOOSE"
            print(f"  !! POSSESSION: {prev_name} -> {cur_name}")
        if state_change:
            print(f"  !! BALL STATE: {prev_ball_state} -> {cur_ball_state}")
        if not ball_moved and cur_ball_state != "stationary":
            print(f"  !! WARN: ball state is {cur_ball_state} but ball didn't move!")

    prev_owner_id = cur_owner_id
    prev_ball_state = cur_ball_state
    prev_ball_pos = state.ball.pos.copy()

print()
print("=== FINAL STATE ===")
print(fmt_state(state))
print()
print("=== ALL PLAYER POSITIONS ===")
for p in state.all_players():
    d = float(np.linalg.norm(p.pos - state.ball.pos))
    print(f"  {p.name:20s} pos=[{p.pos[0]:.1f},{p.pos[1]:.1f}]  vel=[{p.velocity[0]:.3f},{p.velocity[1]:.3f}]  has_ball={p.has_ball}  dist_to_ball={d:.1f}m")
