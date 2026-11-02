"""Test the actual API HTTP round-trip, exactly like the frontend does."""
import httpx

client = httpx.Client(base_url="http://localhost:8000")

r = client.post("/simulate/start")
print("start status:", r.status_code)
state = r.json()
ball = state["ball"]
print(f"tick={state['tick']} ball={ball['pos']} bstate={ball['state']} owner={ball['owner_id']}")

prev_pos = list(ball["pos"])
for i in range(30):
    r = client.post("/simulate/tick", json=state)
    if r.status_code != 200:
        print("ERROR:", r.status_code, r.text[:300])
        break
    state = r.json()
    ball = state["ball"]
    owner_id = ball["owner_id"]
    owner_name = "LOOSE"
    if owner_id is not None:
        for team in [state["home_team"], state["away_team"]]:
            for p in team["players"]:
                if p["player_id"] == owner_id:
                    owner_name = p["name"]

    moved = [round(ball["pos"][0] - prev_pos[0], 2), round(ball["pos"][1] - prev_pos[1], 2)]
    print(
        f"tick={state['tick']:3d}  ball={[round(x,1) for x in ball['pos']]}  "
        f"vel={[round(v,3) for v in ball['velocity']]}  "
        f"bstate={ball['state']:12s}  owner={owner_name:20s}  ball_moved={moved}"
    )
    # Also check if players are moving
    if state["tick"] == 5:
        print("\n  --- Player positions at tick 5 ---")
        for team in [state["home_team"], state["away_team"]]:
            for p in team["players"]:
                print(f"    {p['name']:20s} pos={[round(x,1) for x in p['pos']]}  vel={[round(v,3) for v in p['velocity']]}  has_ball={p['has_ball']}")
        print()

    prev_pos = list(ball["pos"])
