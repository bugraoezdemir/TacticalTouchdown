"""Verbose API test - prints all player positions every tick."""
import httpx

client = httpx.Client(base_url="http://localhost:8000")

r = client.post("/simulate/start")
state = r.json()
print(f"START: tick={state['tick']} ball={[round(x,2) for x in state['ball']['pos']]}")
for team in [state['home_team'], state['away_team']]:
    for p in team['players']:
        print(f"  {p['name']:20s} pos={[round(x,2) for x in p['pos']]} has_ball={p['has_ball']}")

for i in range(10):
    r = client.post("/simulate/tick", json=state)
    state = r.json()
    print(f"\ntick={state['tick']} ball={[round(x,2) for x in state['ball']['pos']]} bstate={state['ball']['state']} owner={state['ball']['owner_id']}")
    for team in [state['home_team'], state['away_team']]:
        for p in team['players']:
            print(f"  {p['name']:20s} pos={[round(x,2) for x in p['pos']]} vel={[round(v,3) for v in p['velocity']]} has_ball={p['has_ball']}")
