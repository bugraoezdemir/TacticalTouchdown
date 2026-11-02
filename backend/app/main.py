from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import simulation, tournament

app = FastAPI(title="Football Simulation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulation.router, prefix="/simulate", tags=["simulation"])
app.include_router(tournament.router, prefix="/tournament", tags=["tournament"])
