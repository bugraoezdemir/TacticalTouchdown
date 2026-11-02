# Tactical Touchdown

A football simulation where you set your team's tactics and manage it through an 18-team season. Every match, including the ones you don't watch, is played by the full engine.

**Play it here: https://bugraoezdemir.github.io/TacticalTouchdown/**

It runs entirely in your browser, with nothing to install and no account. The first load downloads about 12 MB. A computer works best; phones are not tested yet.

Your season is saved in that browser, so it stays on that device, and clearing the site's data deletes it.

The older version of this game is kept under the tag [`v0.0.1`](https://github.com/bugraoezdemir/TacticalTouchdown/tree/v0.0.1).

## What's here

- `backend/app`: the game engine and its API, in Python
- `backend/fb_core`: the slowest parts of the engine, in Rust
- `frontend`: the game's interface, in React
