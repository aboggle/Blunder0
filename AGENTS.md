# AGENTS.md

Blunder 0 (BC0): a UCI chess engine in Python. Single package, no framework, no build tooling beyond PyInstaller.

## Commands

- Setup: `py -m venv .venv` → `.\.venv\Scripts\Activate.ps1` → `python -m pip install -r requirements.txt`. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` first.
- Tests: `python -m pytest tests` (pytest ships in `requirements-dev.txt`, which also pins PyInstaller). There is no CI or lint config.
- Verify UCI changes with the handshake: `@("uci", "isready", "quit") | python -m chess_engine`. Quick search check: `python -c "import chess; from chess_engine.search import Search; print(Search().run(chess.Board(), 4))"`.
- When touching `search.py`, bench before and after: `python -c "import time, chess; from chess_engine.search import Search; t=time.perf_counter(); s=Search(); m,sc=s.run(chess.Board(),6); dt=time.perf_counter()-t; print(m.uci(), sc, f'{s.nodes/dt/1000:.1f}k nps')"` — node count and best move must not change silently.
- Run the engine: `python -m chess_engine` (UCI loop, blocks on stdin). `python -m chess_engine uci --depth N` (depth clamped to 1–30). `engine.py` is a compatibility shim re-exporting `chess_engine.uci.main` only so PyInstaller has an entry point — keep it working.
- Train the neural evaluator: `python -m chess_engine train <csv> --output models\evaluator.npz` (CSV columns `FEN,evaluation`).

## Gotchas

- Scores are white-relative centipawns everywhere in search (`chess_engine/search.py`): mate = 100000, mate-distance pruning keys off that. The neural evaluator (`predict_board`) returns white-relative pawns and is multiplied by 100 before mixing with the handcrafted eval. Training CSV targets are white-relative pawns (`#3` mate scores map to ±20 pawns, clipped to ±20) — a model trained on black-relative data will silently weaken the engine.
- With `EvalFile` loaded the engine uses network output alone; the handcrafted positional terms are skipped because the model was trained on full evaluations (they would double-count). UCI option `Blend Handcrafted` (default false) restores legacy combined scoring.
- `Hash` MB → `hash_mb * 4096` TT entries (Python entries cost ~230 bytes). Before the fix the implied ~230 MB at Hash=64 was ~350% off and Hash=4096 could request tens of GB.
- The Lichess opening book (`chess_engine/opening_book.py`) hits `explorer.lichess.ovh` (not `.org`), needs internet, disables itself for 30 s on any failure, and is only probed up to move 20. UCI option `Lichess Book` turns it off — set it to false when you want the search tested deterministically.
- Model `.npz` outputs (`models/*.npz`) and PyInstaller `.spec` files are gitignored — no trained model or dataset ships with the repo. At startup the engine auto-loads `models/evaluator.npz` when present (relative to the working directory); an explicit `EvalFile` overrides, empty disables.
- `build_exe.ps1` → `dist\bc0.exe` via PyInstaller one-file from `engine.py`; it must `--collect-all chess --collect-all numpy`, and it bundles `models\evaluator.npz` when present (loaded at startup from the PyInstaller `sys._MEIPASS` payload — see `BUNDLED_EVAL_FILE` in `uci.py`). Keep that variable in sync if the resource layout changes.
