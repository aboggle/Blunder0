# Contributing to Blunder 0

Thanks for helping improve BC0. This file covers setup, verification, and the conventions that keep the compact engine healthy.

## Setup

Requires Python 3.10+. From the project root (PowerShell example):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes the runtime requirements plus `pytest` and PyInstaller. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` once in that terminal.

## Verifying changes

There is no CI yet, so run the test suite locally before opening a PR:

```powershell
python -m pytest tests
```

When editing search, evaluation, or the UCI adapter also check the engine boots in UCI mode:

```powershell
@("uci", "isready", "quit") | python -m chess_engine
```

For anything touching `chess_engine/search.py`, compare speed before and after:

```powershell
python -c "import time, chess; from chess_engine.search import Search; t=time.perf_counter(); s=Search(); m,sc=s.run(chess.Board(),6); dt=time.perf_counter()-t; print(m.uci(), sc, f'{s.nodes/dt/1000:.1f}k nps')"
```

Include the before/after numbers in your PR description. Node counts and best moves at a fixed depth must not change unless the change is an intended strength or policy change.

## Guidelines

- Keep the engine pure Python 3.10+ stdlib plus the pinned `requirements.txt` dependencies. Do not add heavy dependencies.
- Use python-chess (`chess`) for all board logic; never reimplement move legality or FEN parsing.
- Scores are white-relative centipawns throughout `search.py` (mate = 100000). Neural-evaluator output is white-relative pawns multiplied by 100. Training CSV targets are white-relative pawns; mate scores (`#3`) map to ±20.
- Keep `engine.py` working as the PyInstaller entry point (a two-line shim over `chess_engine.uci.main`).
- The Lichess opening book must stay strictly optional and fail silently offline (see `opening_book.py`).
- Windows executable builds go through `build_exe.ps1`; do not pass different PyInstaller flags casually — `--collect-all chess --collect-all numpy` is required.

## Submitting changes

1. Fork or branch off `main`.
2. Keep commits focused; write a summary line plus a short body for anything non-trivial.
3. Add tests for bug fixes and for new behaviors (see `tests/`).
4. Update `README.md` for user-visible behavior changes (UCI options, commands).
5. PRs are squash-checked against the test suite; mention any bench numbers.

`Co-authored-by:` trailers are welcome for paired or AI-assisted work.

## License

By contributing you agree that your contributions are licensed under the MIT License (see `LICENSE`).
