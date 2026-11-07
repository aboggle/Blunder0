# Blunder 0 (BC0)

Blunder 0 (BC0) is a UCI-compatible chess engine written in Python. It uses iterative deepening, alpha-beta search, quiescence search, a transposition table, move ordering, and a built-in positional evaluator. An optional NumPy neural evaluator can be trained from FEN/evaluation data.

## Features

- UCI support for chess GUIs such as Arena, Cute Chess, and Banksia
- Explicit negamax alpha-beta search with iterative deepening, aspiration windows, principal-variation search, late-move reductions, and guarded null-move pruning
- Quiescence search, capped check extensions, mate-distance pruning, transposition table, killer/history/countermove ordering
- Positional terms for center control, minor-piece development, king safety, tempo, pawn structure, and piece coordination
- Optional neural evaluator training from CSV data
- Cached Lichess Opening Explorer book for early-game moves, with automatic search fallback

## Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux

## Install

Run these commands from the project directory. On Windows, use PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To run the test suite or build a Windows executable, install the dev extras (pytest and PyInstaller) instead: `python -m pip install -r requirements-dev.txt` (it also includes the runtime requirements).

If PowerShell blocks virtual environment activation, run this once in that terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Command reference

Show the available commands:

```powershell
python -m chess_engine --help
python -m chess_engine train --help
python -m chess_engine uci --help
```

Start UCI mode with the default depth of 8:

```powershell
python -m chess_engine
```

Start UCI mode with a different default depth (valid range: 1–30):

```powershell
python -m chess_engine uci --depth 10
```

The compatibility entry point does the same thing:

```powershell
python engine.py
```

You can send a short UCI handshake from PowerShell to check that the engine starts and responds:

```powershell
@("uci", "isready", "quit") | python -m chess_engine
```

## Use with a chess GUI

Create a UCI engine entry in your chess GUI with:

- **Command / executable:** the Python executable in `.venv` (for example, `.venv\Scripts\python.exe` on Windows)
- **Arguments:** `-m chess_engine uci --depth 8`
- **Working directory:** the BC0 project directory

The engine identifies itself as **Blunder 0**. The GUI can override the default depth and send timed searches. Available UCI options are:

| Option | Default | Description |
| --- | ---: | --- |
| `Depth` | 8 | Default search depth, from 1 to 30 |
| `Hash` | 64 MB | Transposition table size setting |
| `EvalFile` | `models/evaluator.npz` | Path to a trained `.npz` evaluator, loaded automatically at startup when the file exists in the engine's working directory. Set another path to override it; send an empty value to use only the handcrafted evaluation. |
| `Blend Handcrafted` | false | When an `EvalFile` is loaded, add the handcrafted positional terms on top of the network output. The default false applies the network alone, since a trained model already encodes the handcrafted positional knowledge; enable only for models trained on material-only targets. |
| `Lichess Book` | true | Use Lichess's public opening explorer during the opening (requires internet) |
| `Clear Hash` | — | Clear the transposition table |

Common UCI commands sent by GUIs include:

| Command | Purpose |
| --- | --- |
| `uci` | Start UCI identification and list engine options |
| `isready` | Check that the engine is ready |
| `ucinewgame` | Reset the game and clear cached positions |
| `position startpos [moves ...]` | Set a position from the normal starting position |
| `position fen <FEN> [moves ...]` | Set a position from FEN |
| `go depth <n>` | Search to a fixed depth |
| `go nodes <n>` | Search up to a fixed number of nodes |
| `go movetime <ms>` | Search for a fixed number of milliseconds |
| `go wtime <ms> btime <ms> [winc <ms> binc <ms>]` | Search using the remaining clocks and increments |
| `stop` | Stop the current search |
| `quit` | Exit the engine |

## Train a neural evaluator

Training input must be a CSV with `FEN` and `evaluation` columns. Numeric evaluations are in pawn units from White's point of view; for example, `1.5` means White is evaluated at +1.5 pawns. Convert centipawn scores to pawns by dividing by 100. PGN mate scores such as `#3` and `#-3` are also accepted and become decisive targets of +20 and -20 pawns respectively. Numeric and mate targets are clipped to the range -20 to +20.

Example CSV:

```csv
FEN,evaluation
"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",0
"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",#3
```

Train with the defaults (5 epochs, batch size 256, hidden width 64, learning rate 0.001, and 4 BLAS threads):

```powershell
python -m chess_engine train evaluated.csv --output models\evaluator.npz
```

Specify all available training settings:

```powershell
python -m chess_engine train evaluated.csv `
  --output models\evaluator.npz `
  --epochs 5 `
  --batch-size 256 `
  --hidden 64 `
  --learning-rate 0.001 `
  --limit 100000 `
  --seed 7 `
  --chunk-size 8192 `
  --threads 4
```

Options:

- `--output`: output model path (default: `models/evaluator.npz`)
- `--epochs`: training passes over the CSV (default: `5`)
- `--batch-size`: rows per training batch (default: `256`)
- `--hidden`: neural network hidden width (default: `64`)
- `--learning-rate`: optimizer learning rate (default: `0.001`)
- `--limit`: maximum CSV rows to read; `0` means all rows (default: `0`)
- `--seed`: random seed (default: `7`)
- `--chunk-size`: rows buffered for shuffling (default: `8192`; must be at least the batch size)
- `--threads`: BLAS threads used during training (default: `4`)

Training writes `models\evaluator.npz` by default, and the engine loads that file automatically at startup when it exists — no GUI configuration needed. To use a different model, set the `EvalFile` option to its path; to use only the handcrafted evaluation, send an empty `EvalFile` value. Note the engine resolves the model path relative to its working directory, so point a GUI at the BC0 project directory. Model `.npz` files are ignored by Git, so add or distribute a trained model separately if users need one.

## Run the tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests
```

The suite covers evaluation mirror symmetry, perft move-generation counts, mate detection, transposition-table score handling, UCI command robustness, and the Lichess book's offline behavior.

## Build a Windows executable

After installing requirements, run from the project directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build_exe.ps1
```

The script creates `dist\bc0.exe`. Run it directly or add it to a chess GUI as a UCI engine:

```powershell
.\dist\bc0.exe
```

The exe is fully standalone: when a trained `models\evaluator.npz` exists at build time it is bundled into the exe and auto-loaded at startup, so the compiled engine needs no source tree or model files. A freshly retrained `models\evaluator.npz` in the engine's working directory (or an explicit `EvalFile` option) still overrides the bundled one.

## Project files

```text
chess_engine/
  __main__.py   CLI commands
  features.py   Neural evaluator input features
  network.py    Neural evaluator and trainer
  search.py     Position evaluation and search
  uci.py        UCI protocol adapter
  opening_book.py Cached Lichess opening explorer client
tests/          pytest suite for evaluation, search, and UCI handling
engine.py       Compatibility entry point
requirements.txt
requirements-dev.txt  Test and build extras (pytest, PyInstaller)
build_exe.ps1   Windows executable build script
CONTRIBUTING.md Contribution guide
LICENSE         MIT license
```

## Limitations

BC0 is a compact, CPU-oriented engine and is not at Stockfish strength. Playing strength depends on search time and, when enabled, the quality and score conventions of the training data.

## License

MIT — see `LICENSE`. Contributions are welcome; see `CONTRIBUTING.md`.
