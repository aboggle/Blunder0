import shlex
import sys
import threading
import time
from pathlib import Path

import chess
from .search import Search, MATE
from .network import Evaluator
from .opening_book import LichessBook

# Trained models are gitignored, so none ships with the repo. When a user
# has trained one (the `train` command's default output), it is picked up
# automatically; EvalFile overrides, and an empty EvalFile opts out.
DEFAULT_EVAL_FILE = Path("models") / "evaluator.npz"
# PyInstaller extracts --add-data resources to sys._MEIPASS. Compiled builds
# carry the trained model there so the exe runs without any source tree.
BUNDLED_EVAL_FILE = (Path(sys._MEIPASS) / "models" / "evaluator.npz"
                     if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None)
                     else None)


def _unquote(token):
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


class UCI:
    def __init__(self):
        self.depth, self.hash_mb = 8, 64
        self.model_path = ""
        self.evalfile_set = False
        self.book_enabled = True
        # With EvalFile set, use only the network output. Legacy combined
        # scoring (network output plus the handcrafted positional terms)
        # double-counts the positional knowledge a trained model already
        # carries; enabled via the Blend Handcrafted option.
        self.blend = False
        self.book = LichessBook()
        self.evaluator = None
        self.board = chess.Board()
        self.lock = threading.Lock()
        self.worker = None
        self.searcher = None
        self.started = time.monotonic()

    def send(self, line):
        with self.lock:
            print(line, flush=True)

    def info(self, depth, score, nodes, move):
        elapsed = max(1, int((time.monotonic() - self.started) * 1000))
        if abs(score) >= MATE - 100:
            mate = (MATE - abs(score) + 1) // 2
            score_text = f"mate {mate if score > 0 else -mate}"
        else:
            score_text = f"cp {score}"
        self.send(f"info depth {depth} score {score_text} nodes {nodes} nps {nodes * 1000 // elapsed} time {elapsed} pv {move.uci()}")

    def start_go(self, args):
        if self.worker and self.worker.is_alive(): self.stop()
        opts = {args[i]: args[i + 1] for i in range(len(args) - 1)
                if args[i] in ("depth", "nodes", "movetime", "wtime", "btime", "winc", "binc")}
        depth = 30 if "infinite" in args else int(opts.get("depth", self.depth))
        mt = int(opts["movetime"]) / 1000 if "movetime" in opts else None
        node_limit = int(opts["nodes"]) if "nodes" in opts else None
        if mt is None:
            clock_key = "wtime" if self.board.turn else "btime"
            inc_key = "winc" if self.board.turn else "binc"
            if clock_key in opts:
                remaining = int(opts[clock_key]) / 1000
                inc = int(opts.get(inc_key, 0)) / 1000
                mt = max(.05, min(remaining * .7, remaining / 25 + inc * .8))
        board = self.board.copy()
        self.started = time.monotonic()
        if self.searcher is None:
            self.searcher = Search(self.evaluator, self.hash_mb, self.info, self.blend)
        else:
            self.searcher.evaluator = self.evaluator
            self.searcher.blend = self.blend
            # TT entries in Python cost ~230 bytes each, not the ~16 the
            # entry count once implied. 4 KB per entry keeps the dict at or
            # below the advertised Hash size.
            self.searcher.tt_limit = max(10000, self.hash_mb * 4096)
        def work():
            if self.book_enabled:
                move = self.book.probe(board)
                if move is not None:
                    self.send(f"info string Lichess opening book: {move.uci()}")
                    self.send(f"bestmove {move.uci()}")
                    return
            move, _ = self.searcher.run(board, depth, mt, node_limit)
            self.send(f"bestmove {move.uci() if move else '0000'}")
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def stop(self):
        if self.searcher: self.searcher.stop = True
        if self.worker and self.worker.is_alive(): self.worker.join()

    def setoption(self, line):
        try:
            # posix=False keeps Windows drive-letter backslashes intact; the
            # quote stripper restores quoting for names and values with spaces.
            parts = [_unquote(t) for t in shlex.split(line, posix=False)]
            name_at = parts.index("name") + 1
            value_at = parts.index("value") if "value" in parts else len(parts)
            name = " ".join(parts[name_at:value_at]).lower()
            value = " ".join(parts[value_at + 1:]) if value_at < len(parts) else ""
            if name == "depth": self.depth = max(1, min(30, int(value)))
            elif name == "hash": self.hash_mb = max(1, min(4096, int(value)))
            elif name == "evalfile":
                self.evalfile_set = True
                self.model_path = value
                if value:
                    self.evaluator = Evaluator(value)
                else:
                    self.evaluator = None
            elif name == "blend handcrafted": self.blend = value.lower() in ("true", "1", "yes")
            elif name == "lichess book": self.book_enabled = value.lower() in ("true", "1", "yes")
            elif name == "clear hash":
                if self.searcher: self.searcher.tt.clear()
        except Exception as e:
            self.send(f"info string setoption failed: {e}")

    def position(self, args):
        try:
            if args[0] == "startpos":
                board, idx = chess.Board(), 1
            elif args[0] == "fen":
                mi = args.index("moves") if "moves" in args else len(args)
                board, idx = chess.Board(" ".join(args[1:mi])), mi
            else: return
            if idx < len(args) and args[idx] == "moves":
                for token in args[idx + 1:]: board.push_uci(token)
            self.board = board
        except Exception as e:
            self.send(f"info string invalid position: {e}")

    def _load_evaluator(self, path):
        """Load and report a neural evaluator; failures degrade silently."""
        if not path:
            return
        try:
            self.evaluator = Evaluator(path)
            self.send(f"info string Loaded neural evaluator: {path}")
        except Exception as e:
            self.send(f"info string EvalFile load failed: {e}")

    def _resolve_default_eval_file(self):
        """First existing auto-load candidate, else None.

        Working-directory model wins over the bundled one so a freshly
        retrained model beats whatever shipped with a compiled exe.
        """
        for path in (DEFAULT_EVAL_FILE, BUNDLED_EVAL_FILE):
            if path is not None and path.is_file():
                return path
        return None

    def loop(self):
        self.send("id name Blunder 0")
        self.send("id author Shubham")
        self.send("option name Depth type spin default 8 min 1 max 30")
        self.send("option name Hash type spin default 64 min 1 max 4096")
        self.send("option name EvalFile type string default models/evaluator.npz")
        self.send("option name Blend Handcrafted type check default false")
        self.send("option name Lichess Book type check default true")
        self.send("option name Clear Hash type button")
        self.send("uciok")
        # Auto-load the default trained model only when the GUI has not sent
        # (or will not send) an explicit EvalFile option.
        if not self.evalfile_set:
            self._load_evaluator(self._resolve_default_eval_file())
        for raw in sys.stdin:
            line = raw.strip()
            if not line: continue
            cmd, _, tail = line.partition(" ")
            # A malformed command must never kill the engine mid-game: GUIs
            # quit unexpectedly and lose the position if the process exits.
            try:
                if cmd == "isready": self.send("readyok")
                elif cmd == "setoption": self.setoption(tail)
                elif cmd == "ucinewgame":
                    self.stop(); self.board = chess.Board()
                    if self.searcher: self.searcher.tt.clear()
                elif cmd == "position": self.position(tail.split())
                elif cmd == "go": self.start_go(tail.split())
                elif cmd == "stop": self.stop()
                elif cmd == "quit": self.stop(); break
                elif cmd == "debug": pass
            except Exception as e:
                self.send(f"info string error: {e}")


def main():
    UCI().loop()
