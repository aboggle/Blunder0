"""UCI adapter and opening-book tests."""
import io
import time
from unittest.mock import patch

import chess
import numpy as np
import pytest

import chess_engine.uci as uci_mod
from chess_engine.uci import UCI
from chess_engine.opening_book import LichessBook


@pytest.fixture
def tiny_model(tmp_path):
    """A structurally valid (all-zero) Savez model the Evaluator can load."""
    path = tmp_path / "evaluator.npz"
    hidden = 4
    np.savez_compressed(
        path,
        w1=np.zeros((773, hidden), np.float32), b1=np.zeros(hidden, np.float32),
        w2=np.zeros((hidden, hidden), np.float32), b2=np.zeros(hidden, np.float32),
        w3=np.zeros((hidden, 1), np.float32), b3=np.zeros(1, np.float32))
    return path


def _tiny_model_at(path):
    hidden = 1
    np.savez_compressed(
        path,
        w1=np.zeros((773, hidden), np.float32), b1=np.zeros(hidden, np.float32),
        w2=np.zeros((hidden, hidden), np.float32), b2=np.zeros(hidden, np.float32),
        w3=np.zeros((hidden, 1), np.float32), b3=np.zeros(1, np.float32))
    return path


def test_setoption_clamps_ranges():
    uci = UCI()
    uci.setoption("name Depth value 99")
    assert uci.depth == 30
    uci.setoption("name Depth value 0")
    assert uci.depth == 1
    uci.setoption("name Hash value 0")
    assert uci.hash_mb == 1
    uci.setoption("name Lichess Book value false")
    assert uci.book_enabled is False
    uci.setoption("name Blend Handcrafted value true")
    assert uci.blend is True


def test_setoption_bad_evalfile_does_not_crash(capsys):
    uci = UCI()
    uci.setoption("name EvalFile value missing.npz")
    assert uci.model_path == "missing.npz"
    assert uci.evaluator is None
    assert "info string setoption failed" in capsys.readouterr().out


def test_evalfile_accepts_windows_paths():
    uci = UCI()
    uci.setoption(r"name EvalFile value D:\chess\models\evaluator.npz")
    assert uci.model_path == r"D:\chess\models\evaluator.npz"


def test_default_model_autoloaded_when_present(tiny_model, monkeypatch, capsys):
    monkeypatch.setattr(uci_mod, "DEFAULT_EVAL_FILE", tiny_model)
    uci = UCI()
    uci.book_enabled = False
    monkeypatch.setattr("sys.stdin", io.StringIO("quit\n"))
    uci.loop()
    assert uci.evaluator is not None
    out = capsys.readouterr().out
    assert "Loaded neural evaluator" in out
    assert tiny_model.name in out


def test_no_default_model_means_handcrafted_eval(monkeypatch, tmp_path):
    monkeypatch.setattr(uci_mod, "DEFAULT_EVAL_FILE", tmp_path / "missing.npz")
    uci = UCI()
    uci.book_enabled = False
    monkeypatch.setattr("sys.stdin", io.StringIO("quit\n"))
    uci.loop()
    assert uci.evaluator is None


def test_default_resolution_prefers_working_directory(tmp_path, monkeypatch):
    # A freshly retrained model (working dir) beats a bundled one.
    src_model = _tiny_model_at(tmp_path / "evaluator.npz")
    bundled = _tiny_model_at(tmp_path / "bundle.npz")

    monkeypatch.setattr(uci_mod, "DEFAULT_EVAL_FILE", tmp_path / "missing.npz")
    monkeypatch.setattr(uci_mod, "BUNDLED_EVAL_FILE", tmp_path / "missing2.npz")
    assert UCI()._resolve_default_eval_file() is None  # neither exists

    monkeypatch.setattr(uci_mod, "BUNDLED_EVAL_FILE", bundled)
    assert UCI()._resolve_default_eval_file() == bundled  # bundled fallback

    monkeypatch.setattr(uci_mod, "DEFAULT_EVAL_FILE", src_model)
    assert UCI()._resolve_default_eval_file() == src_model  # cwd wins


def test_bundled_model_autoloaded_in_frozen_exe(tmp_path, monkeypatch, capsys):
    """Compiled builds load the --add-data payload from sys._MEIPASS."""
    bundled = _tiny_model_at(tmp_path / "bundle.npz")
    monkeypatch.setattr(uci_mod, "DEFAULT_EVAL_FILE", tmp_path / "missing.npz")
    monkeypatch.setattr(uci_mod, "BUNDLED_EVAL_FILE", bundled)
    uci = UCI()
    uci.book_enabled = False
    monkeypatch.setattr("sys.stdin", io.StringIO("quit\n"))
    uci.loop()
    assert uci.evaluator is not None
    out = capsys.readouterr().out
    assert "Loaded neural evaluator" in out
    assert "bundle" in out


def test_explicit_evalfile_disables_autoload(tiny_model, monkeypatch):
    default = tiny_model
    monkeypatch.setattr(uci_mod, "DEFAULT_EVAL_FILE", default)
    # GUI sent EvalFile explicitly: the default must lose.
    uci = UCI()
    uci.setoption(f"name EvalFile value {default.as_posix()}")
    assert uci.model_path == default.as_posix()
    assert uci.evaluator is not None and uci.evalfile_set

    # An explicitly empty EvalFile opts out entirely (handcrafted eval).
    empty_uci = UCI()
    empty_uci.setoption("name EvalFile value")
    assert empty_uci.evaluator is None and empty_uci.evalfile_set


def test_bad_explicit_evalfile_does_not_fall_back(tiny_model, monkeypatch, capsys):
    """A broken override must not silently re-enable the default model."""
    monkeypatch.setattr(uci_mod, "DEFAULT_EVAL_FILE", tiny_model)
    uci = UCI()
    uci.setoption("name EvalFile value broken.npz")
    assert uci.evaluator is None and uci.evalfile_set
    # The auto-load branch only runs pre-setoption, so this path must not fire.
    assert uci.evaluator is None


def test_position_parsing():
    uci = UCI()
    uci.position("startpos moves e2e4 e7e5".split())
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert uci.board == board
    uci.position("fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".split())
    assert uci.board == chess.Board()
    uci.position("startpos moves e2e4".split())
    board = chess.Board()
    board.push_uci("e2e4")
    assert uci.board == board


def test_loop_survives_malformed_commands(capsys, monkeypatch):
    """Malformed input must produce an info string, never exit the engine."""
    uci = UCI()
    uci.book_enabled = False  # keep the worker offline for the test
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO("position\ngibberish\ngo movetime nonsense\nquit\n"))
    uci.loop()
    out = capsys.readouterr().out
    # The malformed position is rejected inside its handler...
    assert "invalid position" in out
    # ...and the bad go argument hits the loop's exception guard.
    assert "error:" in out


def test_book_disables_itself_when_offline():
    """A failed fetch must set the backoff and the next probe must not retry."""
    book = LichessBook()

    def offline(*args, **kwargs):
        raise OSError("offline")

    with patch("chess_engine.opening_book.urlopen", offline):
        assert book.probe(chess.Board()) is None
    assert book.disabled_until > time.monotonic()
    with patch("chess_engine.opening_book.urlopen") as must_not_be_called:
        assert book.probe(chess.Board()) is None
        must_not_be_called.assert_not_called()


def test_book_skips_positions_beyond_move_20():
    board = chess.Board()
    for _ in range(40):
        board.push(next(iter(board.legal_moves)))
    book = LichessBook()
    with patch("chess_engine.opening_book.urlopen") as must_not_be_called:
        assert book.probe(board) is None
        must_not_be_called.assert_not_called()
