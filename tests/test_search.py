"""Search and evaluation tests: symmetry, mate detection, TT, draw rules."""
import chess
import pytest

from chess_engine.search import Search, MATE, INF

# Standard positions: startpos, Kiwipete, and three other perft classics.
KEY_POSITIONS = [chess.Board(fen) for fen in (
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
)]


@pytest.mark.parametrize("board", KEY_POSITIONS, ids=lambda b: b.fen().split()[0])
def test_eval_mirror_symmetry(board):
    """White-relative scoring must be invariant under board mirroring."""
    search = Search()
    assert search.evaluate(board.copy(stack=False)) == search.evaluate(board.mirror())


def test_perft_startpos():
    """Move generation via python-chess must survive the pinned version."""
    board = chess.Board()
    assert _perft(board, 4) == 197281


def test_perft_kiwipete():
    assert _perft(chess.Board(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"), 3) == 97862


def _perft(board, depth):
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += _perft(board, depth - 1)
        board.pop()
    return total


def test_mate_in_one_found_and_scored():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
    move, score = Search().run(board, depth=3)
    assert move == chess.Move.from_uci("a1a8")
    assert score > MATE - 100


def test_run_returns_legal_move():
    move, _ = Search().run(chess.Board(), depth=4)
    assert move in chess.Board().legal_moves


def test_tt_mate_scores_round_trip():
    for ply in range(0, 40):
        for score in (MATE - 101, -MATE + 101, 123, -123):
            assert Search._from_tt(Search._to_tt(score, ply), ply) == score


def test_draw_repetition_and_insufficient_material():
    search = Search()
    king_end = chess.Board("8/8/8/8/8/8/k1K5/8 w - - 0 1")
    assert search._draw(king_end, 1)

    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    assert board.is_repetition(3) and search._draw(board, 0)
    assert not search._draw(chess.Board(), 0)


def test_qsearch_scores_stalemate_as_draw():
    search = Search()
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert search._qsearch(stalemate, -INF, INF, 0) == 0


def test_hash_option_maps_to_reasonable_entry_count():
    assert Search(hash_mb=64).tt_limit == 64 * 4096
    # The 10k-entry floor keeps tiny hash settings usable.
    assert Search(hash_mb=1).tt_limit == 10000
    # The new accounting keeps the dict near the advertised size: with Python
    # TT entries at ~230 bytes, 64 MB -> ~260k entries ~ tens of MB, and the
    # maximum Hash setting stays in the low GB instead of OOM territory.
    assert Search(hash_mb=4096).tt_limit == 4096 * 4096


class _StubNet:
    def predict_board(self, board):
        return 1.234  # white-relative pawns


def test_neural_eval_blend_option():
    """With EvalFile the output is the raw network; blending adds positional."""
    board = chess.Board()
    board.push_uci("e2e4")
    pure = Search(evaluator=_StubNet()).evaluate(board)
    blended = Search(evaluator=_StubNet(), blend=True).evaluate(board)
    assert pure == -123  # black to move: white +1.234 pawns becomes -123 cp
    assert blended == -143  # +20 cp of handcrafted positional terms on top
    # Note: the blend is not mirror-exact. The handcrafted positional block is
    # white-relative by construction and relies on the material block for
    # color context; the legacy combined scoring had the same property.
