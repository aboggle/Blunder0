import numpy as np
import chess

FEATURES = 12 * 64 + 5


def feature_indices(board: chess.Board) -> list[int]:
    """Sparse input indices for `encode` and the incremental evaluator.

    First 768 cells are 12 piece planes (white unmirrored, black rank-mirrored
    so patterns are shared), then side-to-move at 768 and four castling-rights
    flags at 769-772. The exact same mapping is baked into trained .npz models,
    so this must never be reordered without bumping the model format.
    """
    indices = []
    for square, piece in board.piece_map().items():
        sq = square if piece.color == chess.WHITE else chess.square_mirror(square)
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        indices.append(plane * 64 + sq)
    rights = (board.has_kingside_castling_rights(chess.WHITE),
              board.has_queenside_castling_rights(chess.WHITE),
              board.has_kingside_castling_rights(chess.BLACK),
              board.has_queenside_castling_rights(chess.BLACK))
    indices.extend(769 + i for i, enabled in enumerate(rights) if enabled)
    return indices


def encode(board: chess.Board) -> np.ndarray:
    x = np.zeros(FEATURES, dtype=np.float32)
    for idx in feature_indices(board):
        x[idx] = 1.0
    x[768] = 1.0 if board.turn == chess.WHITE else -1.0
    return x
