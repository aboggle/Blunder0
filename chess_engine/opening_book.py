"""Small, cached client for the public Lichess Opening Explorer."""
import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import chess


class LichessBook:
    """Fetch a popular move for an opening position, with a short timeout."""

    # The Opening Explorer is hosted on the .ovh API domain. The .org host
    # does not serve this endpoint.
    URL = "https://explorer.lichess.ovh/lichess"

    def __init__(self, timeout=1.0, max_entries=2048):
        self.timeout = timeout
        self.max_entries = max_entries
        self.cache = {}
        self.disabled_until = 0.0

    def probe(self, board):
        if board.fullmove_number > 20 or board.is_variant_end():
            return None
        # Ignore clocks and move history: the explorer identifies the position
        # by its piece placement, side to move, and castling/en-passant rights.
        fen = board.fen(en_passant="fen").split()[:4]
        key = " ".join(fen)
        if key in self.cache:
            uci = self.cache[key]
            move = chess.Move.from_uci(uci) if uci else None
            return move if move in board.legal_moves else None
        if time.monotonic() < self.disabled_until:
            return None
        params = urlencode({"variant": "standard", "speeds": "blitz,rapid,classical",
                            "fen": key,
                            "moves": 1, "topGames": 0, "recentGames": 0})
        request = Request(f"{self.URL}?{params}", headers={"User-Agent": "BC0 chess engine"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read(65536))
            rows = data.get("moves", [])
            legal = {move.uci(): move for move in board.legal_moves}
            rows = [row for row in rows if row.get("uci") in legal and
                    int(row.get("white", 0)) + int(row.get("draws", 0)) +
                    int(row.get("black", 0)) > 0]
            if rows:
                # Favor established lines while modestly accounting for results.
                def rank(row):
                    total = int(row.get("white", 0)) + int(row.get("draws", 0)) + int(row.get("black", 0))
                    score = (int(row.get("white", 0)) + int(row.get("draws", 0)) * .5) / total
                    if not board.turn:
                        score = 1.0 - score
                    return total * (0.75 + 0.5 * score)
                selected = max(rows, key=rank)["uci"]
                move = legal[selected]
            else:
                move = None
            self.cache[key] = move.uci() if move else ""
            if len(self.cache) > self.max_entries:
                self.cache.pop(next(iter(self.cache)))
            return move
        except Exception:
            # Avoid repeatedly stalling the engine when offline or rate limited.
            self.disabled_until = time.monotonic() + 30
            return None
