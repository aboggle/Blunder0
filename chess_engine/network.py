from pathlib import Path
import re
import numpy as np
import chess
from .features import FEATURES, feature_indices


def parse_evaluation(value: str) -> float:
    value = value.strip()
    mate = re.fullmatch(r"#([+-]?\d+)", value)
    if mate:
        # Mate scores have no pawn-equivalent value. Use the existing target
        # ceiling so a forced mate is learned as a decisive result.
        return 20.0 if int(mate.group(1)) > 0 else -20.0
    return float(value)

class Evaluator:
    def __init__(self, path: str | Path):
        with np.load(path) as data:
            self.w1, self.b1 = data["w1"], data["b1"]
            self.w2, self.b2 = data["w2"], data["b2"]
            self.w3, self.b3 = data["w3"], data["b3"]

    def predict_board(self, board) -> float:
        indices = feature_indices(board)
        h1 = self.b1.copy()
        if indices:
            h1 += self.w1[indices].sum(axis=0)
        h1 += self.w1[768] * (1.0 if board.turn == chess.WHITE else -1.0)
        h1 = np.tanh(h1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        return float((h2 @ self.w3 + self.b3)[0])


def train(csv_path: str | Path, output: str | Path, epochs=5, batch_size=256,
          hidden=64, learning_rate=0.001, limit=0, seed=7, chunk_size=8192,
          threads=4):
    import csv
    import chess
    from threadpoolctl import threadpool_limits
    from .features import encode

    if epochs < 1 or hidden < 1 or learning_rate <= 0:
        raise ValueError("epochs, hidden width, and learning rate must be positive")
    if batch_size < 1 or chunk_size < batch_size or threads < 1:
        raise ValueError("batch-size and threads must be positive; chunk-size must be >= batch-size")

    def read_chunks():
        rows_x, rows_y = [], []
        seen = 0
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"FEN", "evaluation"}.issubset(reader.fieldnames):
                raise ValueError("CSV must contain FEN and evaluation columns")
            for row in reader:
                if limit and seen >= limit:
                    break
                seen += 1
                try:
                    board = chess.Board(row["FEN"])
                    score = parse_evaluation(row["evaluation"])
                    if not np.isfinite(score):
                        continue
                    rows_x.append(encode(board))
                    rows_y.append(np.clip(score, -20.0, 20.0))
                except (ValueError, TypeError, KeyError):
                    continue
                if len(rows_x) >= chunk_size:
                    yield np.asarray(rows_x, dtype=np.float32), np.asarray(rows_y, dtype=np.float32).reshape(-1, 1)
                    rows_x, rows_y = [], []
        if rows_x:
            yield np.asarray(rows_x, dtype=np.float32), np.asarray(rows_y, dtype=np.float32).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    # Xavier initialization.
    w1 = rng.normal(0, np.sqrt(2 / (FEATURES + hidden)), (FEATURES, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, np.float32)
    w2 = rng.normal(0, np.sqrt(2 / (hidden * 2)), (hidden, hidden)).astype(np.float32)
    b2 = np.zeros(hidden, np.float32)
    w3 = rng.normal(0, np.sqrt(2 / (hidden + 1)), (hidden, 1)).astype(np.float32)
    b3 = np.zeros(1, np.float32)
    params = [w1, b1, w2, b2, w3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    step = 0
    limiter = threadpool_limits(limits=threads)  # Retained for this training process.
    n = 0
    for epoch in range(epochs):
        total_loss = 0.0
        epoch_n = 0
        chunks = 0
        for chunk_x, chunk_y in read_chunks():
            chunks += 1
            order = rng.permutation(len(chunk_x))
            for start in range(0, len(order), batch_size):
                ix = order[start:start + batch_size]
                xb, yb = chunk_x[ix], chunk_y[ix]
                z1 = xb @ w1 + b1; h1 = np.tanh(z1)
                z2 = h1 @ w2 + b2; h2 = np.tanh(z2)
                pred = h2 @ w3 + b3
                err = pred - yb
                # Huber loss limits the influence of outlier evaluations.
                d = np.clip(err, -1.0, 1.0) / len(ix)
                total_loss += float(np.mean(np.where(np.abs(err) < 1, .5 * err ** 2, np.abs(err) - .5))) * len(ix)
                gw3 = h2.T @ d; gb3 = d.sum(axis=0)
                dh2 = d @ w3.T; dz2 = dh2 * (1 - h2 * h2)
                gw2 = h1.T @ dz2; gb2 = dz2.sum(axis=0)
                dh1 = dz2 @ w2.T; dz1 = dh1 * (1 - h1 * h1)
                gw1 = xb.T @ dz1; gb1 = dz1.sum(axis=0)
                step += 1
                for j, grad in enumerate([gw1, gb1, gw2, gb2, gw3, gb3]):
                    m[j] = .9 * m[j] + .1 * grad
                    v[j] = .999 * v[j] + .001 * grad * grad
                    mh = m[j] / (1 - .9 ** step); vh = v[j] / (1 - .999 ** step)
                    params[j] -= learning_rate * mh / (np.sqrt(vh) + 1e-8)
                epoch_n += len(ix)
            if chunks % 10 == 0:
                print(f"epoch {epoch + 1}/{epochs}  processed={epoch_n}  chunks={chunks}", flush=True)
        if epoch_n == 0:
            raise ValueError("No usable training rows were found")
        n = epoch_n
        print(f"epoch {epoch + 1}/{epochs}  huber_loss={total_loss / n:.4f}  positions={n}", flush=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
    print(f"Saved model to {output} ({n} positions)")
