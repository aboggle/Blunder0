import argparse

def main():
    parser = argparse.ArgumentParser(prog="chess-engine", description="Neural-evaluation chess engine")
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("train", help="train an evaluator from a FEN,evaluation CSV")
    p.add_argument("csv", help="CSV path, e.g. evaluated.csv")
    p.add_argument("--output", default="models/evaluator.npz")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--limit", type=int, default=0, help="maximum rows (0 means all)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--chunk-size", type=int, default=8192, help="positions buffered before shuffling/training")
    p.add_argument("--threads", type=int, default=4, help="BLAS worker threads used for training")
    uci = sub.add_parser("uci", help="start the UCI chess engine")
    uci.add_argument("--depth", type=int, default=8, help="default search depth (1-30)")
    args = parser.parse_args()
    if args.command == "train":
        from .network import train
        train(args.csv, args.output, args.epochs, args.batch_size, args.hidden,
              args.learning_rate, args.limit, args.seed, args.chunk_size, args.threads)
    elif args.command is None or args.command == "uci":
        from .uci import UCI
        engine = UCI()
        if args.command == "uci": engine.depth = max(1, min(30, args.depth))
        engine.loop()

if __name__ == "__main__": main()
