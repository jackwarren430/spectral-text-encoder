import argparse
import os

from run_utils import plot_metrics


def main():
    p = argparse.ArgumentParser(description="Plot train/val loss and accuracy from a run folder's metrics.csv.")
    p.add_argument("run_dir", help="Path to a run folder containing metrics.csv")
    args = p.parse_args()

    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")
    if not os.path.exists(os.path.join(args.run_dir, "metrics.csv")):
        raise SystemExit(f"no metrics.csv in {args.run_dir}")

    plot_metrics(args.run_dir)
    print(f"wrote {os.path.join(args.run_dir, 'loss.png')}")
    print(f"wrote {os.path.join(args.run_dir, 'acc.png')}")


if __name__ == "__main__":
    main()
