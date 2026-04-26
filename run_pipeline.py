"""
Unified Runner for Hybrid GTN-EEG Project
Runs training (single split, no CV) followed by evaluation and visualization.
"""

import argparse
import sys
import os
import train
import run_full_evaluation


def run_project():
    parser = argparse.ArgumentParser(
        description="Run complete Hybrid GTN-EEG Pipeline (single train/val/test split)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --synthetic --epochs 20
  python run_pipeline.py --data_dir ./data --epochs 50
  python run_pipeline.py --synthetic --epochs 30 --batch_size 32 --lr 0.0005
"""
    )
    parser.add_argument('--synthetic',   action='store_true',
                        help='Force use of synthetic data')
    parser.add_argument('--data_dir',    type=str,   default='./data',
                        help='Path to real .mat data directory')
    parser.add_argument('--epochs',      type=int,   default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size',  type=int,   default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr',          type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--output_dir',  type=str,   default='checkpoints',
                        help='Root folder for saved checkpoints (default: checkpoints)')
    args = parser.parse_args()

    print("=" * 60)
    print("  HYBRID GTN-EEG PIPELINE  —  Single Split Mode")
    print("=" * 60)

    # ── Build training args ───────────────────────────────────────────
    class TrainArgs:
        def __init__(self):
            self.data_dir   = args.data_dir
            self.epochs     = args.epochs
            self.batch_size = args.batch_size
            self.lr         = args.lr
            self.synthetic  = args.synthetic
            self.output_dir = args.output_dir
            self.no_cv      = True   # <-- always single split, no K-Fold
            self.n_folds    = 1      # not used, but keeps train.main happy

    train_args = TrainArgs()

    # ── Data check ───────────────────────────────────────────────────
    if not args.synthetic:
        if not os.path.exists(args.data_dir) or not os.listdir(args.data_dir):
            print(f"  [WARN] Real data not found in '{args.data_dir}'."
                  f" Switching to SYNTHETIC mode.")
            train_args.synthetic = True
        else:
            print(f"  Data source : real  ({args.data_dir})")
    else:
        print("  Data source : synthetic")

    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Split       : 70% train / 10% val / 20% test  (single run)")
    print()

    # ── Phase 1 : Training ───────────────────────────────────────────
    print(">>> PHASE 1: TRAINING")
    print("-" * 60)
    try:
        save_dir = train.main(train_args)
        print()
        print(f"  Training complete.  Checkpoint saved to: {save_dir}")
    except Exception as e:
        print(f"  [ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ── Phase 2 : Evaluation & Visualisation ─────────────────────────
    print()
    print(">>> PHASE 2: EVALUATION & VISUALISATION")
    print("-" * 60)
    try:
        run_full_evaluation.run_evaluation(save_dir)
    except Exception as e:
        print(f"  [ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Results saved to: {save_dir}")
    print()
    print("  Files generated:")
    import glob
    for f in sorted(glob.glob(os.path.join(save_dir, '*'))):
        print(f"    {os.path.basename(f)}")
    print("=" * 60)


if __name__ == "__main__":
    run_project()