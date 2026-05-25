"""
validate.py
Validasi nilai α, β, γ pilihan terhadap data terekam.

Berguna untuk:
  1. Verifikasi nilai best dari grid_search.py sebelum di-deploy ke Kotlin
  2. Membandingkan beberapa kandidat (mis. best vs nilai default lama)
  3. Cek confusion matrix + precision/recall (selain accuracy)

Cara pakai:
    # Pakai default lama Kotlin
    python validate.py --alpha 0.0015 --beta 0.001 --gamma 0.0005

    # Pakai best hasil grid search
    python validate.py --alpha 0.0020 --beta 0.0010 --gamma 0.0005

    # Bandingkan side-by-side
    python validate.py --compare 0.0015,0.001,0.0005  0.0020,0.001,0.0005
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

T_BASE = 0.24
T_DYN_FLOOR = 0.10


def load_recordings(rec_dir: Path) -> pd.DataFrame:
    csv_files = sorted(rec_dir.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] Tidak ada CSV di {rec_dir}")
        sys.exit(1)
    dfs = []
    for f in csv_files:
        d = pd.read_csv(f)
        d["subject"] = f.stem
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def compute_metrics(df: pd.DataFrame, alpha: float, beta: float, gamma: float) -> dict:
    """Return dict berisi confusion matrix + metrics per orientasi + overall."""
    t_dyn = T_BASE - (alpha * df["yaw"].abs()
                      + beta  * df["pitch"].abs()
                      + gamma * df["roll"].abs())
    t_dyn = t_dyn.clip(lower=T_DYN_FLOOR)
    pred = (df["ear"] < t_dyn).astype(int)
    gt = df["eye_state"]

    def calc(pred_sub, gt_sub):
        tp = int(((pred_sub == 1) & (gt_sub == 1)).sum())
        tn = int(((pred_sub == 0) & (gt_sub == 0)).sum())
        fp = int(((pred_sub == 1) & (gt_sub == 0)).sum())
        fn = int(((pred_sub == 0) & (gt_sub == 1)).sum())
        n = tp + tn + fp + fn
        acc  = (tp + tn) / n if n > 0 else np.nan
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1   = (2 * prec * rec / (prec + rec)
                if prec is not np.nan and rec is not np.nan and (prec + rec) > 0
                else np.nan)
        return dict(tp=tp, tn=tn, fp=fp, fn=fn, n=n,
                    acc=acc, precision=prec, recall=rec, f1=f1)

    result = {"alpha": alpha, "beta": beta, "gamma": gamma,
              "overall": calc(pred, gt)}
    for ori in df["orientation"].unique():
        m = df["orientation"] == ori
        result[ori] = calc(pred[m], gt[m])
    return result


def print_metrics(m: dict, header: str):
    print("\n" + "=" * 80)
    print(header)
    print(f"  α = {m['alpha']:.5f}   β = {m['beta']:.5f}   γ = {m['gamma']:.5f}")
    print("=" * 80)
    cols = "Group         N     TP    TN    FP    FN    Acc      Prec     Rec      F1"
    print(cols)
    print("-" * 80)
    groups = ["overall"] + [k for k in m if k not in ("alpha", "beta", "gamma", "overall")]
    for g in groups:
        d = m[g]
        print(f"{g:<13} {d['n']:<5} {d['tp']:<5} {d['tn']:<5} {d['fp']:<5} {d['fn']:<5} "
              f"{_fmt(d['acc']):<8} {_fmt(d['precision']):<8} "
              f"{_fmt(d['recall']):<8} {_fmt(d['f1']):<8}")


def _fmt(v):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def parse_combo(s: str):
    try:
        parts = [float(x) for x in s.split(",")]
        if len(parts) != 3:
            raise ValueError
        return parts
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Format harus 'alpha,beta,gamma' (mis. '0.0015,0.001,0.0005'), bukan '{s}'"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings-dir", default="recordings")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta",  type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--compare", nargs="+", type=parse_combo,
                        help="Bandingkan beberapa kombinasi, format: 0.0015,0.001,0.0005")
    args = parser.parse_args()

    df = load_recordings(Path(args.recordings_dir))
    print(f"Loaded {len(df)} frame dari {df['subject'].nunique()} subjek")

    if args.compare:
        for combo in args.compare:
            m = compute_metrics(df, *combo)
            print_metrics(m, f"VALIDASI: α={combo[0]} β={combo[1]} γ={combo[2]}")
    else:
        if args.alpha is None or args.beta is None or args.gamma is None:
            print("[ERROR] Berikan --alpha --beta --gamma atau pakai --compare")
            sys.exit(1)
        m = compute_metrics(df, args.alpha, args.beta, args.gamma)
        print_metrics(m, "VALIDASI HASIL")


if __name__ == "__main__":
    main()
