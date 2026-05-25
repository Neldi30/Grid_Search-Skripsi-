"""
grid_search.py
Cari nilai optimal α, β, γ untuk T_dynamic = T_BASE − (α·|yaw| + β·|pitch| + γ·|roll|)

METODOLOGI (defensible untuk skripsi):
  - Train/Test split stratified per (subject, orientation, eye_state) — default 80/20
  - Alternatif: LOSO-CV (Leave-One-Subject-Out) bila ≥ 2 subjek
  - Coarse grid → 2 putaran fine grid iteratif (refine top-3, span ±25% lalu ±10%)
  - Vectorized batch evaluation via numpy broadcasting
  - Baseline: static threshold (α=β=γ=0) untuk pembanding eksplisit
  - Multi-metric: accuracy, balanced_accuracy, F1, precision, recall, specificity
  - Boundary check: warning kalau best param di tepi grid
  - Floor clipping report: % frame yang T_dyn-nya ter-clip ke T_DYN_FLOOR
  - Sort + report konsisten di metrik yang dipilih (primary_metric)
  - Output timestamped (results/YYYYMMDD_HHMMSS/) — eksperimen lama tidak ter-overwrite

Cara pakai:
    python grid_search.py
    python grid_search.py --metric f1                 # objective F1 (cost-sensitive)
    python grid_search.py --split loso                # LOSO-CV (butuh ≥ 2 subjek)
    python grid_search.py --test-size 0.3             # ubah proporsi test
    python grid_search.py --no-fine                   # skip fine search
    python grid_search.py --seed 123                  # reproducibility
"""

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Konstanta ────────────────────────────────────────────────────────────
T_BASE = 0.24
T_DYN_FLOOR = 0.10

COARSE_ALPHA = np.array([0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030])
COARSE_BETA  = np.array([0.0003, 0.0007, 0.0010, 0.0015, 0.0020])
COARSE_GAMMA = np.array([0.0001, 0.0003, 0.0005, 0.0008, 0.0010])

THESIS_ORIENTATIONS = ["frontal", "left", "right"]
REQUIRED_COLUMNS    = ["ear", "yaw", "pitch", "roll", "orientation", "eye_state"]
AVAILABLE_METRICS   = ["accuracy", "balanced_accuracy", "f1"]


# ════════════════════════════════════════════════════════════════════════
# 1. Data loading & validation (issue #11)
# ════════════════════════════════════════════════════════════════════════
def validate_columns(df: pd.DataFrame, source: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[ERROR] {source}: kolom hilang {missing}")
        print(f"        Kolom tersedia: {list(df.columns)}")
        sys.exit(1)
    bad_eye = ~df["eye_state"].isin([0, 1])
    if bad_eye.any():
        print(f"[ERROR] {source}: eye_state harus 0/1, ditemukan {df.loc[bad_eye, 'eye_state'].unique()}")
        sys.exit(1)
    for col in ["ear", "yaw", "pitch", "roll"]:
        if df[col].isna().any():
            print(f"[WARN] {source}: ada NaN di kolom {col} — akan dibuang")


def load_recordings(rec_dir: Path) -> pd.DataFrame:
    csv_files = sorted(rec_dir.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] Tidak ada CSV di {rec_dir}. Jalankan record_data.py dulu.")
        sys.exit(1)
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        validate_columns(df, f.name)
        df["subject"] = f.stem
        dfs.append(df)
        print(f"  loaded {f.name}: {len(df)} frame")
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined["orientation"] != "none"].dropna(
        subset=["ear", "yaw", "pitch", "roll"]).reset_index(drop=True)
    print(f"\nTotal: {len(combined)} frame dari {combined['subject'].nunique()} subjek")
    print("Distribusi (orientation × eye_state):")
    print(combined.groupby(["orientation", "eye_state"]).size().rename("count").to_string())
    return combined


# ════════════════════════════════════════════════════════════════════════
# 2. Train/Test split (issue #1)
# ════════════════════════════════════════════════════════════════════════
def stratified_holdout(df: pd.DataFrame, test_size: float, seed: int):
    """Split stratified per (subject, orientation, eye_state)."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for _, g in df.groupby(["subject", "orientation", "eye_state"]):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def loso_folds(df: pd.DataFrame):
    """Yield (train, test, held_out_subject)."""
    subjects = sorted(df["subject"].unique())
    if len(subjects) < 2:
        print("[ERROR] LOSO-CV butuh ≥ 2 subjek. Pakai --split holdout untuk 1 subjek.")
        sys.exit(1)
    for s in subjects:
        train = df[df["subject"] != s].reset_index(drop=True)
        test  = df[df["subject"] == s].reset_index(drop=True)
        yield train, test, s


# ════════════════════════════════════════════════════════════════════════
# 3. Vectorized batch evaluation (issue #10)
# ════════════════════════════════════════════════════════════════════════
def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b > 0, a / np.maximum(b, 1), np.nan)
    return out


def compute_metrics(pred, gt):
    """pred,gt last axis = frames. Returns dict of arrays (same leading shape)."""
    tp = ((pred == 1) & (gt == 1)).sum(axis=-1)
    tn = ((pred == 0) & (gt == 0)).sum(axis=-1)
    fp = ((pred == 1) & (gt == 0)).sum(axis=-1)
    fn = ((pred == 0) & (gt == 1)).sum(axis=-1)
    n  = tp + tn + fp + fn
    acc  = _safe_div(tp + tn, n)
    prec = _safe_div(tp, tp + fp)
    rec  = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    f1   = _safe_div(2 * prec * rec, prec + rec)
    bal  = (rec + spec) / 2.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, n=n,
                accuracy=acc, precision=prec, recall=rec,
                specificity=spec, f1=f1, balanced_accuracy=bal)


def batch_predict_grid(df: pd.DataFrame, alphas, betas, gammas):
    """
    Vectorized predict untuk semua kombinasi grid (A × B × G) × F frames.
    Returns:
      pred    : (A, B, G, F) int8
      clipped : (A, B, G, F) bool — apakah T_dyn ter-floor
    """
    yaw_abs   = df["yaw"].abs().to_numpy(dtype=np.float64)
    pitch_abs = df["pitch"].abs().to_numpy(dtype=np.float64)
    roll_abs  = df["roll"].abs().to_numpy(dtype=np.float64)
    ear       = df["ear"].to_numpy(dtype=np.float64)

    alphas = np.asarray(alphas, dtype=np.float64)
    betas  = np.asarray(betas,  dtype=np.float64)
    gammas = np.asarray(gammas, dtype=np.float64)

    ac = alphas[:, None] * yaw_abs[None, :]    # (A, F)
    bc = betas[:, None]  * pitch_abs[None, :]  # (B, F)
    gc = gammas[:, None] * roll_abs[None, :]   # (G, F)
    total = (ac[:, None, None, :] +
             bc[None, :, None, :] +
             gc[None, None, :, :])             # (A, B, G, F)

    t_dyn_raw = T_BASE - total
    clipped   = t_dyn_raw < T_DYN_FLOOR
    t_dyn     = np.maximum(t_dyn_raw, T_DYN_FLOOR)
    pred      = (ear[None, None, None, :] < t_dyn).astype(np.int8)
    return pred, clipped


def batch_predict_combos(df: pd.DataFrame, alphas, betas, gammas):
    """Eval N kombinasi non-grid (mismatched) → (N, F) pred + (N,) clip_frac."""
    yaw_abs   = df["yaw"].abs().to_numpy(dtype=np.float64)
    pitch_abs = df["pitch"].abs().to_numpy(dtype=np.float64)
    roll_abs  = df["roll"].abs().to_numpy(dtype=np.float64)
    ear       = df["ear"].to_numpy(dtype=np.float64)

    alphas = np.asarray(alphas, dtype=np.float64)
    betas  = np.asarray(betas,  dtype=np.float64)
    gammas = np.asarray(gammas, dtype=np.float64)

    # Per-combo (N, F)
    contrib = (alphas[:, None] * yaw_abs[None, :] +
               betas[:, None]  * pitch_abs[None, :] +
               gammas[:, None] * roll_abs[None, :])
    t_dyn_raw = T_BASE - contrib
    clipped   = t_dyn_raw < T_DYN_FLOOR
    t_dyn     = np.maximum(t_dyn_raw, T_DYN_FLOOR)
    pred      = (ear[None, :] < t_dyn).astype(np.int8)
    return pred, clipped


# ════════════════════════════════════════════════════════════════════════
# 4. Evaluate → DataFrame (per-orientation + weighted thesis avg, issues #3, #12)
# ════════════════════════════════════════════════════════════════════════
def _per_orientation_metrics(pred, df: pd.DataFrame):
    """Return dict {orientation: metrics_dict}, leading shape sesuai pred."""
    ori_arr = df["orientation"].to_numpy()
    eye_arr = df["eye_state"].to_numpy()
    out = {}
    for ori in np.unique(ori_arr):
        mask = ori_arr == ori
        if mask.sum() == 0:
            continue
        out[ori] = compute_metrics(pred[..., mask], eye_arr[mask])
        out[ori]["n_frames"] = int(mask.sum())
    out["__overall"] = compute_metrics(pred, eye_arr)
    return out


def _flatten_to_rows(pred_shape_grid, metrics_per_ori, alphas, betas, gammas, clip_frac):
    """Convert (A,B,G) tensor metrics → list of rows."""
    A, B, G = pred_shape_grid
    rows = []
    for i in range(A):
        for j in range(B):
            for k in range(G):
                row = {"alpha": float(alphas[i]),
                       "beta":  float(betas[j]),
                       "gamma": float(gammas[k])}
                o = metrics_per_ori["__overall"]
                for m in ["accuracy", "balanced_accuracy", "f1",
                          "precision", "recall", "specificity"]:
                    row[f"overall_{m}"] = float(o[m][i, j, k])
                row["overall_tp"] = int(o["tp"][i, j, k])
                row["overall_tn"] = int(o["tn"][i, j, k])
                row["overall_fp"] = int(o["fp"][i, j, k])
                row["overall_fn"] = int(o["fn"][i, j, k])
                for ori, m in metrics_per_ori.items():
                    if ori == "__overall":
                        continue
                    row[f"acc_{ori}"] = float(m["accuracy"][i, j, k])
                    row[f"f1_{ori}"]  = float(m["f1"][i, j, k])
                row["clip_frac"] = float(clip_frac[i, j, k])
                rows.append(row)
    return rows


def _flatten_combos_to_rows(pred_n, metrics_per_ori, alphas, betas, gammas, clip_frac):
    """Convert (N,) per-combo metrics → list of rows."""
    rows = []
    N = len(alphas)
    for i in range(N):
        row = {"alpha": float(alphas[i]), "beta": float(betas[i]), "gamma": float(gammas[i])}
        o = metrics_per_ori["__overall"]
        for m in ["accuracy", "balanced_accuracy", "f1",
                  "precision", "recall", "specificity"]:
            row[f"overall_{m}"] = float(o[m][i])
        row["overall_tp"] = int(o["tp"][i])
        row["overall_tn"] = int(o["tn"][i])
        row["overall_fp"] = int(o["fp"][i])
        row["overall_fn"] = int(o["fn"][i])
        for ori, m in metrics_per_ori.items():
            if ori == "__overall":
                continue
            row[f"acc_{ori}"] = float(m["accuracy"][i])
            row[f"f1_{ori}"]  = float(m["f1"][i])
        row["clip_frac"] = float(clip_frac[i])
        rows.append(row)
    return rows


def add_thesis_avg(df_rows: pd.DataFrame, ori_n: dict) -> pd.DataFrame:
    """Weighted thesis avg + simple thesis avg (issue #12)."""
    weights = []
    cols = []
    for ori in THESIS_ORIENTATIONS:
        col = f"acc_{ori}"
        if col in df_rows.columns and ori in ori_n and ori_n[ori] > 0:
            weights.append(ori_n[ori])
            cols.append(df_rows[col].to_numpy())
    if not cols:
        df_rows["thesis_avg_weighted"] = np.nan
        df_rows["thesis_avg_simple"]   = np.nan
        return df_rows
    w = np.array(weights, dtype=np.float64)
    M = np.stack(cols, axis=0)  # (n_oris, n_combos)
    df_rows["thesis_avg_weighted"] = (w[:, None] * M).sum(axis=0) / w.sum()
    df_rows["thesis_avg_simple"]   = M.mean(axis=0)
    return df_rows


def evaluate_grid(df, alphas, betas, gammas) -> pd.DataFrame:
    pred, clipped = batch_predict_grid(df, alphas, betas, gammas)
    ori_metrics = _per_orientation_metrics(pred, df)
    clip_frac   = clipped.mean(axis=-1)
    rows = _flatten_to_rows((len(alphas), len(betas), len(gammas)),
                            ori_metrics, alphas, betas, gammas, clip_frac)
    df_out = pd.DataFrame(rows)
    ori_n = {ori: m["n_frames"] for ori, m in ori_metrics.items() if ori != "__overall"}
    return add_thesis_avg(df_out, ori_n)


def evaluate_combos(df, alphas, betas, gammas) -> pd.DataFrame:
    pred, clipped = batch_predict_combos(df, alphas, betas, gammas)
    ori_arr = df["orientation"].to_numpy()
    eye_arr = df["eye_state"].to_numpy()
    metrics = {"__overall": compute_metrics(pred, eye_arr)}
    for ori in np.unique(ori_arr):
        mask = ori_arr == ori
        metrics[ori] = compute_metrics(pred[:, mask], eye_arr[mask])
        metrics[ori]["n_frames"] = int(mask.sum())
    clip_frac = clipped.mean(axis=-1)
    rows = _flatten_combos_to_rows(len(alphas), metrics, alphas, betas, gammas, clip_frac)
    df_out = pd.DataFrame(rows)
    ori_n = {ori: m["n_frames"] for ori, m in metrics.items() if ori != "__overall"}
    return add_thesis_avg(df_out, ori_n)


# ════════════════════════════════════════════════════════════════════════
# 5. Coarse + Iterative Fine Search (issue #6)
# ════════════════════════════════════════════════════════════════════════
def fine_grid_around(row, n=5, span=0.25, eps=1e-5):
    def around(v):
        lo = max(v * (1 - span), eps)
        hi = v * (1 + span)
        return np.linspace(lo, hi, n)
    return around(row["alpha"]), around(row["beta"]), around(row["gamma"])


def search_train(df_train, primary_metric, n_fine_rounds=2, verbose=True):
    sort_key = f"overall_{primary_metric}"

    if verbose:
        print(f"\n>>> COARSE SEARCH "
              f"({len(COARSE_ALPHA)}×{len(COARSE_BETA)}×{len(COARSE_GAMMA)} = "
              f"{len(COARSE_ALPHA)*len(COARSE_BETA)*len(COARSE_GAMMA)} kombinasi)")
    coarse_df = evaluate_grid(df_train, COARSE_ALPHA, COARSE_BETA, COARSE_GAMMA)
    coarse_df["round"] = "coarse"
    coarse_df = coarse_df.sort_values(sort_key, ascending=False).reset_index(drop=True)

    all_rounds = [coarse_df]
    for r in range(n_fine_rounds):
        if verbose:
            print(f"\n>>> FINE SEARCH putaran {r+1}/{n_fine_rounds} — refine top-3")
        round_rows = []
        span = 0.25 if r == 0 else 0.10
        for top_i in range(min(3, len(all_rounds[-1]))):
            row = all_rounds[-1].iloc[top_i]
            a, b, g = fine_grid_around(row, n=5, span=span)
            sub = evaluate_grid(df_train, a, b, g)
            sub["round"] = f"fine_r{r+1}_top{top_i+1}"
            round_rows.append(sub)
        if not round_rows:
            break
        rnd = pd.concat(round_rows, ignore_index=True)
        rnd = rnd.sort_values(sort_key, ascending=False).reset_index(drop=True)
        all_rounds.append(rnd)

    combined = (pd.concat(all_rounds, ignore_index=True)
                  .assign(alpha=lambda d: d["alpha"].round(6),
                          beta =lambda d: d["beta"].round(6),
                          gamma=lambda d: d["gamma"].round(6))
                  .drop_duplicates(subset=["alpha", "beta", "gamma"], keep="first")
                  .sort_values(sort_key, ascending=False)
                  .reset_index(drop=True))
    return combined, coarse_df


def apply_to_test(df_test, train_combined: pd.DataFrame) -> pd.DataFrame:
    """Eval semua kombinasi (yang sudah dipilih dari train) di TEST set.
    Selalu kembalikan DataFrame dengan kolom train_* (dan test_* kalau ada test data)."""
    rename_train = {"train_alpha": "alpha", "train_beta": "beta",
                    "train_gamma": "gamma", "train_round": "round"}
    train_pref = train_combined.add_prefix("train_").rename(columns=rename_train)

    if df_test is None or len(df_test) == 0:
        return train_pref

    a = train_combined["alpha"].to_numpy()
    b = train_combined["beta"].to_numpy()
    g = train_combined["gamma"].to_numpy()
    test_df = evaluate_combos(df_test, a, b, g)
    test_df = test_df.add_prefix("test_").rename(columns={
        "test_alpha": "alpha", "test_beta": "beta", "test_gamma": "gamma"
    })
    return train_pref.merge(test_df, on=["alpha", "beta", "gamma"], how="left")


# ════════════════════════════════════════════════════════════════════════
# 6. Baseline (issue #2)
# ════════════════════════════════════════════════════════════════════════
def baseline_static(df) -> dict:
    """α = β = γ = 0 — threshold tetap T_BASE."""
    a = np.array([0.0]); b = np.array([0.0]); g = np.array([0.0])
    res = evaluate_combos(df, a, b, g).iloc[0]
    return res.to_dict()


# ════════════════════════════════════════════════════════════════════════
# 7. Sensitivity Analysis (consistent metric)
# ════════════════════════════════════════════════════════════════════════
def sensitivity(df_train, df_test, best, varying: str, values, primary_metric):
    """Vary 1 param (di nilai coarse asli), fix 2 di best. Output: train + test."""
    sort_key = f"overall_{primary_metric}"
    params = {"alpha": np.array([best["alpha"]]),
              "beta":  np.array([best["beta"]]),
              "gamma": np.array([best["gamma"]])}
    params[varying] = np.asarray(values, dtype=np.float64)

    train_sens = evaluate_grid(df_train, params["alpha"], params["beta"], params["gamma"])
    test_sens  = apply_to_test(df_test, train_sens)
    return test_sens


# ════════════════════════════════════════════════════════════════════════
# 8. Boundary check (issue #5)
# ════════════════════════════════════════════════════════════════════════
def boundary_check(best, alpha_grid, beta_grid, gamma_grid) -> list:
    msgs = []
    eps = 1e-6
    if np.isclose(best["alpha"], alpha_grid[0], atol=eps):
        msgs.append(f"α best = {best['alpha']:.5f} di TEPI BAWAH grid coarse "
                    f"→ kemungkinan optimum di luar grid. Turunkan COARSE_ALPHA[0].")
    if np.isclose(best["alpha"], alpha_grid[-1], atol=eps):
        msgs.append(f"α best = {best['alpha']:.5f} di TEPI ATAS grid coarse "
                    f"→ naikkan COARSE_ALPHA[-1].")
    if np.isclose(best["beta"], beta_grid[0], atol=eps):
        msgs.append(f"β best = {best['beta']:.5f} di TEPI BAWAH grid coarse.")
    if np.isclose(best["beta"], beta_grid[-1], atol=eps):
        msgs.append(f"β best = {best['beta']:.5f} di TEPI ATAS grid coarse.")
    if np.isclose(best["gamma"], gamma_grid[0], atol=eps):
        msgs.append(f"γ best = {best['gamma']:.5f} di TEPI BAWAH grid coarse.")
    if np.isclose(best["gamma"], gamma_grid[-1], atol=eps):
        msgs.append(f"γ best = {best['gamma']:.5f} di TEPI ATAS grid coarse.")
    return msgs


# ════════════════════════════════════════════════════════════════════════
# 9. Heatmap dari COARSE grid saja (issue #13)
# ════════════════════════════════════════════════════════════════════════
def plot_heatmap(coarse_df, x_col, y_col, value_col, fixed_label, out_path):
    pivot = coarse_df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.4f}" for v in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.4f}" for v in pivot.index])
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(f"{value_col} (coarse grid, {fixed_label})")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if v < 0.85 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# 10. Reporting (issue #4 — sort & report konsisten)
# ════════════════════════════════════════════════════════════════════════
def fmt(v, ndec=4):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{ndec}f}"


def format_thesis_table(df_final, n=4, primary="accuracy"):
    """Tabel 3.5 — top-N berdasar train metric, lapor TEST."""
    lines = []
    lines.append("Tabel 3.5 — Hasil Pengujian Grid Search Parameter α, β, γ")
    lines.append(f"(Diranking berdasar TRAIN {primary}; accuracy dilaporkan pada TEST set)")
    lines.append("-" * 100)
    lines.append(f"{'No':<4}{'α':<10}{'β':<10}{'γ':<10}"
                 f"{'Acc.Front':<12}{'Acc.Left':<12}{'Acc.Right':<12}"
                 f"{'Avg(W)':<10}{'Avg(S)':<10}")
    lines.append("-" * 100)
    for i, (_, r) in enumerate(df_final.head(n).iterrows(), start=1):
        lines.append(
            f"{i:<4}{r['alpha']:<10.5f}{r['beta']:<10.5f}{r['gamma']:<10.5f}"
            f"{fmt(r.get('test_acc_frontal')):<12}"
            f"{fmt(r.get('test_acc_left')):<12}"
            f"{fmt(r.get('test_acc_right')):<12}"
            f"{fmt(r.get('test_thesis_avg_weighted')):<10}"
            f"{fmt(r.get('test_thesis_avg_simple')):<10}"
        )
    lines.append("-" * 100)
    lines.append("Avg(W) = weighted by sample count per orientasi; Avg(S) = simple mean")
    return "\n".join(lines)


def format_extended_metrics(best_row, baseline_train, baseline_test):
    """Confusion-matrix style untuk best & baseline (issue #3)."""
    def block(title, src, prefix):
        lines = [title, "-" * 80,
                 f"{'Metric':<20}{'Train':<15}{'Test':<15}"]
        lines.append("-" * 80)
        for m in ["accuracy", "balanced_accuracy", "f1",
                  "precision", "recall", "specificity"]:
            tr = src.get(f"{prefix}overall_{m}") if prefix else baseline_train.get(m)
            te = src.get(f"test_overall_{m}") if prefix else baseline_test.get(m)
            lines.append(f"{m:<20}{fmt(tr):<15}{fmt(te):<15}")
        return "\n".join(lines)

    out = ["EXTENDED METRICS — BEST kombinasi α,β,γ vs BASELINE static"]
    out.append("=" * 80)
    out.append(block("BEST  (α,β,γ optimal)", best_row, "train_"))
    out.append("")
    # Baseline as single dict — manually
    lines = ["BASELINE (α=β=γ=0, T_dyn ≡ T_BASE)",
             "-" * 80,
             f"{'Metric':<20}{'Train':<15}{'Test':<15}",
             "-" * 80]
    for m in ["accuracy", "balanced_accuracy", "f1",
              "precision", "recall", "specificity"]:
        lines.append(f"{m:<20}"
                     f"{fmt(baseline_train.get(f'overall_{m}')):<15}"
                     f"{fmt(baseline_test.get(f'overall_{m}')):<15}")
    out.append("\n".join(lines))
    return "\n".join(out)


def format_sensitivity_table(df, varying):
    lines = [f"Sensitivity — varying {varying} (fix 2 lain di best)"]
    lines.append("-" * 100)
    lines.append(f"{varying:<10}"
                 f"{'Tr.Acc':<10}{'Te.Acc':<10}"
                 f"{'Te.Front':<11}{'Te.Left':<11}{'Te.Right':<11}"
                 f"{'Te.F1':<10}{'Te.BalAcc':<11}{'Clip%':<8}")
    lines.append("-" * 100)
    for _, r in df.iterrows():
        clip = r.get("test_clip_frac")
        clip_pct = clip * 100.0 if pd.notna(clip) else np.nan
        lines.append(
            f"{r[varying]:<10.5f}"
            f"{fmt(r.get('train_overall_accuracy', r.get('overall_accuracy'))):<10}"
            f"{fmt(r.get('test_overall_accuracy')):<10}"
            f"{fmt(r.get('test_acc_frontal')):<11}"
            f"{fmt(r.get('test_acc_left')):<11}"
            f"{fmt(r.get('test_acc_right')):<11}"
            f"{fmt(r.get('test_overall_f1')):<10}"
            f"{fmt(r.get('test_overall_balanced_accuracy')):<11}"
            f"{fmt(clip_pct, 1):<8}"
        )
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# 11. LOSO-CV runner
# ════════════════════════════════════════════════════════════════════════
def run_loso(df, primary_metric, n_fine_rounds, verbose=True):
    """LOSO-CV. Output ringkasan rata-rata + per-fold."""
    fold_results = []
    for train, test, held_out in loso_folds(df):
        if verbose:
            print(f"\n========== FOLD: held out '{held_out}' "
                  f"(train={len(train)}, test={len(test)}) ==========")
        train_combined, _ = search_train(train, primary_metric, n_fine_rounds, verbose=False)
        final = apply_to_test(test, train_combined)
        sort_key = f"train_overall_{primary_metric}" if f"train_overall_{primary_metric}" in final else f"overall_{primary_metric}"
        final = final.sort_values(sort_key, ascending=False).reset_index(drop=True)
        best = final.iloc[0]
        fold_results.append({
            "held_out_subject": held_out,
            "alpha": best["alpha"],
            "beta":  best["beta"],
            "gamma": best["gamma"],
            "train_accuracy": best.get(f"train_overall_accuracy", best.get("overall_accuracy")),
            "test_accuracy":  best.get("test_overall_accuracy"),
            "test_f1":        best.get("test_overall_f1"),
            "test_balanced_accuracy": best.get("test_overall_balanced_accuracy"),
            "test_acc_frontal": best.get("test_acc_frontal"),
            "test_acc_left":    best.get("test_acc_left"),
            "test_acc_right":   best.get("test_acc_right"),
        })
    return pd.DataFrame(fold_results)


# ════════════════════════════════════════════════════════════════════════
# 12. Main pipeline
# ════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings-dir", default="recordings")
    parser.add_argument("--output-dir",     default="results")
    parser.add_argument("--metric", choices=AVAILABLE_METRICS, default="accuracy",
                        help="Primary metric untuk seleksi best (default: accuracy)")
    parser.add_argument("--split", choices=["holdout", "loso"], default="holdout",
                        help="Strategi train/test (default holdout 80/20)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fine", action="store_true")
    parser.add_argument("--n-fine-rounds", type=int, default=2)
    parser.add_argument("--no-heatmap", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Save metadata (issue #14)
    metadata = {
        "timestamp":  timestamp,
        "args":       vars(args),
        "T_BASE":     T_BASE,
        "T_DYN_FLOOR": T_DYN_FLOOR,
        "coarse_alpha": COARSE_ALPHA.tolist(),
        "coarse_beta":  COARSE_BETA.tolist(),
        "coarse_gamma": COARSE_GAMMA.tolist(),
        "thesis_orientations": THESIS_ORIENTATIONS,
        "versions": {
            "python":     platform.python_version(),
            "numpy":      np.__version__,
            "pandas":     pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }

    # Load data
    print("=" * 80)
    print("GRID SEARCH α, β, γ — T_dynamic Drowsiness Detection")
    print("=" * 80)
    df = load_recordings(Path(args.recordings_dir))

    n_subjects = df["subject"].nunique()
    if args.split == "loso" and n_subjects < 2:
        print("\n[ERROR] LOSO-CV butuh ≥ 2 subjek — fallback ke holdout.")
        args.split = "holdout"

    n_fine = 0 if args.no_fine else args.n_fine_rounds

    # ── LOSO branch ──
    if args.split == "loso":
        print(f"\n[MODE] Leave-One-Subject-Out CV (n_subjects={n_subjects})")
        fold_df = run_loso(df, args.metric, n_fine)
        fold_df.to_csv(out_dir / "loso_per_fold.csv", index=False)

        summary = {
            "alpha_mean": fold_df["alpha"].mean(), "alpha_std": fold_df["alpha"].std(),
            "beta_mean":  fold_df["beta"].mean(),  "beta_std":  fold_df["beta"].std(),
            "gamma_mean": fold_df["gamma"].mean(), "gamma_std": fold_df["gamma"].std(),
            "test_accuracy_mean":          fold_df["test_accuracy"].mean(),
            "test_accuracy_std":           fold_df["test_accuracy"].std(),
            "test_f1_mean":                fold_df["test_f1"].mean(),
            "test_balanced_accuracy_mean": fold_df["test_balanced_accuracy"].mean(),
        }
        print("\n>>> LOSO-CV SUMMARY")
        for k, v in summary.items():
            print(f"  {k:<32} = {fmt(v)}")
        print("\nPer-fold:")
        print(fold_df.to_string(index=False))

        # Final params = average across folds (recommended deploy)
        final_alpha = float(fold_df["alpha"].mean())
        final_beta  = float(fold_df["beta"].mean())
        final_gamma = float(fold_df["gamma"].mean())
        metadata["loso_summary"] = summary
        metadata["final_params"] = dict(alpha=final_alpha, beta=final_beta, gamma=final_gamma)

        baseline_full = baseline_static(df)
        report = []
        report.append("=" * 80)
        report.append(f"LAPORAN LOSO-CV — {timestamp}")
        report.append("=" * 80)
        report.append(f"Subjek (folds): {fold_df['held_out_subject'].tolist()}")
        report.append(f"Primary metric: {args.metric}")
        report.append(f"Folds total   : {len(fold_df)}")
        report.append("")
        report.append("PER-FOLD RESULTS:")
        report.append("-" * 100)
        report.append(f"{'Subject':<14}{'α':<10}{'β':<10}{'γ':<10}"
                      f"{'Te.Acc':<10}{'Te.F1':<10}{'Te.BalAcc':<11}"
                      f"{'Te.Front':<11}{'Te.Left':<11}{'Te.Right':<11}")
        report.append("-" * 100)
        for _, r in fold_df.iterrows():
            report.append(
                f"{r['held_out_subject']:<14}"
                f"{r['alpha']:<10.5f}{r['beta']:<10.5f}{r['gamma']:<10.5f}"
                f"{fmt(r['test_accuracy']):<10}{fmt(r['test_f1']):<10}"
                f"{fmt(r['test_balanced_accuracy']):<11}"
                f"{fmt(r['test_acc_frontal']):<11}"
                f"{fmt(r['test_acc_left']):<11}"
                f"{fmt(r['test_acc_right']):<11}"
            )
        report.append("-" * 100)
        report.append("")
        report.append("AGGREGATE (mean ± std across folds):")
        report.append(f"  α       = {summary['alpha_mean']:.5f} ± {summary['alpha_std']:.5f}")
        report.append(f"  β       = {summary['beta_mean']:.5f} ± {summary['beta_std']:.5f}")
        report.append(f"  γ       = {summary['gamma_mean']:.5f} ± {summary['gamma_std']:.5f}")
        report.append(f"  Te.Acc  = {summary['test_accuracy_mean']:.4f} ± {summary['test_accuracy_std']:.4f}")
        report.append(f"  Te.F1   = {fmt(summary['test_f1_mean'])}")
        report.append(f"  Te.BAcc = {fmt(summary['test_balanced_accuracy_mean'])}")
        report.append("")
        report.append("BASELINE STATIS (α=β=γ=0, T_dyn ≡ T_BASE) di seluruh data:")
        report.append(f"  accuracy          = {fmt(baseline_full.get('overall_accuracy'))}")
        report.append(f"  balanced_accuracy = {fmt(baseline_full.get('overall_balanced_accuracy'))}")
        report.append(f"  f1                = {fmt(baseline_full.get('overall_f1'))}")
        report.append("")
        report.append("=== FINAL PARAMS untuk deploy (rata-rata LOSO folds) ===")
        report.append(f"  ALPHA = {final_alpha:.5f}")
        report.append(f"  BETA  = {final_beta:.5f}")
        report.append(f"  GAMMA = {final_gamma:.5f}")
        report_text = "\n".join(report)
        print("\n" + report_text)
        (out_dir / "thesis_report.txt").write_text(report_text, encoding="utf-8")
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"\nLaporan tersimpan: {out_dir}")
        return

    # ── Holdout branch ──
    print(f"\n[MODE] Stratified holdout — test_size={args.test_size}")
    df_train, df_test = stratified_holdout(df, args.test_size, args.seed)
    print(f"  train: {len(df_train)} frame   test: {len(df_test)} frame")
    metadata["n_train"] = len(df_train)
    metadata["n_test"]  = len(df_test)

    # Search on train
    train_combined, coarse_df = search_train(df_train, args.metric, n_fine, verbose=True)
    print(f"\nTotal kombinasi unik (coarse + fine): {len(train_combined)}")

    # Apply to test
    final = apply_to_test(df_test, train_combined)
    # Sort by TRAIN primary metric (consistent objective — issue #4)
    train_sort_key = f"train_overall_{args.metric}"
    final = final.sort_values(train_sort_key, ascending=False).reset_index(drop=True)

    # Boundary check (issue #5)
    best = final.iloc[0]
    boundary_msgs = boundary_check(best, COARSE_ALPHA, COARSE_BETA, COARSE_GAMMA)

    # Baseline (issue #2)
    baseline_train = baseline_static(df_train)
    baseline_test  = baseline_static(df_test)

    # Save all
    final.to_csv(out_dir / "all_combinations.csv", index=False)
    final.head(10).to_csv(out_dir / "top10.csv", index=False)
    final.head(4).to_csv(out_dir / "tabel_3_5.csv", index=False)

    # Sensitivity (issue #3, #4)
    sens_a = sensitivity(df_train, df_test, best, "alpha", COARSE_ALPHA, args.metric)
    sens_b = sensitivity(df_train, df_test, best, "beta",  COARSE_BETA,  args.metric)
    sens_g = sensitivity(df_train, df_test, best, "gamma", COARSE_GAMMA, args.metric)
    sens_a.to_csv(out_dir / "sensitivity_alpha.csv", index=False)
    sens_b.to_csv(out_dir / "sensitivity_beta.csv",  index=False)
    sens_g.to_csv(out_dir / "sensitivity_gamma.csv", index=False)

    # Heatmaps (issue #13 — coarse only, consistent grid)
    if not args.no_heatmap:
        try:
            heat_ab = coarse_df[np.isclose(coarse_df["gamma"], best["gamma"], atol=1e-6)]
            if len(heat_ab) >= 4:
                plot_heatmap(heat_ab, "alpha", "beta", f"overall_{args.metric}",
                             f"γ={best['gamma']:.4f}", out_dir / "heatmap_alpha_beta.png")
            heat_ag = coarse_df[np.isclose(coarse_df["beta"], best["beta"], atol=1e-6)]
            if len(heat_ag) >= 4:
                plot_heatmap(heat_ag, "alpha", "gamma", f"overall_{args.metric}",
                             f"β={best['beta']:.4f}", out_dir / "heatmap_alpha_gamma.png")
            heat_bg = coarse_df[np.isclose(coarse_df["alpha"], best["alpha"], atol=1e-6)]
            if len(heat_bg) >= 4:
                plot_heatmap(heat_bg, "beta", "gamma", f"overall_{args.metric}",
                             f"α={best['alpha']:.4f}", out_dir / "heatmap_beta_gamma.png")
            # Note: kalau best tidak ada di coarse grid (hasil fine search),
            # heatmap dibuat di nilai coarse TERDEKAT (issue #13)
        except Exception as e:
            print(f"  [WARN] heatmap gagal: {e}")

    # Floor clipping summary (issue #8)
    clip_train = float(best.get("train_clip_frac", 0)) * 100
    clip_test  = float(best.get("test_clip_frac",  0)) * 100

    # ── Build report ──
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"LAPORAN GRID SEARCH α, β, γ — {timestamp}")
    report_lines.append("=" * 80)
    report_lines.append(f"Subjek         : {n_subjects} ({sorted(df['subject'].unique())})")
    report_lines.append(f"Frames train   : {len(df_train)}   test: {len(df_test)}")
    report_lines.append(f"Primary metric : {args.metric}  (digunakan untuk SORT + SELECT best)")
    report_lines.append(f"Split          : stratified holdout (test_size={args.test_size}, seed={args.seed})")
    report_lines.append(f"T_BASE         : {T_BASE} (fixed sesuai metodologi skripsi)")
    report_lines.append(f"T_DYN_FLOOR    : {T_DYN_FLOOR}")
    report_lines.append(f"Total kombinasi diuji: {len(final)}")
    report_lines.append("")
    report_lines.append(f"BEST (rank #1 by train_{args.metric}):")
    report_lines.append(f"  α = {best['alpha']:.5f}")
    report_lines.append(f"  β = {best['beta']:.5f}")
    report_lines.append(f"  γ = {best['gamma']:.5f}")
    report_lines.append(f"  Floor clipping (T_dyn = {T_DYN_FLOOR}): "
                       f"train {clip_train:.1f}% | test {clip_test:.1f}%")

    if boundary_msgs:
        report_lines.append("")
        report_lines.append("⚠ BOUNDARY WARNINGS:")
        for m in boundary_msgs:
            report_lines.append(f"  - {m}")

    report_lines.append("")
    report_lines.append(format_extended_metrics(best, baseline_train, baseline_test))
    report_lines.append("")
    report_lines.append(format_thesis_table(final, n=4, primary=args.metric))
    report_lines.append("")

    # Sensitivity
    report_lines.append(format_sensitivity_table(sens_a, "alpha"))
    report_lines.append("")
    report_lines.append(format_sensitivity_table(sens_b, "beta"))
    report_lines.append("")
    report_lines.append(format_sensitivity_table(sens_g, "gamma"))
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("UPDATE KE KOTLIN")
    report_lines.append("=" * 80)
    report_lines.append("MainActivity.kt:88-91 dan TestingActivity.kt:180-183:")
    report_lines.append("")
    report_lines.append(f"    private val T_BASE  = {T_BASE}")
    report_lines.append(f"    private val ALPHA   = {best['alpha']:.5f}")
    report_lines.append(f"    private val BETA    = {best['beta']:.5f}")
    report_lines.append(f"    private val GAMMA   = {best['gamma']:.5f}")
    report_lines.append("")
    report_lines.append("Lalu jalankan TestingActivity di HP untuk validasi (Tabel 3.7/3.8).")

    report = "\n".join(report_lines)
    print("\n" + report)
    (out_dir / "thesis_report.txt").write_text(report, encoding="utf-8")

    # Save metadata
    metadata["best_params"] = {"alpha": float(best["alpha"]),
                                "beta":  float(best["beta"]),
                                "gamma": float(best["gamma"])}
    metadata["boundary_warnings"] = boundary_msgs
    metadata["baseline_train"] = {k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                                  for k, v in baseline_train.items() if not isinstance(v, (list, dict))}
    metadata["baseline_test"]  = {k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                                  for k, v in baseline_test.items() if not isinstance(v, (list, dict))}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(f"\nSemua output tersimpan di: {out_dir}")


if __name__ == "__main__":
    main()
