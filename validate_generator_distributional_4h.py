"""
validate_generator_distributional_4h.py — Part 3 (re-validation): the 4h counterpart of
validate_generator_distributional.py, same methodology, same acceptance criterion (does a
classifier trained on summary stats distinguish REAL vs SYNTHETIC windows?), run against
the REBUILT 4h generator (generator_4h.py, config/calibration_4h.json) instead of the
original hourly one. Three groups, same-length non-overlapping windows, all 5 coins:
  REAL-TRAIN:    real 4h windows from the calibration TRAIN period
  REAL-HELDOUT:  real 4h windows from the locked held-out period (data/heldout_4h/)
  SYNTHETIC:     fresh generator_4h.py "mix"-regime windows, same length

WINDOW LENGTH: 84 four-hour bars (14 days) -- the SAME real-world calendar duration as the
hourly validation's 336h windows, and the SAME window length calibrate_jumps_4h.py used as
its kurtosis-matching target, so this is an apples-to-apples re-run of the exact test that
found the original blind spots, at the new resolution.

Part 1/2 methodology, feature set, and classifier are otherwise UNCHANGED from the hourly
version (same stat functions imported from calibrate.py where resolution-independent,
same RandomForestClassifier/CV setup) -- differences in the result are due to the
generator, not the test.

Run: python3 -u validate_generator_distributional_4h.py
"""
import json
import time

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from calibrate import excess_kurtosis, acf_abs_lag1, max_drawdown
from calibrate_4h import COINS, load_data, compute_common_window, build_returns, ann_vol_from_returns
from generator_4h import generate_raw, load_calibration

WINDOW_BARS = 84  # 14 days at 4h resolution -- matches calibrate_jumps_4h.py's target window
ACF_LAGS = [1, 2, 3, 6, 12, 24]
QUANTILE_LEVELS = [0.01, 0.05, 0.50, 0.95, 0.99]

N_SYNTHETIC_WINDOWS = 120
SYNTHETIC_SEED_BASE = 950000

RESULTS_PATH = "/tmp/generator_validation_stats_results_4h.json"


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------------
# Real data loading
# ----------------------------------------------------------------------------------
def get_train_returns():
    data = load_data()
    train_start, train_end, train_index, *_ = compute_common_window(data)
    close, returns, dropped = build_returns(data, train_index)
    return returns


def get_heldout_returns():
    HELD_OUT_DIR = "data/heldout_4h"
    raw = {}
    for coin in COINS:
        df = pd.read_csv(f"{HELD_OUT_DIR}/{coin}USDT_4h_heldout.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        raw[coin] = df
    full_index = pd.date_range(min(df.index.min() for df in raw.values()),
                                max(df.index.max() for df in raw.values()), freq="4h", tz="UTC")
    close, returns, dropped = build_returns(raw, full_index)
    return returns


# ----------------------------------------------------------------------------------
# Per-window, per-coin stat computation
# ----------------------------------------------------------------------------------
def _acf_signed_lag1(r):
    return float(r.autocorr(lag=1))


def _acf_abs_lag(r, lag):
    return float(r.abs().autocorr(lag=lag))


def _longest_run(sign_array):
    best_up = best_down = cur_up = cur_down = 0
    for s in sign_array:
        if s > 0:
            cur_up += 1
            cur_down = 0
        elif s < 0:
            cur_down += 1
            cur_up = 0
        else:
            cur_up = cur_down = 0
        best_up = max(best_up, cur_up)
        best_down = max(best_down, cur_down)
    return best_up, best_down


def window_coin_stats(r):
    r = r.dropna()
    out = {
        "ann_vol": ann_vol_from_returns(r),
        "excess_kurtosis": excess_kurtosis(r),
        "acf_abs_lag1": acf_abs_lag1(r),
        "return_acf_lag1": _acf_signed_lag1(r),
        "skew": float(scipy_stats.skew(r)),
    }
    for lag in ACF_LAGS:
        out[f"acf_abs_lag{lag}"] = _acf_abs_lag(r, lag)
    q = r.quantile(QUANTILE_LEVELS)
    for lvl in QUANTILE_LEVELS:
        out[f"q{int(lvl*100)}"] = float(q.loc[lvl])

    price = pd.Series(np.exp(np.concatenate([[0.0], np.cumsum(r.to_numpy())])))
    out["max_drawdown"] = max_drawdown(price)
    up_run, down_run = _longest_run(r.to_numpy())
    out["longest_up_run"] = up_run
    out["longest_down_run"] = down_run
    return out


def chop_windows(returns_df, window_bars):
    n = len(returns_df)
    n_windows = n // window_bars
    return [returns_df.iloc[i * window_bars:(i + 1) * window_bars] for i in range(n_windows)]


def compute_group_stats(windows, label):
    per_window_coin = []
    per_window_features = []
    for wi, w in enumerate(windows):
        coin_stats = {}
        for coin in COINS:
            s = window_coin_stats(w[coin])
            coin_stats[coin] = s
            per_window_coin.append({"window_idx": wi, "coin": coin, **s})

        complete = w.dropna(how="any")
        corr = complete.corr() if len(complete) > 2 else pd.DataFrame(
            np.eye(len(COINS)), index=COINS, columns=COINS)

        feat = {}
        for coin in COINS:
            for k, v in coin_stats[coin].items():
                feat[f"{coin}_{k}"] = v
        for i, c1 in enumerate(COINS):
            for c2 in COINS[i + 1:]:
                feat[f"corr_{c1}_{c2}"] = float(corr.loc[c1, c2])
        per_window_features.append(feat)
    log(f"  {label}: {len(windows)} windows x {len(COINS)} coins = {len(per_window_coin)} window-coin samples")
    return per_window_coin, per_window_features


# ----------------------------------------------------------------------------------
# Synthetic generation
# ----------------------------------------------------------------------------------
def generate_synthetic_windows(n_windows, window_bars, seed_base):
    calib = load_calibration()
    windows = []
    for i in range(n_windows):
        seed = seed_base + i
        dfs, _, _ = generate_raw(seed=seed, n_periods=window_bars, regime_mode="mix", calib=calib)
        close = pd.DataFrame({coin: df["close"].to_numpy() for coin, df in dfs.items()})
        returns = np.log(close).diff()
        windows.append(returns)
    return windows


# ----------------------------------------------------------------------------------
# Part 1: stylized-facts comparison table
# ----------------------------------------------------------------------------------
PART1_STATS = ["ann_vol", "excess_kurtosis", "acf_abs_lag1", "return_acf_lag1", "skew",
               "q1", "q5", "q50", "q95", "q99"]


def part1_table(groups_wc):
    log(f"\n{'='*100}\nPART 1: STYLIZED-FACTS COMPARISON, 4h (mean +/- std across window-coin samples)\n{'='*100}")
    header = f"{'statistic':<20}" + "".join(f"{g+'_mean':>16}{g+'_std':>12}" for g in groups_wc)
    log(header + f"{'FLAG':>8}")
    summary = {}
    for stat in PART1_STATS:
        row = f"{stat:<20}"
        means = {}
        for g, rows in groups_wc.items():
            vals = np.array([r[stat] for r in rows])
            means[g] = float(vals.mean())
            row += f"{vals.mean():>16.4f}{vals.std():>12.4f}"
        lo, hi = sorted([means["REAL-TRAIN"], means["REAL-HELDOUT"]])
        flag = "OUTSIDE" if not (lo <= means["SYNTHETIC"] <= hi) else ""
        log(row + f"{flag:>8}")
        summary[stat] = {"means": means, "real_range": [lo, hi], "flagged": bool(flag)}

    log(f"\n-- vol-clustering DECAY (mean acf_abs at each lag, per group) --")
    header2 = f"{'lag':<8}" + "".join(f"{g:>16}" for g in groups_wc)
    log(header2)
    decay = {}
    for lag in ACF_LAGS:
        row = f"{lag:<8}"
        decay[lag] = {}
        for g, rows in groups_wc.items():
            vals = np.array([r[f"acf_abs_lag{lag}"] for r in rows])
            decay[lag][g] = float(vals.mean())
            row += f"{vals.mean():>16.4f}"
        log(row)

    return summary, decay


def part1_correlation(groups_windows):
    log(f"\n-- MEAN CROSS-COIN CORRELATION MATRIX per group --")
    matrices = {}
    for g, windows in groups_windows.items():
        mats = []
        for w in windows:
            complete = w.dropna(how="any")
            if len(complete) > 2:
                mats.append(complete.corr().to_numpy())
        mean_mat = np.mean(mats, axis=0)
        matrices[g] = mean_mat
        log(f"\n{g}:")
        log("       " + "".join(f"{c:>8}" for c in COINS))
        for i, c1 in enumerate(COINS):
            log(f"{c1:<7}" + "".join(f"{mean_mat[i,j]:>8.3f}" for j in range(len(COINS))))
    return {g: m.tolist() for g, m in matrices.items()}


# ----------------------------------------------------------------------------------
# Part 2: classifier two-sample test
# ----------------------------------------------------------------------------------
def run_classifier(real_features, synth_features, label):
    log(f"\n{'='*100}\nPART 2: CLASSIFIER TWO-SAMPLE TEST, 4h -- {label}\n{'='*100}")
    log(f"n_real={len(real_features)}  n_synthetic={len(synth_features)}")

    feature_names = sorted(real_features[0].keys())
    X_real = np.array([[f[k] for k in feature_names] for f in real_features])
    X_synth = np.array([[f[k] for k in feature_names] for f in synth_features])
    X = np.vstack([X_real, X_synth])
    y = np.array([1] * len(X_real) + [0] * len(X_synth))

    n_splits = min(5, min(np.bincount(y)))
    if n_splits < 2:
        log(f"[SKIP] too few samples in the minority class ({min(np.bincount(y))}) for CV.")
        return None

    clf = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
    auc = roc_auc_score(y, proba)
    log(f"n_splits={n_splits}-fold stratified CV, ROC-AUC = {auc:.4f}")
    log("(0.5 = indistinguishable/generator excellent; 1.0 = trivially separable)")

    clf.fit(X, y)
    importances = sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1])
    log("\nTop 15 feature importances (full-data fit, for interpretation only -- not the CV score):")
    for name, imp in importances[:15]:
        log(f"  {name:<28}{imp:.4f}")

    return {"auc": float(auc), "n_real": len(real_features), "n_synth": len(synth_features),
            "n_splits": n_splits, "top_importances": [(n, float(i)) for n, i in importances[:20]]}


# ----------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    log("Loading real 4h TRAIN returns...")
    train_returns = get_train_returns()
    log(f"  TRAIN: {len(train_returns)} 4h bars")

    log("Loading real 4h HELD-OUT returns...")
    heldout_returns = get_heldout_returns()
    log(f"  HELD-OUT: {len(heldout_returns)} 4h bars")

    train_windows = chop_windows(train_returns, WINDOW_BARS)
    heldout_windows = chop_windows(heldout_returns, WINDOW_BARS)
    log(f"  TRAIN windows: {len(train_windows)}   HELD-OUT windows: {len(heldout_windows)}")

    log(f"\nGenerating {N_SYNTHETIC_WINDOWS} fresh synthetic 4h mix windows ({WINDOW_BARS} bars each)...")
    t1 = time.perf_counter()
    synth_windows = generate_synthetic_windows(N_SYNTHETIC_WINDOWS, WINDOW_BARS, SYNTHETIC_SEED_BASE)
    log(f"  done in {time.perf_counter()-t1:.0f}s")

    log("\nComputing per-window-coin stats for all three groups...")
    train_wc, train_feat = compute_group_stats(train_windows, "REAL-TRAIN")
    heldout_wc, heldout_feat = compute_group_stats(heldout_windows, "REAL-HELDOUT")
    synth_wc, synth_feat = compute_group_stats(synth_windows, "SYNTHETIC")

    groups_wc = {"REAL-TRAIN": train_wc, "REAL-HELDOUT": heldout_wc, "SYNTHETIC": synth_wc}
    groups_windows = {"REAL-TRAIN": train_windows, "REAL-HELDOUT": heldout_windows, "SYNTHETIC": synth_windows}

    part1_summary, part1_decay = part1_table(groups_wc)
    part1_corr = part1_correlation(groups_windows)

    result_a = run_classifier(train_feat, synth_feat, "(a) REAL-TRAIN vs SYNTHETIC")
    result_b = run_classifier(heldout_feat, synth_feat, "(b) REAL-HELDOUT vs SYNTHETIC")

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "window_bars": WINDOW_BARS,
            "n_windows": {"train": len(train_windows), "heldout": len(heldout_windows),
                          "synthetic": len(synth_windows)},
            "part1_summary": part1_summary, "part1_decay": part1_decay, "part1_corr": part1_corr,
            "classifier_a": result_a, "classifier_b": result_b,
        }, f, indent=2)
    log(f"\nWrote full results to {RESULTS_PATH}")
    log(f"\nTotal wall-clock: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
