"""
calibrate_4h.py — measure stylized facts of real 4-HOUR crypto OHLCV data and write
config/calibration_4h.json. Structurally identical pipeline to calibrate.py (same order
of steps, same measurement functions where they're resolution-independent), adapted for
4h bars:
  - DATA_DIR/FILE_TEMPLATE point at data/raw_4h/*_4h_full.csv.
  - PERIODS_PER_YEAR = 6*365 = 2190 (4h bars/day * days/year), replacing calibrate.py's
    ANNUALIZATION_HOURS=8760 everywhere annualization happens.
  - REGIME_MIN_PERIODS = 6 (>=1 day of real data = 6 four-hour bars), replacing
    calibrate.py's REGIME_MIN_PERIODS=24 (1 day = 24 hourly bars) -- same real-world
    "at least 1 day" semantic, expressed in the new bar unit.
  - TRAILING_WINDOW_HOURS/BULL_THRESHOLD/BEAR_THRESHOLD are UNCHANGED: the regime
    trailing window is a pandas TIME-based rolling window ('168h'), which is inherently
    resolution-agnostic (operates on the datetime index, not bar count) -- no change needed.
  - excess_kurtosis/acf_abs_lag1/max_drawdown/fit_student_t are imported VERBATIM from
    calibrate.py (pure statistics, no frequency dependency -- "lag 1" naturally means
    "1 four-hour bar" here, which is exactly what we want). Only ann_vol_from_returns
    needs a resolution-aware reimplementation (it embeds the annualization constant).
  - No hardcoded SANITY_REF (those hourly reference numbers don't apply at 4h resolution
    and re-deriving independent 4h reference values is out of scope) -- instead, a LIGHT
    cross-resolution sanity check: 4h ann_vol per coin should be roughly the same order of
    magnitude as the existing hourly calibration.json's ann_vol (a real, principled check
    we CAN make, vs asserting against numbers nobody has independently verified for 4h).

Run: python3 calibrate_4h.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning

from calibrate import excess_kurtosis, acf_abs_lag1, max_drawdown, fit_student_t

np.random.seed(42)

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
DATA_DIR = "data/raw_4h"
COINS = ["BTC", "ETH", "BNB", "MATIC", "SOL"]
FILE_TEMPLATE = "{coin}USDT_4h_full.csv"

HELD_OUT_MONTHS = 6  # calendar-based, unchanged by resolution
HELD_OUT_DIR = "data/heldout_4h"

BARS_PER_DAY = 6
PERIODS_PER_YEAR = BARS_PER_DAY * 365  # 2190 -- replaces calibrate.py's ANNUALIZATION_HOURS=8760

TRAILING_WINDOW_HOURS = 168  # 7 days -- time-based rolling window, resolution-agnostic
BULL_THRESHOLD = 0.10
BEAR_THRESHOLD = 0.10
REGIME_MIN_PERIODS = 6  # >=1 day of real data = 6 four-hour bars (was 24 hourly bars)

IDIO_VOL_MATCH_TOL = 0.03
IDIO_STABLE_PERSISTENCE = 0.99


def ann_vol_from_returns(r, periods_per_year=PERIODS_PER_YEAR):
    return float(r.std() * np.sqrt(periods_per_year))


# --------------------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------------------
def load_data():
    data = {}
    for coin in COINS:
        path = f"{DATA_DIR}/{FILE_TEMPLATE.format(coin=coin)}"
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        data[coin] = df
    return data


# --------------------------------------------------------------------------------------
# 2. Common window
# --------------------------------------------------------------------------------------
def compute_common_window(data):
    starts = [df.index.min() for df in data.values()]
    ends = [df.index.max() for df in data.values()]
    full_common_start = max(starts)
    full_common_end = min(ends)

    held_out_start = full_common_end - pd.DateOffset(months=HELD_OUT_MONTHS)

    WINDOW_START, WINDOW_END = full_common_start, held_out_start - pd.Timedelta(hours=4)

    full_index = pd.date_range(WINDOW_START, WINDOW_END, freq="4h", tz="UTC")
    return WINDOW_START, WINDOW_END, full_index, full_common_start, full_common_end, held_out_start


# --------------------------------------------------------------------------------------
# 3. Reindex to full 4h grid + gap-safe log returns
# --------------------------------------------------------------------------------------
def build_returns(data, full_index):
    close = pd.DataFrame({coin: df["close"].reindex(full_index) for coin, df in data.items()})
    log_price = np.log(close)
    returns = log_price.diff()

    dropped_per_coin = {}
    for coin in COINS:
        possible = len(full_index) - 1
        valid = returns[coin].notna().sum()
        dropped_per_coin[coin] = int(possible - valid)

    return close, returns, dropped_per_coin


# --------------------------------------------------------------------------------------
# 4. Per-coin stylized facts
# --------------------------------------------------------------------------------------
def fit_garch(returns, label):
    scale = 100.0
    r = returns.dropna()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            am = arch_model(r * scale, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
            res = am.fit(disp="off")
        if res.convergence_flag != 0:
            raise RuntimeError(f"non-zero convergence flag: {res.convergence_flag}")
        omega = float(res.params["omega"]) / (scale ** 2)
        alpha = float(res.params["alpha[1]"])
        beta = float(res.params["beta[1]"])
        return {"omega": omega, "alpha": alpha, "beta": beta, "converged": True}
    except Exception as e:
        print(f"  [WARN] GARCH(1,1) failed to converge for {label} ({e}); "
              f"falling back to autocorrelation diagnostics.")
        abs_r = r.abs()
        sq_r = r ** 2
        return {
            "omega": None, "alpha": None, "beta": None, "converged": False,
            "fallback_acf_abs_ret_lag1": float(abs_r.autocorr(lag=1)),
            "fallback_acf_sq_ret_lag1": float(sq_r.autocorr(lag=1)),
        }


def compute_coin_stats(returns):
    stats_out = {}
    for coin in COINS:
        r = returns[coin].dropna()
        mean_period = float(r.mean())
        std_period = float(r.std())
        ann_vol = ann_vol_from_returns(r)
        excess_kurt = excess_kurtosis(r)
        standardized = (r - mean_period) / std_period
        dof = fit_student_t(standardized)
        garch = fit_garch(r, coin)
        stats_out[coin] = {
            "mean_period_logret": mean_period,
            "ann_vol": ann_vol,
            "excess_kurtosis": excess_kurt,
            "student_t_dof": dof,
            "garch": garch,
            "_std_period": std_period,
        }
    return stats_out


# --------------------------------------------------------------------------------------
# 5. Cross-sectional structure
# --------------------------------------------------------------------------------------
def compute_cross_sectional(returns, coin_stats):
    returns_common = returns.dropna(how="any")
    correlation_matrix = returns_common.corr()

    means = returns_common.mean()
    stds = returns_common.std()
    Z = (returns_common - means) / stds

    cov = np.cov(Z.values, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    v1 = eigvecs[:, 0]
    if v1.sum() < 0:
        v1 = -v1
    explained_var_ratio = float(eigvals[0] / eigvals.sum())

    pc1_scores = Z.values @ v1
    avg_period_vol = float(np.mean([coin_stats[c]["_std_period"] for c in COINS]))
    market_factor_logret = pd.Series(pc1_scores * avg_period_vol, index=returns_common.index)

    betas = {}
    for coin in COINS:
        y = returns_common[coin].values
        x = market_factor_logret.values
        beta, alpha = np.polyfit(x, y, 1)
        resid = y - (alpha + beta * x)
        idio_vol = float(np.std(resid, ddof=1) * np.sqrt(PERIODS_PER_YEAR))
        betas[coin] = {"beta_to_market": float(beta), "idio_vol": idio_vol, "_alpha": float(alpha)}

    return correlation_matrix, market_factor_logret, explained_var_ratio, betas, returns_common


def compute_market_factor_stats(market_factor_logret, explained_var_ratio):
    r = market_factor_logret.dropna()
    mean_period = float(r.mean())
    std_period = float(r.std())
    ann_vol = ann_vol_from_returns(r)
    excess_kurt = excess_kurtosis(r)
    standardized = (r - mean_period) / std_period
    dof = fit_student_t(standardized)
    garch = fit_garch(r, "market_factor")
    return {
        "mean_period_logret": mean_period, "ann_vol": ann_vol, "excess_kurtosis": excess_kurt,
        "explained_var_ratio": explained_var_ratio, "garch": garch, "student_t_dof": dof,
    }


# --------------------------------------------------------------------------------------
# 5b. Idiosyncratic-residual GARCH
# --------------------------------------------------------------------------------------
def compute_idio_garch(returns_common, market_factor_logret, betas):
    idio_out = {}
    print("\nIdio-residual sanity report (annualized vol should match existing idio_vol):")
    for coin in COINS:
        beta = betas[coin]["beta_to_market"]
        alpha_int = betas[coin]["_alpha"]
        resid = (returns_common[coin] - (alpha_int + beta * market_factor_logret)).dropna()

        resid_ann_vol = ann_vol_from_returns(resid)
        resid_kurt = excess_kurtosis(resid)
        resid_acf = acf_abs_lag1(resid)
        existing_idio_vol = betas[coin]["idio_vol"]
        rel_diff = abs(resid_ann_vol - existing_idio_vol) / existing_idio_vol
        vol_match_ok = rel_diff <= IDIO_VOL_MATCH_TOL
        print(f"  {coin}: resid_ann_vol={resid_ann_vol:.4f}  existing_idio_vol={existing_idio_vol:.4f}  "
              f"diff={rel_diff:.2%} {'OK' if vol_match_ok else '[WARN] mismatch'}   "
              f"resid_kurtosis={resid_kurt:.2f}  resid_acf_abs={resid_acf:.4f}")

        standardized = (resid - resid.mean()) / resid.std()
        dof = fit_student_t(standardized)
        garch = fit_garch(resid, f"{coin}_idio")

        if garch["converged"]:
            a_, b_, omega = garch["alpha"], garch["beta"], garch["omega"]
            garch_ok = True
        else:
            a_, b_, omega = 0.0, 0.0, None
            garch_ok = False

        idio_out[coin] = {
            "idio_garch": {"omega": omega, "alpha": a_, "beta": b_},
            "idio_student_t_dof": dof, "idio_persistence": a_ + b_, "idio_garch_ok": garch_ok,
            "_resid_ann_vol": resid_ann_vol, "_resid_kurtosis": resid_kurt,
            "_resid_acf_abs": resid_acf, "_vol_match_ok": vol_match_ok,
        }
    return idio_out


# --------------------------------------------------------------------------------------
# 6. Regimes
# --------------------------------------------------------------------------------------
def compute_regimes(market_factor_logret, full_index):
    factor_full = market_factor_logret.reindex(full_index)
    trailing_return = factor_full.rolling(
        f"{TRAILING_WINDOW_HOURS}h", min_periods=REGIME_MIN_PERIODS
    ).sum()

    labels_order = ["bull", "bear", "sideways"]

    def label_row(x):
        if pd.isna(x):
            return None
        if x > BULL_THRESHOLD:
            return "bull"
        if x < -BEAR_THRESHOLD:
            return "bear"
        return "sideways"

    labels = trailing_return.apply(label_row)
    labeled = labels.dropna()
    n_labeled = len(labeled)
    share_of_time = {r: float((labeled == r).sum() / n_labeled) for r in labels_order}

    params = {}
    for regime in labels_order:
        mask = labels == regime
        r = factor_full[mask].dropna()
        params[regime] = {
            "mean_period_logret": float(r.mean()), "ann_vol": float(r.std() * np.sqrt(PERIODS_PER_YEAR)),
            "share_of_time": share_of_time[regime],
        }

    idx_pos = {r: i for i, r in enumerate(labels_order)}
    counts = np.zeros((3, 3))
    lab_values = labels.values
    for i in range(len(lab_values) - 1):
        a, b = lab_values[i], lab_values[i + 1]
        if a not in idx_pos or b not in idx_pos:
            continue
        counts[idx_pos[a], idx_pos[b]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix = counts / row_sums

    mean_duration_periods = {}
    for r in labels_order:
        p_self = transition_matrix[idx_pos[r], idx_pos[r]]
        mean_duration_periods[r] = float(1.0 / (1.0 - p_self)) if p_self < 1.0 else float("inf")

    return {
        "labels": labels_order, "params": params, "transition_matrix": transition_matrix.tolist(),
        "mean_duration_periods": mean_duration_periods,
        "config": {"trailing_window_hours": TRAILING_WINDOW_HOURS, "bull_threshold": BULL_THRESHOLD,
                   "bear_threshold": BEAR_THRESHOLD},
    }, share_of_time


# --------------------------------------------------------------------------------------
# 7. Light cross-resolution sanity check
# --------------------------------------------------------------------------------------
def cross_resolution_sanity_check(coin_stats, correlation_matrix):
    print("\n" + "=" * 70)
    print("CROSS-RESOLUTION SANITY CHECK (4h vs existing hourly calibration.json)")
    print("=" * 70)
    print("No independently-verified 4h reference numbers exist -- this only checks that")
    print("4h stats are the same rough ORDER OF MAGNITUDE as the hourly ones (loose, +/-50%),")
    print("not that they're expected to match exactly (they shouldn't -- see report).\n")
    try:
        with open("config/calibration.json") as f:
            hourly = json.load(f)
    except FileNotFoundError:
        print("  [SKIP] config/calibration.json (hourly) not found.")
        return
    for coin in COINS:
        h_vol = hourly["coins"][coin]["ann_vol"]
        n_vol = coin_stats[coin]["ann_vol"]
        rel = abs(n_vol - h_vol) / h_vol
        status = "OK" if rel <= 0.5 else "[WARN] large gap"
        print(f"  ann_vol[{coin}]: 4h={n_vol:.2%}  hourly={h_vol:.2%}  rel_diff={rel:.1%}  {status}")
    h_corr = hourly["correlation_matrix"]["BTC"]["ETH"]
    n_corr = correlation_matrix.loc["BTC", "ETH"]
    print(f"  corr(BTC,ETH): 4h={n_corr:.3f}  hourly={h_corr:.3f}")


# --------------------------------------------------------------------------------------
# 8. Write outputs
# --------------------------------------------------------------------------------------
def write_config(window_start, window_end, n_periods, n_returns_dropped_total,
                  coin_stats, market_factor_stats, correlation_matrix,
                  betas, drawdowns, regimes, idio_out, split_info):
    config = {
        "meta": {
            "resolution": "4h",
            "window_start": window_start.isoformat(), "window_end": window_end.isoformat(),
            "n_periods": n_periods, "n_returns_dropped_to_gaps": n_returns_dropped_total,
            "coins": COINS, "periods_per_year": PERIODS_PER_YEAR, "bars_per_day": BARS_PER_DAY,
            "split": {
                "train_start": split_info["train_start"].isoformat(),
                "train_end": split_info["train_end"].isoformat(),
                "heldout_start": split_info["heldout_start"].isoformat(),
                "heldout_end": split_info["heldout_end"].isoformat(),
                "held_out_months": HELD_OUT_MONTHS,
            },
        },
        "coins": {
            coin: {
                "mean_period_logret": coin_stats[coin]["mean_period_logret"],
                "ann_vol": coin_stats[coin]["ann_vol"],
                "excess_kurtosis": coin_stats[coin]["excess_kurtosis"],
                "max_drawdown": drawdowns[coin],
                "student_t_dof": coin_stats[coin]["student_t_dof"],
                "garch": coin_stats[coin]["garch"],
                "beta_to_market": betas[coin]["beta_to_market"],
                "idio_vol": betas[coin]["idio_vol"],
                "idio_garch": idio_out[coin]["idio_garch"],
                "idio_student_t_dof": idio_out[coin]["idio_student_t_dof"],
                "idio_persistence": idio_out[coin]["idio_persistence"],
                "idio_garch_ok": idio_out[coin]["idio_garch_ok"],
            }
            for coin in COINS
        },
        "market_factor": market_factor_stats,
        "correlation_matrix": {
            c1: {c2: float(correlation_matrix.loc[c1, c2]) for c2 in COINS} for c1 in COINS
        },
        "regimes": regimes,
    }
    with open("config/calibration_4h.json", "w") as f:
        json.dump(config, f, indent=2)
    return config


def write_report(window_start, window_end, n_periods, n_returns_dropped_total, dropped_per_coin,
                  coin_stats, market_factor_stats, correlation_matrix, betas, drawdowns,
                  regimes, share_of_time, idio_out):
    lines = []
    lines.append("CALIBRATION REPORT (4-HOUR bars, TRAIN window -- held-out real data excluded)")
    lines.append("=" * 78)
    lines.append(f"TRAIN window: {window_start} -> {window_end}")
    lines.append(f"n_periods (4h bars) in window: {n_periods}")
    lines.append(f"periods_per_year: {PERIODS_PER_YEAR}")
    lines.append(f"Total returns dropped due to gaps: {n_returns_dropped_total}")
    lines.append("Dropped per coin: " + ", ".join(f"{c}={dropped_per_coin[c]}" for c in COINS))
    lines.append("")

    lines.append("PER-COIN STYLIZED FACTS (4h)")
    lines.append("-" * 78)
    header = f"{'coin':<7}{'mean_4h':>12}{'ann_vol':>10}{'kurt':>9}{'max_dd':>9}" \
             f"{'t_dof':>8}{'beta':>8}{'idio_vol':>10}{'garch_ok':>10}"
    lines.append(header)
    for coin in COINS:
        s = coin_stats[coin]
        lines.append(
            f"{coin:<7}{s['mean_period_logret']:>12.6f}{s['ann_vol']:>10.2%}"
            f"{s['excess_kurtosis']:>9.2f}{drawdowns[coin]:>9.2%}{s['student_t_dof']:>8.2f}"
            f"{betas[coin]['beta_to_market']:>8.2f}{betas[coin]['idio_vol']:>10.2%}"
            f"{str(s['garch']['converged']):>10}"
        )
    lines.append("")

    lines.append("GARCH(1,1) PARAMETERS (4h)")
    lines.append("-" * 78)
    for coin in COINS:
        g = coin_stats[coin]["garch"]
        if g["converged"]:
            lines.append(f"{coin:<7} omega={g['omega']:.3e}  alpha={g['alpha']:.4f}  beta={g['beta']:.4f}")
        else:
            lines.append(f"{coin:<7} FAILED TO CONVERGE")
    lines.append("")

    lines.append("CORRELATION MATRIX (4h log returns)")
    lines.append("-" * 78)
    lines.append(f"{'':<7}" + "".join(f"{c:>8}" for c in COINS))
    for c1 in COINS:
        lines.append(f"{c1:<7}" + "".join(f"{correlation_matrix.loc[c1, c2]:>8.3f}" for c2 in COINS))
    lines.append("")

    lines.append(f"REGIMES (trailing_window={TRAILING_WINDOW_HOURS}h, "
                  f"bull_thresh=+{BULL_THRESHOLD:.0%}, bear_thresh=-{BEAR_THRESHOLD:.0%})")
    lines.append("-" * 78)
    lines.append(f"{'regime':<10}{'share':>9}{'mean_4h':>12}{'ann_vol':>10}{'mean_dur_4hbars':>17}")
    for r in regimes["labels"]:
        p = regimes["params"][r]
        lines.append(f"{r:<10}{p['share_of_time']:>9.2%}{p['mean_period_logret']:>12.6f}"
                      f"{p['ann_vol']:>10.2%}{regimes['mean_duration_periods'][r]:>17.1f}")
    lines.append("")

    lines.append("IDIO-RESIDUAL GARCH (4h)")
    lines.append("-" * 78)
    header = f"{'coin':<7}{'resid_ann_vol':>14}{'idio_vol':>10}{'vol_match':>11}" \
             f"{'resid_kurt':>12}{'resid_acf':>11}"
    lines.append(header)
    for coin in COINS:
        d = idio_out[coin]
        lines.append(
            f"{coin:<7}{d['_resid_ann_vol']:>14.2%}{betas[coin]['idio_vol']:>10.2%}"
            f"{str(d['_vol_match_ok']):>11}{d['_resid_kurtosis']:>12.2f}{d['_resid_acf_abs']:>11.4f}"
        )
    lines.append("")
    lines.append("=" * 78)

    report = "\n".join(lines)
    os.makedirs("reports", exist_ok=True)
    with open("reports/calibration_4h_report.txt", "w") as f:
        f.write(report + "\n")
    return report


def write_heldout_readme(held_out_start, held_out_end):
    os.makedirs(HELD_OUT_DIR, exist_ok=True)
    text = f"""# HELD-OUT REAL 4-HOUR DATA -- LOCKED, MEASUREMENT ONLY

This directory contains the last {HELD_OUT_MONTHS} months of REAL recorded 4-hour OHLCV
({held_out_start} -> {held_out_end}) for {", ".join(COINS)}, carved out by calibrate_4h.py
and excluded from calibration_4h.json / generator_4h.py entirely. Same discipline as
data/heldout/ (the hourly held-out set) -- measurement only, never tuning.

Files: {{coin}}USDT_4h_heldout.csv per coin, same columns as data/raw_4h/.
"""
    with open(f"{HELD_OUT_DIR}/README.md", "w") as f:
        f.write(text)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    print("Loading 4h data...")
    data = load_data()

    train_start, train_end, train_index, full_common_start, full_common_end, held_out_start = \
        compute_common_window(data)
    held_out_index = pd.date_range(held_out_start, full_common_end, freq="4h", tz="UTC")
    n_periods_train = len(train_index)
    n_periods_heldout = len(held_out_index)

    print("\n" + "=" * 78)
    print("TRAIN / HELD-OUT SPLIT (4h)")
    print("=" * 78)
    print(f"Full common window: {full_common_start} -> {full_common_end}")
    print(f"TRAIN window:    {train_start} -> {train_end}   ({n_periods_train} 4h bars)")
    print(f"HELD-OUT window: {held_out_start} -> {full_common_end}   ({n_periods_heldout} 4h bars)")

    no_overlap = len(set(train_index) & set(held_out_index)) == 0
    no_gap = (train_index[-1] + pd.Timedelta(hours=4)) == held_out_index[0]
    print(f"No overlap between TRAIN and HELD-OUT: {no_overlap}")
    print(f"No gap at the boundary: {no_gap}")
    assert no_overlap and no_gap, "TRAIN/HELD-OUT split is not contiguous/disjoint -- stopping."

    os.makedirs(HELD_OUT_DIR, exist_ok=True)
    for coin in COINS:
        df = data[coin]
        mask = (df.index >= held_out_start) & (df.index <= full_common_end)
        df.loc[mask].reset_index().to_csv(f"{HELD_OUT_DIR}/{coin}USDT_4h_heldout.csv", index=False)
    write_heldout_readme(held_out_start, full_common_end)
    print(f"\nPersisted held-out 4h OHLCV slices + README to {HELD_OUT_DIR}/")

    close, returns, dropped_per_coin = build_returns(data, train_index)
    n_returns_dropped_total = sum(dropped_per_coin.values())
    print(f"\nTRAIN: n_periods={n_periods_train}  returns_dropped_to_gaps={n_returns_dropped_total}")

    coin_stats = compute_coin_stats(returns)
    drawdowns = {coin: max_drawdown(close[coin]) for coin in COINS}

    correlation_matrix, market_factor_logret, explained_var_ratio, betas, returns_common = \
        compute_cross_sectional(returns, coin_stats)
    market_factor_stats = compute_market_factor_stats(market_factor_logret, explained_var_ratio)

    regimes, share_of_time = compute_regimes(market_factor_logret, train_index)
    print("Regime shares: " + ", ".join(f"{r}={share_of_time[r]:.1%}" for r in regimes["labels"]))

    idio_out = compute_idio_garch(returns_common, market_factor_logret, betas)

    for coin in COINS:
        coin_stats[coin].pop("_std_period", None)
        betas[coin].pop("_alpha", None)

    split_info = {"train_start": train_start, "train_end": train_end,
                  "heldout_start": held_out_start, "heldout_end": full_common_end}
    write_config(train_start, train_end, n_periods_train, n_returns_dropped_total,
                 coin_stats, market_factor_stats, correlation_matrix, betas, drawdowns,
                 regimes, idio_out, split_info)

    report = write_report(train_start, train_end, n_periods_train, n_returns_dropped_total,
                           dropped_per_coin, coin_stats, market_factor_stats, correlation_matrix,
                           betas, drawdowns, regimes, share_of_time, idio_out)
    print("\n" + report)

    cross_resolution_sanity_check(coin_stats, correlation_matrix)

    print("\nWrote config/calibration_4h.json and reports/calibration_4h_report.txt")
    print(f"Wrote {HELD_OUT_DIR}/*.csv + README.md (HELD-OUT, locked, measurement-only)")


if __name__ == "__main__":
    main()
