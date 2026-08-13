"""
calibrate.py — measure stylized facts of real hourly crypto OHLCV data and write
config/calibration.json, a stable contract consumed by a later synthetic-data
generator (not built here). Purely descriptive: no forecasting, no ML.

Run: python3 calibrate.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning

np.random.seed(42)

# --------------------------------------------------------------------------------------
# CONFIG — the only step with real design choices lives here (regime thresholds).
# Everything else below is a direct measurement of the data.
# --------------------------------------------------------------------------------------
DATA_DIR = "data/raw"
COINS = ["BTC", "ETH", "BNB", "MATIC", "SOL"]
FILE_TEMPLATE = "{coin}USDT_1h_full.csv"

# Held-out real-data split (see compute_common_window). The last HELD_OUT_MONTHS of the
# common window are carved out and LOCKED for measurement-only evaluation
# (evaluate_champion.py) -- never seen by calibration, generation, or evolution.
HELD_OUT_MONTHS = 6
HELD_OUT_DIR = "data/heldout"

HOURS_PER_YEAR = 8760  # calendar hours/year, used consistently for annualization denom
ANNUALIZATION_HOURS = 24 * 365  # per spec: ann_vol = std * sqrt(24*365)

# Regime-labeling thresholds (rule-based, transparent). Tuned so all three regimes are
# reasonably represented over the ~4y common window rather than one regime dominating.
TRAILING_WINDOW_HOURS = 168  # 7 days
BULL_THRESHOLD = 0.10  # trailing 7d market-factor return > +10% => "bull"
BEAR_THRESHOLD = 0.10  # trailing 7d market-factor return < -10% => "bear"
REGIME_MIN_PERIODS = 24  # require >=1 day of real data in the trailing window to label

# Idio-residual GARCH sanity/summary thresholds (v2 addition).
IDIO_VOL_MATCH_TOL = 0.03      # resid ann_vol vs existing idio_vol: warn if off by more than this
IDIO_STABLE_PERSISTENCE = 0.99  # idio_persistence below this = "stable", else "near-unit-root"

# Reference values I was given to sanity-check against (measured independently).
SANITY_REF = {
    "n_hours": (35761, 500),  # (expected, absolute tolerance)
    "ann_vol": {  # (expected fraction, absolute tolerance in fraction points)
        "BTC": (0.64, 0.15), "ETH": (0.81, 0.15), "BNB": (0.87, 0.15),
        "MATIC": (1.32, 0.20), "SOL": (1.36, 0.20),
    },
    "excess_kurtosis_range": (8, 35),  # loose band around the stated 13-26
    "corr_BTC_ETH": (0.84, 0.10),
    "corr_other_range": (0.45, 0.80),  # other pairs ~0.56-0.72
    "max_drawdown": {
        "BTC": (-0.77, 0.10), "ETH": (-0.81, 0.10), "BNB": (-0.73, 0.10),
        "MATIC": (-0.89, 0.10), "SOL": (-0.97, 0.10),
    },
}


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
# 2. Common window (computed from data, not hard-coded)
# --------------------------------------------------------------------------------------
def compute_common_window(data):
    starts = [df.index.min() for df in data.values()]
    ends = [df.index.max() for df in data.values()]
    full_common_start = max(starts)
    full_common_end = min(ends)

    # held_out_start is BOTH the first excluded-from-TRAIN hour and the first
    # INCLUDED-in-HELD-OUT hour: TRAIN = [full_common_start, held_out_start), HELD_OUT =
    # [held_out_start, full_common_end] -- contiguous, disjoint, no gap, no overlap.
    held_out_start = full_common_end - pd.DateOffset(months=HELD_OUT_MONTHS)

    # --- Window-selection seam ---------------------------------------------------
    # This IS the one-line change flagged when this file was first written: swap the
    # full common window for a train-only sub-window that locks out the held-out real
    # data (data/heldout/) from calibration entirely.
    WINDOW_START, WINDOW_END = full_common_start, held_out_start - pd.Timedelta(hours=1)
    # -------------------------------------------------------------------------------

    full_index = pd.date_range(WINDOW_START, WINDOW_END, freq="h", tz="UTC")
    return WINDOW_START, WINDOW_END, full_index, full_common_start, full_common_end, held_out_start


# --------------------------------------------------------------------------------------
# 3. Reindex to full hourly grid + gap-safe log returns
# --------------------------------------------------------------------------------------
def build_returns(data, full_index):
    close = pd.DataFrame({coin: df["close"].reindex(full_index) for coin, df in data.items()})
    log_price = np.log(close)

    # .diff() on a NaN-padded grid is automatically gap-safe: if the previous hour
    # is missing (NaN), the diff is NaN and gets dropped. We never fabricate a price.
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
# These three are pulled out as standalone functions (rather than left inline) so that
# validate_generator.py can import and apply the *exact same* math to synthetic data —
# see that file's `measure()`.
def ann_vol_from_returns(r):
    return float(r.std() * np.sqrt(ANNUALIZATION_HOURS))


def excess_kurtosis(r):
    return float(stats.kurtosis(r, fisher=True))


def acf_abs_lag1(r):
    return float(r.abs().autocorr(lag=1))


def max_drawdown(price_series):
    prices = price_series.dropna()
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return float(drawdown.min())


def fit_student_t(standardized_returns):
    # Fix loc=0, scale=1 since the input is already standardized; fit only dof.
    dof, loc, scale = stats.t.fit(standardized_returns, floc=0, fscale=1)
    return float(dof)


def fit_garch(returns, label):
    """Fit GARCH(1,1) via `arch`. Returns dict with omega/alpha/beta (in the return's
    native variance units) plus a `converged` flag. On failure/non-convergence, falls
    back to lag-1 autocorrelation of |r| and r^2, flagged accordingly."""
    scale = 100.0  # `arch` recommends O(1) magnitude returns for numerical stability
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
        mean_hourly = float(r.mean())
        std_hourly = float(r.std())
        ann_vol = ann_vol_from_returns(r)
        excess_kurt = excess_kurtosis(r)
        standardized = (r - mean_hourly) / std_hourly
        dof = fit_student_t(standardized)
        garch = fit_garch(r, coin)
        stats_out[coin] = {
            "mean_hourly_logret": mean_hourly,
            "ann_vol": ann_vol,
            "excess_kurtosis": excess_kurt,
            "student_t_dof": dof,
            "garch": garch,
            "_std_hourly": std_hourly,  # kept internally for market-factor rescaling
        }
    return stats_out


# --------------------------------------------------------------------------------------
# 5. Cross-sectional structure: correlation, market factor (PCA), beta/idio_vol
# --------------------------------------------------------------------------------------
def compute_cross_sectional(returns, coin_stats):
    # Complete-case rows only: PCA/regression need a rectangular matrix with no gaps.
    returns_common = returns.dropna(how="any")

    correlation_matrix = returns_common.corr()

    means = returns_common.mean()
    stds = returns_common.std()
    Z = (returns_common - means) / stds  # standardized returns

    cov = np.cov(Z.values, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    v1 = eigvecs[:, 0]
    if v1.sum() < 0:  # sign convention: PC1 points "up" with the market, not down
        v1 = -v1
    explained_var_ratio = float(eigvals[0] / eigvals.sum())

    pc1_scores = Z.values @ v1  # standardized units, mean~0, var~eigvals[0]

    # Rescale the standardized PC1 score into a "log-return-like" series so it can be
    # GARCH-fit and thresholded (e.g. +-10% / 7d) in the same units as real returns.
    # Design choice: scale by the average per-coin hourly vol (not add back a mean —
    # de-meaning only removes the *full-sample* mean, local trends are preserved).
    avg_hourly_vol = float(np.mean([coin_stats[c]["_std_hourly"] for c in COINS]))
    market_factor_logret = pd.Series(pc1_scores * avg_hourly_vol, index=returns_common.index)

    # Per-coin beta_to_market (OLS with intercept) and idio_vol (annualized residual std).
    # `_alpha` (the OLS intercept) is kept internally so compute_idio_garch() can
    # reconstruct the exact same residual later, rather than re-fitting anything.
    betas = {}
    for coin in COINS:
        y = returns_common[coin].values
        x = market_factor_logret.values
        beta, alpha = np.polyfit(x, y, 1)
        resid = y - (alpha + beta * x)
        idio_vol = float(np.std(resid, ddof=1) * np.sqrt(ANNUALIZATION_HOURS))
        betas[coin] = {"beta_to_market": float(beta), "idio_vol": idio_vol, "_alpha": float(alpha)}

    return correlation_matrix, market_factor_logret, explained_var_ratio, betas, returns_common


def compute_market_factor_stats(market_factor_logret, explained_var_ratio):
    r = market_factor_logret.dropna()
    mean_hourly = float(r.mean())
    std_hourly = float(r.std())
    ann_vol = ann_vol_from_returns(r)
    excess_kurt = excess_kurtosis(r)
    standardized = (r - mean_hourly) / std_hourly
    dof = fit_student_t(standardized)
    garch = fit_garch(r, "market_factor")
    return {
        "mean_hourly_logret": mean_hourly,
        "ann_vol": ann_vol,
        "excess_kurtosis": excess_kurt,
        "explained_var_ratio": explained_var_ratio,
        "garch": garch,
        "student_t_dof": dof,
    }


# --------------------------------------------------------------------------------------
# 5b. Idiosyncratic-residual GARCH (v2 addition)
# --------------------------------------------------------------------------------------
# The v1.5 generator gave each coin's idio noise its own GARCH(1,1), but BORROWED
# alpha/beta from that coin's TOTAL-return GARCH fit (compute_coin_stats' `garch`).
# That fit's persistence is mostly carried by the shared factor (e.g. BNB's total-return
# persistence is ~0.99995 — a near-unit-root value belonging to the market, not to BNB's
# idio channel), and since the idio process gets no regime reset, that borrowed
# persistence let its realized variance drift far from target -> inflated correlations.
# Fix: fit GARCH directly on the idiosyncratic RESIDUAL (return minus what the single
# factor explains), so the idio channel gets its own, much lower persistence.
def compute_idio_garch(returns_common, market_factor_logret, betas):
    idio_out = {}
    print("\nIdio-residual sanity report (annualized vol should match existing idio_vol):")
    for coin in COINS:
        beta = betas[coin]["beta_to_market"]
        alpha_int = betas[coin]["_alpha"]
        # Same residual definition compute_cross_sectional already used for idio_vol —
        # reconstructed here (not re-fit) from the SAME beta/alpha/market_factor.
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
            # Fall back to flat-vol idio (alpha=beta=0) rather than propagate a bad fit.
            a_, b_, omega = 0.0, 0.0, None
            garch_ok = False

        idio_out[coin] = {
            "idio_garch": {"omega": omega, "alpha": a_, "beta": b_},
            "idio_student_t_dof": dof,
            "idio_persistence": a_ + b_,
            "idio_garch_ok": garch_ok,
            "_resid_ann_vol": resid_ann_vol,
            "_resid_kurtosis": resid_kurt,
            "_resid_acf_abs": resid_acf,
            "_vol_match_ok": vol_match_ok,
        }
    return idio_out


# --------------------------------------------------------------------------------------
# 6. Regimes (rule-based on the market factor's trailing return)
# --------------------------------------------------------------------------------------
def compute_regimes(market_factor_logret, full_index):
    # Put the factor back on the full hourly grid (NaN where not all coins had data)
    # and use a *time-based* rolling window ('168h') so trailing return respects
    # calendar gaps rather than just counting rows.
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

    # Per-regime stats computed on the market factor's own hourly log return
    params = {}
    for regime in labels_order:
        mask = labels == regime
        r = factor_full[mask].dropna()
        params[regime] = {
            "mean_hourly_logret": float(r.mean()),
            "ann_vol": float(r.std() * np.sqrt(ANNUALIZATION_HOURS)),
            "share_of_time": share_of_time[regime],
        }

    # Transition matrix: only count transitions between temporally consecutive hours
    # (guaranteed by full_index construction) where both hours are labeled.
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

    mean_duration_hours = {}
    for r in labels_order:
        p_self = transition_matrix[idx_pos[r], idx_pos[r]]
        mean_duration_hours[r] = float(1.0 / (1.0 - p_self)) if p_self < 1.0 else float("inf")

    return {
        "labels": labels_order,
        "params": params,
        "transition_matrix": transition_matrix.tolist(),
        "mean_duration_hours": mean_duration_hours,
        "config": {
            "trailing_window_hours": TRAILING_WINDOW_HOURS,
            "bull_threshold": BULL_THRESHOLD,
            "bear_threshold": BEAR_THRESHOLD,
        },
    }, share_of_time


# --------------------------------------------------------------------------------------
# 7. Sanity checks against independently-measured reference values
# --------------------------------------------------------------------------------------
def sanity_check(n_hours, coin_stats, correlation_matrix, drawdowns):
    print("\n" + "=" * 70)
    print("SANITY CHECK")
    print("=" * 70)

    def check(name, value, expected, tol, fmt="{:.4f}"):
        ok = abs(value - expected) <= tol
        status = "PASS" if ok else "WARN"
        print(f"[{status}] {name}: got {fmt.format(value)}, "
              f"expected {fmt.format(expected)} +/- {fmt.format(tol)}")
        return ok

    exp_n, tol_n = SANITY_REF["n_hours"]
    check("common window n_hours", n_hours, exp_n, tol_n, fmt="{:.0f}")

    for coin in COINS:
        exp, tol = SANITY_REF["ann_vol"][coin]
        check(f"ann_vol[{coin}]", coin_stats[coin]["ann_vol"], exp, tol)

    lo, hi = SANITY_REF["excess_kurtosis_range"]
    for coin in COINS:
        k = coin_stats[coin]["excess_kurtosis"]
        ok = lo <= k <= hi
        print(f"[{'PASS' if ok else 'WARN'}] excess_kurtosis[{coin}]: got {k:.2f}, "
              f"expected in [{lo}, {hi}]")

    exp, tol = SANITY_REF["corr_BTC_ETH"]
    check("corr(BTC,ETH)", correlation_matrix.loc["BTC", "ETH"], exp, tol)

    lo, hi = SANITY_REF["corr_other_range"]
    for i, c1 in enumerate(COINS):
        for c2 in COINS[i + 1:]:
            if {c1, c2} == {"BTC", "ETH"}:
                continue
            v = correlation_matrix.loc[c1, c2]
            ok = lo <= v <= hi
            print(f"[{'PASS' if ok else 'WARN'}] corr({c1},{c2}): got {v:.3f}, "
                  f"expected in [{lo}, {hi}]")

    for coin in COINS:
        exp, tol = SANITY_REF["max_drawdown"][coin]
        check(f"max_drawdown[{coin}]", drawdowns[coin], exp, tol)


# --------------------------------------------------------------------------------------
# 8. Write outputs
# --------------------------------------------------------------------------------------
def write_config(window_start, window_end, n_hours, n_returns_dropped_total,
                  coin_stats, market_factor_stats, correlation_matrix,
                  betas, drawdowns, regimes, idio_out, split_info):
    config = {
        "meta": {
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "n_hours": n_hours,
            "n_returns_dropped_to_gaps": n_returns_dropped_total,
            "coins": COINS,
            "hours_per_year": HOURS_PER_YEAR,
            # Provenance: calibration.json is computed ONLY on [train_start, train_end].
            # [heldout_start, heldout_end] is real, recorded data locked in data/heldout/
            # -- never used for calibration, generation, or evolution. See
            # data/heldout/README.md and evaluate_champion.py.
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
                "mean_hourly_logret": coin_stats[coin]["mean_hourly_logret"],
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
    with open("config/calibration.json", "w") as f:
        json.dump(config, f, indent=2)
    return config


def write_report(window_start, window_end, n_hours, n_returns_dropped_total,
                  dropped_per_coin, coin_stats, market_factor_stats,
                  correlation_matrix, betas, drawdowns, regimes, share_of_time, idio_out):
    lines = []
    lines.append("CALIBRATION REPORT (TRAIN window -- held-out real data excluded)")
    lines.append("=" * 78)
    lines.append(f"TRAIN window: {window_start} -> {window_end}")
    lines.append(f"n_hours in window: {n_hours}")
    lines.append(f"Total returns dropped due to gaps: {n_returns_dropped_total}")
    lines.append("Dropped per coin: " + ", ".join(f"{c}={dropped_per_coin[c]}" for c in COINS))
    lines.append("")

    lines.append("PER-COIN STYLIZED FACTS")
    lines.append("-" * 78)
    header = f"{'coin':<7}{'mean_hr':>12}{'ann_vol':>10}{'kurt':>9}{'max_dd':>9}" \
             f"{'t_dof':>8}{'beta':>8}{'idio_vol':>10}{'garch_ok':>10}"
    lines.append(header)
    for coin in COINS:
        s = coin_stats[coin]
        lines.append(
            f"{coin:<7}{s['mean_hourly_logret']:>12.6f}{s['ann_vol']:>10.2%}"
            f"{s['excess_kurtosis']:>9.2f}{drawdowns[coin]:>9.2%}{s['student_t_dof']:>8.2f}"
            f"{betas[coin]['beta_to_market']:>8.2f}{betas[coin]['idio_vol']:>10.2%}"
            f"{str(s['garch']['converged']):>10}"
        )
    lines.append("")

    lines.append("GARCH(1,1) PARAMETERS")
    lines.append("-" * 78)
    for coin in COINS:
        g = coin_stats[coin]["garch"]
        if g["converged"]:
            lines.append(f"{coin:<7} omega={g['omega']:.3e}  alpha={g['alpha']:.4f}  beta={g['beta']:.4f}")
        else:
            lines.append(f"{coin:<7} FAILED TO CONVERGE -> fallback: "
                          f"acf|r|={g['fallback_acf_abs_ret_lag1']:.4f}  "
                          f"acf(r^2)={g['fallback_acf_sq_ret_lag1']:.4f}")
    g = market_factor_stats["garch"]
    if g["converged"]:
        lines.append(f"{'MARKET':<7} omega={g['omega']:.3e}  alpha={g['alpha']:.4f}  beta={g['beta']:.4f}")
    else:
        lines.append(f"{'MARKET':<7} FAILED TO CONVERGE -> fallback: "
                      f"acf|r|={g['fallback_acf_abs_ret_lag1']:.4f}  "
                      f"acf(r^2)={g['fallback_acf_sq_ret_lag1']:.4f}")
    lines.append("")

    lines.append("MARKET FACTOR (PC1 of standardized returns)")
    lines.append("-" * 78)
    lines.append(f"explained_var_ratio: {market_factor_stats['explained_var_ratio']:.2%}")
    lines.append(f"ann_vol: {market_factor_stats['ann_vol']:.2%}   "
                 f"excess_kurtosis: {market_factor_stats['excess_kurtosis']:.2f}   "
                 f"student_t_dof: {market_factor_stats['student_t_dof']:.2f}")
    lines.append("")

    lines.append("CORRELATION MATRIX (hourly log returns)")
    lines.append("-" * 78)
    lines.append(f"{'':<7}" + "".join(f"{c:>8}" for c in COINS))
    for c1 in COINS:
        lines.append(f"{c1:<7}" + "".join(f"{correlation_matrix.loc[c1, c2]:>8.3f}" for c2 in COINS))
    lines.append("")

    lines.append(f"REGIMES (trailing_window={TRAILING_WINDOW_HOURS}h, "
                  f"bull_thresh=+{BULL_THRESHOLD:.0%}, bear_thresh=-{BEAR_THRESHOLD:.0%})")
    lines.append("-" * 78)
    lines.append(f"{'regime':<10}{'share':>9}{'mean_hr':>12}{'ann_vol':>10}{'mean_dur_h':>12}")
    for r in regimes["labels"]:
        p = regimes["params"][r]
        lines.append(f"{r:<10}{p['share_of_time']:>9.2%}{p['mean_hourly_logret']:>12.6f}"
                      f"{p['ann_vol']:>10.2%}{regimes['mean_duration_hours'][r]:>12.1f}")
    lines.append("")
    lines.append("Transition matrix (rows=from, cols=to), order=" + str(regimes["labels"]))
    for i, r in enumerate(regimes["labels"]):
        lines.append(f"{r:<10}" + "".join(f"{p:>10.3f}" for p in regimes["transition_matrix"][i]))
    lines.append("")

    lines.append("IDIO-RESIDUAL GARCH (v2 — fit on return minus beta*market_factor, not borrowed)")
    lines.append("-" * 78)
    lines.append("Sanity: resid ann_vol should match existing idio_vol (same residual, reconstructed).")
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
    lines.append("Idio persistence vs total-return persistence (expect idio << total: the factor")
    lines.append(f"carries most of the persistence). Stable = idio_persistence < {IDIO_STABLE_PERSISTENCE}.")
    header2 = f"{'coin':<7}{'idio_persist':>13}{'total_persist':>14}{'idio_t_dof':>11}{'idio_garch_ok':>14}   status"
    lines.append(header2)
    for coin in COINS:
        d = idio_out[coin]
        g = coin_stats[coin]["garch"]
        total_persist = (g["alpha"] + g["beta"]) if g["converged"] else float("nan")
        status = "stable" if d["idio_persistence"] < IDIO_STABLE_PERSISTENCE \
            else "still near-unit-root -- will need a reset in generator"
        lines.append(
            f"{coin:<7}{d['idio_persistence']:>13.5f}{total_persist:>14.5f}"
            f"{d['idio_student_t_dof']:>11.2f}{str(d['idio_garch_ok']):>14}   {status}"
        )
    lines.append("")
    lines.append("=" * 78)

    report = "\n".join(lines)
    with open("reports/calibration_report.txt", "w") as f:
        f.write(report + "\n")
    return report


# --------------------------------------------------------------------------------------
# Convenience wrapper for other scripts (e.g. validate_generator.py) that just need the
# real gap-safe hourly log returns, without recomputing coin_stats/regimes/etc.
# --------------------------------------------------------------------------------------
def get_real_returns():
    data = load_data()
    # Only the TRAIN-window pieces are used here -- returns unpacked into `_` are the
    # full-common-window/held-out boundaries, not needed by this convenience wrapper's
    # external contract (kept as a 5-tuple, unchanged, so existing callers -- e.g.
    # validate_generator.py -- don't need to change).
    window_start, window_end, full_index, _, _, _ = compute_common_window(data)
    close, returns, dropped_per_coin = build_returns(data, full_index)
    return returns, close, window_start, window_end, full_index


# --------------------------------------------------------------------------------------
# 9. Held-out split: run the full measurement pipeline for an arbitrary window (used
# twice -- once on the full common window for BEFORE/AFTER comparison only, once on the
# TRAIN-only window, which is what actually gets written to calibration.json).
# --------------------------------------------------------------------------------------
def compute_calibration_bundle(data, window_start, window_end, full_index, label=""):
    """Pure computation (no file writes, no sanity_check) -- reuses every existing
    per-window measurement function unchanged, in the same order main() always called
    them in. Returns a dict with everything write_config/write_report/sanity_check need.
    """
    if label:
        print(f"\n--- Calibration bundle: {label} ---")

    n_hours = len(full_index)
    close, returns, dropped_per_coin = build_returns(data, full_index)
    n_returns_dropped_total = sum(dropped_per_coin.values())
    print(f"  n_hours={n_hours}  returns_dropped_to_gaps={n_returns_dropped_total}")

    coin_stats = compute_coin_stats(returns)
    drawdowns = {coin: max_drawdown(close[coin]) for coin in COINS}

    correlation_matrix, market_factor_logret, explained_var_ratio, betas, returns_common = \
        compute_cross_sectional(returns, coin_stats)
    market_factor_stats = compute_market_factor_stats(market_factor_logret, explained_var_ratio)

    regimes, share_of_time = compute_regimes(market_factor_logret, full_index)
    print("  Regime shares: " + ", ".join(f"{r}={share_of_time[r]:.1%}" for r in regimes["labels"]))

    idio_out = compute_idio_garch(returns_common, market_factor_logret, betas)

    # strip internal-only fields (kept above for market-factor rescaling / idio-residual
    # reconstruction) before handing the bundle to callers.
    for coin in COINS:
        coin_stats[coin].pop("_std_hourly", None)
        betas[coin].pop("_alpha", None)

    return {
        "window_start": window_start, "window_end": window_end, "n_hours": n_hours,
        "n_returns_dropped_total": n_returns_dropped_total, "dropped_per_coin": dropped_per_coin,
        "coin_stats": coin_stats, "drawdowns": drawdowns,
        "correlation_matrix": correlation_matrix, "betas": betas,
        "market_factor_stats": market_factor_stats,
        "regimes": regimes, "share_of_time": share_of_time, "idio_out": idio_out,
    }


def print_before_after_comparison(bundle_before, bundle_after):
    """Surfaces how much recalibrating on TRAIN-only (excluding the held-out 6 months)
    moved the key stats, relative to the full-window calibration. Printed only -- never
    used to re-tune anything (see this file's module docstring / the task discipline)."""
    print("\n" + "=" * 78)
    print("BEFORE (full window) vs AFTER (TRAIN-only, excludes held-out) -- key stats")
    print("=" * 78)
    header = (f"{'coin':<7}{'ann_vol_before':>15}{'ann_vol_after':>14}{'kurt_before':>13}"
              f"{'kurt_after':>12}{'beta_before':>13}{'beta_after':>12}")
    print(header)
    for coin in COINS:
        cb, ca = bundle_before["coin_stats"][coin], bundle_after["coin_stats"][coin]
        bb, ba = bundle_before["betas"][coin], bundle_after["betas"][coin]
        print(f"{coin:<7}{cb['ann_vol']:>15.2%}{ca['ann_vol']:>14.2%}"
              f"{cb['excess_kurtosis']:>13.2f}{ca['excess_kurtosis']:>12.2f}"
              f"{bb['beta_to_market']:>13.2f}{ba['beta_to_market']:>12.2f}")

    print()
    print(f"{'regime':<10}{'share_before':>14}{'share_after':>13}")
    for r in bundle_before["regimes"]["labels"]:
        sb = bundle_before["share_of_time"][r]
        sa = bundle_after["share_of_time"][r]
        print(f"{r:<10}{sb:>14.1%}{sa:>13.1%}")

    print()
    print("(Not re-tuned to compensate -- if a stat moved a lot, the held-out months were")
    print("likely an unusual regime relative to the rest of the window; that's useful to")
    print("know, not something to correct for.)")


def write_heldout_readme(held_out_start, held_out_end):
    os.makedirs(HELD_OUT_DIR, exist_ok=True)
    text = f"""# HELD-OUT REAL DATA -- LOCKED, MEASUREMENT ONLY

This directory contains the last {HELD_OUT_MONTHS} months of REAL recorded hourly OHLCV
({held_out_start} -> {held_out_end}) for {", ".join(COINS)}, carved out of the full common
window by calibrate.py and excluded from everything else in this pipeline.

DO NOT use this data for:
  - calibration (calibrate.py only ever calibrates on the TRAIN window, data/raw/ minus
    this slice -- see config/calibration.json's meta.split for the exact boundaries)
  - synthetic-data generation (generator.py is calibrated on TRAIN only)
  - evolution / training / hyperparameter tuning of any kind

The ONLY legitimate use of this data is evaluate_champion.py, run ONCE per finished
evolution run's best genome, to MEASURE (never tune) how it performs on real market
history it has never seen, directly or indirectly through the generator. If a champion
fails here, the fix is to revise the pipeline (calibrator/generator/evolution) and judge
the revision using GENERATED-world performance -- then come back to this held-out set
only for a fresh final check on the NEW champion. Never iterate against this data; every
look costs some of its power as an unbiased judge.

Files: {{coin}}USDT_1h_heldout.csv per coin, one per coin in {COINS}, same columns as
data/raw/ (timestamp,open,high,low,close,volume) -- real recorded rows only, gaps
preserved as-is (not filled or reindexed).
"""
    with open(f"{HELD_OUT_DIR}/README.md", "w") as f:
        f.write(text)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    print("Loading data...")
    data = load_data()

    train_start, train_end, train_index, full_common_start, full_common_end, held_out_start = \
        compute_common_window(data)
    held_out_index = pd.date_range(held_out_start, full_common_end, freq="h", tz="UTC")
    n_hours_train = len(train_index)
    n_hours_heldout = len(held_out_index)

    print("\n" + "=" * 78)
    print("TRAIN / HELD-OUT SPLIT")
    print("=" * 78)
    print(f"Full common window: {full_common_start} -> {full_common_end}")
    print(f"TRAIN window:    {train_start} -> {train_end}   ({n_hours_train} hours)")
    print(f"HELD-OUT window: {held_out_start} -> {full_common_end}   ({n_hours_heldout} hours)")

    no_overlap = len(set(train_index) & set(held_out_index)) == 0
    no_gap = (train_index[-1] + pd.Timedelta(hours=1)) == held_out_index[0]
    print(f"No overlap between TRAIN and HELD-OUT: {no_overlap}")
    print(f"No gap at the boundary (train's last hour + 1h == held-out's first hour): {no_gap}")
    assert no_overlap and no_gap, "TRAIN/HELD-OUT split is not contiguous/disjoint -- stopping."

    # --- Persist the locked held-out real OHLCV slice (real recorded rows only) ---
    os.makedirs(HELD_OUT_DIR, exist_ok=True)
    for coin in COINS:
        df = data[coin]
        mask = (df.index >= held_out_start) & (df.index <= full_common_end)
        df.loc[mask].reset_index().to_csv(f"{HELD_OUT_DIR}/{coin}USDT_1h_heldout.csv", index=False)
    write_heldout_readme(held_out_start, full_common_end)
    print(f"\nPersisted held-out OHLCV slices + README to {HELD_OUT_DIR}/ "
          f"(gaps preserved, not reindexed/filled).")

    # --- BEFORE (full window) vs AFTER (TRAIN-only) calibration bundles -----------
    full_index_all = pd.date_range(full_common_start, full_common_end, freq="h", tz="UTC")
    bundle_before = compute_calibration_bundle(
        data, full_common_start, full_common_end, full_index_all,
        label="FULL window (BEFORE, comparison only -- NOT written)")
    bundle_after = compute_calibration_bundle(
        data, train_start, train_end, train_index,
        label="TRAIN-only window (AFTER, this IS what gets written)")

    print_before_after_comparison(bundle_before, bundle_after)

    # --- Write outputs from the TRAIN-only bundle ONLY -----------------------------
    split_info = {
        "train_start": train_start, "train_end": train_end,
        "heldout_start": held_out_start, "heldout_end": full_common_end,
    }
    write_config(bundle_after["window_start"], bundle_after["window_end"], bundle_after["n_hours"],
                 bundle_after["n_returns_dropped_total"], bundle_after["coin_stats"],
                 bundle_after["market_factor_stats"], bundle_after["correlation_matrix"],
                 bundle_after["betas"], bundle_after["drawdowns"], bundle_after["regimes"],
                 bundle_after["idio_out"], split_info)

    report = write_report(bundle_after["window_start"], bundle_after["window_end"], bundle_after["n_hours"],
                           bundle_after["n_returns_dropped_total"], bundle_after["dropped_per_coin"],
                           bundle_after["coin_stats"], bundle_after["market_factor_stats"],
                           bundle_after["correlation_matrix"], bundle_after["betas"],
                           bundle_after["drawdowns"], bundle_after["regimes"],
                           bundle_after["share_of_time"], bundle_after["idio_out"])
    print("\n" + report)

    sanity_check(bundle_after["n_hours"], bundle_after["coin_stats"],
                 bundle_after["correlation_matrix"], bundle_after["drawdowns"])

    print("\nWrote config/calibration.json and reports/calibration_report.txt (TRAIN-only)")
    print(f"Wrote {HELD_OUT_DIR}/*.csv + README.md (HELD-OUT, locked, measurement-only)")


if __name__ == "__main__":
    main()
