"""
generator_4h.py — synthetic 4-HOUR OHLCV generator, rebuilt from generator.py's
hourly design on config/calibration_4h.json, with THREE fixes for blind spots a
statistical validation found in the hourly generator (see
validate_generator_distributional.py's prior report): excess kurtosis ~5x too low,
cross-coin correlation ~10-15pts too low, vol-clustering decay shape wrong.

Same four-layer structure as generator.py:
  1. Regime      — "pure" or "mix", unchanged design (durations now in 4h-bar units).
  2. Market factor — GARCH(1,1)-t, variance-targeted per regime. Unchanged design.
  3. Coins       — beta*factor + CORRELATED idio GARCH(1,1) diffusion + a per-coin
                   compound-Poisson JUMP component. This is where all three fixes live:
       FIX 1 (fat tails): each coin's idio return gets an added jump term
         J_t * xi_t, J_t ~ Bernoulli(p_jump), xi_t ~ N(0, jump_std^2), calibrated (by
         simulation search, see calibrate_jumps_4h.py) so generated WINDOW-level excess
         kurtosis matches real 4h window-level kurtosis -- NOT just a long-sample
         aggregate (that's what let the old generator's kurtosis gap go unnoticed).
       FIX 2 (cross-coin correlation): the diffusion shocks across the 5 coins are no
         longer independent -- they're drawn as eps_t ~ Cholesky(idio_corr) @ w_t,
         w_t iid N(0,1), where idio_corr is the RESIDUAL correlation needed on top of
         the shared beta*factor exposure so the TOTAL (factor + idio) correlation
         matches the calibrated empirical matrix (see compute_idio_correlation()).
       FIX 3 (vol-clustering shape): persistence cap + periodic reset kept from the
         hourly design (same mechanism, converted to 4h-bar units), re-tuned against
         the 4h ACF-decay target -- see JUMP/CLUSTERING calibration report.
  4. Prices/OHLC — unchanged design, 4h bar spacing.

Jump parameters (jump_prob, jump_var_frac per coin) are NOT derived analytically --
compound jump-diffusion + GARCH calibration doesn't have a clean closed form, so they're
found by simulation search (calibrate_jumps_4h.py) and stored back into
config/calibration_4h.json under each coin's block. generate_raw() reads them from
there, same "everything from one file" contract as generator.py.

Run: python3 generator_4h.py   (writes a demo 4h "mix" world to data/synthetic_4h/)
"""

import json
import os

import numpy as np
import pandas as pd

CALIBRATION_PATH = "config/calibration_4h.json"
COINS = ["BTC", "ETH", "BNB", "MATIC", "SOL"]
REGIME_NAMES = ["bull", "bear", "sideways"]

BARS_PER_DAY = 6

# Layer 1 (mix mode): mean regime duration, 4h-bar units. Was 672 HOURS (~4 weeks) in the
# hourly generator; same real-world duration, expressed in 4h bars: 672/4 = 168.
MIX_MEAN_DURATION_PERIODS = 168

CRASH_CORR_STRENGTH_DEFAULT = 0.0

# Layer 3: idio-channel safeguards, converted to 4h-bar units (was PERSISTENCE_CAP=0.995,
# IDIO_RESET_PERIOD=336 HOURS=14 days in the hourly generator -- same real-world reset
# cadence, expressed in 4h bars: 336/4 = 84).
PERSISTENCE_CAP = 0.995
IDIO_RESET_PERIOD = 84

# FIX 3 (round 2): vol-clustering was too weak at lag 1 (~0.025 generated vs ~0.09 real,
# on a LONG sample) even though the CALIBRATED idio-residual GARCH alpha/beta, measured
# in isolation, reproduces plenty of clustering (~0.23 at lag 1). The gap is largely JUMP
# DILUTION: with jump_var_frac~0.4-0.7 (needed for Fix 1's kurtosis), most of a period's
# variance comes from an i.i.d., non-clustering jump, which swamps the diffusion's own
# autocorrelation once mixed into the observed total return.
# idio_alpha_boost (calibration_4h.json per-coin, DEFAULT_IDIO_ALPHA_BOOST fallback here)
# scales the diffusion's ARCH term (alpha) before re-pinning to PERSISTENCE_CAP, to
# compensate. HONESTLY REPORTED, NOT FULLY SOLVED: a round-2 attempt at aggressive,
# per-coin boosts (tuned against LONG, e.g. 50000-period, samples) reached the long-sample
# ACF target closely, but REGRESSED the actual acceptance metric -- classifier AUC
# computed on the real SHORT 84-bar windows went UP (worse), not down, because acf_abs
# at n=84 is a high-variance, apparently-biased-downward statistic that a long-sample
# calibration doesn't optimize for (see generate_market_factor()'s docstring for the
# larger version of this same lesson, which is why the market-factor dampening was
# reverted too). Left at a conservative uniform boost=2.0 (round 1's originally-shipped,
# validated value) rather than the more aggressive round-2 search result.
DEFAULT_IDIO_ALPHA_BOOST = 1.0

# Fix 1 defaults (overridden per-coin by calibration_4h.json's "jump_prob"/"jump_var_frac"
# once calibrate_jumps_4h.py has run; these are only a fallback for a coin with no
# calibrated jump params yet).
DEFAULT_JUMP_PROB = 0.0
DEFAULT_JUMP_VAR_FRAC = 0.0

# Safety cap on jump_var_frac (Fix 1 x Fix 2 interaction): correlation (Fix 2) is only
# carried by the diffusion sub-component of idio (jumps are independent noise -- see
# compute_idio_correlation's dilution-compensation math). As jump_var_frac -> 1, the
# diffusion's share of variance -> 0, so the correlation boost needed to compensate for
# jump dilution blows up and eventually can't be met even at the +-0.999 correlation
# clip. Empirically, jump_var_frac=0.95 for BTC pushed 2 correlation pairs (BTC-MATIC,
# BTC-SOL) outside the +-0.05 tolerance; capping at 0.85 (also the search grid's ceiling)
# keeps enough diffusion variance for Fix 2's compensation to actually work.
MAX_JUMP_VAR_FRAC = 0.85

# Layer 4: intra-bar OHLC shape (un-calibrated approximation, same as generator.py).
OHLC_WICK_FACTOR = 0.5
OHLC_LOW_FLOOR_FRAC = 0.5

VOLUME_BASE = {"BTC": 5_000, "ETH": 80_000, "BNB": 20_000, "MATIC": 5_000_000, "SOL": 300_000}
VOLUME_VOL_SENSITIVITY = 3.0
VOLUME_NOISE_RANGE = (0.7, 1.3)

FALLBACK_START_PRICE = {
    "BTC": 11751.79, "ETH": 391.79, "BNB": 22.1702, "MATIC": 0.02463, "SOL": 2.9515,
}


# --------------------------------------------------------------------------------------
# Calibration + price-anchor loading
# --------------------------------------------------------------------------------------
def load_calibration(path=CALIBRATION_PATH):
    with open(path) as f:
        return json.load(f)


def load_start_prices(calib):
    window_start = pd.Timestamp(calib["meta"]["window_start"])
    prices = {}
    for coin in COINS:
        path = f"data/raw_4h/{coin}USDT_4h_full.csv"
        price = None
        if os.path.exists(path):
            df = pd.read_csv(path, usecols=["timestamp", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            row = df.loc[df["timestamp"] == window_start]
            if len(row):
                price = float(row["close"].iloc[0])
        prices[coin] = price if price is not None else FALLBACK_START_PRICE[coin]
    return prices


# --------------------------------------------------------------------------------------
# Layer 1 — Regime schedule (unchanged design from generator.py, 4h-bar units)
# --------------------------------------------------------------------------------------
def build_regime_schedule(rng, n_periods, regime_mode, regime, calib):
    if regime_mode == "pure":
        if regime not in REGIME_NAMES:
            raise ValueError(f"regime_mode='pure' requires regime in {REGIME_NAMES}, got {regime!r}")
        return np.full(n_periods, regime, dtype=object)

    if regime_mode != "mix":
        raise ValueError(f"regime_mode must be 'pure' or 'mix', got {regime_mode!r}")

    shares = {r: calib["regimes"]["params"][r]["share_of_time"] for r in REGIME_NAMES}
    weights = np.array([shares[r] for r in REGIME_NAMES])
    weights = weights / weights.sum()

    schedule = np.empty(n_periods, dtype=object)
    current = rng.choice(REGIME_NAMES, p=weights)
    t = 0
    while t < n_periods:
        duration = int(max(1, round(rng.exponential(MIX_MEAN_DURATION_PERIODS))))
        duration = min(duration, n_periods - t)
        schedule[t:t + duration] = current
        t += duration
        if t >= n_periods:
            break
        others = [r for r in REGIME_NAMES if r != current]
        w = np.array([shares[r] for r in others])
        w = w / w.sum()
        current = rng.choice(others, p=w)
    return schedule


# --------------------------------------------------------------------------------------
# Layer 2 — Market factor (unchanged design, periods_per_year swapped in)
# --------------------------------------------------------------------------------------
# FIX 3 (round 2) ATTEMPTED AND REVERTED -- reported honestly rather than shipped in a
# regressed state. Diagnosis: the market factor's raw fitted GARCH (alpha~0.073,
# beta~0.920, persistence~0.99, untouched by any cap/reset unlike the idio channel)
# dominates every coin's total clustering via beta_i^2*Var(factor) -- in isolation it
# gives lag-1 acf_abs~0.25 (real target ~0.09), and idio_alpha_boost alone had almost no
# effect on the total (idio's variance share is too small once jump_var_frac diverts
# 40-70% of it to Fix 1's jumps). Dampening the factor to alpha=0.05/beta=0.7 (persistence
# 0.75) DID fix the factor-in-isolation and LONG-SAMPLE (50000-period) ACF numbers -- but
# the actual acceptance test measures acf_abs on SHORT 84-bar windows (matching real
# held-out's fixed 6-month length), where this estimator has enormous sampling
# variance/bias (empirically: mean 0.017-0.036 across 120 such windows for the DAMPENED
# generator, vs the SAME generator's long-sample estimate of ~0.05-0.09 -- a large,
# window-length-dependent gap that a long-sample calibration search cannot see or fix).
# Shipping the dampened factor + aggressive per-coin idio boosts RAISED both AUCs (0.910
# ->0.949 train, 0.876->0.919 held-out) -- a real regression, not an improvement -- so
# both changes were reverted. Left here as a documented dead end: whoever revisits Fix 3
# needs a calibration search that optimizes the SHORT-WINDOW statistic directly (the one
# the classifier actually sees), not a long-sample proxy for it.
def generate_market_factor(rng, n_periods, regime_schedule, calib, periods_per_year):
    garch = calib["market_factor"]["garch"]
    alpha, beta = garch["alpha"], garch["beta"]
    dof = calib["market_factor"]["student_t_dof"]
    regime_params = calib["regimes"]["params"]

    z = rng.standard_t(dof, size=n_periods) * np.sqrt((dof - 2) / dof)

    f = np.empty(n_periods)
    sigma2 = np.empty(n_periods)
    a = np.empty(n_periods)

    prev_regime = None
    omega = None
    for t in range(n_periods):
        regime = regime_schedule[t]
        if regime != prev_regime:
            ann_vol_target = regime_params[regime]["ann_vol"]
            target_var = (ann_vol_target / np.sqrt(periods_per_year)) ** 2
            omega = target_var * (1 - alpha - beta)
            sigma2_t = target_var
            prev_regime = regime
        else:
            sigma2_t = omega + alpha * a[t - 1] ** 2 + beta * sigma2[t - 1]

        sigma2[t] = sigma2_t
        a_t = np.sqrt(sigma2_t) * z[t]
        a[t] = a_t
        mean_key = "mean_period_logret" if "mean_period_logret" in regime_params[regime] else "mean_hourly_logret"
        f[t] = regime_params[regime][mean_key] + a_t

    return f, sigma2


# --------------------------------------------------------------------------------------
# Layer 3 — Per-coin returns: beta*factor + CORRELATED idio GARCH + JUMPS
# --------------------------------------------------------------------------------------
def cap_idio_persistence(alpha, beta):
    persistence = alpha + beta
    if persistence > PERSISTENCE_CAP:
        ratio = PERSISTENCE_CAP / persistence
        alpha *= ratio
        beta *= ratio
    return alpha, beta


def nearest_psd_correlation(corr):
    """Eigenvalue-clip + renormalize to the nearest valid (PSD, unit-diagonal)
    correlation matrix. Needed because target_corr MINUS the beta*beta'*var_factor
    rank-1 term (see compute_idio_correlation) is not guaranteed PSD in general, even
    though the original empirical correlation matrix is."""
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals_clipped = np.clip(eigvals, 1e-8, None)
    psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    d = np.sqrt(np.diag(psd))
    psd = psd / np.outer(d, d)
    np.fill_diagonal(psd, 1.0)
    return psd


def compute_idio_correlation(calib):
    """FIX 2: the residual idio correlation matrix needed ON TOP OF the shared
    beta*factor exposure so total correlation (factor-induced + idio-induced) matches
    the calibrated empirical correlation matrix. Derivation (unconditional, annualized
    units -- correlation is unit-invariant so this holds regardless of the
    hourly/4h/period convention used, as long as it's applied consistently):
      Cov(r_i,r_j) = beta_i*beta_j*Var(factor) + Cov(idio_i,idio_j)
      => Cov(idio_i,idio_j) = Corr_target(i,j)*sqrt(Var(r_i)Var(r_j)) - beta_i*beta_j*Var(factor)
      idio_total_corr(i,j) = Cov(idio_i,idio_j) / (idio_vol_i * idio_vol_j)
    idio_total_corr is the target for the WHOLE idio channel (diffusion + jump
    combined). Jumps (Fix 1) are independent, uncorrelated noise, so only the
    DIFFUSION sub-component can actually carry correlation -- if jump_var_frac of a
    coin's idio variance is jump noise, mixing an uncorrelated jump into an
    idio_total_corr-correlated diffusion would DILUTE the realized total correlation
    by sqrt((1-jump_var_frac_i)(1-jump_var_frac_j)). So the diffusion-level
    correlation fed to Cholesky is BOOSTED by the inverse of that factor, to
    compensate in advance and land back on idio_total_corr once jumps are mixed in.
    Returns the Cholesky factor L (diffusion_corr = L @ L.T) of the PSD-regularized,
    jump-compensated result, ready to correlate a vector of iid N(0,1) draws each
    step: w_t = L @ eps_t.
    """
    betas = {c: calib["coins"][c]["beta_to_market"] for c in COINS}
    idio_vol = {c: calib["coins"][c]["idio_vol"] for c in COINS}
    jump_var_frac = {c: min(calib["coins"][c].get("jump_var_frac", 0.0), MAX_JUMP_VAR_FRAC) for c in COINS}
    target_corr = calib["correlation_matrix"]
    var_factor = calib["market_factor"]["ann_vol"] ** 2
    var_total = {c: betas[c] ** 2 * var_factor + idio_vol[c] ** 2 for c in COINS}

    n = len(COINS)
    diffusion_corr = np.eye(n)
    for i, ci in enumerate(COINS):
        for j, cj in enumerate(COINS):
            if i == j:
                continue
            cov_total_ij = target_corr[ci][cj] * np.sqrt(var_total[ci] * var_total[cj])
            cov_idio_ij = cov_total_ij - betas[ci] * betas[cj] * var_factor
            idio_total_corr_ij = cov_idio_ij / (idio_vol[ci] * idio_vol[cj])
            dilution = np.sqrt((1 - jump_var_frac[ci]) * (1 - jump_var_frac[cj]))
            diffusion_corr[i, j] = idio_total_corr_ij / dilution if dilution > 1e-6 else idio_total_corr_ij
    diffusion_corr = np.clip(diffusion_corr, -0.999, 0.999)
    np.fill_diagonal(diffusion_corr, 1.0)

    diffusion_corr_psd = nearest_psd_correlation(diffusion_corr)
    L = np.linalg.cholesky(diffusion_corr_psd)
    return diffusion_corr_psd, L


def generate_coin_returns(rng, market_factor, regime_schedule, calib, periods_per_year,
                           crash_corr_strength):
    n_periods = len(market_factor)
    is_bear = regime_schedule == "bear"
    _, L = compute_idio_correlation(calib)  # FIX 2: Cholesky of the residual idio correlation

    # -- per-coin GARCH state setup --
    state = {}
    for coin in COINS:
        c = calib["coins"][coin]
        target_idio_var = (c["idio_vol"] / np.sqrt(periods_per_year)) ** 2
        jump_prob = c.get("jump_prob", DEFAULT_JUMP_PROB)
        jump_var_frac = min(c.get("jump_var_frac", DEFAULT_JUMP_VAR_FRAC), MAX_JUMP_VAR_FRAC)
        # Variance budget split (FIX 1): total idio variance stays exactly at the
        # calibrated target regardless of jump params -- jump_var_frac of it goes to
        # jumps, the rest to the GARCH diffusion, so adding jumps never inflates ann_vol
        # past calibration.
        diffusion_var_target = target_idio_var * (1.0 - jump_var_frac)
        jump_var_target = target_idio_var * jump_var_frac
        jump_std = float(np.sqrt(jump_var_target / jump_prob)) if jump_prob > 0 else 0.0

        alpha_boost = c.get("idio_alpha_boost", DEFAULT_IDIO_ALPHA_BOOST)
        if c.get("idio_garch_ok"):
            ia = c["idio_garch"]["alpha"] * alpha_boost
        else:
            ia = 0.0
        # FIX 3: boosted alpha, THEN re-pinned to PERSISTENCE_CAP by giving the
        # remainder to beta (not cap_idio_persistence()'s ratio-preserving cap -- that
        # would partially undo the boost by shrinking alpha back down too; this matches
        # the exact per-coin search that found idio_alpha_boost, see its docstring above).
        ia = min(ia, PERSISTENCE_CAP - 1e-3)
        ib = PERSISTENCE_CAP - ia
        persistence = ia + ib
        omega = diffusion_var_target * (1 - persistence)

        state[coin] = {
            "beta": c["beta_to_market"], "sigma2": diffusion_var_target, "omega": omega,
            "alpha": ia, "beta_g": ib, "diffusion_var_target": diffusion_var_target,
            "jump_prob": jump_prob, "jump_std": jump_std,
            "idio_diffusion": np.empty(n_periods), "idio_total": np.empty(n_periods),
        }

    # -- main loop: draw correlated N(0,1) diffusion shocks, apply GARCH scaling, add jumps --
    for t in range(n_periods):
        eps_t = rng.standard_normal(len(COINS))
        w_t = L @ eps_t  # correlated unit-variance shocks, Corr(w) == idio_corr by construction
        for ci, coin in enumerate(COINS):
            s = state[coin]
            if t == 0:
                sigma2_t = s["diffusion_var_target"]
            elif t % IDIO_RESET_PERIOD == 0:
                sigma2_t = s["diffusion_var_target"]
            else:
                prev_shock = s["idio_diffusion"][t - 1]
                sigma2_t = s["omega"] + s["alpha"] * prev_shock ** 2 + s["beta_g"] * s["sigma2"]
            s["sigma2"] = sigma2_t
            diffusion_t = np.sqrt(sigma2_t) * w_t[ci]
            s["idio_diffusion"][t] = diffusion_t

            jump_t = 0.0
            if s["jump_prob"] > 0 and rng.random() < s["jump_prob"]:
                jump_t = rng.normal(0.0, s["jump_std"])
            s["idio_total"][t] = diffusion_t + jump_t

    coin_returns = {}
    for coin in COINS:
        s = state[coin]
        beta = s["beta"]
        if crash_corr_strength > 0:
            beta_eff = np.where(is_bear, beta + crash_corr_strength * (1 - beta), beta)
        else:
            beta_eff = beta
        coin_returns[coin] = beta_eff * market_factor + s["idio_total"]
    return coin_returns


# --------------------------------------------------------------------------------------
# Layer 4 — Prices, OHLC synthesis, volume placeholder, honesty guards (unchanged design)
# --------------------------------------------------------------------------------------
def returns_to_ohlcv(coin_returns, start_prices, start_time, rng):
    n_periods = len(next(iter(coin_returns.values())))
    idx = pd.date_range(start_time, periods=n_periods, freq="4h", tz="UTC")

    dfs = {}
    for coin, r in coin_returns.items():
        log_price = np.log(start_prices[coin]) + np.cumsum(r)
        close = np.exp(log_price)

        open_ = np.empty(n_periods)
        open_[0] = start_prices[coin]
        open_[1:] = close[:-1]

        hi_base = np.maximum(open_, close)
        lo_base = np.minimum(open_, close)
        wick = OHLC_WICK_FACTOR * np.abs(r) * close
        high = hi_base + wick
        low = np.maximum(lo_base - wick, lo_base * OHLC_LOW_FLOOR_FRAC)

        base_vol = VOLUME_BASE[coin]
        ret_scale = np.std(r) if np.std(r) > 0 else 1.0
        volume = base_vol * (1.0 + VOLUME_VOL_SENSITIVITY * np.abs(r) / ret_scale)
        volume *= rng.uniform(*VOLUME_NOISE_RANGE, size=n_periods)

        df = pd.DataFrame({
            "timestamp": idx, "open": open_, "high": high, "low": low,
            "close": close, "volume": volume,
        })

        vals = df[["open", "high", "low", "close", "volume"]].to_numpy()
        assert np.all(np.isfinite(vals)), f"{coin}: non-finite values generated"
        assert (df["close"] > 0).all() and (df["low"] > 0).all(), f"{coin}: non-positive price"
        assert df["timestamp"].is_monotonic_increasing, f"{coin}: timestamps not monotonic"

        dfs[coin] = df
    return dfs


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def generate_raw(seed, n_periods, regime_mode="mix", regime=None,
                  crash_corr_strength=CRASH_CORR_STRENGTH_DEFAULT, calib=None,
                  start_time=None):
    if calib is None:
        calib = load_calibration()
    periods_per_year = calib["meta"].get("periods_per_year", calib["meta"].get("hours_per_year"))

    rng = np.random.default_rng(seed)

    regime_schedule = build_regime_schedule(rng, n_periods, regime_mode, regime, calib)
    market_factor, _ = generate_market_factor(rng, n_periods, regime_schedule, calib, periods_per_year)
    coin_returns = generate_coin_returns(rng, market_factor, regime_schedule, calib,
                                          periods_per_year, crash_corr_strength)

    start_prices = load_start_prices(calib)
    if start_time is None:
        start_time = pd.Timestamp(calib["meta"]["window_start"])

    dfs = returns_to_ohlcv(coin_returns, start_prices, start_time, rng)
    return dfs, regime_schedule, market_factor


def generate(seed, n_periods, regime_mode="mix", regime=None,
             crash_corr_strength=CRASH_CORR_STRENGTH_DEFAULT, out_dir="data/synthetic_4h",
             calib=None, write_csv=True):
    dfs, regime_schedule, market_factor = generate_raw(
        seed, n_periods, regime_mode, regime, crash_corr_strength, calib
    )

    if write_csv:
        os.makedirs(out_dir, exist_ok=True)
        for coin, df in dfs.items():
            df.to_csv(os.path.join(out_dir, f"{coin}USDT_4h_synthetic.csv"), index=False)

    combined = pd.concat(
        [df.assign(coin=coin)[["timestamp", "coin", "open", "high", "low", "close", "volume"]]
         for coin, df in dfs.items()],
        ignore_index=True,
    )
    return combined


if __name__ == "__main__":
    print("Demo: generating a 1-year 'mix' world (seed=42) -> data/synthetic_4h/")
    demo = generate(seed=42, n_periods=2190, regime_mode="mix", out_dir="data/synthetic_4h")
    n_coins = demo["coin"].nunique()
    print(f"Wrote {n_coins} coins x {len(demo) // n_coins} 4h-bars to data/synthetic_4h/")
    print(demo.head())
