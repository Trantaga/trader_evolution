"""
generator.py — synthetic hourly OHLCV generator driven by config/calibration.json.

Four stacked layers:
  1. Regime      — "pure" (whole run is one regime) or "mix" (regimes switch over time).
  2. Market factor — GARCH(1,1) volatility engine, one shared shock series.
  3. Coins       — beta * factor + idiosyncratic GARCH(1,1)-t noise, per coin (driven by
                   calibration's idio-RESIDUAL GARCH fit, with a persistence cap + periodic
                   variance reset — see PERSISTENCE_CAP / IDIO_RESET_PERIOD below).
  4. Prices/OHLC — compound log returns into prices, synthesize OHLC + a volume placeholder.

Same seed -> identical output (reproducible). Different seed -> a fresh, statistically
equivalent world — this is how later steps (evolution) get new data every generation.

Run: python3 generator.py   (writes a 1-year "mix" demo world to data/synthetic/)
Or:  from generator import generate; generate(seed=0, n_hours=8760, regime_mode="pure",
     regime="bull")
"""

import json
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# CONFIG — tunable constants that are NOT in calibration.json (deliberately un-calibrated,
# clearly labeled). Everything else the generator uses is read from calibration.json.
# --------------------------------------------------------------------------------------
CALIBRATION_PATH = "config/calibration.json"
COINS = ["BTC", "ETH", "BNB", "MATIC", "SOL"]
REGIME_NAMES = ["bull", "bear", "sideways"]

# Layer 1 (mix mode): mean regime duration. Deliberately NOT the empirical transition
# matrix from calibration.json (~1-day implied durations, too chattery) — a tunable,
# much longer default so fitness evaluation sees sustained regimes.
MIX_MEAN_DURATION_HOURS = 672  # ~4 weeks

# Layer 3: crash-correlation knob. Regime-conditional correlations were never measured
# by the calibrator, so this is an explicit, off-by-default guess: beta_eff = beta +
# strength*(1-beta) during bear regimes, pulling coins toward moving together in a crash.
CRASH_CORR_STRENGTH_DEFAULT = 0.0  # in [0, 1]

# Layer 3: idio-channel safeguards for calibration.json's idio_garch (fit on the
# idio RESIDUAL, not borrowed from the total-return GARCH — see calibrate.py's
# compute_idio_garch). That fit lands at/above the unit-root boundary for 4/5 coins
# (a genuine QMLE-near-IGARCH result, not a bug), so it needs both of these:
PERSISTENCE_CAP = 0.995   # cap alpha+beta (ratio-preserving) so omega>0 and the path can't explode
IDIO_RESET_PERIOD = 336   # hours (~2 weeks); periodically re-anchor sigma_idio^2 to its target —
                           # for a near-IGARCH process E[sigma^2_t] drifts (usually downward) from
                           # target without this, same idea as the market factor's regime-switch reset

# Layer 4: intra-hour OHLC shape (un-calibrated approximation). Extra high/low "wick"
# beyond the close-to-close move, as a fraction of |hourly log return| * price.
OHLC_WICK_FACTOR = 0.5
OHLC_LOW_FLOOR_FRAC = 0.5  # never let the low wick eat more than half of (open,close) range base

# Layer 4: volume placeholder. NOT calibrated — just a rough per-coin base level, scaled
# up when |return| is large (crude proxy for "volume spikes with volatility"), so the
# column exists and isn't nonsensical. Do not use this for anything quantitative.
VOLUME_BASE = {"BTC": 5_000, "ETH": 80_000, "BNB": 20_000, "MATIC": 5_000_000, "SOL": 300_000}
VOLUME_VOL_SENSITIVITY = 3.0
VOLUME_NOISE_RANGE = (0.7, 1.3)  # multiplicative uniform noise on top of the vol-linked base

# Fallback start-price anchors (real close price at calibration window_start, hardcoded
# in case data/raw/ isn't available at generation time). Only used as a scale anchor —
# never fed back into the stylized facts.
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
    """Anchor each coin's synthetic price path to its real close at the calibration
    window's start hour (read from data/raw/ if present), falling back to a hardcoded
    constant otherwise. Purely a scale anchor — has no effect on returns/vol/etc."""
    window_start = pd.Timestamp(calib["meta"]["window_start"])
    prices = {}
    for coin in COINS:
        path = f"data/raw/{coin}USDT_1h_full.csv"
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
# Layer 1 — Regime schedule
# --------------------------------------------------------------------------------------
def build_regime_schedule(rng, n_hours, regime_mode, regime, calib):
    if regime_mode == "pure":
        if regime not in REGIME_NAMES:
            raise ValueError(f"regime_mode='pure' requires regime in {REGIME_NAMES}, got {regime!r}")
        return np.full(n_hours, regime, dtype=object)

    if regime_mode != "mix":
        raise ValueError(f"regime_mode must be 'pure' or 'mix', got {regime_mode!r}")

    shares = {r: calib["regimes"]["params"][r]["share_of_time"] for r in REGIME_NAMES}
    weights = np.array([shares[r] for r in REGIME_NAMES])
    weights = weights / weights.sum()

    schedule = np.empty(n_hours, dtype=object)
    current = rng.choice(REGIME_NAMES, p=weights)
    t = 0
    while t < n_hours:
        duration = int(max(1, round(rng.exponential(MIX_MEAN_DURATION_HOURS))))
        duration = min(duration, n_hours - t)
        schedule[t:t + duration] = current
        t += duration
        if t >= n_hours:
            break
        # Step change: next regime sampled among the *other* two, weighted by their
        # calibration shares (not the empirical, too-chattery transition matrix).
        others = [r for r in REGIME_NAMES if r != current]
        w = np.array([shares[r] for r in others])
        w = w / w.sum()
        current = rng.choice(others, p=w)
    return schedule


# --------------------------------------------------------------------------------------
# Layer 2 — Market factor (GARCH(1,1) volatility engine, variance-targeted per regime)
# --------------------------------------------------------------------------------------
def generate_market_factor(rng, n_hours, regime_schedule, calib, hours_per_year):
    garch = calib["market_factor"]["garch"]
    alpha, beta = garch["alpha"], garch["beta"]
    dof = calib["market_factor"]["student_t_dof"]
    regime_params = calib["regimes"]["params"]

    # Pre-draw standardized Student-t innovations (unit variance): a raw t(dof) draw has
    # variance dof/(dof-2), so scale by sqrt((dof-2)/dof) to normalize it to 1.
    z = rng.standard_t(dof, size=n_hours) * np.sqrt((dof - 2) / dof)

    f = np.empty(n_hours)       # market factor hourly log return
    sigma2 = np.empty(n_hours)  # GARCH conditional variance
    a = np.empty(n_hours)       # shock a_t = sigma_t * z_t

    prev_regime = None
    omega = None
    for t in range(n_hours):
        regime = regime_schedule[t]
        if regime != prev_regime:
            # *** CRITICAL: variance targeting, not the fitted omega ***
            # The fitted GARCH is near-unit-root (alpha+beta ~ 1), so its implied
            # unconditional variance omega/(1-alpha-beta) is inflated/unstable. Pin the
            # unconditional variance to this regime's calibrated target instead; only
            # omega is recomputed — alpha/beta (the clustering shape) are kept as-is.
            ann_vol_target = regime_params[regime]["ann_vol"]
            target_var = (ann_vol_target / np.sqrt(hours_per_year)) ** 2
            omega = target_var * (1 - alpha - beta)
            sigma2_t = target_var  # reset conditional variance to the target on switch
            prev_regime = regime
        else:
            sigma2_t = omega + alpha * a[t - 1] ** 2 + beta * sigma2[t - 1]

        sigma2[t] = sigma2_t
        a_t = np.sqrt(sigma2_t) * z[t]
        a[t] = a_t
        f[t] = regime_params[regime]["mean_hourly_logret"] + a_t

    return f, sigma2


# --------------------------------------------------------------------------------------
# Layer 3 — Per-coin returns: beta * factor + idiosyncratic GARCH(1,1)-t noise
# --------------------------------------------------------------------------------------
def cap_idio_persistence(alpha, beta):
    """Cap alpha+beta at PERSISTENCE_CAP, preserving their ratio (so the clustering shape
    is otherwise unchanged). Needed because calibration.json's idio_garch is fit on the
    idio residual itself and lands at/above the unit-root boundary for 4/5 coins (BTC is
    slightly above 1.0) — a genuine QMLE-near-IGARCH result, not a calibration bug. Pulled
    out as its own function so validate_generator.py can report the capped value per coin
    using the exact same math, not a re-derivation.
    """
    persistence = alpha + beta
    if persistence > PERSISTENCE_CAP:
        ratio = PERSISTENCE_CAP / persistence
        alpha *= ratio
        beta *= ratio
    return alpha, beta


def generate_idio_garch(rng, n_hours, alpha, beta, dof, target_var, reset_period=IDIO_RESET_PERIOD):
    """Per-coin idiosyncratic GARCH(1,1)-t process, driven by the coin's OWN idio-RESIDUAL
    GARCH shape (calibrated on return-minus-beta*factor — see calibrate.py's
    compute_idio_garch), not borrowed from the total-return fit as in v1.5. Two
    safeguards on top of it:
      1) cap_idio_persistence(): keeps alpha+beta < 1 so omega > 0 and the recursion
         can't explode.
      2) periodic variance reset every `reset_period` hours (default IDIO_RESET_PERIOD):
         for a near-IGARCH process (which this is, per (1)), E[sigma^2_t] only stays at
         its target if periodically re-anchored — left alone, realized variance drifts
         (typically downward) over a long run. This is the actual fix for v1.5's
         correlation-inflation bug: it mirrors the reset that already protects the market
         factor (Layer 2), just on a fixed period instead of a regime switch, since the
         idio target is constant across regimes (we never measured regime-conditional
         idio vol). `reset_period` is exposed (not hardcoded to the module constant) so
         validate_generator.py can run a with/without-reset counterfactual using this
         exact same function, rather than duplicating the recursion.
    """
    alpha, beta = cap_idio_persistence(alpha, beta)
    persistence = alpha + beta
    omega = target_var * (1 - persistence)

    z = rng.standard_t(dof, size=n_hours) * np.sqrt((dof - 2) / dof)
    idio = np.empty(n_hours)
    sigma2 = np.empty(n_hours)

    sigma2[0] = target_var  # initialize at the target, same convention as Layer 2
    idio[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, n_hours):
        if t % reset_period == 0:
            sigma2[t] = target_var  # periodic anchor -- see rationale in the docstring
        else:
            sigma2[t] = omega + alpha * idio[t - 1] ** 2 + beta * sigma2[t - 1]
        idio[t] = np.sqrt(sigma2[t]) * z[t]
    return idio


def generate_coin_returns(rng, market_factor, regime_schedule, calib, hours_per_year,
                           crash_corr_strength):
    n_hours = len(market_factor)
    is_bear = regime_schedule == "bear"

    coin_returns = {}
    for coin in COINS:
        c = calib["coins"][coin]
        beta = c["beta_to_market"]
        dof = c["idio_student_t_dof"]
        target_idio_var = (c["idio_vol"] / np.sqrt(hours_per_year)) ** 2

        if c.get("idio_garch_ok"):
            ia, ib = c["idio_garch"]["alpha"], c["idio_garch"]["beta"]
        else:
            # Fallback: flat-vol idio (alpha=beta=0 -> generate_idio_garch reduces to
            # constant sigma_idio^2 = target_var) if the calibrator's residual GARCH fit
            # never converged for this coin.
            ia, ib = 0.0, 0.0

        idio = generate_idio_garch(rng, n_hours, ia, ib, dof, target_idio_var)

        if crash_corr_strength > 0:
            beta_eff = np.where(is_bear, beta + crash_corr_strength * (1 - beta), beta)
        else:
            beta_eff = beta

        coin_returns[coin] = beta_eff * market_factor + idio
    return coin_returns


# --------------------------------------------------------------------------------------
# Layer 4 — Prices, OHLC synthesis, volume placeholder, honesty guards
# --------------------------------------------------------------------------------------
def returns_to_ohlcv(coin_returns, start_prices, start_time, rng):
    n_hours = len(next(iter(coin_returns.values())))
    idx = pd.date_range(start_time, periods=n_hours, freq="h", tz="UTC")

    dfs = {}
    for coin, r in coin_returns.items():
        # Generate in log-space so price is strictly positive by construction.
        log_price = np.log(start_prices[coin]) + np.cumsum(r)
        close = np.exp(log_price)

        open_ = np.empty(n_hours)
        open_[0] = start_prices[coin]
        open_[1:] = close[:-1]

        hi_base = np.maximum(open_, close)
        lo_base = np.minimum(open_, close)
        wick = OHLC_WICK_FACTOR * np.abs(r) * close  # simple approximation, not calibrated
        high = hi_base + wick
        low = np.maximum(lo_base - wick, lo_base * OHLC_LOW_FLOOR_FRAC)

        base_vol = VOLUME_BASE[coin]
        ret_scale = np.std(r) if np.std(r) > 0 else 1.0
        volume = base_vol * (1.0 + VOLUME_VOL_SENSITIVITY * np.abs(r) / ret_scale)
        volume *= rng.uniform(*VOLUME_NOISE_RANGE, size=n_hours)

        df = pd.DataFrame({
            "timestamp": idx, "open": open_, "high": high, "low": low,
            "close": close, "volume": volume,
        })

        # Honesty guards.
        vals = df[["open", "high", "low", "close", "volume"]].to_numpy()
        assert np.all(np.isfinite(vals)), f"{coin}: non-finite values generated"
        assert (df["close"] > 0).all() and (df["low"] > 0).all(), f"{coin}: non-positive price"
        assert df["timestamp"].is_monotonic_increasing, f"{coin}: timestamps not monotonic"

        dfs[coin] = df
    return dfs


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def generate_raw(seed, n_hours, regime_mode="mix", regime=None,
                  crash_corr_strength=CRASH_CORR_STRENGTH_DEFAULT, calib=None,
                  start_time=None):
    """Core simulation. Returns (dfs, regime_schedule, market_factor):
      dfs             -- {coin: DataFrame[timestamp,open,high,low,close,volume]}
      regime_schedule -- length-n_hours array of regime labels
      market_factor   -- length-n_hours array of the market factor's hourly log returns
    Exposed (not just `generate`) so validate_generator.py can inspect the regime
    schedule and factor series directly, without re-parsing written CSVs.
    """
    if calib is None:
        calib = load_calibration()
    hours_per_year = calib["meta"]["hours_per_year"]

    rng = np.random.default_rng(seed)

    regime_schedule = build_regime_schedule(rng, n_hours, regime_mode, regime, calib)
    market_factor, _ = generate_market_factor(rng, n_hours, regime_schedule, calib, hours_per_year)
    coin_returns = generate_coin_returns(rng, market_factor, regime_schedule, calib,
                                          hours_per_year, crash_corr_strength)

    start_prices = load_start_prices(calib)
    if start_time is None:
        start_time = pd.Timestamp(calib["meta"]["window_start"])

    dfs = returns_to_ohlcv(coin_returns, start_prices, start_time, rng)
    return dfs, regime_schedule, market_factor


def generate(seed, n_hours, regime_mode="mix", regime=None,
             crash_corr_strength=CRASH_CORR_STRENGTH_DEFAULT, out_dir="data/synthetic",
             calib=None, write_csv=True):
    """Generate one synthetic world. Writes 5 per-coin CSVs (real Binance format:
    timestamp,open,high,low,close,volume) to out_dir, and returns a single combined
    long-format DataFrame (columns: timestamp, coin, open, high, low, close, volume).
    Same seed -> identical world; different seed -> a fresh statistically-equivalent one.
    """
    dfs, regime_schedule, market_factor = generate_raw(
        seed, n_hours, regime_mode, regime, crash_corr_strength, calib
    )

    if write_csv:
        os.makedirs(out_dir, exist_ok=True)
        for coin, df in dfs.items():
            df.to_csv(os.path.join(out_dir, f"{coin}USDT_1h_synthetic.csv"), index=False)

    combined = pd.concat(
        [df.assign(coin=coin)[["timestamp", "coin", "open", "high", "low", "close", "volume"]]
         for coin, df in dfs.items()],
        ignore_index=True,
    )
    return combined


if __name__ == "__main__":
    print("Demo: generating a 1-year 'mix' world (seed=42) -> data/synthetic/")
    demo = generate(seed=42, n_hours=8760, regime_mode="mix", out_dir="data/synthetic")
    n_coins = demo["coin"].nunique()
    print(f"Wrote {n_coins} coins x {len(demo) // n_coins} hours to data/synthetic/")
    print(demo.head())
