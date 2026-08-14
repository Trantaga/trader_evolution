"""
sensors.py — LOOK-AHEAD-SAFE sensor construction for the trading agent's brain.

At decision time for hour t, a sensor may use price data ONLY through the previous
closed bar (index <= t-1). Nothing here may touch open/high/low/close of hour t or
later. This is enforced structurally: all price-derived sensors are built from a
`.shift(1)` of the close series (so index t always maps to close[t-1]) composed with
causal pandas rolling/diff operations, which by construction only ever read indices
<= t. self_test.py's LOOK-AHEAD TEST empirically double-checks this by mutating all
bars strictly after a decision point and confirming nothing before it changes. The
new cross-sectional sensors below (rel_strength_short/medium, breadth) are causal in
the SAME sense: at hour t they only combine each coin's OWN <=t-1 return with every
OTHER coin's <=t-1 return -- comparing coins to each other at a fixed past instant,
never reading ahead in time for any coin.

real-02 CHANGE (v2 sensor layout -- NOT compatible with real-01 genomes/checkpoints,
which were bred against the v1 42-sensor layout below; node ids shift because the new
per-coin sensors are inserted before holding_frac/unrealized_pl in each coin's block,
and a new global sensor is inserted before bias. Do not attempt to load a real-01
genome against this layout -- its SENSOR-sourced connection ids would silently point
at different, wrong sensors. real-02 starts from a fresh random-genome population.):
  - added 3 per-coin sensors: rel_strength_short, rel_strength_medium, mom_long
  - added 1 global sensor: breadth
  - N_SENSORS: 42 -> 58 (5 coins * (8+3) + (2+1) = 55+3)
Motivation: real-01's champion could defend capital (lowest drawdown of all held-out
baselines) but couldn't pick winners (lost money in a falling market where one of five
coins was actually up, because every sensor described a coin in ISOLATION -- nothing
told the network which coin was relatively strong). These sensors are cross-sectional:
they compare each coin against the other 4 at the same past hour.

Sensor vector layout (fixed order — this IS the SENSOR node set), exposed via
SENSOR_NAMES / N_SENSORS:
  for each coin in COINS (11 sensors each, in this order):
    ret_1h, ret_6h, ret_24h, ret_168h   -- normalized log returns, momentum at 4 scales
    price_vs_sma168                     -- close[t-1]/SMA_168(close) - 1, normalized
    vol_24h                             -- std of last 24 log returns, normalized
    rel_strength_short                  -- cross-sectional z-score of ret_24h vs the
                                            other 4 coins at t-1, tanh-squashed
    rel_strength_medium                 -- cross-sectional z-score of ret_168h vs the
                                            other 4 coins at t-1, tanh-squashed
    mom_long                            -- 720h (~30d) log return, normalized/tanh like
                                            the other per-coin momentum sensors
    holding_frac                        -- fraction of portfolio value in this coin, [0,1]
    unrealized_pl                       -- mark vs avg entry using close[t-1], normalized
  then 3 global sensors:
    cash_frac                           -- fraction of portfolio value in cash, [0,1]
    breadth                             -- 2*(fraction of the 5 coins with positive
                                            24h return at t-1) - 1, in [-1,1]
    bias                                -- constant 1.0
"""

import numpy as np
import pandas as pd

COINS = ["BTC", "ETH", "BNB", "MATIC", "SOL"]

RETURN_HORIZONS = [1, 6, 24, 168]   # hours; multi-scale per-coin momentum
MOM_LONG_HORIZON = 720               # hours (~30d); same normalization style as RETURN_HORIZONS
SMA_WINDOW = 168                     # hours
VOL_WINDOW = 24                      # hours

# First hour index t for which ALL sensors are computable from data <= t-1. Binding
# constraint is now the 720h mom_long return (needs index t-1-720 >= 0 => t >= 721),
# up from 169 in the v1 (pre-real-02) sensor set.
FIRST_DECIDABLE_T = max(max(RETURN_HORIZONS), MOM_LONG_HORIZON, SMA_WINDOW, VOL_WINDOW) + 1

# Normalization constants (deliberately NOT calibrated to any specific coin -- a single
# fixed reference scale so the tanh network sees comparably-sized inputs across coins and
# horizons; tunable, documented here rather than derived from calibration.json since this
# is about network conditioning, not measuring the real market).
NORM_HOURLY_VOL = 0.02    # rough order-of-magnitude hourly vol used only for sensor scaling
UNREALIZED_PL_NORM = 0.5  # a +/-50% unrealized move saturates the tanh squash

# Cross-sectional sensors (rel_strength_short/medium, breadth) compare each coin's
# 24h/168h return against the OTHER 4 coins' returns at the SAME past hour t-1 -- never
# across time, so this adds no look-ahead. eps floors the z-score's std denominator so a
# freak hour where all 5 coins have (numerically) identical returns doesn't divide by ~0.
CROSS_SECTIONAL_EPS = 1e-8

GLOBAL_KEY = "_global"  # precompute_price_sensors' key for the (non-per-coin) breadth array

PRICE_SENSOR_NAMES = [
    "ret_1h", "ret_6h", "ret_24h", "ret_168h", "price_vs_sma168", "vol_24h",
    "rel_strength_short", "rel_strength_medium", "mom_long",
]
PORTFOLIO_SENSOR_NAMES = ["holding_frac", "unrealized_pl"]
GLOBAL_SENSOR_NAMES = ["cash_frac", "breadth", "bias"]

SENSOR_NAMES = [
    f"{coin}_{name}" for coin in COINS for name in (PRICE_SENSOR_NAMES + PORTFOLIO_SENSOR_NAMES)
] + GLOBAL_SENSOR_NAMES
N_SENSORS = len(SENSOR_NAMES)
BIAS_SENSOR_ID = N_SENSORS - 1  # last sensor is always the constant-1.0 bias node


def _cross_sectional_z(raw_by_coin):
    """raw_by_coin: dict[coin] -> pandas Series of a per-coin raw (pre-tanh) return.
    Returns dict[coin] -> np.array, each coin's tanh-squashed z-score against the other
    4 coins AT THE SAME t (axis=0 stacks coins, so mean/std are taken cross-sectionally
    at each t independently -- never across time)."""
    mat = np.stack([raw_by_coin[c].to_numpy() for c in COINS], axis=0)  # (5, T)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    z = (mat - mean[None, :]) / (std[None, :] + CROSS_SECTIONAL_EPS)
    return {c: np.tanh(z[i]) for i, c in enumerate(COINS)}


def precompute_price_sensors(closes_by_coin):
    """Vectorized, look-ahead-safe precompute of the price-derived (non-portfolio)
    sensors for every hour t, for every coin, PLUS the global breadth sensor. This is a
    pure function of price history -- independent of trading decisions -- so it is
    computed ONCE per world rather than inside the per-hour simulator loop.

    Returns dict[coin] -> dict[name] -> np.array(T) for every coin in COINS, PLUS
    dict[GLOBAL_KEY] -> {"breadth": np.array(T)}. NaN before FIRST_DECIDABLE_T (the
    simulator never reads those indices).

    Look-ahead safety: every per-coin array is built from `shifted = close.shift(1)`
    (shifted[t] == close[t-1]) composed with .rolling()/.diff(), both causal by
    construction. The cross-sectional sensors (rel_strength_short/medium, breadth)
    combine several coins' raw returns at the SAME t -- comparing across coins, never
    across time -- so they inherit the same look-ahead safety from their per-coin inputs.
    """
    raw_by_h = {h: {} for h in RETURN_HORIZONS + [MOM_LONG_HORIZON]}
    price_vs_sma = {}
    vol24 = {}

    for coin in COINS:
        close = pd.Series(np.asarray(closes_by_coin[coin], dtype=float))
        shifted = close.shift(1)  # shifted[t] == close[t-1]

        for h in RETURN_HORIZONS + [MOM_LONG_HORIZON]:
            raw_by_h[h][coin] = np.log(shifted) - np.log(shifted.shift(h))

        sma = shifted.rolling(SMA_WINDOW).mean()
        price_vs_sma_raw = shifted / sma - 1
        price_vs_sma[coin] = np.tanh(price_vs_sma_raw / (NORM_HOURLY_VOL * np.sqrt(SMA_WINDOW)))

        log_ret = np.log(close).diff()       # log_ret[t] = return realized DURING hour t
        log_ret_shifted = log_ret.shift(1)    # log_ret_shifted[t] = return realized during t-1
        vol24_raw = log_ret_shifted.rolling(VOL_WINDOW).std()
        vol24[coin] = np.tanh(vol24_raw / NORM_HOURLY_VOL)

    rel_strength_short = _cross_sectional_z(raw_by_h[24])
    rel_strength_medium = _cross_sectional_z(raw_by_h[168])

    ret24_mat = np.stack([raw_by_h[24][c].to_numpy() for c in COINS], axis=0)  # (5, T)
    breadth = 2.0 * np.mean(ret24_mat > 0, axis=0) - 1.0

    out = {}
    for coin in COINS:
        rets = {h: np.tanh(raw_by_h[h][coin] / (NORM_HOURLY_VOL * np.sqrt(h))) for h in RETURN_HORIZONS}
        mom_long = np.tanh(raw_by_h[MOM_LONG_HORIZON][coin]
                            / (NORM_HOURLY_VOL * np.sqrt(MOM_LONG_HORIZON)))
        out[coin] = {
            "ret_1h": rets[1].to_numpy(), "ret_6h": rets[6].to_numpy(),
            "ret_24h": rets[24].to_numpy(), "ret_168h": rets[168].to_numpy(),
            "price_vs_sma168": price_vs_sma[coin].to_numpy(), "vol_24h": vol24[coin].to_numpy(),
            "rel_strength_short": rel_strength_short[coin],
            "rel_strength_medium": rel_strength_medium[coin],
            "mom_long": mom_long.to_numpy(),
        }
    out[GLOBAL_KEY] = {"breadth": breadth}
    return out


def compute_sensor_vector(t, price_sensors, closes_by_coin, cash, positions):
    """Assemble the full sensor vector for decision hour t, in the fixed SENSOR_NAMES
    order. `closes_by_coin[coin][t-1]` is used ONLY to mark the portfolio (holding_frac,
    unrealized_pl, cash_frac) -- still index <= t-1, still look-ahead safe.

    positions: dict[coin] -> {"qty": float, "avg_entry": float or None}.
    """
    mark_prices = {coin: closes_by_coin[coin][t - 1] for coin in COINS}
    position_values = {coin: positions[coin]["qty"] * mark_prices[coin] for coin in COINS}
    total_equity = cash + sum(position_values.values())

    vec = np.empty(N_SENSORS)
    i = 0
    for coin in COINS:
        ps = price_sensors[coin]
        vec[i] = ps["ret_1h"][t]; i += 1
        vec[i] = ps["ret_6h"][t]; i += 1
        vec[i] = ps["ret_24h"][t]; i += 1
        vec[i] = ps["ret_168h"][t]; i += 1
        vec[i] = ps["price_vs_sma168"][t]; i += 1
        vec[i] = ps["vol_24h"][t]; i += 1
        vec[i] = ps["rel_strength_short"][t]; i += 1
        vec[i] = ps["rel_strength_medium"][t]; i += 1
        vec[i] = ps["mom_long"][t]; i += 1

        vec[i] = position_values[coin] / total_equity if total_equity > 0 else 0.0
        i += 1

        qty = positions[coin]["qty"]
        avg_entry = positions[coin]["avg_entry"]
        if qty > 0 and avg_entry:
            unrealized = (mark_prices[coin] - avg_entry) / avg_entry
            vec[i] = np.tanh(unrealized / UNREALIZED_PL_NORM)
        else:
            vec[i] = 0.0
        i += 1

    vec[i] = cash / total_equity if total_equity > 0 else 0.0
    i += 1
    vec[i] = price_sensors[GLOBAL_KEY]["breadth"][t]
    i += 1
    vec[i] = 1.0  # bias
    i += 1

    assert i == N_SENSORS
    return vec
