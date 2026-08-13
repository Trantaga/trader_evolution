"""
sensors.py — LOOK-AHEAD-SAFE sensor construction for the trading agent's brain.

At decision time for hour t, a sensor may use price data ONLY through the previous
closed bar (index <= t-1). Nothing here may touch open/high/low/close of hour t or
later. This is enforced structurally: all price-derived sensors are built from a
`.shift(1)` of the close series (so index t always maps to close[t-1]) composed with
causal pandas rolling/diff operations, which by construction only ever read indices
<= t. self_test.py's LOOK-AHEAD TEST empirically double-checks this by mutating all
bars strictly after a decision point and confirming nothing before it changes.

Sensor vector layout (fixed order — this IS the SENSOR node set), exposed via
SENSOR_NAMES / N_SENSORS:
  for each coin in COINS (8 sensors each, in this order):
    ret_1h, ret_6h, ret_24h, ret_168h   -- normalized log returns, momentum at 4 scales
    price_vs_sma168                     -- close[t-1]/SMA_168(close) - 1, normalized
    vol_24h                             -- std of last 24 log returns, normalized
    holding_frac                        -- fraction of portfolio value in this coin, [0,1]
    unrealized_pl                       -- mark vs avg entry using close[t-1], normalized
  then 2 global sensors:
    cash_frac                           -- fraction of portfolio value in cash, [0,1]
    bias                                -- constant 1.0
"""

import numpy as np
import pandas as pd

COINS = ["BTC", "ETH", "BNB", "MATIC", "SOL"]

RETURN_HORIZONS = [1, 6, 24, 168]   # hours; multi-scale momentum
SMA_WINDOW = 168                     # hours
VOL_WINDOW = 24                      # hours

# First hour index t for which ALL sensors are computable from data <= t-1.
# Binding constraint is the 168h return / SMA window (needs index t-1-168 >= 0 => t >= 169).
FIRST_DECIDABLE_T = max(max(RETURN_HORIZONS), SMA_WINDOW, VOL_WINDOW) + 1

# Normalization constants (deliberately NOT calibrated to any specific coin -- a single
# fixed reference scale so the tanh network sees comparably-sized inputs across coins and
# horizons; tunable, documented here rather than derived from calibration.json since this
# is about network conditioning, not measuring the real market).
NORM_HOURLY_VOL = 0.02    # rough order-of-magnitude hourly vol used only for sensor scaling
UNREALIZED_PL_NORM = 0.5  # a +/-50% unrealized move saturates the tanh squash

PRICE_SENSOR_NAMES = ["ret_1h", "ret_6h", "ret_24h", "ret_168h", "price_vs_sma168", "vol_24h"]
PORTFOLIO_SENSOR_NAMES = ["holding_frac", "unrealized_pl"]
GLOBAL_SENSOR_NAMES = ["cash_frac", "bias"]

SENSOR_NAMES = [
    f"{coin}_{name}" for coin in COINS for name in (PRICE_SENSOR_NAMES + PORTFOLIO_SENSOR_NAMES)
] + GLOBAL_SENSOR_NAMES
N_SENSORS = len(SENSOR_NAMES)
BIAS_SENSOR_ID = N_SENSORS - 1  # last sensor is always the constant-1.0 bias node


def precompute_price_sensors(closes_by_coin):
    """Vectorized, look-ahead-safe precompute of the price-derived (non-portfolio)
    sensors for every hour t, for every coin. This is a pure function of price history
    -- independent of trading decisions -- so it is computed ONCE per world rather than
    inside the per-hour simulator loop. This is the main vectorization opportunity in
    this module; the portfolio-state sensors below still must be computed live each hour
    since they depend on the agent's own trades (cash/position state).

    Returns dict[coin] -> dict[name] -> np.array(T); NaN before FIRST_DECIDABLE_T (the
    simulator never reads those indices).

    Look-ahead safety: every array below is built from `shifted = close.shift(1)`
    (shifted[t] == close[t-1]) composed with .rolling()/.diff(), both of which are
    causal by construction (value at t depends only on indices <= t of their input).
    So price_sensors[coin][name][t] is, by construction, a pure function of
    close[<=t-1] regardless of what any index >= t contains.
    """
    out = {}
    for coin, close_arr in closes_by_coin.items():
        close = pd.Series(np.asarray(close_arr, dtype=float))
        shifted = close.shift(1)  # shifted[t] == close[t-1]

        rets = {}
        for h in RETURN_HORIZONS:
            raw = np.log(shifted) - np.log(shifted.shift(h))
            rets[h] = np.tanh(raw / (NORM_HOURLY_VOL * np.sqrt(h)))

        sma = shifted.rolling(SMA_WINDOW).mean()
        price_vs_sma_raw = shifted / sma - 1
        price_vs_sma = np.tanh(price_vs_sma_raw / (NORM_HOURLY_VOL * np.sqrt(SMA_WINDOW)))

        log_ret = np.log(close).diff()       # log_ret[t] = return realized DURING hour t
        log_ret_shifted = log_ret.shift(1)    # log_ret_shifted[t] = return realized during t-1
        vol24_raw = log_ret_shifted.rolling(VOL_WINDOW).std()
        vol24 = np.tanh(vol24_raw / NORM_HOURLY_VOL)

        out[coin] = {
            "ret_1h": rets[1].to_numpy(), "ret_6h": rets[6].to_numpy(),
            "ret_24h": rets[24].to_numpy(), "ret_168h": rets[168].to_numpy(),
            "price_vs_sma168": price_vs_sma.to_numpy(), "vol_24h": vol24.to_numpy(),
        }
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
    vec[i] = 1.0  # bias
    i += 1

    assert i == N_SENSORS
    return vec
