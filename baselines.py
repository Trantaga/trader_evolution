"""
baselines.py — trivial, fixed-parameter (NO tuning, NO evolution) rule-based trading
strategies, run under the EXACT SAME execution mechanics as the champion (simulator.py):
decide on data <= t-1, fill at open[t], SELLS before BUYS, equal-split of free cash
across simultaneous buy signals, same FEE_RATE/SLIPPAGE/STARTING_CAPITAL/CASH_EPSILON.
Built to answer one question: does the evolved champion beat DUMB, mechanical rules a
10-line script could implement? See champion_selection.py/evaluate_champion.py for the
champion side of that comparison; this module only supplies the baselines.

backtest_rule() is the shared engine (mirrors simulator.simulate()'s trading-mechanics
loop line-for-line, just swapping the NN forward pass for an arbitrary decide_fn) so
every rule below is executed under IDENTICAL mechanics to the champion -- apples-to-apples.

Each rule's signal is precomputed ONCE per world via a causal shift(1)+rolling pipeline
(same look-ahead-safe pattern sensors.py uses: shifted[t] == close[t-1], so a rolling
window ending at index t is a pure function of data <= t-1).

FIXED PARAMETERS (documented here, never tuned to any data):
  MA_CROSSOVER_PARAMS = [(24, 168), (168, 720)]   -- (fast_hours, slow_hours)
  PRICE_VS_SMA_WINDOW = 720                        -- hours (~30d)
  MOMENTUM_LOOKBACK_HOURS = 720                     -- hours (~30d trailing return)
  MOMENTUM_REBALANCE_HOURS = 168                    -- hours (weekly)
  RSI_PERIOD = 14                                   -- hours, simple (non-Wilder) rolling mean
  RSI_BUY_BELOW = 30.0
  RSI_SELL_ABOVE = 70.0
"""

import numpy as np
import pandas as pd

from sensors import COINS, FIRST_DECIDABLE_T
from simulator import Trade, FEE_RATE, SLIPPAGE, STARTING_CAPITAL, CASH_EPSILON, _extract_aligned_arrays

MA_CROSSOVER_PARAMS = [(24, 168), (168, 720)]
PRICE_VS_SMA_WINDOW = 720
MOMENTUM_LOOKBACK_HOURS = 720
MOMENTUM_REBALANCE_HOURS = 168
RSI_PERIOD = 14
RSI_BUY_BELOW = 30.0
RSI_SELL_ABOVE = 70.0


# --------------------------------------------------------------------------------------
# Shared execution engine -- mirrors simulator.simulate()'s mechanics exactly, with an
# arbitrary decide_fn(t, closes, opens, positions) -> (buy_coins, sell_coins) in place of
# the NN forward pass.
# --------------------------------------------------------------------------------------
def backtest_rule(world_ohlcv, decide_fn, first_decidable_t=FIRST_DECIDABLE_T):
    T, opens, closes = _extract_aligned_arrays(world_ohlcv)
    cash = STARTING_CAPITAL
    positions = {c: {"qty": 0.0, "avg_entry": None} for c in COINS}
    equity_curve = np.empty(T)
    trades = []

    for t in range(T):
        if t >= first_decidable_t:
            buy_coins, sell_coins = decide_fn(t, closes, opens, positions)
            sell_coins = [c for c in sell_coins if positions[c]["qty"] > 0]
            buy_coins = [c for c in buy_coins if c not in sell_coins]
            open_t = {c: opens[c][t] for c in COINS}

            for coin in sell_coins:
                qty = positions[coin]["qty"]
                fill_price = open_t[coin] * (1 - SLIPPAGE)
                notional = qty * fill_price
                fee = notional * FEE_RATE
                cash += notional - fee
                trades.append(Trade(t, coin, "SELL", fill_price, qty, fee))
                positions[coin] = {"qty": 0.0, "avg_entry": None}

            if buy_coins and cash > CASH_EPSILON:
                alloc = cash / len(buy_coins)
                for coin in buy_coins:
                    fill_price = open_t[coin] * (1 + SLIPPAGE)
                    notional = alloc / (1 + FEE_RATE)
                    fee = alloc - notional
                    new_qty = notional / fill_price
                    if new_qty <= 0:
                        continue
                    old = positions[coin]
                    if old["qty"] > 0:
                        total_qty = old["qty"] + new_qty
                        avg_entry = (old["qty"] * old["avg_entry"] + new_qty * fill_price) / total_qty
                        positions[coin] = {"qty": total_qty, "avg_entry": avg_entry}
                    else:
                        positions[coin] = {"qty": new_qty, "avg_entry": fill_price}
                    cash -= alloc
                    trades.append(Trade(t, coin, "BUY", fill_price, new_qty, fee))

        equity_curve[t] = cash + sum(positions[c]["qty"] * closes[c][t] for c in COINS)

    return {"equity_curve": equity_curve, "trades": trades, "n_trades": len(trades),
            "final_value": float(equity_curve[-1])}


# --------------------------------------------------------------------------------------
# Rule 2: MA CROSSOVER -- long when fast SMA crosses above slow SMA, exit on cross below.
# --------------------------------------------------------------------------------------
def _crossover_signals(closes_by_coin, fast_w, slow_w):
    """dict[coin] -> (cross_up: np.bool array(T), cross_down: np.bool array(T)).
    fast/slow SMAs computed on shifted[t]==close[t-1] (causal). If fast is ALREADY above
    slow at the first bar both SMAs become defined, that counts as a cross_up there too
    (fillna(False) treats the undefined prior state as "was not above") -- i.e. the rule
    enters immediately if already in an uptrend when it becomes computable, rather than
    silently waiting for a fresh crossing that may never come in a short world."""
    out = {}
    for coin, close_arr in closes_by_coin.items():
        close = pd.Series(np.asarray(close_arr, dtype=float))
        shifted = close.shift(1)
        fast = shifted.rolling(fast_w).mean()
        slow = shifted.rolling(slow_w).mean()
        above = fast > slow
        prev_above = above.shift(1).fillna(False)
        cross_up = (above & ~prev_above).to_numpy()
        cross_down = (~above & prev_above).to_numpy()
        out[coin] = (cross_up, cross_down)
    return out


def ma_crossover_portfolio(world_ohlcv, fast_w, slow_w):
    closes = {c: world_ohlcv[c]["close"].to_numpy(dtype=float) for c in COINS}
    signals = _crossover_signals(closes, fast_w, slow_w)

    def decide(t, closes_, opens_, positions):
        buy_coins = [c for c in COINS if signals[c][0][t]]
        sell_coins = [c for c in COINS if signals[c][1][t] and positions[c]["qty"] > 0]
        return buy_coins, sell_coins

    return backtest_rule(world_ohlcv, decide)


# --------------------------------------------------------------------------------------
# Rule 3: PRICE vs LONG SMA -- hold when price > SMA_long, else cash. Level-based (every
# bar), not a one-time crossing event.
# --------------------------------------------------------------------------------------
def price_vs_sma_portfolio(world_ohlcv, sma_window=PRICE_VS_SMA_WINDOW):
    closes = {c: world_ohlcv[c]["close"].to_numpy(dtype=float) for c in COINS}
    above = {}
    for coin, close_arr in closes.items():
        close = pd.Series(np.asarray(close_arr, dtype=float))
        shifted = close.shift(1)
        sma = shifted.rolling(sma_window).mean()
        above[coin] = (shifted > sma).to_numpy()

    def decide(t, closes_, opens_, positions):
        buy_coins = [c for c in COINS if above[c][t] and positions[c]["qty"] == 0]
        sell_coins = [c for c in COINS if not above[c][t] and positions[c]["qty"] > 0]
        return buy_coins, sell_coins

    return backtest_rule(world_ohlcv, decide)


# --------------------------------------------------------------------------------------
# Rule 4: CROSS-SECTIONAL MOMENTUM -- weekly rebalance into the single best-trailing-
# return coin; cash if the best trailing return is <=0.
# --------------------------------------------------------------------------------------
def cross_sectional_momentum(world_ohlcv, lookback_h=MOMENTUM_LOOKBACK_HOURS,
                              rebalance_h=MOMENTUM_REBALANCE_HOURS,
                              first_decidable_t=FIRST_DECIDABLE_T):
    closes = {c: world_ohlcv[c]["close"].to_numpy(dtype=float) for c in COINS}
    mom = {}
    for coin, close_arr in closes.items():
        close = pd.Series(np.asarray(close_arr, dtype=float))
        shifted = close.shift(1)
        mom[coin] = (shifted / shifted.shift(lookback_h) - 1.0).to_numpy()

    def decide(t, closes_, opens_, positions):
        if (t - first_decidable_t) % rebalance_h != 0:
            return [], []
        scores = {c: mom[c][t] for c in COINS}
        best_coin = max(scores, key=scores.get)
        target = best_coin if scores[best_coin] > 0 else None
        held = [c for c in COINS if positions[c]["qty"] > 0]
        sell_coins = [c for c in held if c != target]
        buy_coins = [target] if (target is not None and target not in held) else []
        return buy_coins, sell_coins

    return backtest_rule(world_ohlcv, decide)


# --------------------------------------------------------------------------------------
# Rule 5: RSI MEAN-REVERSION -- buy when RSI<30, sell when RSI>70 (standard 14-period,
# simple/non-Wilder rolling-mean RSI). Level-based (every bar), non-trend-following.
# --------------------------------------------------------------------------------------
def _rsi(closes_by_coin, period=RSI_PERIOD):
    out = {}
    for coin, close_arr in closes_by_coin.items():
        close = pd.Series(np.asarray(close_arr, dtype=float))
        shifted = close.shift(1)
        delta = shifted.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = (100 - 100 / (1 + rs)).fillna(50.0)  # no observed loss -> neutral, not NaN
        out[coin] = rsi.to_numpy()
    return out


def rsi_mean_reversion_portfolio(world_ohlcv, period=RSI_PERIOD, buy_below=RSI_BUY_BELOW,
                                  sell_above=RSI_SELL_ABOVE):
    closes = {c: world_ohlcv[c]["close"].to_numpy(dtype=float) for c in COINS}
    rsi = _rsi(closes, period)

    def decide(t, closes_, opens_, positions):
        buy_coins = [c for c in COINS if rsi[c][t] < buy_below and positions[c]["qty"] == 0]
        sell_coins = [c for c in COINS if rsi[c][t] > sell_above and positions[c]["qty"] > 0]
        return buy_coins, sell_coins

    return backtest_rule(world_ohlcv, decide)


# --------------------------------------------------------------------------------------
# Registry: (label, callable(world_ohlcv) -> sim_result dict)
# --------------------------------------------------------------------------------------
def build_baseline_registry():
    return [
        ("MA_CROSSOVER_24_168", lambda w: ma_crossover_portfolio(w, 24, 168)),
        ("MA_CROSSOVER_168_720", lambda w: ma_crossover_portfolio(w, 168, 720)),
        ("PRICE_VS_SMA720", lambda w: price_vs_sma_portfolio(w, PRICE_VS_SMA_WINDOW)),
        ("CROSS_SECTIONAL_MOMENTUM", lambda w: cross_sectional_momentum(w)),
        ("RSI_MEAN_REVERSION", lambda w: rsi_mean_reversion_portfolio(w)),
    ]
