"""
simulator.py — run one agent (genome) through one world (synthetic OHLCV) and produce
an equity curve + trade log. Sizing scheme "A": all-in swapping. No fitness/ranking here
-- that is the next (evolution) step.

Execution/accounting rules (all tunable constants below):
  - Decisions for hour t use sensors built from data <= t-1 (sensors.py).
  - Fills for hour t always use open[t] -- open[t] is never a sensor input, so using it
    for fills introduces no look-ahead.
  - Per hour: SELLS first (liquidate the full position in any SELL-signaled coin,
    freeing cash), THEN BUYS (split ALL currently-free cash equally across every
    BUY-signaled coin).
  - Every fill pays FEE_RATE on notional, and SLIPPAGE as a strictly worse fill price
    (buys fill above open[t], sells fill below open[t]).
  - Guards: cash can never go negative (buys spend exactly the allocated cash, never
    more), a coin can't be sold unless held, zero-size orders are skipped, BUYs are
    ignored when free cash is ~0 (CASH_EPSILON).
  - Equity is marked to market at close[t] every hour.

Performance note (not implemented -- see task scope): the per-hour Python loop here
(trade decoding, sell/buy bookkeeping) is the main remaining cost after
precompute_price_sensors() already vectorized the price-derived sensors. If this needs
to be faster for evolution (thousands of genomes x worlds), the next win would be
batching brain.forward() across many genomes at once (a small matrix multiply instead of
per-genome Python loops), not touched here.
"""

from collections import namedtuple

import numpy as np

from brain import build_network, forward
from sensors import COINS, FIRST_DECIDABLE_T, precompute_price_sensors, compute_sensor_vector

BUY_THRESH = 0.5      # action_k > +BUY_THRESH  => BUY signal
SELL_THRESH = 0.5     # action_k < -SELL_THRESH => SELL signal
FEE_RATE = 0.001       # 0.1% of notional, every executed trade
SLIPPAGE = 0.0005      # 0.05%, modeled as a worse fill price (buys higher, sells lower)
STARTING_CAPITAL = 1000.0
CASH_EPSILON = 1e-6    # ignore BUYs when free cash is below this

Trade = namedtuple("Trade", ["t", "coin", "side", "price", "size", "fee"])

DEFAULT_PARAMS = {
    "buy_thresh": BUY_THRESH, "sell_thresh": SELL_THRESH,
    "fee_rate": FEE_RATE, "slippage": SLIPPAGE,
    "starting_capital": STARTING_CAPITAL, "cash_epsilon": CASH_EPSILON,
}


def _extract_aligned_arrays(world_ohlcv):
    # audited (determinism): `lengths` dict-comprehension order traces back to COINS-
    # ordered upstream construction (generator.py), so it's deterministic regardless of
    # hash seed. `set(lengths.values())` below is only used for a cardinality check
    # (len(...)==1), never iterated -- and `next(iter(...))` is order-invariant anyway
    # since the assertion guarantees every value is identical. Safe as-is.
    lengths = {coin: len(df) for coin, df in world_ohlcv.items()}
    assert len(set(lengths.values())) == 1, f"world coins have mismatched lengths: {lengths}"
    T = next(iter(lengths.values()))
    timestamps = None
    for coin, df in world_ohlcv.items():
        ts = df["timestamp"].to_numpy()
        if timestamps is None:
            timestamps = ts
        else:
            assert np.array_equal(ts, timestamps), f"world coins are not aligned on timestamp ({coin})"
    opens = {coin: df["open"].to_numpy(dtype=float) for coin, df in world_ohlcv.items()}
    closes = {coin: df["close"].to_numpy(dtype=float) for coin, df in world_ohlcv.items()}
    return T, opens, closes


def simulate(genome, world_ohlcv, params=None):
    """Deterministic given (genome, world_ohlcv, params). Returns:
      {equity_curve: np.array(T), trades: list[Trade], n_trades: int, final_value: float}
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    T, opens, closes = _extract_aligned_arrays(world_ohlcv)
    assert T > FIRST_DECIDABLE_T, "world too short for the agent's sensor lookback"

    price_sensors = precompute_price_sensors(closes)
    network = build_network(genome)

    cash = p["starting_capital"]
    positions = {coin: {"qty": 0.0, "avg_entry": None} for coin in COINS}
    equity_curve = np.empty(T)
    trades = []

    for t in range(T):
        if t >= FIRST_DECIDABLE_T:
            sensor_vec = compute_sensor_vector(t, price_sensors, closes, cash, positions)
            actions = forward(network, sensor_vec)

            sell_coins = [c for i, c in enumerate(COINS)
                          if actions[i] < -p["sell_thresh"] and positions[c]["qty"] > 0]
            buy_coins = [c for i, c in enumerate(COINS)
                         if actions[i] > p["buy_thresh"] and c not in sell_coins]
            # A single action_k can't be both > +thresh and < -thresh at once, so no coin
            # can generate BUY and SELL simultaneously -- `c not in sell_coins` is just a
            # defensive restatement of that invariant, not load-bearing.

            open_t = {c: opens[c][t] for c in COINS}

            # -- SELLS first: liquidate the full position, freeing cash for buys below --
            for coin in sell_coins:
                qty = positions[coin]["qty"]
                fill_price = open_t[coin] * (1 - p["slippage"])
                notional = qty * fill_price
                fee = notional * p["fee_rate"]
                cash += notional - fee
                trades.append(Trade(t, coin, "SELL", fill_price, qty, fee))
                positions[coin] = {"qty": 0.0, "avg_entry": None}

            # -- BUYS: split ALL currently-free cash equally across BUY-signaled coins --
            if buy_coins and cash > p["cash_epsilon"]:
                alloc = cash / len(buy_coins)
                for coin in buy_coins:
                    fill_price = open_t[coin] * (1 + p["slippage"])
                    notional = alloc / (1 + p["fee_rate"])  # notional*(1+fee_rate) == alloc spent
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

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "n_trades": len(trades),
        "final_value": float(equity_curve[-1]),
    }
