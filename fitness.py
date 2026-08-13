"""
fitness.py — per-genome fitness: Sortino ratio per world (annualized), a hard drawdown
cap that overrides the score if breached, mean-aggregation across a genome's N_WORLDS.
Degeneracy is explicitly guarded: doing nothing must score 0, not a large/undefined number.
"""

import numpy as np

from simulator import simulate, STARTING_CAPITAL
from generator import generate_raw

HOURS_PER_YEAR = 8760

# Degeneracy guard: if BOTH downside deviation and mean excess return are ~0 (e.g. a
# never-trading bot, or any perfectly flat equity curve), fitness contribution is 0, not
# a large or undefined number.
DEGENERACY_RETURN_EPS = 1e-8
DEGENERACY_DOWNSIDE_EPS = 1e-8

# Hard drawdown cap ("catastrophe guard"): if a world's max drawdown exceeds this, that
# world's score is floored to -DD_BREACH_PENALTY regardless of what Sortino would say --
# a bot that rode a position down past the cap "failed" that world even if it recovered.
DD_CAP = 0.60
DD_BREACH_PENALTY = 5.0

# A genome whose mean trade count across its worlds is below this is counted as
# "degenerate" (~never-trading) for telemetry purposes only (not for scoring).
DEGENERATE_TRADE_THRESHOLD = 1.0

# Aggregator across a genome's N_WORLDS scores. Default: MEAN. Alternative to try later:
# np.min, which rewards worst-case robustness harder than averaging does (a genome that
# is merely mediocre on every world would beat one that's brilliant on 4 and catastrophic
# on 1, under min-aggregation, unlike under mean).
DEFAULT_AGGREGATOR = np.mean


def build_world_composition(n_worlds):
    """World-set regime composition: deliberately spans regimes so a bull-only strategy
    is punished. Tunable; the first min(n_worlds,3) slots are one pure bull/bear/sideways
    world each, the rest are "mix". For n_worlds=5 this reproduces the task's example:
    ["bull","bear","sideways","mix","mix"].
    """
    base = ["bull", "bear", "sideways"]
    if n_worlds <= len(base):
        return base[:n_worlds]
    return base + ["mix"] * (n_worlds - len(base))


def generate_world(seed, n_hours, regime):
    """regime in {"bull","bear","sideways","mix"}. Thin wrapper over generator.generate_raw
    so evolution.py doesn't need to know the pure/mix calling-convention difference."""
    if regime == "mix":
        dfs, _, _ = generate_raw(seed=seed, n_hours=n_hours, regime_mode="mix")
    else:
        dfs, _, _ = generate_raw(seed=seed, n_hours=n_hours, regime_mode="pure", regime=regime)
    return dfs


def max_drawdown(equity_curve):
    equity_curve = np.asarray(equity_curve)
    cummax = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - cummax) / cummax
    return float(-dd.min())


def sortino_detail(equity_curve, hours_per_year=HOURS_PER_YEAR):
    """Same computation as sortino_ratio(), but also returns the intermediates
    (mean_hourly_return, downside_deviation before/after the epsilon floor, whether the
    degenerate-guard fired) -- for diagnostics/telemetry only. sortino_ratio() is a thin
    wrapper around this; behavior of every existing caller is unchanged."""
    equity_curve = np.asarray(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    mean_ret = float(returns.mean())
    downside = returns[returns < 0]
    downside_dev = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 0.0

    degenerate = downside_dev < DEGENERACY_DOWNSIDE_EPS and abs(mean_ret) < DEGENERACY_RETURN_EPS
    downside_dev_floored = downside_dev
    if degenerate:
        sortino = 0.0
    else:
        if downside_dev < DEGENERACY_DOWNSIDE_EPS:
            downside_dev_floored = DEGENERACY_DOWNSIDE_EPS  # avoid div-by-zero without faking an infinite score
        annualized_mean = mean_ret * hours_per_year
        annualized_downside = downside_dev_floored * np.sqrt(hours_per_year)
        sortino = annualized_mean / annualized_downside

    return {
        "sortino": sortino,
        "mean_hourly_return": mean_ret,
        "downside_deviation_hourly": downside_dev,            # raw, before any epsilon floor
        "downside_deviation_floored": downside_dev_floored,   # what was actually used in the ratio
        "degenerate": degenerate,
        "hours_per_year": hours_per_year,
    }


def sortino_ratio(equity_curve, hours_per_year=HOURS_PER_YEAR):
    """Mean hourly return / downside deviation, annualized (MAR=0). Guards: a
    doing-nothing outcome scores exactly 0 (never reward a never-trade / flat bot); a
    positive-drift, zero-observed-downside outcome is floored, not divided by zero or
    turned into an infinite score.
    """
    return sortino_detail(equity_curve, hours_per_year)["sortino"]


def final_cash(trades, starting_capital=STARTING_CAPITAL):
    """Reconstruct ending cash from the trade log (simulate() doesn't expose it
    directly) -- used only for the cash_frac diversity signal, not for scoring."""
    cash = starting_capital
    for tr in trades:
        if tr.side == "BUY":
            cash -= tr.size * tr.price + tr.fee
        else:
            cash += tr.size * tr.price - tr.fee
    return cash


def score_world(sim_result, dd_cap=DD_CAP, dd_breach_penalty=DD_BREACH_PENALTY):
    equity = sim_result["equity_curve"]
    dd = max_drawdown(equity)
    breached = dd > dd_cap
    score = -dd_breach_penalty if breached else sortino_ratio(equity)
    cash = final_cash(sim_result["trades"])
    cash_frac = cash / sim_result["final_value"] if sim_result["final_value"] > 0 else 1.0
    return {
        "score": score, "max_drawdown": dd, "dd_breached": breached,
        "n_trades": sim_result["n_trades"], "final_value": sim_result["final_value"],
        "cash_frac": cash_frac,
    }


def evaluate_genome(genome, worlds, aggregator=DEFAULT_AGGREGATOR, params=None):
    """worlds: list of world_ohlcv dicts (the SAME set for every genome in a generation).
    Returns {fitness, per_world, mean_n_trades, mean_max_drawdown, mean_cash_frac}.

    NOTE: this per-genome, per-world simulate() call is the hot spot the NEXT
    (performance) step will vectorize/batch across genomes -- deliberately left as a
    naive Python loop here so correctness is easy to trust.
    """
    per_world = [score_world(simulate(genome, world, params)) for world in worlds]
    scores = np.array([w["score"] for w in per_world])
    return {
        "fitness": float(aggregator(scores)),
        "per_world": per_world,
        "mean_n_trades": float(np.mean([w["n_trades"] for w in per_world])),
        "mean_max_drawdown": float(np.mean([w["max_drawdown"] for w in per_world])),
        "mean_cash_frac": float(np.mean([w["cash_frac"] for w in per_world])),
    }
