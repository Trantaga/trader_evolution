"""
fitness.py — per-genome fitness: Sortino ratio per world (annualized), a hard drawdown
cap that overrides the score if breached, mean-aggregation across a genome's N_WORLDS.
Degeneracy is explicitly guarded: doing nothing must score 0, not a large/undefined number.

real-02 BUG FIX (DD_FLOOR/SORTINO_CAP below): a genome that happens to have ZERO
losing hours on one evaluation world (downside_deviation_hourly == 0.0, not just small)
but a small positive mean return produced a Sortino ratio of ~124,527 on that single
world -- the old DEGENERACY_DOWNSIDE_EPS=1e-8 floor was too small to matter (annualizing
1e-8 still divides by a near-zero number). Averaged into a 15-world mean, this alone
dragged best_fitness to 8300.33, poisoning best_fitness_so_far and causing evolution.py's
plateau criterion to stop the run on a false signal while the population median was still
improving normally. DD_FLOOR/SORTINO_CAP close this off structurally; see sortino_detail().
"""

import numpy as np

from simulator import simulate, STARTING_CAPITAL
from generator import generate_raw, load_calibration

HOURS_PER_YEAR = 8760

# Degeneracy guard: if BOTH downside deviation and mean excess return are ~0 (e.g. a
# never-trading bot, or any perfectly flat equity curve), fitness contribution is 0, not
# a large or undefined number.
DEGENERACY_RETURN_EPS = 1e-8
DEGENERACY_DOWNSIDE_EPS = 1e-8


def _typical_hourly_vol_from_calibration():
    """Mean, across all calibrated coins, of ann_vol / sqrt(HOURS_PER_YEAR) -- i.e. each
    coin's calibrated annualized vol converted back to an hourly scale, then averaged.
    Computed from config/calibration.json (BTC/ETH/BNB/MATIC/SOL as of this writing) ~=
    0.011 (1.1% per hour) -- used only to derive DD_FLOOR below, a fixed constant computed
    once at import time, not re-read per call."""
    calib = load_calibration()
    ann_vols = [c["ann_vol"] for c in calib["coins"].values()]
    hourly_vols = [v / np.sqrt(HOURS_PER_YEAR) for v in ann_vols]
    return float(np.mean(hourly_vols))


TYPICAL_HOURLY_VOL = _typical_hourly_vol_from_calibration()  # ~0.011 as of this calibration

# DD_FLOOR: the downside-deviation floor used in the Sortino ratio's denominator, so a
# genuinely-zero (or near-zero) observed downside on one world can't blow the ratio up
# toward infinity -- see module docstring for the real-02 gen-170 incident this fixes.
# Set to 5% of TYPICAL_HOURLY_VOL (~0.00055): small enough that every NORMAL genome's
# downside_deviation_hourly (observed ~0.006-0.015 across real-02's worlds, see the
# gen-170 diagnostic) is well above the floor and therefore completely unaffected --
# only the pathological near-zero-downside case is floored. See fitness's test-run report
# for the observed firing rate.
DD_FLOOR_FRACTION = 0.05
DD_FLOOR = DD_FLOOR_FRACTION * TYPICAL_HOURLY_VOL

# SORTINO_CAP: belt-and-suspenders hard clamp on the PER-WORLD Sortino ratio, applied
# after DD_FLOOR. DD_FLOOR alone closes off the exact zero-downside case (see module
# docstring); this additionally bounds any other pathological combination (e.g. a
# short/degenerate world with an unusually large mean return relative to even the
# floored downside) from producing an outsized single-world score that would dominate a
# mean-aggregated fitness. +-10 comfortably exceeds every NORMAL per-world score observed
# in real-02's telemetry (typically in [-5, +8] -- see compact.jsonl), so it never fires
# on ordinary genomes.
SORTINO_CAP = 10.0

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
    (mean_hourly_return, downside_deviation before/after DD_FLOOR, whether the
    degenerate-guard/DD_FLOOR/SORTINO_CAP fired) -- for diagnostics/telemetry only.
    sortino_ratio() is a thin wrapper around this; behavior of every existing caller is
    unchanged except for the DD_FLOOR/SORTINO_CAP fix itself (see module docstring).

    downside_dev_used = max(downside_dev, DD_FLOOR): DEGENERACY_DOWNSIDE_EPS (1e-8) is
    still used for the exact "doing nothing" degenerate guard below (a flat/never-trading
    equity curve scores exactly 0.0, not DD_FLOOR-bounded) -- DD_FLOOR is a SEPARATE,
    much larger floor specifically for the ratio's denominator once a genome is judged
    non-degenerate (i.e. it has some real return signal, just happened to have zero or
    near-zero downside on this particular world).
    """
    equity_curve = np.asarray(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    mean_ret = float(returns.mean())
    downside = returns[returns < 0]
    downside_dev = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 0.0

    degenerate = downside_dev < DEGENERACY_DOWNSIDE_EPS and abs(mean_ret) < DEGENERACY_RETURN_EPS
    downside_dev_floored = downside_dev
    floor_fired = False
    clamp_fired = False
    if degenerate:
        sortino = 0.0
    else:
        downside_dev_floored = max(downside_dev, DD_FLOOR)
        floor_fired = bool(downside_dev_floored != downside_dev)
        annualized_mean = mean_ret * hours_per_year
        # float(): np.sqrt(hours_per_year) is np.float64, which would otherwise make
        # `sortino` (and therefore `clamped != sortino` below) a numpy scalar -- np.bool_
        # is NOT json-serializable (unlike Python bool), which broke telemetry.
        # write_full_log()'s per-genome dump the first time this was tried untested.
        annualized_downside = float(downside_dev_floored * np.sqrt(hours_per_year))
        sortino = annualized_mean / annualized_downside
        clamped = float(np.clip(sortino, -SORTINO_CAP, SORTINO_CAP))
        clamp_fired = bool(clamped != sortino)
        sortino = clamped

    return {
        "sortino": sortino,
        "mean_hourly_return": mean_ret,
        "downside_deviation_hourly": downside_dev,            # raw, before DD_FLOOR
        "downside_deviation_floored": downside_dev_floored,   # what was actually used in the ratio
        "degenerate": degenerate,
        "floor_fired": floor_fired,    # DD_FLOOR changed the denominator (non-degenerate case only)
        "clamp_fired": clamp_fired,    # SORTINO_CAP changed the final ratio
        "hours_per_year": hours_per_year,
    }


def sortino_ratio(equity_curve, hours_per_year=HOURS_PER_YEAR):
    """Mean hourly return / downside deviation, annualized (MAR=0). Guards: a
    doing-nothing outcome scores exactly 0 (never reward a never-trade / flat bot); a
    near-zero-observed-downside outcome has its denominator floored at DD_FLOOR (not an
    epsilon too small to matter -- see module docstring for the incident this fixes), and
    the final ratio is clamped to +-SORTINO_CAP as a belt-and-suspenders bound.
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
    """floor_fired/clamp_fired: whether sortino_detail's DD_FLOOR/SORTINO_CAP changed
    this world's score -- False (not just absent) when the drawdown cap was breached,
    since sortino is never even computed in that case (the -dd_breach_penalty score
    doesn't go through sortino_detail at all)."""
    equity = sim_result["equity_curve"]
    dd = max_drawdown(equity)
    breached = dd > dd_cap
    if breached:
        score, floor_fired, clamp_fired = -dd_breach_penalty, False, False
    else:
        detail = sortino_detail(equity)
        score, floor_fired, clamp_fired = detail["sortino"], detail["floor_fired"], detail["clamp_fired"]
    cash = final_cash(sim_result["trades"])
    cash_frac = cash / sim_result["final_value"] if sim_result["final_value"] > 0 else 1.0
    return {
        "score": score, "max_drawdown": dd, "dd_breached": breached,
        "n_trades": sim_result["n_trades"], "final_value": sim_result["final_value"],
        "cash_frac": cash_frac, "floor_fired": floor_fired, "clamp_fired": clamp_fired,
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
