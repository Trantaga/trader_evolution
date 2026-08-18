"""
fitness.py — per-genome fitness: Sortino ratio per world (annualized) with a GRADUATED
drawdown penalty (subtracted, not a hard cliff -- see DD_SOFT/PENALTY_SCALE), mean-
aggregation across a genome's N_WORLDS. Degeneracy is explicitly guarded: doing nothing
must score 0, not a large/undefined number.

real-02 BUG FIX (DD_FLOOR/SORTINO_CAP below): a genome that happens to have ZERO
losing hours on one evaluation world (downside_deviation_hourly == 0.0, not just small)
but a small positive mean return produced a Sortino ratio of ~124,527 on that single
world -- the old DEGENERACY_DOWNSIDE_EPS=1e-8 floor was too small to matter (annualizing
1e-8 still divides by a near-zero number). Averaged into a 15-world mean, this alone
dragged best_fitness to 8300.33, poisoning best_fitness_so_far and causing evolution.py's
plateau criterion to stop the run on a false signal while the population median was still
improving normally. DD_FLOOR/SORTINO_CAP close this off structurally; see sortino_detail().

real-03 BUG FIX (DD_SOFT/PENALTY_SCALE below, replacing the old hard DD_CAP cliff): once
the Sortino fix above was in place, real-03 (otherwise a clean restart of real-02) never
learned -- median fitness stayed flat ~-3.4 for 121 generations. Root cause: the OLD
scoring floored ANY world with drawdown > DD_CAP=0.60 to a FLAT score of -DD_BREACH_PENALTY
(-5.0), discarding all Sortino information for that world. real-03's population started at
mean_max_drawdown=0.82 (already past the cliff) and never got under it -- meaning most of
the population scored the SAME flat -5.0 on most worlds most generations, leaving
selection almost no gradient to climb. (real-02 escaped this only by an accident of the
OLD, buggy Sortino formula: a lucky near-zero-downside genome got a wildly inflated
reward, which happened to correlate with low-drawdown/conservative behavior, providing an
extra-strong -- unintended -- selective push under the cliff. The Sortino fix correctly
removed that accidental exploit, which exposed this latent cliff problem.) See
drawdown_penalty()/score_world() for the graduated replacement: a smooth, differentiable
penalty that still punishes deep drawdowns hard, but preserves a real ranking gradient
across the whole 0.40-1.0 drawdown range instead of collapsing it all to one flat value.

real-05 CHANGE (GENERATOR_PERIODS_PER_YEAR below): generated training worlds now come
from generator_4h.py (4-hour bars), but evaluate_champion.py's held-out CHECK still runs
against REAL, still-HOURLY recorded data (data/heldout/*.csv) via this SAME module's
sortino_ratio()/sortino_detail() -- so a single global annualization constant can't
serve both. HOURS_PER_YEAR (8760) stays the DEFAULT, unchanged, specifically so
evaluate_champion.py's hourly held-out annualization keeps working correctly without
any code change there. GENERATOR_PERIODS_PER_YEAR (2190) is a SEPARATE constant, passed
EXPLICITLY by score_world()/generate_world() (the functions that touch GENERATED, now
4h-resolution worlds) -- see their call sites. Using the wrong one for either path would
silently mis-annualize Sortino by a factor of ~2x (mean term 4x, downside term
sqrt(4)=2x, net ~2x) -- exactly the kind of bug this split is designed to prevent.
"""

import numpy as np

from simulator import simulate, STARTING_CAPITAL
from generator import load_calibration
from generator_4h import generate_raw as generate_raw_4h, load_calibration as load_calibration_4h

HOURS_PER_YEAR = 8760              # REAL hourly data (evaluate_champion.py's held-out check)
GENERATOR_PERIODS_PER_YEAR = 2190  # GENERATED 4h-bar worlds (real-05+, generator_4h.py)

# Degeneracy guard: if BOTH downside deviation and mean excess return are ~0 (e.g. a
# never-trading bot, or any perfectly flat equity curve), fitness contribution is 0, not
# a large or undefined number.
DEGENERACY_RETURN_EPS = 1e-8
DEGENERACY_DOWNSIDE_EPS = 1e-8


def _typical_period_vol_from_calibration():
    """Mean, across all calibrated coins, of ann_vol / sqrt(GENERATOR_PERIODS_PER_YEAR)
    -- i.e. each coin's calibrated annualized vol converted back to a PER-4H-BAR scale,
    then averaged. Computed from config/calibration_4h.json (the ACTIVE generator's own
    calibration, not the hourly one -- DD_FLOOR must be scaled to whatever resolution
    score_world() actually evaluates) -- used only to derive DD_FLOOR below, a fixed
    constant computed once at import time, not re-read per call."""
    calib = load_calibration_4h()
    ann_vols = [c["ann_vol"] for c in calib["coins"].values()]
    period_vols = [v / np.sqrt(GENERATOR_PERIODS_PER_YEAR) for v in ann_vols]
    return float(np.mean(period_vols))


TYPICAL_HOURLY_VOL = _typical_period_vol_from_calibration()  # ~0.011 as of the 4h calibration (name kept for continuity; now a per-4h-bar value, see docstring above)

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

# Graduated drawdown penalty (replaces the old hard DD_CAP cliff -- see module docstring's
# real-03 bug-fix note): SUBTRACTED from the Sortino score, not a replacement of it, so a
# deep-drawdown world still contributes a genuine ranking signal instead of collapsing to
# one flat value. Below DD_SOFT, no penalty at all (Sortino alone). Above DD_SOFT, the
# penalty grows as a square of how far past DD_SOFT the drawdown is (relative to the
# remaining 1.0-DD_SOFT headroom), so it starts gently and accelerates toward the worst
# case (dd -> 1.0, total wipeout) -- e.g. dd=0.60 costs ~0.78, dd=0.70 costs ~1.75,
# dd=0.85 costs ~3.94, dd=0.90 costs ~4.86 (comparable to the old flat -5.0, by design, at
# the genuinely catastrophic end) -- see this module's own test-run report for the
# observed spread across a real population's 0.60-0.85 drawdown band.
DD_SOFT = 0.40
PENALTY_SCALE = 7.0

# DD_CAP: no longer used for scoring (see drawdown_penalty() above) -- kept ONLY as the
# threshold score_world() reports "dd_breached" against, for telemetry/diagnostics (e.g.
# "what fraction of the population is past the old danger line this generation").
DD_CAP = 0.60


def drawdown_penalty(dd, dd_soft=DD_SOFT, penalty_scale=PENALTY_SCALE):
    """0 at/below dd_soft; grows as penalty_scale * ((dd-dd_soft)/(1-dd_soft))**2 above it.
    Bounded by construction (max_drawdown() is always < 1.0, since equity can approach but
    never reach exactly 0), so this saturates near penalty_scale as dd -> 1.0 -- no
    separate explicit clamp needed."""
    if dd <= dd_soft:
        return 0.0
    x = (dd - dd_soft) / (1.0 - dd_soft)
    return penalty_scale * x * x

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
    """regime in {"bull","bear","sideways","mix"}. Thin wrapper over generator_4h.generate_raw
    so evolution.py doesn't need to know the pure/mix calling-convention difference.

    real-05 CHANGE: now backed by generator_4h.py (4-hour bars), not the original hourly
    generator.py. The `n_hours` PARAMETER NAME is kept unchanged (evolution.py/
    checkpoint.py/telemetry.py all call this with `world_length_hours`-style arguments,
    and this task is scoped to leave those untouched) -- but the VALUE it's called with
    for real-05 is a 4h-BAR COUNT, not an hour count (e.g. real-04's
    pure_world_length_hours=2160/mix_world_length_hours=17520 become real-05's
    pure_world_length_hours=540/mix_world_length_hours=4380 -- same real-world spans,
    /4 for the new bar resolution). Passed straight through to generator_4h.generate_raw's
    own `n_periods` parameter, which is genuinely bar-count-native.
    """
    if regime == "mix":
        dfs, _, _ = generate_raw_4h(seed=seed, n_periods=n_hours, regime_mode="mix")
    else:
        dfs, _, _ = generate_raw_4h(seed=seed, n_periods=n_hours, regime_mode="pure", regime=regime)
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


def score_world(sim_result, dd_soft=DD_SOFT, penalty_scale=PENALTY_SCALE, dd_cap=DD_CAP,
                 periods_per_year=GENERATOR_PERIODS_PER_YEAR):
    """score = Sortino (via sortino_detail, ALWAYS computed now -- see module docstring's
    real-03 bug-fix note) MINUS a graduated drawdown_penalty(). dd_breached is now purely
    informational (dd > dd_cap), no longer changes the score.

    real-05: periods_per_year defaults to GENERATOR_PERIODS_PER_YEAR (2190, 4h bars) --
    this function is ONLY ever called on GENERATED worlds (evolution.py/fast_evaluator.py
    via evaluate_genome/_evaluate_population), never on real hourly held-out data (that
    path uses sortino_ratio() directly with ITS OWN default -- see module docstring).
    Explicit, not relying on sortino_detail's own HOURS_PER_YEAR default, which would
    silently mis-annualize by ~2x if this world came from generator_4h.py.
    """
    equity = sim_result["equity_curve"]
    dd = max_drawdown(equity)
    detail = sortino_detail(equity, hours_per_year=periods_per_year)
    penalty = drawdown_penalty(dd, dd_soft, penalty_scale)
    score = detail["sortino"] - penalty
    cash = final_cash(sim_result["trades"])
    cash_frac = cash / sim_result["final_value"] if sim_result["final_value"] > 0 else 1.0
    return {
        "score": score, "max_drawdown": dd, "dd_breached": dd > dd_cap,
        "drawdown_penalty": penalty,
        "n_trades": sim_result["n_trades"], "final_value": sim_result["final_value"],
        "cash_frac": cash_frac, "floor_fired": detail["floor_fired"], "clamp_fired": detail["clamp_fired"],
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
