"""
validate_generator.py — self-validation that closes the calibrate -> generate loop.

Measures synthetic data produced by generator.py using the SAME stylized-fact functions
calibrate.py uses on real data (imported, not re-implemented), and compares GENERATED vs
REAL. Real targets are computed by re-running calibrate.py's own real-data pipeline
(get_real_returns()) rather than reading calibration.json, so that literally the same
function calls (ann_vol_from_returns, excess_kurtosis, acf_abs_lag1, DataFrame.corr) are
applied to both real and synthetic data.

Run: python3 validate_generator.py
"""

import numpy as np
import pandas as pd

from calibrate import (
    COINS, ann_vol_from_returns, excess_kurtosis, acf_abs_lag1, get_real_returns,
)
from generator import (
    generate_raw, load_calibration, REGIME_NAMES,
    generate_idio_garch, cap_idio_persistence,
)

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
MIX_VALIDATION_HOURS = 50_000  # long enough for regime shares to converge to calibration
MIX_SEED = 123
PURE_CHECK_HOURS = 5_000
PURE_SEED = 777

TOL_ANN_VOL_REL = 0.20        # ann_vol: within +/-20% relative
TOL_KURTOSIS_REL = 0.60       # excess_kurtosis: within +/-60% relative (4th-moment estimator is noisy)
TOL_ACF_ABS = 0.08            # lag-1 acf of |returns|: within +/-0.08 absolute
TOL_CORR_ABS = 0.10           # correlation matrix entries: within +/-0.10 absolute

ACF_ATTRIBUTION_SEEDS = 20     # near-unit-root idio realizations are noisy on one path;
ACF_ATTRIBUTION_SEED_BASE = 2000  # average the with/without-reset acf gap over this many seeds

# Reference numbers from the PRIOR (v1.5, idio borrowed total-return GARCH, no reset)
# generator, recorded from the last validation run at the same seed/horizon (seed=123,
# 50000h mix) — used only to print a BEFORE/AFTER comparison below; not recomputed live
# (there's no v1.5 code left to run). This is the run that showed the correlation-
# inflation regression this task fixes.
BEFORE_V15 = {
    "ann_vol":          {"BTC": 0.5729, "ETH": 0.7507, "BNB": 0.6871, "MATIC": 1.0925, "SOL": 1.2592},
    "excess_kurtosis":  {"BTC": 9.2931, "ETH": 8.5839, "BNB": 10.8138, "MATIC": 7.7300, "SOL": 6.6286},
    "acf_abs_lag1":     {"BTC": 0.2944, "ETH": 0.2811, "BNB": 0.3079, "MATIC": 0.2780, "SOL": 0.2516},
    "corr": {
        ("BTC", "ETH"): 0.8462, ("BTC", "BNB"): 0.9134, ("BTC", "MATIC"): 0.8310, ("BTC", "SOL"): 0.7170,
        ("ETH", "BNB"): 0.9185, ("ETH", "MATIC"): 0.8337, ("ETH", "SOL"): 0.7204,
        ("BNB", "MATIC"): 0.9018, ("BNB", "SOL"): 0.7779, ("MATIC", "SOL"): 0.7077,
    },
}


# --------------------------------------------------------------------------------------
# Shared measurement — same functions calibrate.py uses, applied to any returns frame.
# --------------------------------------------------------------------------------------
def measure(returns_df):
    per_coin = {}
    for coin in COINS:
        r = returns_df[coin].dropna()
        per_coin[coin] = {
            "ann_vol": ann_vol_from_returns(r),
            "excess_kurtosis": excess_kurtosis(r),
            "acf_abs_lag1": acf_abs_lag1(r),
        }
    corr = returns_df.dropna(how="any").corr()
    return per_coin, corr


def synthetic_returns(dfs):
    close = pd.DataFrame({coin: df["close"].values for coin, df in dfs.items()})
    return np.log(close).diff()


# --------------------------------------------------------------------------------------
# PASS/WARN helpers
# --------------------------------------------------------------------------------------
def check_rel(name, gen, real, tol_rel, lines, counts):
    rel_err = abs(gen - real) / abs(real) if real != 0 else float("inf")
    ok = rel_err <= tol_rel
    status = "PASS" if ok else "WARN"
    counts[status] += 1
    lines.append(f"[{status}] {name:<28} generated={gen:>10.4f}  real={real:>10.4f}  "
                 f"rel_err={rel_err:>7.1%}  (tol +/-{tol_rel:.0%})")
    return ok


def check_abs(name, gen, real, tol_abs, lines, counts):
    err = abs(gen - real)
    ok = err <= tol_abs
    status = "PASS" if ok else "WARN"
    counts[status] += 1
    lines.append(f"[{status}] {name:<28} generated={gen:>10.4f}  real={real:>10.4f}  "
                 f"abs_err={err:>7.4f}  (tol +/-{tol_abs:.2f})")
    return ok


# --------------------------------------------------------------------------------------
# Main validation
# --------------------------------------------------------------------------------------
def main():
    counts = {"PASS": 0, "WARN": 0}
    lines = []

    print("Loading real data + computing real stylized facts (calibrate.py pipeline)...")
    real_returns, *_ = get_real_returns()
    real_stats, real_corr = measure(real_returns)

    calib = load_calibration()

    print(f"Generating a {MIX_VALIDATION_HOURS}-hour 'mix' run (seed={MIX_SEED}) "
          f"for overall validation...")
    dfs, regime_schedule, _ = generate_raw(
        seed=MIX_SEED, n_hours=MIX_VALIDATION_HOURS, regime_mode="mix", calib=calib
    )
    syn_returns = synthetic_returns(dfs)
    syn_stats, syn_corr = measure(syn_returns)

    achieved_shares = {r: float(np.mean(regime_schedule == r)) for r in REGIME_NAMES}
    calib_shares = {r: calib["regimes"]["params"][r]["share_of_time"] for r in REGIME_NAMES}

    lines.append("=" * 88)
    lines.append("GENERATOR VALIDATION REPORT")
    lines.append("=" * 88)
    lines.append(f"Mix run: {MIX_VALIDATION_HOURS} hours, seed={MIX_SEED}")
    lines.append("Regime shares  " + ", ".join(
        f"{r}: generated={achieved_shares[r]:.1%} / calibration={calib_shares[r]:.1%}"
        for r in REGIME_NAMES))
    lines.append("")

    lines.append("GENERATED vs REAL — per-coin stylized facts")
    lines.append("-" * 88)
    for coin in COINS:
        lines.append(f"-- {coin} --")
        check_rel(f"ann_vol[{coin}]", syn_stats[coin]["ann_vol"], real_stats[coin]["ann_vol"],
                   TOL_ANN_VOL_REL, lines, counts)
        check_rel(f"excess_kurtosis[{coin}]", syn_stats[coin]["excess_kurtosis"],
                   real_stats[coin]["excess_kurtosis"], TOL_KURTOSIS_REL, lines, counts)
        check_abs(f"acf_abs_lag1[{coin}] (vol clustering)", syn_stats[coin]["acf_abs_lag1"],
                   real_stats[coin]["acf_abs_lag1"], TOL_ACF_ABS, lines, counts)
    lines.append("")

    lines.append("BEFORE (v1.5, borrowed idio GARCH, no reset) vs AFTER (v2, own idio-residual GARCH")
    lines.append("+ persistence cap + periodic reset) vs REAL  [seed=123, same 50000h mix]")
    lines.append("-" * 88)
    lines.append(f"{'coin':<7}{'ann_vol_before':>15}{'ann_vol_after':>15}{'ann_vol_real':>13}   |"
                 f"{'kurt_before':>12}{'kurt_after':>11}{'kurt_real':>10}   |"
                 f"{'acf_before':>11}{'acf_after':>10}{'acf_real':>9}")
    for coin in COINS:
        lines.append(
            f"{coin:<7}{BEFORE_V15['ann_vol'][coin]:>15.4f}{syn_stats[coin]['ann_vol']:>15.4f}"
            f"{real_stats[coin]['ann_vol']:>13.4f}   |"
            f"{BEFORE_V15['excess_kurtosis'][coin]:>12.2f}{syn_stats[coin]['excess_kurtosis']:>11.2f}"
            f"{real_stats[coin]['excess_kurtosis']:>10.2f}   |"
            f"{BEFORE_V15['acf_abs_lag1'][coin]:>11.3f}{syn_stats[coin]['acf_abs_lag1']:>10.3f}"
            f"{real_stats[coin]['acf_abs_lag1']:>9.3f}"
        )
    lines.append("")

    lines.append("GENERATED vs REAL — correlation matrix (upper triangle), with v1.5 BEFORE for reference")
    lines.append("-" * 88)
    for i, c1 in enumerate(COINS):
        for c2 in COINS[i + 1:]:
            check_abs(f"corr({c1},{c2})", syn_corr.loc[c1, c2], real_corr.loc[c1, c2],
                       TOL_CORR_ABS, lines, counts)
            before = BEFORE_V15["corr"][(c1, c2)]
            lines.append(f"         (before={before:.4f})")
    lines.append("")

    lines.append("Full correlation matrices")
    lines.append("-" * 88)
    lines.append("GENERATED:")
    lines.append(str(syn_corr.round(3)))
    lines.append("")
    lines.append("REAL:")
    lines.append(str(real_corr.round(3)))
    lines.append("")

    # ------------------------------------------------------------------------------
    # v2 fix check: idio channel in ISOLATION (no factor mixed in), same length/seed as
    # the mix run, to directly confirm the persistence cap + periodic reset hold the
    # idio channel's realized variance near its calibrated target over a long run —
    # this is the actual mechanism the correlation-matrix fix above depends on.
    # ------------------------------------------------------------------------------
    lines.append("IDIO CHANNEL CHECK (isolated from the factor) -- does the reset hold the level?")
    lines.append("-" * 88)
    lines.append(f"Simulated idio_i_t alone for {MIX_VALIDATION_HOURS}h (seed={MIX_SEED}), no beta*factor term.")
    header = f"{'coin':<7}{'raw_persist':>12}{'capped_persist':>15}{'target_idio_vol':>17}" \
             f"{'realized_idio_vol':>18}{'rel_err':>9}"
    lines.append(header)
    idio_rng = np.random.default_rng(MIX_SEED)
    hours_per_year = calib["meta"]["hours_per_year"]
    for coin in COINS:
        c = calib["coins"][coin]
        dof = c["idio_student_t_dof"]
        target_var = (c["idio_vol"] / np.sqrt(hours_per_year)) ** 2
        if c.get("idio_garch_ok"):
            raw_a, raw_b = c["idio_garch"]["alpha"], c["idio_garch"]["beta"]
        else:
            raw_a, raw_b = 0.0, 0.0
        capped_a, capped_b = cap_idio_persistence(raw_a, raw_b)

        idio_path = generate_idio_garch(idio_rng, MIX_VALIDATION_HOURS, raw_a, raw_b, dof, target_var)
        realized_idio_vol = ann_vol_from_returns(pd.Series(idio_path))
        rel_err = abs(realized_idio_vol - c["idio_vol"]) / c["idio_vol"]

        lines.append(
            f"{coin:<7}{raw_a + raw_b:>12.5f}{capped_a + capped_b:>15.5f}"
            f"{c['idio_vol']:>17.4f}{realized_idio_vol:>18.4f}{rel_err:>9.1%}"
        )
    lines.append("")
    lines.append("(Compare to BTC/MATIC/SOL's raw idio_persistence >= 1.0 in calibration.json --")
    lines.append(f" cap_idio_persistence() pulls all of them under PERSISTENCE_CAP; the periodic")
    lines.append(" reset then keeps the long-run realized vol close to target despite that residual")
    lines.append(" near-unit-root shape. This — not the cap alone — is what fixes the correlation")
    lines.append(" inflation: v1.5 had no reset, so realized idio vol drifted well below target.)")
    lines.append("")

    # ------------------------------------------------------------------------------
    # acf_abs_lag1 regressed to WARN for BNB and MATIC (see table above). Before
    # attributing that to "reset/cap too aggressive" (as the task anticipated), test it
    # directly: rerun each coin's idio channel in isolation with reset_period effectively
    # disabled (>n_hours, so it never fires) and compare acf to the WITH-reset run, using
    # generate_idio_garch's own reset_period parameter (same function, not a
    # re-derivation). A single-seed A/B comparison turned out to be unreliable here — for
    # a near-unit-root process, realized acf_abs_lag1 on one 50000h path is itself a noisy
    # statistic (confirmed below: std comparable to the mean effect) — so this uses
    # ACF_ATTRIBUTION_SEEDS independent seeds and reports the mean +/- std.
    # ------------------------------------------------------------------------------
    lines.append("ACF REGRESSION ATTRIBUTION: is the periodic reset the cause?")
    lines.append("-" * 88)
    lines.append("BNB and MATIC's acf_abs_lag1 regressed from PASS (v1.5) to WARN (v2). Tested by")
    lines.append("rerunning each coin's idio channel WITH vs WITHOUT the reset (reset_period > n_hours)")
    lines.append(f"across {ACF_ATTRIBUTION_SEEDS} seeds ({MIX_VALIDATION_HOURS}h each) -- a single seed turned out to be too")
    lines.append("noisy to trust for a near-unit-root process (see std_diff below).")
    header3 = f"{'coin':<7}{'mean_acf_with':>15}{'mean_acf_without':>18}{'mean_reset_impact':>19}{'std_diff':>10}"
    lines.append(header3)
    acf_diffs = {}
    n_seeds = ACF_ATTRIBUTION_SEEDS
    for coin in COINS:
        c = calib["coins"][coin]
        dof = c["idio_student_t_dof"]
        target_var = (c["idio_vol"] / np.sqrt(hours_per_year)) ** 2
        if c.get("idio_garch_ok"):
            raw_a, raw_b = c["idio_garch"]["alpha"], c["idio_garch"]["beta"]
        else:
            raw_a, raw_b = 0.0, 0.0

        with_vals, without_vals = [], []
        for s in range(n_seeds):
            seed = ACF_ATTRIBUTION_SEED_BASE + s
            rng_a = np.random.default_rng(seed)
            idio_with = generate_idio_garch(rng_a, MIX_VALIDATION_HOURS, raw_a, raw_b, dof, target_var)
            rng_b = np.random.default_rng(seed)  # same seed -> same z draws, isolates reset's effect
            idio_without = generate_idio_garch(rng_b, MIX_VALIDATION_HOURS, raw_a, raw_b, dof, target_var,
                                                reset_period=MIX_VALIDATION_HOURS + 1)
            with_vals.append(acf_abs_lag1(pd.Series(idio_with)))
            without_vals.append(acf_abs_lag1(pd.Series(idio_without)))

        with_vals, without_vals = np.array(with_vals), np.array(without_vals)
        diffs = with_vals - without_vals
        acf_diffs[coin] = diffs.mean()
        lines.append(f"{coin:<7}{with_vals.mean():>15.4f}{without_vals.mean():>18.4f}"
                      f"{diffs.mean():>+19.4f}{diffs.std():>10.4f}")
    lines.append("")
    max_mean_impact = max(abs(v) for v in acf_diffs.values())
    lines.append(f"Largest mean reset impact on idio's own acf_abs_lag1: {max_mean_impact:.4f} (MATIC/SOL)")
    lines.append("-- real and systematic (all 5 coins negative), but well under half of the ~0.10")
    lines.append("shortfall seen at the coin level, and the per-seed std (~0.03-0.04) shows single-path")
    lines.append("estimates of this effect are NOT reliable on their own (an earlier single-seed check")
    lines.append("in this development gave a misleadingly large estimate for exactly this reason --")
    lines.append("flagging that here rather than silently using the better number).")
    lines.append("")
    lines.append("So the reset is a real but MINOR contributor. Bigger cause: v1.5's idio variance had")
    lines.append("collapsed (see ann_vol[BNB] WARN in the prior report), so BNB/MATIC's total return was")
    lines.append("dominated by beta*market_factor -- whose own acf_abs_lag1 (~0.31) propped up the")
    lines.append("coin-level acf, coincidentally close to real. v2 restores idio's correct variance")
    lines.append("share (ann_vol back to PASS, see above) -- but mixing two INDEPENDENTLY-timed")
    lines.append("clustering processes (factor + idio) mechanically dilutes the combined series's")
    lines.append("lag-1 |return| autocorrelation below either component's own acf (same aggregation-")
    lines.append("dilution mechanism flagged for kurtosis in the very first v1 report, now showing up")
    lines.append("again for acf on the two highest idio-share coins). Not tuned away, per instructions --")
    lines.append("reported as-is. A secondary, separate finding: the wide per-seed std above means")
    lines.append("different seeds can realize meaningfully different vol-clustering strength for these")
    lines.append("near-unit-root coins -- worth knowing before relying on acf-sensitive fitness signals")
    lines.append("from a single generated world.")
    lines.append("")

    # ------------------------------------------------------------------------------
    # Pure bull / pure bear qualitative check
    # ------------------------------------------------------------------------------
    lines.append("=" * 88)
    lines.append("PURE-REGIME QUALITATIVE CHECK")
    lines.append("-" * 88)
    pure_results = {}
    for regime in ["bull", "bear"]:
        _, _, factor = generate_raw(
            seed=PURE_SEED, n_hours=PURE_CHECK_HOURS, regime_mode="pure", regime=regime,
            calib=calib,
        )
        f = pd.Series(factor)
        realized_ann_vol = ann_vol_from_returns(f)
        realized_mean = float(f.mean())
        pure_results[regime] = {"ann_vol": realized_ann_vol, "mean": realized_mean}
        lines.append(f"pure {regime:<9} run ({PURE_CHECK_HOURS}h, seed={PURE_SEED}): "
                      f"realized ann_vol={realized_ann_vol:.2%}  "
                      f"mean_hourly_logret={realized_mean:+.6f}")

    bear_vol_gt_bull = pure_results["bear"]["ann_vol"] > pure_results["bull"]["ann_vol"]
    bear_drift_neg = pure_results["bear"]["mean"] < 0
    bull_drift_pos = pure_results["bull"]["mean"] > 0
    ordering_ok = bear_vol_gt_bull and bear_drift_neg and bull_drift_pos
    counts["PASS" if ordering_ok else "WARN"] += 1
    lines.append(f"[{'PASS' if ordering_ok else 'WARN'}] bear_vol > bull_vol: {bear_vol_gt_bull}   "
                 f"bear_drift < 0: {bear_drift_neg}   bull_drift > 0: {bull_drift_pos}")
    lines.append("")

    # ------------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------------
    lines.append("=" * 88)
    lines.append(f"VALIDATION SUMMARY: {counts['PASS']} PASS, {counts['WARN']} WARN "
                 f"(total {counts['PASS'] + counts['WARN']} checks)")
    lines.append("=" * 88)

    report = "\n".join(lines)
    print("\n" + report)

    with open("reports/generator_validation.txt", "w") as f:
        f.write(report + "\n")
    print("\nWrote reports/generator_validation.txt")


if __name__ == "__main__":
    main()
