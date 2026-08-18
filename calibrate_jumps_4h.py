"""
calibrate_jumps_4h.py — FIX 1's calibration step: find (jump_prob, jump_var_frac) per
coin, by simulation search, so generator_4h.py's compound-Poisson jump component makes
generated WINDOW-level excess kurtosis match REAL 4h window-level excess kurtosis. This
is a second calibration pass ON TOP OF calibrate_4h.py's analytic fit (GARCH/beta/
correlation), because jump-diffusion + GARCH doesn't have a clean closed-form MLE --
simulate-and-search is standard practice for this class of model.

TARGET: real 4h data chopped into WINDOW_BARS=84 (14-day) non-overlapping windows, mean
excess kurtosis per coin across those windows -- NOT the long-sample aggregate kurtosis
(that's what masked the original blind spot; see validate_generator_distributional.py).

SEARCH: coarse grid over (jump_prob, jump_var_frac) per coin, generating N_WINDOWS_COARSE
fresh 84-bar mix windows per candidate (fixed seed block, deterministic), picking the
candidate whose mean generated window-kurtosis is closest to the real target; then a
refinement pass with more windows on the winner for an accurate final report.

Writes the tuned params back into config/calibration_4h.json (coins.<coin>.jump_prob /
coins.<coin>.jump_var_frac), so generate_raw() picks them up automatically.

Run: python3 -u calibrate_jumps_4h.py
"""
import json
import time

import numpy as np
import pandas as pd

from calibrate_4h import load_data, compute_common_window, build_returns, COINS
from calibrate import excess_kurtosis

WINDOW_BARS = 84  # 14 days at 4h resolution -- same convention Part 3's validation will use

# Window-level excess kurtosis is a noisy 4th-moment estimator (well known -- see
# calibrate.py/validate_generator.py's own comments on this). An earlier version of this
# search used 10 windows for the grid pass and 40 for refinement, and the two disagreed
# wildly on the SAME parameters (e.g. BTC: 10-window estimate said kurt=4.19, the
# 40-window estimate for the identical params said kurt=2.29) -- the coarse pass was
# picking noise, not signal. Fixed by using the SAME, larger window count for every grid
# point (no separate under-sampled "coarse" stage), so every candidate is compared on
# equal, adequate footing.
P_JUMP_GRID = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
# 0.85 ceiling (not e.g. 0.95): generator_4h.MAX_JUMP_VAR_FRAC caps it there anyway --
# correlation (Fix 2) is only carried by the diffusion sub-component of idio, and an
# earlier search picking jump_var_frac=0.95 for BTC left too little diffusion variance
# for Fix 2's dilution-compensation to keep 2 correlation pairs within +-0.05 tolerance.
JUMP_VAR_FRAC_GRID = [0.5, 0.7, 0.85]
N_WINDOWS_SEARCH = 60
N_WINDOWS_REFINE = 200
SEARCH_SEED_BASE = 700000

CALIB_PATH = "config/calibration_4h.json"


def log(msg):
    print(msg, flush=True)


def real_window_kurtosis_targets():
    data = load_data()
    train_start, train_end, train_index, *_ = compute_common_window(data)
    close, returns, dropped = build_returns(data, train_index)
    n_windows = len(returns) // WINDOW_BARS
    targets = {}
    for coin in COINS:
        kurts = []
        for i in range(n_windows):
            r = returns[coin].iloc[i * WINDOW_BARS:(i + 1) * WINDOW_BARS].dropna()
            if len(r) >= 20:
                kurts.append(excess_kurtosis(r))
        targets[coin] = float(np.mean(kurts))
    return targets, n_windows


def simulate_window_kurtosis(calib, coin, jump_prob, jump_var_frac, n_windows, seed_base):
    """Generates n_windows fresh 84-bar mix worlds with candidate jump params for ONE
    coin (other coins' jump params temporarily left at whatever's already in calib --
    irrelevant here since we only measure this coin's own returns), returns the mean
    generated window-level excess kurtosis for that coin."""
    import generator_4h as g4h
    trial_calib = json.loads(json.dumps(calib))  # deep copy, don't mutate the shared dict
    trial_calib["coins"][coin]["jump_prob"] = jump_prob
    trial_calib["coins"][coin]["jump_var_frac"] = jump_var_frac

    kurts = []
    for i in range(n_windows):
        seed = seed_base + i
        dfs, _, _ = g4h.generate_raw(seed=seed, n_periods=WINDOW_BARS, regime_mode="mix", calib=trial_calib)
        close = dfs[coin]["close"].to_numpy()
        r = pd.Series(np.log(close)).diff().dropna()
        kurts.append(excess_kurtosis(r))
    return float(np.mean(kurts))


def search_coin(calib, coin, target_kurt):
    log(f"\n-- {coin}: target window-kurtosis = {target_kurt:.3f} --")
    best = None
    t0 = time.perf_counter()
    for jf in JUMP_VAR_FRAC_GRID:
        for pj in P_JUMP_GRID:
            seed_base = SEARCH_SEED_BASE + hash((coin, jf, pj)) % 100000
            k = simulate_window_kurtosis(calib, coin, pj, jf, N_WINDOWS_SEARCH, seed_base)
            err = abs(k - target_kurt)
            log(f"    p_jump={pj:<8}jump_var_frac={jf:<6}kurt={k:>7.3f}  err={err:.3f}")
            if best is None or err < best["err"]:
                best = {"jump_prob": pj, "jump_var_frac": jf, "kurt": k, "err": err}
    log(f"  grid search done in {time.perf_counter()-t0:.0f}s ({N_WINDOWS_SEARCH} windows/candidate) -- "
        f"best: p_jump={best['jump_prob']} jump_var_frac={best['jump_var_frac']} "
        f"kurt={best['kurt']:.3f} (err={best['err']:.3f})")

    t0 = time.perf_counter()
    refined_kurt = simulate_window_kurtosis(calib, coin, best["jump_prob"], best["jump_var_frac"],
                                             N_WINDOWS_REFINE, SEARCH_SEED_BASE + 999999)
    log(f"  refined ({N_WINDOWS_REFINE} windows, {time.perf_counter()-t0:.0f}s): "
        f"generated kurt={refined_kurt:.3f}  target={target_kurt:.3f}")
    best["refined_kurt"] = refined_kurt
    return best


def main():
    log("Computing real 4h window-level kurtosis targets...")
    targets, n_real_windows = real_window_kurtosis_targets()
    for coin, k in targets.items():
        log(f"  {coin}: target={k:.3f}")

    with open(CALIB_PATH) as f:
        calib = json.load(f)

    results = {}
    for coin in COINS:
        results[coin] = search_coin(calib, coin, targets[coin])
        calib["coins"][coin]["jump_prob"] = results[coin]["jump_prob"]
        calib["coins"][coin]["jump_var_frac"] = results[coin]["jump_var_frac"]
        # persist incrementally so a partial run isn't wasted
        with open(CALIB_PATH, "w") as f:
            json.dump(calib, f, indent=2)

    log(f"\n{'='*78}\nFIX 1 SUMMARY: generated vs real window-level excess kurtosis\n{'='*78}")
    log(f"{'coin':<8}{'target':>10}{'generated':>12}{'jump_prob':>12}{'jump_var_frac':>15}")
    for coin in COINS:
        r = results[coin]
        log(f"{coin:<8}{targets[coin]:>10.3f}{r['refined_kurt']:>12.3f}{r['jump_prob']:>12}{r['jump_var_frac']:>15}")

    log(f"\nWrote tuned jump_prob/jump_var_frac into {CALIB_PATH}")

    # -- quick correlation re-check now that jumps are calibrated (dilution-compensation
    # in compute_idio_correlation should keep this within tolerance -- verify directly) --
    import generator_4h as g4h
    calib_final = g4h.load_calibration()
    dfs, _, _ = g4h.generate_raw(seed=555, n_periods=20000, regime_mode="mix", calib=calib_final)
    close = pd.DataFrame({c: dfs[c]["close"].to_numpy() for c in COINS})
    returns = np.log(close).diff().dropna()
    achieved_corr = returns.corr()
    target_corr = pd.DataFrame(calib_final["correlation_matrix"])
    log(f"\n{'='*78}\nFIX 2 RE-CHECK (post-jump-calibration): achieved vs target correlation\n{'='*78}")
    max_dev = 0.0
    for i, c1 in enumerate(COINS):
        for c2 in COINS[i + 1:]:
            a, t = achieved_corr.loc[c1, c2], target_corr.loc[c1, c2]
            dev = abs(a - t)
            max_dev = max(max_dev, dev)
            flag = "OUTSIDE +/-0.05" if dev > 0.05 else ""
            log(f"  corr({c1},{c2}): achieved={a:.3f}  target={t:.3f}  dev={dev:.3f}  {flag}")
    log(f"\nMax deviation across all pairs: {max_dev:.3f}  ({'PASS' if max_dev <= 0.05 else 'FAIL'} +/-0.05 tolerance)")


if __name__ == "__main__":
    main()
