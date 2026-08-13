# HELD-OUT REAL DATA -- LOCKED, MEASUREMENT ONLY

This directory contains the last 6 months of REAL recorded hourly OHLCV
(2024-03-10 02:00:00+00:00 -> 2024-09-10 02:00:00+00:00) for BTC, ETH, BNB, MATIC, SOL, carved out of the full common
window by calibrate.py and excluded from everything else in this pipeline.

DO NOT use this data for:
  - calibration (calibrate.py only ever calibrates on the TRAIN window, data/raw/ minus
    this slice -- see config/calibration.json's meta.split for the exact boundaries)
  - synthetic-data generation (generator.py is calibrated on TRAIN only)
  - evolution / training / hyperparameter tuning of any kind

The ONLY legitimate use of this data is evaluate_champion.py, run ONCE per finished
evolution run's best genome, to MEASURE (never tune) how it performs on real market
history it has never seen, directly or indirectly through the generator. If a champion
fails here, the fix is to revise the pipeline (calibrator/generator/evolution) and judge
the revision using GENERATED-world performance -- then come back to this held-out set
only for a fresh final check on the NEW champion. Never iterate against this data; every
look costs some of its power as an unbiased judge.

Files: {coin}USDT_1h_heldout.csv per coin, one per coin in ['BTC', 'ETH', 'BNB', 'MATIC', 'SOL'], same columns as
data/raw/ (timestamp,open,high,low,close,volume) -- real recorded rows only, gaps
preserved as-is (not filled or reindexed).
