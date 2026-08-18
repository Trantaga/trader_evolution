# HELD-OUT REAL 4-HOUR DATA -- LOCKED, MEASUREMENT ONLY

This directory contains the last 6 months of REAL recorded 4-hour OHLCV
(2024-03-10 00:00:00+00:00 -> 2024-09-10 00:00:00+00:00) for BTC, ETH, BNB, MATIC, SOL, carved out by calibrate_4h.py
and excluded from calibration_4h.json / generator_4h.py entirely. Same discipline as
data/heldout/ (the hourly held-out set) -- measurement only, never tuning.

Files: {coin}USDT_4h_heldout.csv per coin, same columns as data/raw_4h/.
