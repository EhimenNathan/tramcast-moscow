# Research notebook (scripts)
Exploratory and validation scripts behind the production model (run from `work/` with labels in place).
* `floor.py` — irreducible-noise ceiling (≈0.92) · `noise.py`, `lowo.py` — stable-regime LOWO CV
* `wx.py` — weather regression (robust, split replication) · `hol.py` — holiday ratios
* `cands.py`, `stack.py` — candidate library + exact LP (WAPE-optimal) stacking
* `dzoo*.py`, `ada*.py`, `gru.py`, `zoo2_gru.py` (GRU + drift-symmetric synthetic augmentation), `zoo2_prophet.py`, `ensemble.py` — model tournament (raw vs drift-neutral)
* `kalman2.py` — hierarchical local-level (Kalman) filter · `shape_lr.py`, `sp_tests.py` — SVD / harmonic (tidal) shape smoothing, Hampel
* `shape_w.py` — seasonal (winter) shape interpolation · `forecast*.py`, `make_v2.py` — earlier submission builders
