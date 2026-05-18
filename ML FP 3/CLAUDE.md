# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project nature

This is a **single-notebook research project** for a Machine Learning course (Spring 2026), not a software library. All code lives in `wta_analysis_final.ipynb` (143 cells, ~3.7 MB, already executed). There is no test suite, no linter config, no build step. `wta_analysis_final.BACKUP.ipynb` is a manual backup — do not edit it as part of normal work.

## Environment & execution

- Python 3.13 (conda env recommended; see README §Reproducibility).
- `pip install -r requirements.txt` — pinned versions matter (torch 2.12.0, xgboost 2.1.1, lightgbm 4.6.0, catboost 1.2.10, shap 0.51.0).
- Run with `jupyter notebook wta_analysis_final.ipynb` → Kernel → Restart & Run All. End-to-end ~25–30 min on CPU; bottlenecks are XGBoost grid search (~1 min), Random Forest tuning (~40s), Hybrid LSTM grid search (~3.5 min).
- `SEED = 42` / `random_state=42` are used throughout for reproducibility.

## Pipeline architecture (notebook flow)

The notebook is structured as a linear pipeline across numbered markdown sections (§1–§14). Cells **must execute in order** — later cells depend on `df_all`, the engineered feature frame, and the fitted pipeline objects from earlier cells.

Conceptual stages:

1. **Load** (§2, §4): concatenates `data/wta_matches_2000.csv` … `wta_matches_2024.csv` into `df_all`. Includes a data-quality audit (dtypes, ranges, duplicates, date validity, missing rates).
2. **Feature engineering** (§5): builds pre-match features — surface-specific Elo (Hard/Clay/Grass), rolling form, rolling serve stats (Sipko 2015), common-opponent features (Knottenbelt 2012), momentum/fatigue. These are computed causally over time using the **2000–2014 burn-in window** (not part of train or test).
3. **Modeling dataset** (§7): symmetric train/test construction (each match emitted twice with player order swapped) + a unified sklearn preprocessing pipeline (imputation + scaling) reused by every classical model.
4. **Models** (§9, §11): three Elo baselines, six ML models (LogReg, calibrated LinearSVC, RF, XGBoost, LightGBM, CatBoost), and two NN models (MLP, Hybrid Siamese Bi-LSTM).
5. **Evaluation** (§10): bootstrap CIs (B=1000), CV-vs-test consistency, calibration, learning/validation curves, ROC, feature importance, sensitivity & ablation studies.
6. **Interpretation** (§10.5, §11.5, §12): SHAP global/local, PCA.

### The temporal split (load-bearing)

- **Burn-in:** 2000–2014 → initializes Elo ratings and rolling statistics only. Never used as train or test.
- **Train:** 2015–2022.
- **Test:** 2023–2024.

Any change to this split must propagate through Elo init, rolling features, leakage audit (§8.2), and CV folds (`TimeSeriesSplit`). Do not introduce a random shuffle split.

### MetricsRegistry — single source of truth

Cell 2 defines a `MetricsRegistry` class that persists to `results/final_metrics.json`. Every model cell writes its metrics via `registry.set(name, ...)` + `registry.save()`, and every downstream comparison/table/plot reads from `registry.get(name)`. **Do not hardcode metric numbers in markdown or downstream cells** — go through the registry. The registry starts empty on each full run and is rebuilt from scratch.

`results/bootstrap_ci_day2.json` is the analogous persisted store for bootstrap CIs.

## Known caveats

- **§9.3.1 VIF Ablation:** the cell is a placeholder with `RUN_ABLATION = False`. The numbers in the surrounding markdown come from a prior run that still included features `elo_diff` and `elo_surface_diff`; those features were subsequently dropped in the feature-selection cell upstream based on that ablation. To re-run, set the flag to `True` — do not silently change the markdown without re-executing.
- **LSTM grid search** (§11.3) emits a `multiprocessing.ResourceTracker` warning on Python 3.13 / macOS during DataLoader teardown. It is benign and does not affect results.
- `df_all` is mutated in place by the data-quality audit (duplicate dropping). Re-running cells out of order can produce inconsistent state — restart the kernel.

## Outputs

Generated into `results/`:
- `final_metrics.json` — all model metrics (regenerated each run via MetricsRegistry).
- `bootstrap_ci_day2.json` — bootstrap 95% CIs.
- `figures/*.png` — 14 figures referenced from the notebook markdown.
