"""
temporal_core.py — temporal generalisation of PV power estimation at a single plant.

Data : FMI Helsinki PV plant, hourly, 2016–2021.
Cases: A_5yr  — train 2016–2020, test 2021
       B_1yr  — train 2020,      test 2021
Models: kPOA (physical reference), MLR, KNN, SVR, RandomForest, XGBoost, LSTM.
"""

from __future__ import annotations

import os
import sys
import time
import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
log = logging.getLogger("temporal")


def section(title: str, char: str = "=", width: int = 72) -> None:
    log.info(char * width); log.info(title); log.info(char * width)


# ════════════════════════════════════════════════════════════════════════
# 1) Configuration
# ════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class TemporalConfig:
    base_dir: str = "."
    data_filename: str = os.path.join("data", "Helsinki_MultiYear.csv")
    results_folder_name: str = "results"

    time_col: str = "utctime"
    site_col: str = "Site_ID"
    year_col: str = "Year"
    target_col: str = "Power_PU"

    # Split
    test_year: int = 2021
    long_history: tuple = (2016, 2017, 2018, 2019, 2020)
    short_history: tuple = (2020,)
    case_long: str = "A_5yr"
    case_short: str = "B_1yr"

    # Features
    base_features: tuple = (
        "HOUR_SIN", "HOUR_COS", "MONTH_SIN", "MONTH_COS",
        "GHI", "Temp_Ambient", "Wind_Speed", "Humidity",
    )
    poa_feature: str = "Irradiance_POA"
    cyclic_features: tuple = ("HOUR_SIN", "HOUR_COS", "MONTH_SIN", "MONTH_COS")

    # Lag features (XGBoost and LSTM only)
    lag_hours: tuple = (1, 2, 3, 24)
    lag_source_cols: tuple = ("Irradiance_POA", "GHI", "Temp_Ambient")
    lag_algorithms: tuple = ("XGBoost", "LSTM")

    # Scaling (fitted on training years only)
    scaled_algorithms: tuple = ("KNN", "SVR", "LSTM")

    # LSTM
    lstm_lookback: int = 24
    lstm_units: int = 64
    lstm_epochs: int = 60
    lstm_batch_size: int = 256
    lstm_max_gap_fill_hours: int = 3
    lstm_seeds: tuple = (42, 43, 44, 45, 46)

    # Data-quality flags
    imputed_target_col: str = "Power_Actual_W_Imputed"
    outage_max_daily_pu: float = 0.01        # daily sum of Power_PU below this = no output
    outage_min_daily_poa: float = 1000.0     # daily sum of POA (Wh/m²) above this = sunny
    outage_min_mean_temp: float = 5.0        # daily mean temperature (°C)
    outage_max_hourly_pu: float = 0.001      # hourly output below this = zero
    outage_min_hourly_poa: float = 100.0     # hourly POA (W/m²) above this = sunny

    # Evaluation: daytime hours only (solar elevation > 0°)
    day_min_elevation: float = 0.0
    elevation_col: str = "Solar_Elevation"

    # Statistics
    bootstrap_n: int = 5000
    reference_model: str = "kPOA"

    daytime_ghi_threshold: float = 20.0
    mape_min_power_pu: float = 0.05

    permutation_repeats: int = 3
    permutation_max_samples: int = 3000

    # Hyperparameter tuning (inner split inside the long training history)
    tune_algorithms: tuple = ("KNN", "SVR")
    inner_validation_year: int = 2020
    inner_train_years: tuple = (2016, 2017, 2018, 2019)
    tuning_subsample: int | None = 8000

    param_grids: dict = field(default_factory=lambda: {
        "KNN": [{"n_neighbors": k, "weights": w}
                for k in (15, 50, 150) for w in ("uniform", "distance")],
        "SVR": [{"C": c, "gamma": g} for c in (1.0, 10.0, 100.0) for g in ("scale", 0.01)],
    })

    random_seed: int = 42

    @property
    def data_path(self) -> str:
        return os.path.join(self.base_dir, self.data_filename)

    @property
    def results_dir(self) -> str:
        return os.path.join(self.base_dir, self.results_folder_name)

    @property
    def predictions_dir(self) -> str:
        return os.path.join(self.results_dir, "predictions")

    @property
    def models_dir(self) -> str:
        return os.path.join(self.results_dir, "models")


CFG = TemporalConfig()
np.random.seed(CFG.random_seed)

KNOWN_ALGOS = ("kPOA", "MLR", "KNN", "RandomForest", "XGBoost", "SVR", "LSTM")


def _out(folder: str, name: str) -> str:
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


# ════════════════════════════════════════════════════════════════════════
# 2) Data and cases
# ════════════════════════════════════════════════════════════════════════
def add_quality_flags(df: pd.DataFrame, cfg: TemporalConfig = CFG,
                      quiet: bool = False) -> pd.DataFrame:
    """Adds `Outage`, `Target_Measured`, `Target_Valid` and `Daytime`."""
    df = df.copy()
    day = df[cfg.time_col].dt.date
    g = df.groupby(day)
    e = g[cfg.target_col].transform("sum")
    h = g[cfg.poa_feature].transform("sum")
    tmp = g["Temp_Ambient"].transform("mean")
    warm = tmp > cfg.outage_min_mean_temp
    day_out = (e < cfg.outage_max_daily_pu) & (h > cfg.outage_min_daily_poa) & warm
    hour_out = ((df[cfg.target_col] < cfg.outage_max_hourly_pu)
                & (df[cfg.poa_feature] > cfg.outage_min_hourly_poa) & warm)
    imputed = (df[cfg.imputed_target_col].astype(bool) if cfg.imputed_target_col in df
               else pd.Series(False, index=df.index))
    df["Outage"] = ((day_out | hour_out) & ~imputed).to_numpy()
    df["Target_Measured"] = ~imputed.to_numpy()
    df["Target_Valid"] = df["Target_Measured"] & ~df["Outage"]
    if cfg.elevation_col in df:
        df["Daytime"] = (df[cfg.elevation_col] > cfg.day_min_elevation).to_numpy()
    else:
        df["Daytime"] = (df["GHI"] > 0).to_numpy()

    if quiet:
        return df
    section("Data-quality flags", char="-")
    s = df.groupby(cfg.year_col).agg(
        missing_target_h=("Target_Measured", lambda x: int((~x).sum())),
        outage_h=("Outage", "sum"),
        valid_h=("Target_Valid", "sum"),
        valid_daytime_h=("Daytime", lambda x: int((x & df.loc[x.index, "Target_Valid"]).sum())))
    print(s.to_string())
    od = sorted({d for d, o in zip(day, df["Outage"]) if o})
    if od:
        print(f"\nOutage days ({len(od)}): " + ", ".join(str(d) for d in od))
    return df


def build_cases(cfg: TemporalConfig = CFG) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Reads the multi-year file and returns {case: (train, test)}."""
    section("Building cases")
    df = (pd.read_csv(cfg.data_path, parse_dates=[cfg.time_col])
          .sort_values(cfg.time_col).reset_index(drop=True))
    df = add_quality_flags(df, cfg)

    test = df[df[cfg.year_col] == cfg.test_year].reset_index(drop=True)
    cases = {
        cfg.case_long: (df[df[cfg.year_col].isin(cfg.long_history)].reset_index(drop=True), test),
        cfg.case_short: (df[df[cfg.year_col].isin(cfg.short_history)].reset_index(drop=True), test),
    }
    for name, (tr, te) in cases.items():
        log.info("%-8s train %s rows (%s) | test %s rows (%d)", name, f"{len(tr):,}",
                 "–".join(str(y) for y in sorted(tr[cfg.year_col].unique())),
                 f"{len(te):,}", cfg.test_year)
        overlap = set(tr[cfg.time_col]) & set(te[cfg.time_col])
        if overlap:
            raise AssertionError(f"temporal overlap in {name}: {len(overlap)} hours")
        if tr[cfg.year_col].max() >= cfg.test_year:
            raise AssertionError(f"{name} trains on a year at or after the test year")
    return cases


# ════════════════════════════════════════════════════════════════════════
# 3) Features and models
# ════════════════════════════════════════════════════════════════════════
def add_lag_features(df: pd.DataFrame, cfg: TemporalConfig) -> pd.DataFrame:
    df = df.sort_values([cfg.site_col, cfg.time_col]).copy()
    for col in cfg.lag_source_cols:
        for lag in cfg.lag_hours:
            df[f"{col}_lag{lag}h"] = df.groupby(cfg.site_col)[col].shift(lag)
    return df


def lag_feature_names(cfg: TemporalConfig = CFG) -> list[str]:
    return [f"{c}_lag{l}h" for c in cfg.lag_source_cols for l in cfg.lag_hours]


def build_feature_list(cfg: TemporalConfig, algorithm: str) -> list[str]:
    features = list(cfg.base_features) + [cfg.poa_feature]
    if algorithm in cfg.lag_algorithms:
        features += lag_feature_names(cfg)
    return features


def scale_if_needed(train_df, test_df, features, algorithm, cfg):
    """StandardScaler fitted on training years only; cyclic features are not scaled."""
    from sklearn.preprocessing import StandardScaler

    X_tr = train_df[features].to_numpy(dtype=float)
    X_te = test_df[features].to_numpy(dtype=float)
    if algorithm not in cfg.scaled_algorithms:
        return X_tr, X_te, None

    mask = np.array([f not in cfg.cyclic_features for f in features])
    scaler = StandardScaler().fit(X_tr[:, mask])
    X_tr, X_te = X_tr.copy(), X_te.copy()
    X_tr[:, mask] = scaler.transform(X_tr[:, mask])
    X_te[:, mask] = scaler.transform(X_te[:, mask])
    return X_tr, X_te, scaler


def build_model(algorithm: str, cfg: TemporalConfig = CFG, params: dict | None = None):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR

    p = dict(params or {})
    if algorithm == "MLR":
        return LinearRegression()
    if algorithm == "RandomForest":
        return RandomForestRegressor(**{"n_estimators": 300, "min_samples_leaf": 2, **p},
                                     n_jobs=-1, random_state=cfg.random_seed)
    if algorithm == "KNN":
        return KNeighborsRegressor(**{"n_neighbors": 15, "weights": "distance", **p}, n_jobs=-1)
    if algorithm == "SVR":
        return SVR(kernel="rbf", **{"C": 10.0, "epsilon": 0.01, "gamma": "scale", **p})
    if algorithm == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(**{"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
                               "subsample": 0.8, "colsample_bytree": 0.8, **p},
                            random_state=cfg.random_seed, n_jobs=-1, verbosity=0)
    raise ValueError(f"Unknown algorithm: {algorithm}")


# ════════════════════════════════════════════════════════════════════════
# 4) LSTM
# ════════════════════════════════════════════════════════════════════════
def _reindex_hourly(site_df, cfg, feature_cols):
    """Continuous hourly index; only features are interpolated, never the target."""
    site = site_df[cfg.site_col].iloc[0]
    df = site_df.set_index(cfg.time_col).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz=df.index.tz)
    n_missing, n_total = len(full) - len(df), len(full)

    df = df.reindex(full)
    df[list(feature_cols)] = df[list(feature_cols)].interpolate(
        method="linear", limit=cfg.lstm_max_gap_fill_hours, limit_direction="both")
    df[cfg.site_col] = site
    df.index.name = cfg.time_col
    return df.reset_index(), n_missing, n_total


def build_lstm_dataset(df, features, cfg, quiet: bool = False):
    """Windows [t-lookback+1, t]; windows containing NaN are dropped."""
    Xs, ys, ds, ts = [], [], [], []
    for site, sdf in df.groupby(cfg.site_col):
        sdf = sdf.sort_values(cfg.time_col).reset_index(drop=True)
        rdf, n_missing, n_total = _reindex_hourly(sdf, cfg, features)

        X = rdf[features].to_numpy(dtype=float)
        y = rdf[cfg.target_col].to_numpy(dtype=float)
        day = (rdf["GHI_raw"].to_numpy(dtype=float) > cfg.daytime_ghi_threshold
               if "GHI_raw" in rdf.columns else np.ones(len(rdf), dtype=bool))

        times = rdf[cfg.time_col].to_numpy()
        lb = cfg.lstm_lookback
        for end in range(lb - 1, len(X)):
            w, tgt = X[end - lb + 1:end + 1], y[end]
            if np.isnan(w).any() or np.isnan(tgt):
                continue
            Xs.append(w); ys.append(tgt); ds.append(day[end]); ts.append(times[end])

        if not quiet:
            log.info("   [%s] hours %s | missing %s (%.1f%%) | valid windows %s", site,
                     f"{n_total:,}", f"{n_missing:,}",
                     100 * n_missing / n_total if n_total else 0, f"{len(Xs):,}")

    if not Xs:
        n = len(features)
        return (np.empty((0, cfg.lstm_lookback, n)), np.empty((0,)),
                np.empty((0,), bool), np.empty((0,), "datetime64[ns]"))
    return np.stack(Xs), np.array(ys), np.array(ds, bool), np.array(ts)


def build_lstm_model(n_features: int, cfg: TemporalConfig = CFG, seed: int | None = None):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.keras.utils.set_random_seed(cfg.random_seed if seed is None else seed)
    model = models.Sequential([
        layers.Input(shape=(cfg.lstm_lookback, n_features)),
        layers.LSTM(cfg.lstm_units),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ════════════════════════════════════════════════════════════════════════
# 5) Metrics and feature importance
# ════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, cfg, daytime_mask=None) -> dict:
    """R², RMSE, MAE, nRMSE/nMAE (% of nominal power) and daytime MAPE/sMAPE."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    m = {"R2": float(r2_score(y_true, y_pred)),
         "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
         "MAE": float(mean_absolute_error(y_true, y_pred))}
    m["nMAE_pct"] = m["MAE"] * 100.0
    m["nRMSE_pct"] = m["RMSE"] * 100.0

    mask = y_true > cfg.mape_min_power_pu
    if daytime_mask is not None:
        mask = mask & daytime_mask

    if mask.sum():
        yt, yp = y_true[mask], y_pred[mask]
        m["MAPE_daytime"] = float(np.mean(np.abs((yt - yp) / yt)) * 100)
        m["sMAPE_daytime"] = float(np.mean(2 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp))) * 100)
        m["n_mape_points"] = int(mask.sum())
    else:
        m["MAPE_daytime"] = m["sMAPE_daytime"] = np.nan
        m["n_mape_points"] = 0
    return m


def permutation_feature_importance(predict_fn: Callable, X, y_true, feature_names,
                                   is_sequence=False, n_repeats=3, seed=42,
                                   max_samples=None) -> dict[str, float]:
    """Mean drop in R² when one feature is permuted."""
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(seed)
    if max_samples and X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X, y_true = X[idx], y_true[idx]

    base = r2_score(y_true, predict_fn(X))
    out = {}
    for i, feat in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            p = rng.permutation(X.shape[0])
            if is_sequence:
                Xp[:, :, i] = Xp[p][:, :, i]
            else:
                Xp[:, i] = Xp[p, i]
            drops.append(base - r2_score(y_true, predict_fn(Xp)))
        out[feat] = float(np.mean(drops))
    return out


# ════════════════════════════════════════════════════════════════════════
# 6) Training and evaluation
# ════════════════════════════════════════════════════════════════════════
def _flags(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    valid = df["Target_Valid"].to_numpy(bool) if "Target_Valid" in df else np.ones(n, bool)
    meas = df["Target_Measured"].to_numpy(bool) if "Target_Measured" in df else np.ones(n, bool)
    day = df["Daytime"].to_numpy(bool) if "Daytime" in df else np.ones(n, bool)
    return valid, meas, day


def run_single_case(algorithm, case_name, train_df, test_df, cfg: TemporalConfig = CFG,
                    params: dict | None = None, quiet: bool = False,
                    return_score_only: bool = False, seed: int | None = None,
                    tag: str = "", with_importance: bool = True):
    """Trains one algorithm on one case. Returns (metrics, importance).

    Training uses `Target_Valid` hours only. Predictions are clipped at zero.
    Main metrics (`Eval_Main`): valid daytime hours.
    `InclOutage_*` metrics (`Eval_Full`): measured daytime hours, outage included.
    """
    import joblib

    features = build_feature_list(cfg, algorithm)
    if not quiet:
        log.info("-> %s | %s", algorithm, case_name + (f" {tag}" if tag else ""))

    train_df, test_df = train_df.copy(), test_df.copy()
    if algorithm in cfg.lag_algorithms:
        train_df = add_lag_features(train_df, cfg)
        test_df = add_lag_features(test_df, cfg)
        train_df = train_df.dropna(subset=features + [cfg.target_col])
        test_df = test_df.dropna(subset=features + [cfg.target_col])

    if algorithm == "LSTM":
        tr_valid, _, _ = _flags(train_df)
        train_df.loc[~tr_valid, cfg.target_col] = np.nan

        X_tr_s, X_te_s, _ = scale_if_needed(train_df, test_df, features, algorithm, cfg)
        tr_s, te_s = train_df.copy(), test_df.copy()
        tr_s[features], te_s[features] = X_tr_s, X_te_s
        tr_s["GHI_raw"] = train_df["GHI"].to_numpy()
        te_s["GHI_raw"] = test_df["GHI"].to_numpy()

        Xtr, ytr, _, _ = build_lstm_dataset(tr_s, features, cfg, quiet=quiet)
        Xte, yte, dte, tte = build_lstm_dataset(te_s, features, cfg, quiet=quiet)
        if len(Xte) == 0:
            log.warning("No valid windows; skipped.")
            return {}, {}

        model = build_lstm_model(len(features), cfg, seed)
        _t = time.time()
        model.fit(Xtr, ytr, epochs=cfg.lstm_epochs, batch_size=cfg.lstm_batch_size, verbose=0)
        fit_s = time.time() - _t

        _t = time.time()
        y_pred = np.maximum(0.0, model.predict(Xte, verbose=0).ravel())
        pred_s = time.time() - _t
        y_true, daytime, n_train, times = yte, dte, len(ytr), tte

        v_all, m_all, d_all = _flags(test_df)
        fl = pd.DataFrame({"v": v_all, "m": m_all, "d": d_all},
                          index=test_df[cfg.time_col].to_numpy())
        al = fl.reindex(pd.to_datetime(times))
        is_day = al["d"].fillna(False).to_numpy(bool)
        valid = al["v"].fillna(False).to_numpy(bool) & is_day
        measured = al["m"].fillna(False).to_numpy(bool) & is_day

        if return_score_only:
            return compute_metrics(y_true[valid], y_pred[valid], cfg, daytime[valid])["R2"]

        model.save(_out(cfg.models_dir, f"LSTM_{case_name}{tag}.keras"))
        importance = (permutation_feature_importance(
            lambda X: model.predict(X, verbose=0).ravel(), Xte[valid], y_true[valid], features,
            True, cfg.permutation_repeats, cfg.random_seed, cfg.permutation_max_samples)
            if with_importance else {})

    else:
        tr_valid, _, _ = _flags(train_df)
        train_df = train_df[tr_valid]
        valid, measured, is_day = _flags(test_df)
        valid, measured = valid & is_day, measured & is_day

        X_tr, X_te, scaler = scale_if_needed(train_df, test_df, features, algorithm, cfg)
        y_train = train_df[cfg.target_col].to_numpy(dtype=float)
        y_true = test_df[cfg.target_col].to_numpy(dtype=float)

        if algorithm == "kPOA":
            # Least squares through the origin on the training years
            poa_tr = train_df[cfg.poa_feature].to_numpy(dtype=float)
            _t = time.time()
            k = float((poa_tr * y_train).sum() / (poa_tr ** 2).sum())
            fit_s = time.time() - _t
            _t = time.time()
            y_pred = np.maximum(0.0, k * test_df[cfg.poa_feature].to_numpy(dtype=float))
            pred_s = time.time() - _t
            params = {"k": round(k, 7), "k_x1000": round(k * 1000, 3)}
            model = None
        else:
            model = build_model(algorithm, cfg, params)
            _t = time.time()
            model.fit(X_tr, y_train)
            fit_s = time.time() - _t
            _t = time.time()
            y_pred = np.maximum(0.0, model.predict(X_te))
            pred_s = time.time() - _t

        daytime = (test_df["GHI"] > cfg.daytime_ghi_threshold).to_numpy()
        n_train = len(y_train)
        times = test_df[cfg.time_col].to_numpy()
        if return_score_only:
            return compute_metrics(y_true[valid], y_pred[valid], cfg, daytime[valid])["R2"]

        if model is not None:
            joblib.dump({"model": model, "scaler": scaler, "features": features},
                        _out(cfg.models_dir, f"{algorithm}_{case_name}{tag}.joblib"))
            importance = (permutation_feature_importance(
                model.predict, X_te[valid], y_true[valid], features, False,
                cfg.permutation_repeats, cfg.random_seed, cfg.permutation_max_samples)
                if with_importance else {})
        else:
            importance = {}

    metrics = compute_metrics(y_true[valid], y_pred[valid], cfg, daytime[valid])
    full = compute_metrics(y_true[measured], y_pred[measured], cfg, daytime[measured])
    metrics.update({f"InclOutage_{k}": full[k]
                    for k in ("R2", "RMSE", "MAE", "nRMSE_pct", "nMAE_pct")})
    metrics["n_valid"] = int(valid.sum())
    metrics["n_measured"] = int(measured.sum())
    metrics["Bias"] = float(np.mean(y_pred[valid] - y_true[valid]))

    pd.DataFrame({cfg.time_col: times, "y_true": y_true, "y_pred": y_pred,
                  "Eval_Main": valid, "Eval_Full": measured}).to_csv(
        _out(cfg.predictions_dir, f"{algorithm}_{case_name}{tag}.csv"), index=False)

    result = {"Model": algorithm, "Case": case_name,
              "Params": str(params) if params else "default",
              "Seed": seed if seed is not None else cfg.random_seed,
              "n_train": int(n_train), "n_test": int(len(y_true)),
              "Train_s": round(fit_s, 3), "Predict_s": round(pred_s, 3),
              **metrics}
    log.info("   R²=%.3f  RMSE=%.4f  MAE=%.4f  |  train %.1fs  predict %.2fs",
             metrics["R2"], metrics["RMSE"], metrics["MAE"], fit_s, pred_s)
    return result, importance


def tune_on_training_years(algorithm, cases, cfg: TemporalConfig = CFG):
    """Grid search on the inner split (train `inner_train_years`, validate
    `inner_validation_year`). The test year is never used. Returns (params, inner R²)."""
    grid = cfg.param_grids.get(algorithm)
    if algorithm not in cfg.tune_algorithms or not grid:
        log.info("   %s: no tuning (fixed values)", algorithm)
        return {}, None

    train_long = cases[cfg.case_long][0]
    inner_tr = train_long[train_long[cfg.year_col].isin(cfg.inner_train_years)]
    inner_va = train_long[train_long[cfg.year_col] == cfg.inner_validation_year]
    if inner_tr.empty or inner_va.empty:
        log.warning("   inner split empty; fixed values kept.")
        return {}, None

    if cfg.tuning_subsample and len(inner_tr) > cfg.tuning_subsample:
        inner_tr = inner_tr.sample(cfg.tuning_subsample, random_state=cfg.random_seed)

    log.info("   %s | %d combinations | inner train %s -> validation %d", algorithm, len(grid),
             "–".join(str(y) for y in cfg.inner_train_years), cfg.inner_validation_year)

    scores = []
    for params in grid:
        try:
            r2 = run_single_case(algorithm, "inner", inner_tr, inner_va, cfg,
                                 params, quiet=True, return_score_only=True)
        except Exception as exc:                          # noqa: BLE001
            log.warning("      x %s -> %s", params, exc)
            continue
        scores.append((r2, params))
        log.info("      %-45s R²=%.4f", str(params), r2)

    if not scores:
        return {}, None
    best_r2, best = max(scores, key=lambda x: x[0])
    log.info("   best: %s  (inner R² = %.4f)", best, best_r2)
    return best, best_r2


def run_algorithm(algorithm: str, cases: dict, cfg: TemporalConfig = CFG,
                  tune: bool = True):
    """Runs one algorithm on both cases with the same tuned parameters."""
    section(f"{algorithm} | {len(cases)} cases")

    params, inner_r2 = ({}, None)
    if tune:
        params, inner_r2 = tune_on_training_years(algorithm, cases, cfg)

    results, importances = [], []
    for name, (tr, te) in cases.items():
        if algorithm == "LSTM":
            # Several seeds; importance on the first seed only.
            # The median-R² run is copied to `LSTM_<case>.csv`.
            runs = []
            for i, s in enumerate(cfg.lstm_seeds):
                r, imp = run_single_case(algorithm, name, tr, te, cfg, params, seed=s,
                                         tag=f"_s{s}", with_importance=(i == 0))
                if not r:
                    continue
                r["Inner_R2"] = inner_r2
                results.append(r); runs.append(r)
                importances += [{"Model": algorithm, "Case": name, "Feature": f, "Importance": v}
                                for f, v in imp.items()]
            if runs:
                r2 = np.array([x["R2"] for x in runs])
                med = runs[int(np.argsort(r2)[len(r2) // 2])]["Seed"]
                src = os.path.join(cfg.predictions_dir, f"LSTM_{name}_s{med}.csv")
                pd.read_csv(src).to_csv(os.path.join(cfg.predictions_dir, f"LSTM_{name}.csv"),
                                        index=False)
                log.info("   LSTM %s: R² = %.4f ± %.4f (%d seeds) | median seed %s", name,
                         r2.mean(), r2.std(ddof=1) if len(r2) > 1 else 0, len(r2), med)
            continue

        r, imp = run_single_case(algorithm, name, tr, te, cfg, params)
        if r:
            r["Inner_R2"] = inner_r2
            results.append(r)
            importances += [{"Model": algorithm, "Case": name, "Feature": f, "Importance": v}
                            for f, v in imp.items()]

    res_df, imp_df = pd.DataFrame(results), pd.DataFrame(importances)
    res_df.to_csv(_out(cfg.results_dir, f"results_{algorithm}.csv"), index=False)
    imp_df.to_csv(_out(cfg.results_dir, f"importance_{algorithm}.csv"), index=False)
    return res_df, imp_df


def run_all(cases: dict, cfg: TemporalConfig = CFG, algorithms=KNOWN_ALGOS, tune: bool = True):
    res, imp = [], []
    for algo in algorithms:
        r, i = run_algorithm(algo, cases, cfg, tune=tune)
        res.append(r); imp.append(i)
    results = pd.concat(res, ignore_index=True)
    importance = pd.concat(imp, ignore_index=True)
    results.to_csv(_out(cfg.results_dir, "all_results.csv"), index=False)
    importance.to_csv(_out(cfg.results_dir, "all_importance.csv"), index=False)
    return results, importance


# ════════════════════════════════════════════════════════════════════════
# 7) Loading saved results
# ════════════════════════════════════════════════════════════════════════
def _load_saved(kind: str, cfg: TemporalConfig = CFG) -> pd.DataFrame:
    """Reads `{kind}_<algorithm>.csv` by explicit name (never by glob)."""
    frames = []
    for algo in KNOWN_ALGOS:
        f = os.path.join(cfg.results_dir, f"{kind}_{algo}.csv")
        if not os.path.exists(f):
            continue
        try:
            d = pd.read_csv(f)
        except pd.errors.EmptyDataError:            # kPOA has no importance
            continue
        if not d.empty:
            frames.append(d)
    if not frames:
        raise RuntimeError(f"No saved `{kind}_*.csv` files in {cfg.results_dir}")
    return pd.concat(frames, ignore_index=True)


def load_results(cfg: TemporalConfig = CFG) -> pd.DataFrame:
    return _load_saved("results", cfg)


def load_importance(cfg: TemporalConfig = CFG) -> pd.DataFrame:
    return _load_saved("importance", cfg)


def load_predictions(algorithm: str, cfg: TemporalConfig = CFG,
                     case: str | None = None) -> pd.DataFrame:
    case = case or cfg.case_long
    f = os.path.join(cfg.predictions_dir, f"{algorithm}_{case}.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f)
    d = pd.read_csv(f)
    d[cfg.time_col] = pd.to_datetime(d[cfg.time_col])
    return d.sort_values(cfg.time_col).reset_index(drop=True)


def _available(cfg: TemporalConfig, case: str) -> list[str]:
    return [m for m in KNOWN_ALGOS
            if os.path.exists(os.path.join(cfg.predictions_dir, f"{m}_{case}.csv"))]


# ════════════════════════════════════════════════════════════════════════
# 8) Statistics
# ════════════════════════════════════════════════════════════════════════
def compare_cases(results: pd.DataFrame | None = None,
                  cfg: TemporalConfig = CFG) -> pd.DataFrame:
    """R² per model and case (LSTM: mean over seeds); Diff = A − B."""
    if results is None:
        results = load_results(cfg)
    section("Case comparison (R²)")
    pivot = results.pivot_table(index="Model", columns="Case", values="R2")
    pivot["Diff"] = pivot[cfg.case_long] - pivot[cfg.case_short]
    pivot = pivot.sort_values(cfg.case_long, ascending=False)
    print(pivot.round(4).to_string())
    pivot.to_csv(_out(cfg.results_dir, "case_comparison.csv"))
    return pivot


def _aligned_predictions(models, cases, cfg: TemporalConfig = CFG,
                         mask_col: str = "Eval_Main") -> pd.DataFrame:
    """Aligns predictions on the timestamp; keeps hours present in all and flagged by `mask_col`."""
    out = None
    for m in models:
        for c in cases:
            d = load_predictions(m, cfg, c)
            keep = d[mask_col].astype(bool) if mask_col in d else True
            d = d[keep][[cfg.time_col, "y_true", "y_pred"]].rename(
                columns={"y_pred": f"{m}|{c}", "y_true": f"y|{m}|{c}"})
            out = d if out is None else out.merge(d, on=cfg.time_col, how="inner")
    ycols = [c for c in out.columns if c.startswith("y|")]
    out["y"] = out[ycols[0]]
    return out.drop(columns=ycols).sort_values(cfg.time_col).reset_index(drop=True)


def daily_bootstrap(err_a: np.ndarray, err_b: np.ndarray, days: np.ndarray,
                    n_boot: int, seed: int) -> dict:
    """Daily block bootstrap of ΔRMSE = RMSE(a) − RMSE(b) and ΔMAE. Negative = a is better."""
    codes, idx = np.unique(days, return_inverse=True)
    nd = len(codes)
    cnt = np.bincount(idx, minlength=nd).astype(float)
    sa, sb = np.bincount(idx, err_a ** 2, nd), np.bincount(idx, err_b ** 2, nd)
    aa, ab = np.bincount(idx, np.abs(err_a), nd), np.bincount(idx, np.abs(err_b), nd)

    rng = np.random.default_rng(seed)
    d_rmse, d_mae = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, nd, nd)
        n = cnt[s].sum()
        d_rmse[i] = np.sqrt(sa[s].sum() / n) - np.sqrt(sb[s].sum() / n)
        d_mae[i] = aa[s].sum() / n - ab[s].sum() / n

    obs_rmse = np.sqrt(np.mean(err_a ** 2)) - np.sqrt(np.mean(err_b ** 2))
    obs_mae = np.mean(np.abs(err_a)) - np.mean(np.abs(err_b))
    p_r = min(1.0, 2 * min((d_rmse <= 0).mean(), (d_rmse >= 0).mean()))
    p_m = min(1.0, 2 * min((d_mae <= 0).mean(), (d_mae >= 0).mean()))
    lo_r, hi_r = np.percentile(d_rmse, [2.5, 97.5])
    lo_m, hi_m = np.percentile(d_mae, [2.5, 97.5])
    return {"dRMSE": obs_rmse, "dRMSE_lo": lo_r, "dRMSE_hi": hi_r, "p_RMSE": p_r,
            "dMAE": obs_mae, "dMAE_lo": lo_m, "dMAE_hi": hi_m, "p_MAE": p_m,
            "n_days": nd}


def bootstrap_compare(cfg: TemporalConfig = CFG, models=None,
                      mask_col: str = "Eval_Main") -> pd.DataFrame:
    """Daily block bootstrap for three families:
    vs_reference (each model − kPOA, case A), A_minus_B (each model), vs_best (case A)."""
    if models is None:
        models = _available(cfg, cfg.case_long)
    A, B = cfg.case_long, cfg.case_short
    ref = cfg.reference_model

    pairs = []
    if ref in models:
        pairs += [("vs_reference", (m, A), (ref, A)) for m in models if m != ref]
    pairs += [("A_minus_B", (m, A), (m, B)) for m in models
              if os.path.exists(os.path.join(cfg.predictions_dir, f"{m}_{B}.csv"))]
    rmse_a = {}
    for m in models:
        d = _aligned_predictions([m], [A], cfg, mask_col)
        rmse_a[m] = float(np.sqrt(np.mean((d[f"{m}|{A}"] - d["y"]) ** 2)))
    best = min((m for m in models if m != ref), key=rmse_a.get)
    pairs += [("vs_best", (best, A), (m, A)) for m in models if m not in (best, ref)]

    rows = []
    for fam, (ma, ca), (mb, cb) in pairs:
        d = _aligned_predictions(sorted({ma, mb}), sorted({ca, cb}), cfg, mask_col)
        days = d[cfg.time_col].dt.date.astype(str).to_numpy()
        ea = (d[f"{ma}|{ca}"] - d["y"]).to_numpy()
        eb = (d[f"{mb}|{cb}"] - d["y"]).to_numpy()
        r = daily_bootstrap(ea, eb, days, cfg.bootstrap_n, cfg.random_seed)
        r.update({"Family": fam, "Comparison": f"{ma}[{ca}] − {mb}[{cb}]",
                  "Signif_RMSE": not (r["dRMSE_lo"] <= 0 <= r["dRMSE_hi"]),
                  "Signif_MAE": not (r["dMAE_lo"] <= 0 <= r["dMAE_hi"])})
        rows.append(r)

    out = pd.DataFrame(rows)[["Family", "Comparison", "dRMSE", "dRMSE_lo", "dRMSE_hi",
                              "p_RMSE", "Signif_RMSE", "dMAE", "dMAE_lo", "dMAE_hi",
                              "p_MAE", "Signif_MAE", "n_days"]]
    section(f"Daily block bootstrap ({cfg.bootstrap_n:,} resamples, mask: {mask_col})")
    with pd.option_context("display.width", 200):
        print(out.round(4).to_string(index=False))
    out.to_csv(_out(cfg.results_dir, f"bootstrap_{mask_col}.csv"), index=False)
    return out


def friedman_compare(cfg: TemporalConfig = CFG, models=None, metric: str = "RMSE",
                     block: str = "D", mask_col: str = "Eval_Main",
                     case: str | None = None, min_daily_poa: float = 0.0) -> dict:
    """Friedman (Iman–Davenport F) on per-block errors, then pairwise Wilcoxon with Holm.
    block: "D" (day) or "W" (week). `min_daily_poa` > 0 drops days below that POA sum (Wh/m²)."""
    import itertools
    from scipy.stats import friedmanchisquare, wilcoxon, f as f_dist

    case = case or cfg.case_long
    if models is None:
        models = _available(cfg, case)
    d = _aligned_predictions(models, [case], cfg, mask_col)
    n_days_all = d[cfg.time_col].dt.date.nunique()
    if min_daily_poa > 0:
        data = pd.read_csv(cfg.data_path, usecols=[cfg.time_col, cfg.poa_feature],
                           parse_dates=[cfg.time_col])
        h = data.groupby(data[cfg.time_col].dt.date)[cfg.poa_feature].sum()
        keep_days = set(h[h >= min_daily_poa].index)
        d = d[d[cfg.time_col].dt.date.isin(keep_days)]
        log.info("   days with POA < %.0f Wh/m² dropped: %d / %d days kept",
                 min_daily_poa, d[cfg.time_col].dt.date.nunique(), n_days_all)
    key = d[cfg.time_col].dt.to_period(block)
    err = pd.DataFrame({m: d[f"{m}|{case}"] - d["y"] for m in models})
    if metric == "RMSE":
        per = np.sqrt((err ** 2).groupby(key).mean())
    elif metric == "MAE":
        per = err.abs().groupby(key).mean()
    else:
        raise ValueError("metric must be 'RMSE' or 'MAE'")
    per = per[(per.sum(axis=1) > 0)]

    n, k = per.shape
    chi2, p_chi = friedmanchisquare(*[per[m].to_numpy() for m in models])
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2)     # Iman–Davenport
    p_ff = float(f_dist.sf(ff, k - 1, (k - 1) * (n - 1)))
    ranks = per.rank(axis=1).mean().sort_values()

    rows = []
    for a, b in itertools.combinations(models, 2):
        diff = per[a] - per[b]
        stat, p = wilcoxon(per[a], per[b], zero_method="wilcox")
        rows.append({"Pair": f"{a} − {b}", "median_diff": float(diff.median()),
                     "a_better_blocks_pct": float((diff < 0).mean() * 100), "p": float(p)})
    pw = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m_ = len(pw)
    adj = np.maximum.accumulate([(m_ - i) * p for i, p in enumerate(pw["p"])])
    pw["p_Holm"] = np.minimum(1.0, adj)
    pw["Signif"] = pw["p_Holm"] < 0.05

    name = {"D": "day", "W": "week"}.get(block, block)
    section(f"Friedman + Wilcoxon/Holm — {metric}, block = {name} (n = {n}), {case}")
    print("Mean rank (1 = best):")
    print(ranks.round(2).to_string())
    print(f"\nFriedman χ² = {chi2:.1f}, p = {p_chi:.2e} | Iman–Davenport F = {ff:.2f}, p = {p_ff:.2e}")
    with pd.option_context("display.width", 200):
        print("\nPairwise Wilcoxon (Holm-adjusted):")
        print(pw.round(4).to_string(index=False))

    tag = (f"{metric}_{block}_{mask_col}_{case}"
           + (f"_poa{int(min_daily_poa)}" if min_daily_poa else ""))
    ranks.to_csv(_out(cfg.results_dir, f"friedman_ranks_{tag}.csv"))
    pw.to_csv(_out(cfg.results_dir, f"wilcoxon_holm_{tag}.csv"), index=False)
    return {"ranks": ranks, "F": ff, "p": p_ff, "chi2": chi2, "n_blocks": n, "pairwise": pw}


def monthly_errors(cfg: TemporalConfig = CFG, models=None,
                   mask_col: str = "Eval_Main") -> dict[str, pd.DataFrame]:
    """Monthly RMSE and the monthly share (%) of the total squared error, for both cases."""
    if models is None:
        models = _available(cfg, cfg.case_long)
    rmse, share = {}, {}
    for m in models:
        for c in (cfg.case_long, cfg.case_short):
            if not os.path.exists(os.path.join(cfg.predictions_dir, f"{m}_{c}.csv")):
                continue
            d = load_predictions(m, cfg, c)
            if mask_col in d:
                d = d[d[mask_col].astype(bool)]
            se = (d["y_pred"] - d["y_true"]) ** 2
            mon = d[cfg.time_col].dt.month
            rmse[f"{m}|{c}"] = np.sqrt(se.groupby(mon).mean())
            share[f"{m}|{c}"] = 100 * se.groupby(mon).sum() / se.sum()

    rmse, share = pd.DataFrame(rmse), pd.DataFrame(share)
    rmse.index.name = share.index.name = "Month"
    diff = pd.DataFrame({m: rmse[f"{m}|{cfg.case_long}"] - rmse[f"{m}|{cfg.case_short}"]
                         for m in models
                         if f"{m}|{cfg.case_long}" in rmse and f"{m}|{cfg.case_short}" in rmse})
    section(f"Monthly errors ({mask_col})")
    print("Monthly RMSE:")
    print(rmse.round(4).to_string())
    print("\nShare of total squared error (%):")
    print(share.round(1).to_string())
    rmse.to_csv(_out(cfg.results_dir, f"monthly_rmse_{mask_col}.csv"))
    share.to_csv(_out(cfg.results_dir, f"monthly_share_{mask_col}.csv"))
    return {"rmse": rmse, "share": share, "diff": diff}


def outage_effect(cfg: TemporalConfig = CFG, models=None) -> pd.DataFrame:
    """Share of the outage hours in the total squared error (case A)."""
    if models is None:
        models = _available(cfg, cfg.case_long)
    rows = []
    for m in models:
        d = load_predictions(m, cfg, cfg.case_long)
        d = d[d["Eval_Full"].astype(bool)]
        out = ~d["Eval_Main"].astype(bool)
        se = (d["y_pred"] - d["y_true"]) ** 2
        rows.append({"Model": m, "outage_h": int(out.sum()),
                     "outage_h_pct": 100 * out.mean(),
                     "sq_error_share_pct": 100 * se[out].sum() / se.sum(),
                     "RMSE_incl_outage": float(np.sqrt(se.mean())),
                     "RMSE_excl_outage": float(np.sqrt(se[~out].mean()))})
    t_ = pd.DataFrame(rows).set_index("Model")
    section("Effect of the outage on the metrics (case A)")
    print(t_.round(4).to_string())
    t_.to_csv(_out(cfg.results_dir, "outage_effect.csv"))
    return t_


def outage_threshold_sensitivity(cfg: TemporalConfig = CFG) -> pd.DataFrame:
    """Outage detection under 3^5 = 243 threshold combinations."""
    import itertools
    from dataclasses import replace

    df = pd.read_csv(cfg.data_path, parse_dates=[cfg.time_col])
    rows = []
    for dpu, dpoa, hpu, hpoa, tmp in itertools.product(
            [0.005, 0.01, 0.02], [500, 1000, 1500], [0.0005, 0.001, 0.005],
            [50, 100, 200], [3, 5, 7]):
        c = replace(cfg, outage_max_daily_pu=dpu, outage_min_daily_poa=dpoa,
                    outage_max_hourly_pu=hpu, outage_min_hourly_poa=hpoa,
                    outage_min_mean_temp=tmp)
        o = add_quality_flags(df, c, quiet=True)
        o = o[o["Outage"]]
        rows.append(dict(daily_pu=dpu, daily_poa=dpoa, hourly_pu=hpu, hourly_poa=hpoa,
                         mean_temp=tmp, hours=len(o), daytime_hours=int(o["Daytime"].sum()),
                         days=o[cfg.time_col].dt.date.nunique(),
                         years=tuple(sorted(o[cfg.year_col].unique()))))
    out = pd.DataFrame(rows)
    section("Outage threshold sensitivity")
    print(out.groupby(["hours", "daytime_hours", "days", "years"]).size().to_string())
    out.to_csv(_out(cfg.results_dir, "outage_threshold_sensitivity.csv"), index=False)
    return out


# ════════════════════════════════════════════════════════════════════════
# 9) Article tables and figures
# ════════════════════════════════════════════════════════════════════════
_C_LONG, _C_SHORT, _C_GRID = "#2a78d6", "#e8673a", "#e6e4df"


def yearly_stability(cfg: TemporalConfig = CFG) -> pd.DataFrame:
    """Table 2: yearly performance indicators and data quality (valid hours only)."""
    df = add_quality_flags(pd.read_csv(cfg.data_path, parse_dates=[cfg.time_col]), cfg, quiet=True)
    meas = df[df["Target_Valid"]]

    day = meas[meas[cfg.poa_feature] > 50]
    ratio = (day["Power_Actual_W"] / day[cfg.poa_feature]).groupby(day[cfg.year_col]).median()
    peak = meas.groupby(cfg.year_col)[cfg.target_col].max() * 100
    corr = day.groupby(cfg.year_col).apply(lambda g: g[cfg.poa_feature].corr(g["Power_Actual_W"]))
    annual = df.groupby(cfg.year_col)[cfg.poa_feature].sum() / 1000

    t_ = pd.DataFrame({
        "Output_POA_ratio": ratio.round(2),
        "Peak_output_pct": peak.round(1),
        "r_POA_P": corr.round(3),
        "Annual_POA_kWh_m2": annual.round(1),
        "Missing_target_h": (~df["Target_Measured"]).groupby(df[cfg.year_col]).sum(),
        "Outage_h": df["Outage"].groupby(df[cfg.year_col]).sum(),
        "Valid_daytime_h": (df["Target_Valid"] & df["Daytime"]).groupby(df[cfg.year_col]).sum(),
    })
    cv = ratio.std() / ratio.median()
    section("Table 2 — yearly stability and data quality")
    print(t_.to_string())
    print(f"\nCoefficient of variation of the output/POA ratio: {cv:.3f}")
    t_.to_csv(_out(cfg.results_dir, "table_yearly_stability.csv"))
    return t_


def article_tables(cfg: TemporalConfig = CFG, results: pd.DataFrame | None = None) -> dict:
    """Main results (case A) and the training-history comparison (A vs B).
    Requires `bootstrap_compare` to have been run for the ΔRMSE intervals."""
    if results is None:
        results = load_results(cfg)
    A, B = cfg.case_long, cfg.case_short

    def agg(case):
        sub = results[results["Case"] == case]
        return sub.groupby("Model").mean(numeric_only=True), sub.groupby("Model")["R2"].std()

    mA, sA = agg(A)
    ref = mA.loc[cfg.reference_model, "RMSE"] if cfg.reference_model in mA.index else np.nan

    def r2txt(m, s, name):
        return (f"{m.loc[name, 'R2']:.3f} ± {s.loc[name]:.3f}"
                if pd.notna(s.get(name)) and s.get(name, 0) > 0 else f"{m.loc[name, 'R2']:.3f}")

    order = mA.sort_values("RMSE").index
    main = pd.DataFrame({
        "R2": [r2txt(mA, sA, n) for n in order],
        "nRMSE_pct": mA.loc[order, "nRMSE_pct"].round(2).values,
        "nMAE_pct": mA.loc[order, "nMAE_pct"].round(2).values,
        "MAPE_pct": mA.loc[order, "MAPE_daytime"].round(1).values,
        "Bias": mA.loc[order, "Bias"].round(4).values,
        "SS": (1 - mA.loc[order, "RMSE"] / ref).round(2).values,
        "Train_s": mA.loc[order, "Train_s"].round(1).values,
        "R2_incl_outage": mA.loc[order, "InclOutage_R2"].round(3).values,
        "nRMSE_incl_outage_pct": mA.loc[order, "InclOutage_nRMSE_pct"].round(2).values,
    }, index=order)
    main.loc[cfg.reference_model, "SS"] = np.nan

    mB, sB = agg(B)
    ci = {}
    boot_f = os.path.join(cfg.results_dir, "bootstrap_Eval_Main.csv")
    if os.path.exists(boot_f):
        bt = pd.read_csv(boot_f)
        for _, r in bt[bt["Family"] == "A_minus_B"].iterrows():
            name = r["Comparison"].split("[")[0]
            star = "*" if (r["Signif_RMSE"] and abs(r["dRMSE"]) >= 5e-5) else ""
            ci[name] = f"{r['dRMSE']:+.4f} [{r['dRMSE_lo']:+.4f}, {r['dRMSE_hi']:+.4f}]{star}"
    feb = {}
    for n in order:
        vals = []
        for c in (A, B):
            if os.path.exists(os.path.join(cfg.predictions_dir, f"{n}_{c}.csv")):
                d = load_predictions(n, cfg, c)
                d = d[d["Eval_Main"].astype(bool) & (d[cfg.time_col].dt.month == 2)]
                vals.append(np.sqrt(np.mean((d["y_pred"] - d["y_true"]) ** 2)))
        feb[n] = " / ".join(f"{v:.3f}" for v in vals)
    history = pd.DataFrame({
        "R2_A": [r2txt(mA, sA, n) for n in order],
        "R2_B": [r2txt(mB, sB, n) for n in order],
        "nRMSE_A/B": [f"{mA.loc[n, 'nRMSE_pct']:.2f} / {mB.loc[n, 'nRMSE_pct']:.2f}" for n in order],
        "nMAE_A/B": [f"{mA.loc[n, 'nMAE_pct']:.2f} / {mB.loc[n, 'nMAE_pct']:.2f}" for n in order],
        "dRMSE_A-B_[95CI]": [ci.get(n, "") for n in order],
        "Feb_RMSE_A/B": [feb[n] for n in order],
    }, index=order)

    section("Article tables")
    print("Main results (case A, valid daytime hours):")
    print(main.to_string())
    print("\nTraining-history length (A = 5 years, B = 1 year):")
    print(history.to_string())
    main.to_csv(_out(cfg.results_dir, "table_main_results.csv"))
    history.to_csv(_out(cfg.results_dir, "table_history_length.csv"))
    return {"main": main, "history": history}


def importance_table(importance: pd.DataFrame | None = None,
                     cfg: TemporalConfig = CFG) -> dict[str, pd.DataFrame]:
    """Permutation importance (drop in R²) per case, features sorted by their maximum."""
    if importance is None:
        importance = load_importance(cfg)
    wide = importance.pivot_table(index="Feature", columns=["Case", "Model"], values="Importance")
    out = {}
    section("Permutation importance (drop in R²)")
    for case in (cfg.case_long, cfg.case_short):
        t_ = wide[case]
        t_ = t_.loc[t_.max(axis=1).sort_values(ascending=False).index]
        print(f"\n{case}")
        with pd.option_context("display.width", 250):
            print(t_.round(3).to_string())
        t_.to_csv(_out(cfg.results_dir, f"table_importance_{case}.csv"))
        out[case] = t_
    return out


def plot_daily_energy(algorithm: str = "XGBoost", cfg: TemporalConfig = CFG, dpi: int = 300,
                      window: int = 7):
    """Figure 1 — daily output in the test year (kWh/kW, 7-day moving average):
    measured, case A and case B. Grey band: outage days."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    a = load_predictions(algorithm, cfg, cfg.case_long)
    b = load_predictions(algorithm, cfg, cfg.case_short)
    d = a.groupby(a[cfg.time_col].dt.date).agg(y=("y_true", "sum"), pa=("y_pred", "sum"))
    d["pb"] = b.groupby(b[cfg.time_col].dt.date)["y_pred"].sum().reindex(d.index).to_numpy()
    d.index = pd.to_datetime(d.index)
    s = d.rolling(window, center=True, min_periods=4).mean()

    data = add_quality_flags(pd.read_csv(cfg.data_path, parse_dates=[cfg.time_col]), cfg, quiet=True)
    t_ = data[data[cfg.year_col] == cfg.test_year]
    g = t_.groupby(t_[cfg.time_col].dt.date)["Outage"].any()
    out_days = pd.to_datetime(g[g].index)

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=dpi)
    if len(out_days):
        ax.axvspan(out_days.min(), out_days.max() + pd.Timedelta(days=1), color=_C_GRID, lw=0)
    short = "–".join(map(str, cfg.short_history))
    long_ = f"{cfg.long_history[0]}–{cfg.long_history[-1]}"
    lb, = ax.plot(s.index, s["pb"], color=_C_SHORT, lw=1.2, label=f"{algorithm}, 1 year ({short})")
    la, = ax.plot(s.index, s["pa"], color=_C_LONG, lw=1.2, label=f"{algorithm}, 5 years ({long_})")
    lo, = ax.plot(s.index, s["y"], color="black", lw=1.2, label="Measured")

    ax.set_xlim(pd.Timestamp(f"{cfg.test_year}-01-01"), pd.Timestamp(f"{cfg.test_year}-12-31"))
    ax.set_ylim(0, 8)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
    if len(out_days):
        mid = out_days.min() + (out_days.max() - out_days.min()) / 2
        ax.text(mid, 7.6, "System outage", ha="center", fontsize=8, color="#555")
    ax.set_ylabel(f"Daily output (kWh/kW)\n{window}-day moving average")
    ax.set_xlabel(f"Month ({cfg.test_year})")
    ax.legend(handles=[lo, la, lb], frameon=False, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.grid(axis="y", color=_C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(_out(cfg.results_dir, f"figure1_daily_output_{algorithm}.png"), facecolor="white")
    plt.show()
    return fig


def plot_monthly_rmse(models=("kPOA", "MLR", "XGBoost", "SVR", "RandomForest", "KNN", "LSTM"),
                      cfg: TemporalConfig = CFG, mask_col: str = "Eval_Main", dpi: int = 300):
    """Figure 2 — monthly RMSE for case A (5 years) and case B (1 year), one panel per model."""
    import matplotlib.pyplot as plt

    names = {"kPOA": "k·POA", "RandomForest": "RF"}
    short = "–".join(map(str, cfg.short_history))
    long_ = f"{cfg.long_history[0]}–{cfg.long_history[-1]}"
    x, w = np.arange(1, 13), 0.38

    fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.2), dpi=dpi, sharey=True)
    for ax, m in zip(axs.ravel(), models):
        for c, col, off, lab in ((cfg.case_long, _C_LONG, -w / 2 - 0.01, f"5 years ({long_})"),
                                 (cfg.case_short, _C_SHORT, w / 2 + 0.01, f"1 year ({short})")):
            d = load_predictions(m, cfg, c)
            d = d[d[mask_col].astype(bool)]
            r = np.sqrt(((d["y_pred"] - d["y_true"]) ** 2).groupby(d[cfg.time_col].dt.month).mean())
            ax.bar(x + off, r.reindex(x).to_numpy(), w, color=col, label=lab,
                   edgecolor="white", lw=0.5)
        ax.set_title(names.get(m, m), fontsize=10, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) if i % 2 else "" for i in x], fontsize=7)
        ax.grid(axis="y", color=_C_GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ax in axs[1, :3]:
        ax.set_xlabel("Month")
    for ax in axs[:, 0]:
        ax.set_ylabel("RMSE (P$_{PU}$)")
    h, l = axs[0, 0].get_legend_handles_labels()
    axs[1, 3].axis("off")
    axs[1, 3].legend(h, l, frameon=False, fontsize=9, loc="center")
    fig.tight_layout()
    fig.savefig(_out(cfg.results_dir, "figure2_monthly_rmse.png"), facecolor="white")
    plt.show()
    return fig
