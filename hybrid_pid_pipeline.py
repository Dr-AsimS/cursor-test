#!/usr/bin/env python3

# Uses the merged 30-min file to train a SARIMAX+XGBoost hybrid and produce
# adaptive PID setpoint/gain recommendations.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import math

# ========= CONFIG: adjusted to the merged file produced in this workspace =========
INFILE = "/workspace/merged_pig_weather.csv"  # merged CSV we created
TIMECOL = "timestamp"                          # index after read
Y_COL   = "Indoor-Temp(°C)"                    # indoor target
NUM_EXOG = [
    "Humidity(%)", "CO2(ppm)",
    "temperature_2m (°C)", "shortwave_radiation (W/m²)",
    "wind_speed_10m (km/h)", "relative_humidity_2m (%)"
]
CAT_EXOG = ["Weather"]                         # categorical from pig file

# SARIMAX seasonal period for 30-min data (daily seasonality)
S = 48
SARIMAX_ORDERS = [(1,0,1)]               # small shortlist; you can add (1,1,1) etc.
SARIMAX_SEAS   = [(0,1,1,S)]

# XGB defaults (good starting point)
XGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, n_jobs=4
)

# Lags & rollings (30-min steps)
LAGS     = [1,2,3,6,12,24,48]
ROLLINGS = [3,6,12,24]

# Adaptive PID mapping settings (demo values)
BASE_SETPOINT = 22.0          # °C
LOOKAHEAD_STEPS = 4           # 4 * 30 min = 2 hours
DELTA_HOT  = +1.0             # forecasted rise threshold
DELTA_COLD = -1.0             # forecasted drop threshold
# ===================================================================


def read_data(path):
    p = Path(path)
    if p.suffix.lower() == ".xlsx":
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    # timestamp handling
    if TIMECOL in df.columns:
        df[TIMECOL] = pd.to_datetime(df[TIMECOL])
        df = df.set_index(TIMECOL)
    df = df.sort_index()
    return df


def make_time_features(df):
    # cyclic encodings
    minutes = df.index.hour * 60 + df.index.minute
    hour_frac = minutes / (24*60)
    df["hour_sin"] = np.sin(2*np.pi*hour_frac)
    df["hour_cos"] = np.cos(2*np.pi*hour_frac)
    dow = df.index.dayofweek / 7.0
    df["dow_sin"]  = np.sin(2*np.pi*dow)
    df["dow_cos"]  = np.cos(2*np.pi*dow)
    return df


def build_feature_matrix(df):
    df = df.copy()

    # ensure target numeric
    if Y_COL in df.columns:
        df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")

    # coerce numeric exog
    for col in [c for c in NUM_EXOG if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # one-hot weather categorical (optional)
    if CAT_EXOG:
        cats = [c for c in CAT_EXOG if c in df.columns]
        if cats:
            dummies = pd.get_dummies(df[cats], drop_first=True, dtype=float)
            df = pd.concat([df.drop(columns=cats), dummies], axis=1)

    df = make_time_features(df)

    # SARIMAX exog = numeric exog + time features + any weather dummies
    sarimax_exog_cols = []
    sarimax_exog_cols += [c for c in NUM_EXOG if c in df.columns]
    sarimax_exog_cols += ["hour_sin","hour_cos","dow_sin","dow_cos"]
    # add any dummy columns made above
    dummy_cols = [c for c in df.columns if any(d in c for d in ["Weather ", "Weather_"])]
    sarimax_exog_cols += [c for c in dummy_cols if c not in sarimax_exog_cols]

    # Fill missing exog values (time interpolation, then ffill/bfill)
    if sarimax_exog_cols:
        df[sarimax_exog_cols] = df[sarimax_exog_cols].interpolate(method="time").ffill().bfill()

    # XGB features = SARIMAX exog + lags + rollings of target
    for k in LAGS:
        df[f"y_lag_{k}"] = df[Y_COL].shift(k)
    for w in ROLLINGS:
        df[f"y_rollmean_{w}"] = df[Y_COL].rolling(w, min_periods=w).mean().shift(1)

    xgb_cols = sarimax_exog_cols + \
               [f"y_lag_{k}" for k in LAGS] + \
               [f"y_rollmean_{w}" for w in ROLLINGS]

    # Optionally fill XGB exog NaNs (XGBoost can handle NaNs, but safer to fill)
    if xgb_cols:
        df[xgb_cols] = df[xgb_cols].ffill().bfill()

    return df, sarimax_exog_cols, xgb_cols


def time_split(df, train_frac=0.8):
    split_idx = int(len(df)*train_frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def fit_best_sarimax(y_train, exog_train):
    best = None
    best_aic = np.inf
    for order in SARIMAX_ORDERS:
        for seas in SARIMAX_SEAS:
            try:
                m = SARIMAX(y_train, exog=exog_train, order=order,
                            seasonal_order=seas, enforce_stationarity=False,
                            enforce_invertibility=False)
                res = m.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best = res
            except Exception as e:
                print(f"SARIMAX {order}x{seas} failed: {e}")
    if best is None:
        raise RuntimeError("All SARIMAX fits failed.")
    return best


def train_hybrid(df):
    # Build features
    df_fe, sarimax_cols, xgb_cols = build_feature_matrix(df)

    # Drop rows that became NaN due to lags/rollings (only for training portion)
    df_train, df_test = time_split(df_fe.dropna(subset=[Y_COL]), train_frac=0.8)

    # SARIMAX
    y_tr  = df_train[Y_COL].astype(float)
    Xs_tr = df_train[sarimax_cols]
    sarimax_res = fit_best_sarimax(y_tr, Xs_tr)

    # SARIMAX predictions
    yhat_tr = sarimax_res.fittedvalues
    Xs_te = df_test[sarimax_cols]
    # Ensure no NaNs in forecast exog
    Xs_te = Xs_te.ffill().bfill()
    yhat_te = sarimax_res.get_forecast(steps=len(df_test), exog=Xs_te).predicted_mean
    y_te = df_test[Y_COL].astype(float)

    # Residuals for Stage 2
    resid_tr = (y_tr - yhat_tr).dropna()

    # Align XGB features with residuals (lags/rollings already on df_train)
    Xx_tr = df_train.loc[resid_tr.index, xgb_cols].astype(float)

    # Train XGB on residuals
    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(Xx_tr, resid_tr)

    # XGB residual predictions on test
    Xx_te = df_test[xgb_cols].astype(float)
    resid_hat_te = pd.Series(xgb.predict(Xx_te), index=Xx_te.index)

    # Final hybrid prediction
    yhat_hybrid_te = (yhat_te + resid_hat_te).rename("yhat_hybrid")

    # Metrics (robust RMSE calculation)
    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred)/np.maximum(np.abs(y_true), 1e-6))) * 100
        r2 = r2_score(y_true, y_pred)
        return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2)

    m_sar = metrics(y_te, yhat_te)
    m_hyb = metrics(y_te, yhat_hybrid_te)

    return {
        "df_train": df_train, "df_test": df_test,
        "sarimax_res": sarimax_res, "xgb": xgb,
        "y_test": y_te, "yhat_sarimax_test": yhat_te,
        "yhat_hybrid_test": yhat_hybrid_te,
        "metrics_sarimax": m_sar, "metrics_hybrid": m_hyb
    }


def adaptive_pid_schedule(y_pred, base_setpoint=BASE_SETPOINT, lookahead=LOOKAHEAD_STEPS):
    """
    Simple rule-based mapping from 2h-ahead forecasted delta to setpoint & gain multipliers.
    Returns a DataFrame aligned to y_pred.index (except the last lookahead steps).
    """
    df = pd.DataFrame({"y_pred": y_pred})
    df["y_pred_fwd"] = df["y_pred"].shift(-lookahead)
    df["delta_2h"] = df["y_pred_fwd"] - df["y_pred"]

    def rule(delta):
        # returns (setpoint_adjust, kp_mult, ki_mult, kd_mult)
        if pd.isna(delta):
            return (0.0, 1.0, 1.0, 1.0)
        if delta >= DELTA_HOT:   # expect warming → pre-cool
            return (-0.5, 0.8, 0.8, 1.2)
        if delta <= DELTA_COLD:  # expect cooling → pre-heat
            return (+0.5, 1.2, 1.1, 1.0)
        if abs(delta) <= 0.3:    # near steady
            return (0.0, 1.0, 1.0, 1.0)
        # moderate change
        return (-0.2, 0.9, 0.9, 1.1) if delta > 0 else (+0.2, 1.1, 1.05, 1.0)

    out = df["delta_2h"].apply(rule).apply(pd.Series)
    out.columns = ["T_set_adjust", "Kp_mult", "Ki_mult", "Kd_mult"]
    out["T_set"] = base_setpoint + out["T_set_adjust"]
    return pd.concat([df, out], axis=1).dropna(subset=["y_pred_fwd"])


def main():
    df = read_data(INFILE)
    results = train_hybrid(df)

    print("\n=== Test metrics ===")
    print("SARIMAX :", results["metrics_sarimax"])
    print("Hybrid  :", results["metrics_hybrid"])

    # Plot predictions vs actual on test
    y_te = results["y_test"]
    y_sar = results["yhat_sarimax_test"]
    y_hyb = results["yhat_hybrid_test"]

    plt.figure(figsize=(12,4))
    y_te.plot(label="Actual", linewidth=1)
    y_sar.plot(label="SARIMAX", linewidth=1)
    y_hyb.plot(label="Hybrid", linewidth=1)
    plt.legend(); plt.title("Test window: Actual vs Predictions")
    plt.tight_layout(); plt.savefig("pred_vs_actual_test.png", dpi=160)
    print("Saved plot: pred_vs_actual_test.png")

    # Adaptive PID schedule on test predictions
    sched = adaptive_pid_schedule(y_hyb)
    sched_path = "adaptive_pid_schedule.csv"
    sched.to_csv(sched_path, index=True)
    print(f"Saved PID schedule: {sched_path}")
    print(sched.head())

if __name__ == "__main__":
    main()