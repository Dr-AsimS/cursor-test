#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

EXCEL_PATH = Path("/workspace/pig_farm_7_2024.xlsx")
SHEET_NAME = "pig_farm_7_2024"
TARGET_COL = "Indoor-Temp(°C)"
DATE_COL = "Date"
TIME_COL = "Time"
HUM_COL = "Humidity(%)"
CO2_COL = "CO2(ppm)"
WEATHER_COL = "Weather"
SEASONAL_PERIOD = 48  # 30-min data -> 48 steps per day


def build_timestamp(df: pd.DataFrame) -> pd.Series:
	# Excel epoch (Windows 1900 base)
	epoch = pd.Timestamp("1899-12-30")
	date_serial = df[DATE_COL].astype(float)
	time_fraction = df[TIME_COL].astype(float)
	# Convert fractional day to integer seconds to remove floating jitter
	seconds_in_day = 24 * 60 * 60
	sec_of_day = np.rint(time_fraction * seconds_in_day).astype(int)
	return epoch + pd.to_timedelta(date_serial, unit="D") + pd.to_timedelta(sec_of_day, unit="s")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
	result = df.copy()
	result["hour"] = result.index.hour + result.index.minute / 60.0
	result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24.0)
	result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24.0)
	result["dow"] = result.index.dayofweek
	result["dow_sin"] = np.sin(2 * np.pi * result["dow"] / 7.0)
	result["dow_cos"] = np.cos(2 * np.pi * result["dow"] / 7.0)
	return result


def prepare_data() -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if not EXCEL_PATH.exists():
		print(f"Error: file not found: {EXCEL_PATH}", file=sys.stderr)
		sys.exit(1)

	df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
	# Drop fully empty rows if any
	df = df.dropna(how="all")

	# Coerce Date/Time to numeric and filter valid rows only
	df[DATE_COL] = pd.to_numeric(df.get(DATE_COL), errors="coerce")
	df[TIME_COL] = pd.to_numeric(df.get(TIME_COL), errors="coerce")
	df = df[df[DATE_COL].notna() & df[TIME_COL].notna()].copy()

	# Timestamp and index
	df["timestamp"] = build_timestamp(df)
	df = df.set_index("timestamp").sort_index()
	# Remove any duplicate timestamps if present
	df = df[~df.index.duplicated(keep="first")]

	# Keep only relevant columns and coerce numeric
	for col in (TARGET_COL, HUM_COL, CO2_COL, DATE_COL, TIME_COL):
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")

	# Feature engineering
	df = add_time_features(df)

	# One-hot encode weather for model inputs (ensure float dtype)
	if WEATHER_COL in df.columns:
		weather_dummies = pd.get_dummies(df[WEATHER_COL].astype(str).str.strip(), prefix="weather", drop_first=True, dtype=float)
	else:
		weather_dummies = pd.DataFrame(index=df.index)

	# Ensure exogenous base columns exist even if missing in data
	base_exog_cols = []
	for col in (HUM_COL, CO2_COL, "hour_sin", "hour_cos", "dow_sin", "dow_cos"):
		if col in df.columns:
			base_exog_cols.append(col)

	# Exogenous for SARIMAX (no target lags): humidity, co2, hour sin/cos, dow sin/cos, weather dummies
	exog_sarimax = pd.concat([
		df[base_exog_cols],
		weather_dummies
	], axis=1)
	# Force numeric float dtype
	exog_sarimax = exog_sarimax.apply(pd.to_numeric, errors="coerce").astype(float)

	# XGB features: exog + target lags
	features_xgb = exog_sarimax.copy()
	for lag in (1, 2, SEASONAL_PERIOD):
		features_xgb[f"y_lag_{lag}"] = df[TARGET_COL].shift(lag)

	# Align datasets: drop rows with any NaNs in required inputs
	y = df[TARGET_COL].copy().astype(float)

	# For SARIMAX, keep rows where y and exog are available
	mask_sarimax = y.notna() & exog_sarimax.notna().all(axis=1)
	y_sarimax = y[mask_sarimax]
	exog_sarimax = exog_sarimax[mask_sarimax]

	# For XGB, drop rows where lagged features are NaN
	features_xgb = features_xgb.apply(pd.to_numeric, errors="coerce")
	mask_xgb = y.notna() & features_xgb.notna().all(axis=1)
	features_xgb = features_xgb[mask_xgb].astype(float)
	return y_sarimax, exog_sarimax, y, features_xgb


def time_based_split(index: pd.DatetimeIndex, test_size: float = 0.2) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
	cutoff = index.min() + (index.max() - index.min()) * (1 - test_size)
	train_idx = index[index <= cutoff]
	test_idx = index[index > cutoff]
	return train_idx, test_idx


def main() -> None:
	y_sarimax, exog_sarimax, y_full, features_xgb = prepare_data()

	# Time split based on sarimax-eligible data for model stability
	train_idx, test_idx = time_based_split(y_sarimax.index, test_size=0.2)
	if len(test_idx) == 0:
		print("Not enough data for a test split.", file=sys.stderr)
		sys.exit(2)

	# Split SARIMAX inputs
	y_train = y_sarimax.loc[train_idx]
	y_test = y_sarimax.loc[test_idx]
	exog_train = exog_sarimax.loc[train_idx]
	exog_test = exog_sarimax.loc[test_idx]

	print(f"Training points: {len(y_train)}, Test points: {len(y_test)}")
	print(f"SARIMAX exog dims: train={exog_train.shape}, test={exog_test.shape}")

	# Fit SARIMAX
	order = (1, 0, 1)
	seasonal_order = (0, 1, 1, SEASONAL_PERIOD)
	try:
		model = SARIMAX(
			y_train.astype(float),
			exog=exog_train.astype(float),
			order=order,
			seasonal_order=seasonal_order,
			enforce_stationarity=False,
			enforce_invertibility=False,
		)
		fit_res = model.fit(disp=False)
	except Exception as exc:
		print("SARIMAX failed with initial order, falling back to simpler order...", file=sys.stderr)
		print(str(exc), file=sys.stderr)
		model = SARIMAX(y_train.astype(float), exog=exog_train.astype(float), order=(1, 0, 0), seasonal_order=(0, 1, 1, SEASONAL_PERIOD), enforce_stationarity=False, enforce_invertibility=False)
		fit_res = model.fit(disp=False)

	# In-sample fitted for train residuals
	fitted_train = fit_res.fittedvalues
	residual_train = (y_train - fitted_train).dropna()

	# Stage 2: XGB on residuals with lag features
	# Align XGB features to training residual index (safe intersection)
	train_align_index = features_xgb.index.intersection(residual_train.index)
	features_xgb_train = features_xgb.loc[train_align_index]
	residual_train = residual_train.loc[train_align_index]

	# Ensure test features exist and align to test index
	test_align_index = test_idx.intersection(features_xgb.index)
	features_xgb_test = features_xgb.loc[test_align_index]
	# Align y for the same period
	y_test_aligned = y_full.loc[test_align_index]

	xgb = XGBRegressor(
		n_estimators=500,
		max_depth=6,
		learning_rate=0.05,
		subsample=0.8,
		colsample_bytree=0.8,
		random_state=42,
		n_jobs=4,
	)
	xgb.fit(features_xgb_train.values, residual_train.values)

	# Forecast SARIMAX over the test_align_index
	exog_pred = exog_sarimax.loc[test_align_index]
	sarimax_pred_test = fit_res.predict(start=test_align_index[0], end=test_align_index[-1], exog=exog_pred)

	# Predict residuals with XGB
	resid_pred_test = xgb.predict(features_xgb_test.values)
	resid_pred_test = pd.Series(resid_pred_test, index=test_align_index)

	# Final prediction
	final_pred_test = sarimax_pred_test.add(resid_pred_test, fill_value=0.0)

	# Metrics
	mae = mean_absolute_error(y_test_aligned, final_pred_test.loc[y_test_aligned.index])
	mse = mean_squared_error(y_test_aligned, final_pred_test.loc[y_test_aligned.index])
	rmse = math.sqrt(mse)
	mape = (np.abs((y_test_aligned - final_pred_test.loc[y_test_aligned.index]) / y_test_aligned).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100

	print("\nHybrid model performance on test subset:")
	print(f"MAE:  {mae:.3f}")
	print(f"RMSE: {rmse:.3f}")
	print(f"MAPE: {mape:.2f}%")

	# Compare to SARIMAX alone
	yhat_sarimax_only = sarimax_pred_test.loc[y_test_aligned.index]
	mse_sarimax = mean_squared_error(y_test_aligned, yhat_sarimax_only)
	rmse_sarimax = math.sqrt(mse_sarimax)
	mae_sarimax = mean_absolute_error(y_test_aligned, yhat_sarimax_only)
	print("\nSARIMAX-only benchmark on same period:")
	print(f"MAE:  {mae_sarimax:.3f}")
	print(f"RMSE: {rmse_sarimax:.3f}")

	# Feature importances (top 10)
	importances = xgb.feature_importances_
	feat_names = list(features_xgb.columns)
	order_idx = np.argsort(importances)[::-1]
	print("\nXGB feature importances (top 10):")
	for rank, idx in enumerate(order_idx[:10], start=1):
		print(f"  {rank:2d}. {feat_names[idx]}: {importances[idx]:.4f}")

	# Save predictions
	out = pd.DataFrame({
		"y_true": y_test_aligned,
		"sarimax": yhat_sarimax_only,
		"xgb_resid": resid_pred_test.loc[y_test_aligned.index],
		"hybrid_pred": final_pred_test.loc[y_test_aligned.index],
	})
	out.to_csv("/workspace/hybrid_predictions.csv", index_label="timestamp")
	print("\nSaved predictions to /workspace/hybrid_predictions.csv")


if __name__ == "__main__":
	main()