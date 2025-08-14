#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

EXCEL_PATH = Path("/workspace/pig_farm_7_2024.xlsx")
SHEET_NAME = "pig_farm_7_2024"
TARGET_COL = "Indoor-Temp(°C)"
DATE_COL = "Date"
TIME_COL = "Time"
HUM_COL = "Humidity(%)"
CO2_COL = "CO2(ppm)"
WEATHER_COL = "Weather"
SEASONAL_PERIOD = 48

N_FOLDS_DEFAULT = 4
VAL_RATIO_DEFAULT = 0.125  # 12.5% per fold
RANDOM_SEED = 42

@dataclass
class SarimaxSpec:
	order: Tuple[int, int, int]
	seasonal_order: Tuple[int, int, int, int]
	description: str

@dataclass
class XgbSpec:
	params: Dict[str, object]
	description: str


def build_timestamp(df: pd.DataFrame) -> pd.Series:
	epoch = pd.Timestamp("1899-12-30")
	date_serial = pd.to_numeric(df[DATE_COL], errors="coerce").astype(float)
	time_fraction = pd.to_numeric(df[TIME_COL], errors="coerce").astype(float).fillna(0.0)
	seconds_in_day = 24 * 60 * 60
	sec_of_day = np.rint(time_fraction * seconds_in_day).astype(np.int64)
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


def prepare_dataset() -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Index]:
	if not EXCEL_PATH.exists():
		print(f"Error: file not found: {EXCEL_PATH}", file=sys.stderr)
		sys.exit(1)

	df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl").dropna(how="all")
	df[DATE_COL] = pd.to_numeric(df.get(DATE_COL), errors="coerce")
	df[TIME_COL] = pd.to_numeric(df.get(TIME_COL), errors="coerce")
	df = df[df[DATE_COL].notna() & df[TIME_COL].notna()].copy()

	df["ts"] = build_timestamp(df)
	df = df.set_index("ts").sort_index()
	df = df[~df.index.duplicated(keep="first")]

	for col in (TARGET_COL, HUM_COL, CO2_COL, DATE_COL, TIME_COL):
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")

	df = add_time_features(df)

	# Weather dummies
	if WEATHER_COL in df.columns:
		weather_dummies = pd.get_dummies(df[WEATHER_COL].astype(str).str.strip(), prefix="weather", drop_first=True, dtype=float)
	else:
		weather_dummies = pd.DataFrame(index=df.index)

	# Exogenous base
	base_cols = [c for c in [HUM_COL, CO2_COL, "hour_sin", "hour_cos", "dow_sin", "dow_cos"] if c in df.columns]
	exog_sarimax = pd.concat([df[base_cols], weather_dummies], axis=1)
	exog_sarimax = exog_sarimax.apply(pd.to_numeric, errors="coerce").astype(float)

	y = df[TARGET_COL].astype(float)

	# XGB features: add lags and rolling stats on target
	features_xgb = exog_sarimax.copy()
	lag_list = [1, 2, 3, 6, 12, 24, 48]
	for lag in lag_list:
		features_xgb[f"y_lag_{lag}"] = y.shift(lag)
	for win in [3, 6, 12, 24]:
		features_xgb[f"y_rollmean_{win}"] = y.rolling(win).mean()

	# Masks
	mask_sarimax = y.notna() & exog_sarimax.notna().all(axis=1)
	y_sarimax = y[mask_sarimax]
	exog_sarimax = exog_sarimax[mask_sarimax]

	features_xgb = features_xgb.apply(pd.to_numeric, errors="coerce")
	mask_xgb = y.notna() & features_xgb.notna().all(axis=1)
	features_xgb = features_xgb[mask_xgb]

	# Common index union for stable folds
	common_index = y_sarimax.index.intersection(features_xgb.index)
	return y_sarimax.loc[common_index], exog_sarimax.loc[common_index], y.loc[common_index], features_xgb.loc[common_index], common_index


def make_folds(index: pd.Index, n_folds: int, val_ratio: float) -> List[Tuple[pd.Index, pd.Index]]:
	n = len(index)
	val_len = max(1, int(n * val_ratio))
	folds: List[Tuple[pd.Index, pd.Index]] = []
	# Start with at least 50% for the first train
	first_train_len = max(int(n * 0.5), val_len * 2)
	for i in range(n_folds):
		train_end = min(first_train_len + i * val_len, n - val_len)
		train_idx = index[:train_end]
		val_idx = index[train_end:train_end + val_len]
		if len(val_idx) == 0:
			break
		folds.append((train_idx, val_idx))
	return folds


def shortlist_sarimax(y_train: pd.Series, exog_train: pd.DataFrame, candidates: List[SarimaxSpec], top_k: int = 2, maxiter: int = 100) -> List[SarimaxSpec]:
	results: List[Tuple[float, SarimaxSpec]] = []
	for spec in candidates:
		try:
			model = SARIMAX(y_train.astype(float), exog=exog_train.astype(float), order=spec.order, seasonal_order=spec.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
			res = model.fit(disp=False, maxiter=maxiter)
			aic = float(res.aic)
			results.append((aic, spec))
			print(f"SARIMAX candidate {spec.description} AIC={aic:.1f}")
		except Exception as exc:
			print(f"SARIMAX candidate {spec.description} failed: {exc}", file=sys.stderr)
			continue
	results.sort(key=lambda x: x[0])
	return [spec for _, spec in results[:top_k]]


def evaluate_combo(y_all: pd.Series, exog_all: pd.DataFrame, features_all: pd.DataFrame, folds: List[Tuple[pd.Index, pd.Index]], sarimax_spec: SarimaxSpec, xgb_spec: XgbSpec, maxiter: int = 100) -> Tuple[float, float]:
	"""Return (avg_rmse, avg_mae) across folds."""
	rmses: List[float] = []
	maes: List[float] = []
	for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
		try:
			y_train = y_all.loc[train_idx]
			exog_train = exog_all.loc[train_idx]
			y_val = y_all.loc[val_idx]
			exog_val = exog_all.loc[val_idx]

			# Fit SARIMAX on train
			model = SARIMAX(y_train.astype(float), exog=exog_train.astype(float), order=sarimax_spec.order, seasonal_order=sarimax_spec.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
			res = model.fit(disp=False, maxiter=maxiter)
			# Forecast on val
			yhat_val_sarimax = res.predict(start=val_idx[0], end=val_idx[-1], exog=exog_val)

			# Residuals on train (for XGB target)
			residual_train = (y_train - res.fittedvalues).dropna()
			# Align XGB train features to residual_train
			feat_train = features_all.loc[residual_train.index]
			# XGB validation target residual = y_val - yhat_val_sarimax
			residual_val_target = (y_val - yhat_val_sarimax).astype(float)
			feat_val = features_all.loc[val_idx]

			# Train XGB (no early stopping to ensure compatibility)
			xgb = XGBRegressor(
				objective="reg:squarederror",
				random_state=RANDOM_SEED,
				n_jobs=4,
				eval_metric="rmse",
				**xgb_spec.params,
			)
			xgb.fit(feat_train.values, residual_train.values)

			# Predict residuals on val and assemble hybrid
			resid_pred_val = xgb.predict(feat_val.values)
			yhat_val_hybrid = yhat_val_sarimax.add(pd.Series(resid_pred_val, index=val_idx), fill_value=0.0)

			mse = mean_squared_error(y_val, yhat_val_hybrid)
			rmse = math.sqrt(mse)
			mae = mean_absolute_error(y_val, yhat_val_hybrid)
			rmses.append(rmse)
			maes.append(mae)
			print(f"  Fold {fold_idx}: RMSE={rmse:.3f} MAE={mae:.3f}")
		except Exception as exc:
			print(f"  Fold {fold_idx} failed: {exc}", file=sys.stderr)
			continue

	avg_rmse = float(np.mean(rmses)) if rmses else float("inf")
	avg_mae = float(np.mean(maes)) if maes else float("inf")
	return avg_rmse, avg_mae


def main() -> None:
	quick = "--quick" in sys.argv
	n_folds = 2 if quick else N_FOLDS_DEFAULT
	val_ratio = 0.2 if quick else VAL_RATIO_DEFAULT
	fit_maxiter = 50 if quick else 100
	print(f"Quick mode: {quick}. Folds={n_folds}, val_ratio={val_ratio}, maxiter={fit_maxiter}")

	# Prepare
	y_all, exog_all, y_raw, features_all, common_index = prepare_dataset()
	# Standardize features for XGB using train statistics per fold? To simplify, standardize globally (tree-based models are insensitive), so skip scaling.

	# Build folds
	folds = make_folds(common_index, n_folds, val_ratio)
	print(f"Using {len(folds)} expanding folds")
	for i, (tr, va) in enumerate(folds, start=1):
		print(f"  Fold {i}: train={len(tr)}, val={len(va)}")

	# SARIMAX candidate list
	sarimax_candidates = [
		SarimaxSpec((1,0,1), (0,1,1,SEASONAL_PERIOD), "(1,0,1)x(0,1,1,48)"),
		SarimaxSpec((2,0,1), (0,1,1,SEASONAL_PERIOD), "(2,0,1)x(0,1,1,48)"),
		SarimaxSpec((1,1,1), (0,1,1,SEASONAL_PERIOD), "(1,1,1)x(0,1,1,48)"),
	]
	# Shortlist by AIC on first fold
	first_train_idx, _ = folds[0]
	shortlisted = shortlist_sarimax(y_all.loc[first_train_idx], exog_all.loc[first_train_idx], sarimax_candidates, top_k=(1 if quick else 2), maxiter=fit_maxiter)
	print("Shortlisted SARIMAX specs:")
	for s in shortlisted:
		print(f"  - {s.description}")

	# XGB grid
	if quick:
		xgb_grid: List[XgbSpec] = [
			XgbSpec({"n_estimators": 800, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "gamma": 0.0}, "d6 lr0.05 mcw1 ss0.8 cs0.8 lam1 g0"),
			XgbSpec({"n_estimators": 1200, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.9, "reg_lambda": 1.0, "gamma": 0.0}, "d6 lr0.03 mcw1 ss0.8 cs0.9 lam1 g0"),
		]
	else:
		xgb_grid = [
			XgbSpec({"n_estimators": 1500, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "gamma": 0.0}, "d6 lr0.05 mcw1 ss0.8 cs0.8 lam1 g0"),
			XgbSpec({"n_estimators": 2000, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.9, "reg_lambda": 1.0, "gamma": 0.0}, "d6 lr0.03 mcw1 ss0.8 cs0.9 lam1 g0"),
			XgbSpec({"n_estimators": 2500, "learning_rate": 0.02, "max_depth": 8, "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 5.0, "gamma": 1.0}, "d8 lr0.02 mcw5 ss0.7 cs0.7 lam5 g1"),
		]

	results: List[Tuple[float, float, SarimaxSpec, XgbSpec]] = []
	for s_spec in shortlisted:
		for x_spec in xgb_grid:
			print(f"\nEvaluating combo: SARIMAX {s_spec.description} + XGB [{x_spec.description}]")
			avg_rmse, avg_mae = evaluate_combo(y_all, exog_all, features_all, folds, s_spec, x_spec, maxiter=fit_maxiter)
			print(f"  -> CV Avg RMSE={avg_rmse:.3f}, MAE={avg_mae:.3f}")
			results.append((avg_rmse, avg_mae, s_spec, x_spec))

	results.sort(key=lambda x: x[0])
	print("\nTop results (by CV RMSE):")
	for rank, (rmse, mae, s_spec, x_spec) in enumerate(results[:5], start=1):
		print(f"  {rank:2d}. RMSE={rmse:.3f} MAE={mae:.3f} | SARIMAX {s_spec.description} | XGB {x_spec.description}")

	best_rmse, best_mae, best_s, best_x = results[0]
	print("\nBest configuration:")
	print(f"  SARIMAX: {best_s.description} order={best_s.order}, seasonal_order={best_s.seasonal_order}")
	print(f"  XGB: {best_x.description} params={best_x.params}")
	print(f"  CV Avg: RMSE={best_rmse:.3f}, MAE={best_mae:.3f}")


if __name__ == "__main__":
	main()