#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error

EXCEL_PATH = Path("/workspace/pig_farm_7_2024.xlsx")
SHEET_NAME = "pig_farm_7_2024"
TARGET_COL = "Indoor-Temp(°C)"
DATE_COL = "Date"
TIME_COL = "Time"
HUM_COL = "Humidity(%)"
CO2_COL = "CO2(ppm)"
WEATHER_COL = "Weather"
SEASONAL_PERIOD = 48  # 30-min data -> 48 steps per day
SEQ_LEN = 48
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
RANDOM_SEED = 42


def set_seeds(seed: int = 42) -> None:
	rng = np.random.default_rng(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	# No direct numpy global seed due to new Generator API; this is fine for our use


def build_timestamp(df: pd.DataFrame) -> pd.Series:
	# Excel epoch (Windows 1900 base)
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


def time_based_split(index: pd.DatetimeIndex, test_size: float = 0.2) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
	cutoff = index.min() + (index.max() - index.min()) * (1 - test_size)
	train_idx = index[index <= cutoff]
	test_idx = index[index > cutoff]
	return train_idx, test_idx


def prepare_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex]:
	if not EXCEL_PATH.exists():
		print(f"Error: file not found: {EXCEL_PATH}", file=sys.stderr)
		sys.exit(1)

	df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
	df = df.dropna(how="all")

	# Ensure Date/Time numeric and keep only valid rows before building timestamps
	df[DATE_COL] = pd.to_numeric(df.get(DATE_COL), errors="coerce")
	df[TIME_COL] = pd.to_numeric(df.get(TIME_COL), errors="coerce")
	df = df[df[DATE_COL].notna() & df[TIME_COL].notna()].copy()

	# Build timestamp index
	df["ds"] = build_timestamp(df)
	df = df.set_index("ds").sort_index()
	df = df[~df.index.duplicated(keep="first")]

	# Ensure numeric for relevant columns
	for col in (TARGET_COL, HUM_COL, CO2_COL, DATE_COL, TIME_COL):
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")

	# Add time features
	df = add_time_features(df)

	# One-hot encode Weather
	if WEATHER_COL in df.columns:
		weather_dummies = pd.get_dummies(df[WEATHER_COL].astype(str).str.strip(), prefix="weather", drop_first=True, dtype=float)
	else:
		weather_dummies = pd.DataFrame(index=df.index)

	# Feature matrix for residual model
	base_cols = [c for c in [HUM_COL, CO2_COL, "hour_sin", "hour_cos", "dow_sin", "dow_cos"] if c in df.columns]
	features = pd.concat([df[base_cols], weather_dummies], axis=1)

	# Target series
	y = df[TARGET_COL].astype(float)

	# Drop rows where target or features missing
	mask = y.notna() & features.notna().all(axis=1)
	y = y[mask]
	features = features[mask]

	# Split indices
	train_idx, test_idx = time_based_split(y.index, test_size=0.2)

	return features, y, df, train_idx, test_idx


class SeqDataset(Dataset):
	def __init__(self, feature_array: np.ndarray, target_array: np.ndarray, indices: List[int], seq_len: int) -> None:
		self.X = feature_array
		self.y = target_array
		self.indices = indices
		self.seq_len = seq_len

	def __len__(self) -> int:
		return len(self.indices)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		t = self.indices[idx]
		x_seq = self.X[t - self.seq_len:t]
		y_t = self.y[t]
		return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_t, dtype=torch.float32)


class CNNTransformerRegressor(nn.Module):
	def __init__(self, num_features: int, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2) -> None:
		super().__init__()
		self.input_proj = nn.Linear(num_features, embed_dim)
		self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
		encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=128, batch_first=True)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.dropout = nn.Dropout(0.1)
		self.head = nn.Linear(embed_dim, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (batch, seq, features)
		x = self.input_proj(x)  # (batch, seq, embed)
		x = x.transpose(1, 2)  # (batch, embed, seq)
		x = torch.relu(self.conv1(x))
		x = x.transpose(1, 2)  # (batch, seq, embed)
		x = self.transformer(x)
		x = x.mean(dim=1)  # global average pooling over time
		x = self.dropout(x)
		out = self.head(x).squeeze(-1)
		return out


def fit_prophet(train_df: pd.DataFrame, full_ds: pd.Series) -> Tuple[pd.Series, pd.Series]:
	# Prepare Prophet dataframe
	train_p = pd.DataFrame({
		"ds": train_df.index,
		"y": train_df[TARGET_COL].values,
	})
	try:
		from prophet import Prophet  # type: ignore
		m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False, seasonality_mode="additive")
		m.fit(train_p)
		# Predict on full timeline to get train and test yhat
		future = pd.DataFrame({"ds": full_ds})
		forecast = m.predict(future)
		yhat_full = pd.Series(forecast["yhat"].values, index=future["ds"].values)
		return yhat_full.loc[train_df.index], yhat_full
	except Exception as exc:
		print("Prophet failed; falling back to seasonal naive baseline for trend/seasonality.", file=sys.stderr)
		print(str(exc), file=sys.stderr)
		# Fallback: seasonal naive using previous day's value (48 steps)
		y_full = train_df[TARGET_COL].copy()
		# Build full yhat by shifting by 48 across the entire index
		full_index = full_ds
		# Align y to full index with NaN for missing
		# For training horizon, use a centered rolling mean as rough trend proxy
		yhat_full = y_full.reindex(full_index).shift(SEASONAL_PERIOD)
		# Fill initial NaNs with overall mean
		fill_value = float(y_full.mean()) if y_full.notna().any() else 0.0
		yhat_full = yhat_full.fillna(fill_value)
		return yhat_full.loc[train_df.index], yhat_full


def main() -> None:
	set_seeds(RANDOM_SEED)

	features, y, df_all, train_idx, test_idx = prepare_data()

	# Prophet stage
	train_df = pd.DataFrame({TARGET_COL: y.loc[train_idx]}, index=train_idx)
	yhat_train_prophet, yhat_full_prophet = fit_prophet(train_df, y.index)
	yhat_test_prophet = yhat_full_prophet.loc[test_idx]

	# Residuals for training
	residual_train = (y.loc[yhat_train_prophet.index] - yhat_train_prophet).astype(float)

	# Standardize features using training stats only
	feat_train = features.loc[train_idx]
	feat_test = features.loc[test_idx]
	mean_vec = feat_train.mean(axis=0)
	std_vec = feat_train.std(axis=0).replace(0.0, 1.0)
	features_std = (features - mean_vec) / std_vec

	# Build sequence arrays over the full standardized feature matrix
	X_all = features_std.values.astype(np.float32)
	# Align target residual array over full timeline; we will index with integer positions
	all_times = features_std.index
	# Map indices to integer positions
	pos_map: Dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(all_times)}

	# Build training indices where residual exists and we have enough history
	train_positions: List[int] = []
	residual_array = np.full((len(all_times),), np.nan, dtype=np.float32)
	for ts, val in residual_train.items():
		if ts in pos_map:
			residual_array[pos_map[ts]] = float(val)
	for i in range(SEQ_LEN, len(all_times)):
		ts_i = all_times[i]
		if ts_i in residual_train.index and not np.isnan(residual_array[i]):
			train_positions.append(i)

	if len(train_positions) == 0:
		print("No training positions available for residual model.", file=sys.stderr)
		sys.exit(2)

	# Build dataset and dataloader
	train_dataset = SeqDataset(X_all, residual_array, train_positions, SEQ_LEN)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

	# Model
	num_features = X_all.shape[1]
	model = CNNTransformerRegressor(num_features=num_features, embed_dim=64, num_heads=4, num_layers=2)
	device = torch.device("cpu")
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=LR)
	criterion = nn.MSELoss()

	# Train
	model.train()
	for epoch in range(EPOCHS):
		running_loss = 0.0
		for batch_X, batch_y in train_loader:
			batch_X = batch_X.to(device)
			batch_y = batch_y.to(device)
			optimizer.zero_grad()
			pred = model(batch_X)
			loss = criterion(pred, batch_y)
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * batch_X.size(0)
		avg_loss = running_loss / len(train_dataset)
		if (epoch + 1) % 5 == 0 or epoch == 0:
			print(f"Epoch {epoch+1}/{EPOCHS} - train MSE: {avg_loss:.5f}")

	# Inference on test: build residual predictions per test time step
	model.eval()
	with torch.no_grad():
		pred_residuals_test: Dict[pd.Timestamp, float] = {}
		for ts in test_idx:
			if ts not in pos_map:
				continue
			pos = pos_map[ts]
			if pos < SEQ_LEN:
				continue
			x_seq = X_all[pos - SEQ_LEN:pos]
			x_tensor = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)
			y_res = model(x_tensor).cpu().numpy().reshape(-1)[0]
			pred_residuals_test[ts] = float(y_res)

	# Align predictions
	resid_pred_series = pd.Series(pred_residuals_test)
	resid_pred_series = resid_pred_series.reindex(test_idx).ffill().bfill()

	# Final hybrid prediction
	y_true_test = y.loc[test_idx]
	y_pred_hybrid = yhat_test_prophet.loc[test_idx].add(resid_pred_series, fill_value=0.0)

	# Metrics
	mae = mean_absolute_error(y_true_test, y_pred_hybrid)
	mse = mean_squared_error(y_true_test, y_pred_hybrid)
	rmse = math.sqrt(mse)
	mape = (np.abs((y_true_test - y_pred_hybrid) / y_true_test).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100
	ss_tot = ((y_true_test - y_true_test.mean()) ** 2).sum()
	ss_res = ((y_true_test - y_pred_hybrid) ** 2).sum()
	r2 = float('nan') if ss_tot == 0 else 1 - ss_res / ss_tot

	print("\nProphet + CNN-Transformer hybrid performance on test subset:")
	print(f"MAE:  {mae:.3f}")
	print(f"RMSE: {rmse:.3f}")
	print(f"MAPE: {mape:.2f}%")
	print(f"R2:   {r2:.4f}")

	# Prophet-only benchmark
	mae_prophet = mean_absolute_error(y_true_test, yhat_test_prophet.loc[test_idx])
	mse_prophet = mean_squared_error(y_true_test, yhat_test_prophet.loc[test_idx])
	rmse_prophet = math.sqrt(mse_prophet)
	ss_res_p = ((y_true_test - yhat_test_prophet.loc[test_idx]) ** 2).sum()
	r2_prophet = float('nan') if ss_tot == 0 else 1 - ss_res_p / ss_tot
	print("\nProphet-only benchmark:")
	print(f"MAE:  {mae_prophet:.3f}")
	print(f"RMSE: {rmse_prophet:.3f}")
	print(f"R2:   {r2_prophet:.4f}")

	# Save predictions
	out = pd.DataFrame({
		"y_true": y_true_test,
		"prophet": yhat_test_prophet.loc[test_idx],
		"cnn_trans_resid": resid_pred_series.loc[test_idx],
		"hybrid_pred": y_pred_hybrid.loc[test_idx],
	})
	out.to_csv("/workspace/hybrid_prophet_cnn_transformer_predictions.csv", index_label="timestamp")
	print("\nSaved predictions to /workspace/hybrid_prophet_cnn_transformer_predictions.csv")


if __name__ == "__main__":
	main()