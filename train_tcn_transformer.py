#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

SEQ_LEN = 96  # 2 days context
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
RANDOM_SEED = 42


def set_seeds(seed: int = 42) -> None:
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


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


def prepare_data() -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, pd.DatetimeIndex]:
	if not EXCEL_PATH.exists():
		print(f"Error: file not found: {EXCEL_PATH}", file=sys.stderr)
		sys.exit(1)

	df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl").dropna(how="all")
	df[DATE_COL] = pd.to_numeric(df.get(DATE_COL), errors="coerce")
	df[TIME_COL] = pd.to_numeric(df.get(TIME_COL), errors="coerce")
	df = df[df[DATE_COL].notna() & df[TIME_COL].notna()].copy()

	df["timestamp"] = build_timestamp(df)
	df = df.set_index("timestamp").sort_index()
	df = df[~df.index.duplicated(keep="first")]

	for col in (TARGET_COL, HUM_COL, CO2_COL):
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")

	df = add_time_features(df)

	if WEATHER_COL in df.columns:
		weather_dummies = pd.get_dummies(df[WEATHER_COL].astype(str).str.strip(), prefix="weather", drop_first=True, dtype=float)
	else:
		weather_dummies = pd.DataFrame(index=df.index)

	base_cols = [c for c in [HUM_COL, CO2_COL, "hour_sin", "hour_cos", "dow_sin", "dow_cos"] if c in df.columns]
	features = pd.concat([df[base_cols], weather_dummies], axis=1)

	y = df[TARGET_COL].astype(float)
	mask = y.notna() & features.notna().all(axis=1)
	features = features[mask]
	y = y[mask]

	# Include past target as an additional feature channel (will be shifted inside dataset)
	features["y"] = y

	train_idx, test_idx = time_based_split(y.index, test_size=0.2)
	return features, y, train_idx, test_idx


class SeqDataset(Dataset):
	def __init__(self, features: pd.DataFrame, y: pd.Series, positions: List[int], seq_len: int) -> None:
		self.X = features.values.astype(np.float32)
		self.y = y.values.astype(np.float32)
		self.index = features.index
		self.positions = positions
		self.seq_len = seq_len

	def __len__(self) -> int:
		return len(self.positions)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		pos = self.positions[idx]
		x_seq = self.X[pos - self.seq_len:pos]
		y_t = self.y[pos]
		return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_t, dtype=torch.float32)


class TCNBlock(nn.Module):
	def __init__(self, channels: int, dilation: int) -> None:
		super().__init__()
		self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
		self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
		self.norm1 = nn.BatchNorm1d(channels)
		self.norm2 = nn.BatchNorm1d(channels)
		self.dropout = nn.Dropout(0.1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		res = x
		x = torch.relu(self.norm1(self.conv1(x)))
		x = self.dropout(x)
		x = torch.relu(self.norm2(self.conv2(x)))
		return x + res


class TCNTransformer(nn.Module):
	def __init__(self, num_features: int, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2) -> None:
		super().__init__()
		self.input_proj = nn.Linear(num_features, embed_dim)
		self.tcn1 = TCNBlock(embed_dim, dilation=1)
		self.tcn2 = TCNBlock(embed_dim, dilation=2)
		self.tcn3 = TCNBlock(embed_dim, dilation=4)
		self.tcn4 = TCNBlock(embed_dim, dilation=8)
		encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=128, batch_first=True)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.head = nn.Sequential(
			nn.LayerNorm(embed_dim),
			nn.Dropout(0.1),
			nn.Linear(embed_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (batch, seq, features)
		x = self.input_proj(x)  # (batch, seq, embed)
		x = x.transpose(1, 2)  # (batch, embed, seq)
		x = self.tcn1(x)
		x = self.tcn2(x)
		x = self.tcn3(x)
		x = self.tcn4(x)
		x = x.transpose(1, 2)  # (batch, seq, embed)
		x = self.transformer(x)
		x = x.mean(dim=1)
		return self.head(x).squeeze(-1)


def main() -> None:
	set_seeds(RANDOM_SEED)
	features, y, train_idx, test_idx = prepare_data()

	# Standardize features (train statistics only)
	mean_vec = features.loc[train_idx].mean(axis=0)
	std_vec = features.loc[train_idx].std(axis=0).replace(0.0, 1.0)
	features_std = (features - mean_vec) / std_vec

	# Build integer position maps
	all_times = features_std.index
	pos_map: Dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(all_times)}

	train_positions: List[int] = []
	test_positions: List[int] = []
	for ts in train_idx:
		if ts in pos_map and pos_map[ts] >= SEQ_LEN:
			train_positions.append(pos_map[ts])
	for ts in test_idx:
		if ts in pos_map and pos_map[ts] >= SEQ_LEN:
			test_positions.append(pos_map[ts])

	train_dataset = SeqDataset(features_std, y, train_positions, SEQ_LEN)
	test_dataset = SeqDataset(features_std, y, test_positions, SEQ_LEN)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

	num_features = features_std.shape[1]
	device = torch.device("cpu")
	model = TCNTransformer(num_features=num_features, embed_dim=64, num_heads=4, num_layers=2).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=LR)
	criterion = nn.MSELoss()

	# Train
	model.train()
	for epoch in range(EPOCHS):
		running = 0.0
		count = 0
		for batch_X, batch_y in train_loader:
			batch_X = batch_X.to(device)
			batch_y = batch_y.to(device)
			optimizer.zero_grad()
			pred = model(batch_X)
			loss = criterion(pred, batch_y)
			loss.backward()
			optimizer.step()
			running += loss.item() * batch_X.size(0)
			count += batch_X.size(0)
		if (epoch + 1) % 5 == 0 or epoch == 0:
			print(f"Epoch {epoch+1}/{EPOCHS} - train MSE: {running / max(1,count):.5f}")

	# Evaluate
	model.eval()
	preds_list: List[float] = []
	y_list: List[float] = []
	with torch.no_grad():
		for batch_X, batch_y in test_loader:
			batch_X = batch_X.to(device)
			pred = model(batch_X).cpu().numpy().reshape(-1)
			preds_list.append(pred)
			y_list.append(batch_y.numpy().reshape(-1))
		y_true = np.concatenate(y_list)
		y_pred = np.concatenate(preds_list)

	mae = mean_absolute_error(y_true, y_pred)
	mse = mean_squared_error(y_true, y_pred)
	rmse = math.sqrt(mse)
	ss_tot = ((y_true - y_true.mean()) ** 2).sum()
	ss_res = ((y_true - y_pred) ** 2).sum()
	r2 = float('nan') if ss_tot == 0 else 1 - ss_res / ss_tot

	print("\nTCN+Transformer performance on test subset:")
	print(f"MAE:  {mae:.3f}")
	print(f"RMSE: {rmse:.3f}")
	print(f"R2:   {r2:.4f}")

	# Save predictions with timestamps
	test_times = [all_times[pos] for pos in test_positions]
	out = pd.DataFrame({
		"y_true": y_true,
		"y_pred": y_pred,
	}, index=pd.Index(test_times, name="timestamp"))
	out.to_csv("/workspace/tcn_transformer_predictions.csv")
	print("Saved predictions to /workspace/tcn_transformer_predictions.csv")


if __name__ == "__main__":
	main()