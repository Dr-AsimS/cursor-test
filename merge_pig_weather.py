#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
import math

import numpy as np
import pandas as pd

PIG_PATH = Path("/workspace/pig_farm_7_2024.xlsx")
WEATHER_PATH = Path("/workspace/weather data 7.xlsx")
PIG_SHEET = "pig_farm_7_2024"
CSV_OUT = Path("/workspace/merged_pig_weather.csv")
XLSX_OUT = Path("/workspace/merged_pig_weather.xlsx")


def build_pig_timestamp(df: pd.DataFrame) -> pd.DataFrame:
	# Ensure numeric for Date and Time
	df["Date"] = pd.to_numeric(df.get("Date"), errors="coerce")
	df["Time"] = pd.to_numeric(df.get("Time"), errors="coerce")
	df = df[df["Date"].notna() & df["Time"].notna()].copy()
	epoch = pd.Timestamp("1899-12-30")
	df["timestamp"] = epoch + pd.to_timedelta(df["Date"] + df["Time"], unit="D")
	# Round to nearest minute
	df["timestamp"] = df["timestamp"].dt.round("min")
	# Index and sort
	df = df.set_index("timestamp").sort_index()
	# Drop duplicate timestamps, keep first
	df = df[~df.index.duplicated(keep="first")]
	return df


def load_pig() -> pd.DataFrame:
	if not PIG_PATH.exists():
		print(f"Pig file not found: {PIG_PATH}", file=sys.stderr)
		sys.exit(1)
	df = pd.read_excel(PIG_PATH, sheet_name=PIG_SHEET, engine="openpyxl").dropna(how="all")
	return build_pig_timestamp(df)


def load_and_resample_weather() -> pd.DataFrame:
	if not WEATHER_PATH.exists():
		print(f"Weather file not found: {WEATHER_PATH}", file=sys.stderr)
		sys.exit(1)
	df = pd.read_excel(WEATHER_PATH, sheet_name=0, engine="openpyxl").dropna(how="all")
	# Parse time and round to minute
	if "time" not in df.columns:
		print("Weather file missing 'time' column", file=sys.stderr)
		sys.exit(2)
	df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
	df = df[df["timestamp"].notna()].copy()
	df["timestamp"] = df["timestamp"].dt.round("min")
	df = df.set_index("timestamp").sort_index()
	# Separate numeric and non-numeric (excluding the original 'time' string)
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	non_numeric_cols = [c for c in df.columns if c not in numeric_cols and c != "time"]
	# Resample numeric to 30-minute with time interpolation
	df_num_30 = df[numeric_cols].resample("30min").interpolate(method="time") if numeric_cols else pd.DataFrame(index=pd.date_range(df.index.min(), df.index.max(), freq="30min"))
	# Resample non-numeric with forward fill/backfill
	if non_numeric_cols:
		df_nonnum_30 = df[non_numeric_cols].resample("30min").ffill().bfill()
		weather_30 = pd.concat([df_num_30, df_nonnum_30], axis=1)
	else:
		weather_30 = df_num_30
	# Ensure sorted and no duplicate index
	weather_30 = weather_30.sort_index()
	weather_30 = weather_30[~weather_30.index.duplicated(keep="first")]
	return weather_30


def main() -> None:
	pig_df = load_pig()
	weather_df_30 = load_and_resample_weather()

	# Left-join weather columns onto pig by timestamp
	merged = pig_df.merge(weather_df_30, left_index=True, right_index=True, how="left", suffixes=("", "_weather"))

	print(f"Pig rows: {len(pig_df)}, Weather rows (30m): {len(weather_df_30)}, Merged rows: {len(merged)}")
	print("Merged columns:")
	print(list(merged.columns))
	print("\nPreview (first 10 rows):")
	print(merged.head(10).to_string())

	# Export to CSV and Excel (include timestamp as a column)
	merged_out = merged.reset_index()
	merged_out.to_csv(CSV_OUT, index=False)
	merged_out.to_excel(XLSX_OUT, index=False)
	print(f"\nSaved merged dataset to: {CSV_OUT} and {XLSX_OUT}")


if __name__ == "__main__":
	main()