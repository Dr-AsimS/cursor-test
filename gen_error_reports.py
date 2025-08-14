#!/usr/bin/env python3

from pathlib import Path
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hybrid_pid_pipeline import read_data, train_hybrid, INFILE


def main() -> None:
	# Load data and train/evaluate
	df = read_data(INFILE)
	results = train_hybrid(df)

	# Build predictions + errors table
	y_te = results["y_test"]
	y_sar = results["yhat_sarimax_test"]
	y_hyb = results["yhat_hybrid_test"]

	pred_df = pd.DataFrame({
		"actual": y_te,
		"sarimax_pred": y_sar,
		"hybrid_pred": y_hyb,
	}).dropna()

	pred_df["err_sarimax"] = pred_df["actual"] - pred_df["sarimax_pred"]
	pred_df["err_hybrid"]  = pred_df["actual"] - pred_df["hybrid_pred"]
	pred_df["ae_sarimax"]  = pred_df["err_sarimax"].abs()
	pred_df["ae_hybrid"]   = pred_df["err_hybrid"].abs()
	pred_df["ape_hybrid_%"] = (pred_df["ae_hybrid"] / pred_df["actual"].clip(lower=1e-6)) * 100

	pred_path = "predictions_with_errors.csv"
	pred_df.to_csv(pred_path)
	print(f"Saved: {pred_path}")

	# Error-over-time plot (hybrid)
	plt.figure(figsize=(12,4))
	pred_df["err_hybrid"].plot(lw=1)
	plt.axhline(0, ls="--", color="gray")
	plt.title("Hybrid Prediction Error Over Time (Actual - Predicted)")
	plt.ylabel("Error (°C)"); plt.xlabel("Time")
	plt.tight_layout(); plt.savefig("error_over_time_hybrid.png", dpi=160); plt.close()

	# Error histogram (hybrid)
	plt.figure(figsize=(6,4))
	pred_df["err_hybrid"].hist(bins=40)
	plt.title("Hybrid Error Distribution"); plt.xlabel("Error (°C)"); plt.ylabel("Count")
	plt.tight_layout(); plt.savefig("error_hist_hybrid.png", dpi=160); plt.close()

	# Actual vs Predicted scatter (hybrid)
	plt.figure(figsize=(5,5))
	plt.scatter(pred_df["actual"], pred_df["hybrid_pred"], s=10, alpha=0.6)
	m_min = float(pred_df["actual"].min())
	m_max = float(pred_df["actual"].max())
	plt.plot([m_min, m_max], [m_min, m_max], "k--", lw=1)  # 45° line
	plt.xlabel("Actual (°C)"); plt.ylabel("Predicted (°C)")
	plt.title("Actual vs Predicted (Hybrid)")
	plt.tight_layout(); plt.savefig("actual_vs_pred_scatter_hybrid.png", dpi=160); plt.close()

	# Export everything to exports/
	os.makedirs("exports", exist_ok=True)
	for f in [
		"pred_vs_actual_test.png",
		"adaptive_pid_schedule.csv",
		"predictions_with_errors.csv",
		"error_over_time_hybrid.png",
		"error_hist_hybrid.png",
		"actual_vs_pred_scatter_hybrid.png",
	]:
		if os.path.exists(f):
			shutil.copy(f, f"exports/{f}")
	print("Exported metrics & error plots to exports/")


if __name__ == "__main__":
	main()