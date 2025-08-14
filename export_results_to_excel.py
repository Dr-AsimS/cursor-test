#!/usr/bin/env python3

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


OUTPUT_XLSX = Path("/workspace/model_results.xlsx")


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
	mask = y_true.notna() & y_pred.notna()
	y = y_true[mask].astype(float)
	yhat = y_pred[mask].astype(float)
	if len(y) == 0:
		return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE%": float("nan"), "R2": float("nan")}
	mae = float((y - yhat).abs().mean())
	mse = float(((y - yhat) ** 2).mean())
	rmse = math.sqrt(mse)
	mape = float((np.abs((y - yhat) / y).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100)
	ss_tot = float(((y - y.mean()) ** 2).sum())
	ss_res = float(((y - yhat) ** 2).sum())
	r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
	return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "R2": r2}


def main() -> None:
	metrics_rows = []

	with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
		# Hybrid SARIMAX+XGB predictions
		hybrid_csv = Path("/workspace/hybrid_predictions.csv")
		if hybrid_csv.exists():
			df_h = pd.read_csv(hybrid_csv)
			# Write predictions
			df_h.to_excel(writer, sheet_name="hybrid_predictions", index=False)
			# Metrics for hybrid and sarimax-only
			if {"y_true", "hybrid_pred"}.issubset(df_h.columns):
				m = compute_metrics(df_h["y_true"], df_h["hybrid_pred"])
				metrics_rows.append({"model": "Hybrid (SARIMAX+XGB)", **m})
			if {"y_true", "sarimax"}.issubset(df_h.columns):
				m = compute_metrics(df_h["y_true"], df_h["sarimax"])
				metrics_rows.append({"model": "SARIMAX only", **m})

		# Prophet + CNN-Transformer predictions
		prophet_cnn_csv = Path("/workspace/hybrid_prophet_cnn_transformer_predictions.csv")
		if prophet_cnn_csv.exists():
			df_p = pd.read_csv(prophet_cnn_csv)
			df_p.to_excel(writer, sheet_name="prophet_cnn_transformer", index=False)
			if {"y_true", "hybrid_pred"}.issubset(df_p.columns):
				m = compute_metrics(df_p["y_true"], df_p["hybrid_pred"])
				metrics_rows.append({"model": "Prophet+CNN-Transformer", **m})
			if {"y_true", "prophet"}.issubset(df_p.columns):
				m = compute_metrics(df_p["y_true"], df_p["prophet"])
				metrics_rows.append({"model": "Prophet only", **m})

		# TCN+Transformer predictions
		tcn_csv = Path("/workspace/tcn_transformer_predictions.csv")
		if tcn_csv.exists():
			df_t = pd.read_csv(tcn_csv)
			df_t.to_excel(writer, sheet_name="tcn_transformer", index=True)
			if {"y_true", "y_pred"}.issubset(df_t.columns):
				m = compute_metrics(df_t["y_true"], df_t["y_pred"])
				metrics_rows.append({"model": "TCN+Transformer", **m})

		# Metrics sheet
		if metrics_rows:
			df_metrics = pd.DataFrame(metrics_rows)
			order = ["model", "MAE", "RMSE", "MAPE%", "R2"]
			df_metrics = df_metrics.reindex(columns=order)
			df_metrics.to_excel(writer, sheet_name="metrics", index=False)

	print(f"Saved Excel to {OUTPUT_XLSX}")


if __name__ == "__main__":
	main()