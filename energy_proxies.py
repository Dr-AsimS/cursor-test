#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Inputs
MERGED_30_PATH = Path("/workspace/merged_pig_weather_30min.csv")
MERGED_PATH    = Path("/workspace/merged_pig_weather.csv")
PRED_PATH      = Path("/workspace/predictions_with_errors.csv")
SCHED_PATH     = Path("/workspace/adaptive_pid_schedule.csv")

# Columns
TIMECOL = "timestamp"
Y_COL   = "Indoor-Temp(°C)"
CO2_COL = "CO2(ppm)"

# Comfort band
BAND_HALF = 0.5
FIXED_SETPOINT = 22.0

# CO2 decay estimation parameters
C_OUT = 420.0  # ppm
VOLUME_M3 = 3000.0  # Assumed barn volume (m^3)
SPF_KW_PER_CMS = 1.0  # kW per (m^3/s)

# Outputs
EXPORT_DIR = Path("/workspace/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = EXPORT_DIR / "energy_proxies_summary.csv"
POWER_DAILY_CSV = EXPORT_DIR / "energy_proxies_fan_power_daily.csv"
PLOT_TIME_IN_BAND = EXPORT_DIR / "energy_proxies_time_in_band.png"
PLOT_IAE_BARS = EXPORT_DIR / "energy_proxies_iae_bars.png"
PLOT_FAN_POWER_DAILY = EXPORT_DIR / "energy_proxies_fan_power_daily.png"


def load_merged() -> pd.DataFrame:
    if MERGED_30_PATH.exists():
        df = pd.read_csv(MERGED_30_PATH)
    else:
        df = pd.read_csv(MERGED_PATH)
    if TIMECOL in df.columns:
        df[TIMECOL] = pd.to_datetime(df[TIMECOL])
        df = df.set_index(TIMECOL)
    df = df.sort_index()
    return df


def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PRED_PATH, index_col=0, parse_dates=True)
    return df.sort_index()


def load_schedule() -> pd.DataFrame:
    df = pd.read_csv(SCHED_PATH, index_col=0, parse_dates=True)
    # Expect columns: y_pred, y_pred_fwd, delta_2h, T_set_adjust, Kp_mult, Ki_mult, Kd_mult, T_set
    return df.sort_index()


def compute_delta_t_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.5
    dt = pd.Series(index).diff().dropna().median()
    # Convert to hours
    return float(dt.total_seconds() / 3600.0)


def comfort_and_iae(y: pd.Series, set_series: pd.Series, band_half: float = BAND_HALF) -> dict:
    idx = y.index.intersection(set_series.index)
    y = y.loc[idx].astype(float)
    s = set_series.loc[idx].astype(float)
    dt_h = compute_delta_t_hours(y.index)

    # Comfort in band: abs(y - s) <= band_half
    in_band = (y.sub(s).abs() <= band_half)
    comfort_pct = float(in_band.mean() * 100.0)

    err = y.sub(s)
    iae_total = float(err.abs().sum() * dt_h)
    iae_heat = float(err[err < 0].abs().sum() * dt_h)
    iae_cool = float(err[err > 0].abs().sum() * dt_h)

    return dict(comfort_pct=comfort_pct, IAE_total=iae_total, IAE_heat=iae_heat, IAE_cool=iae_cool)


def estimate_airflow_power_from_co2(df_merged: pd.DataFrame, c_out: float = C_OUT, volume_m3: float = VOLUME_M3, spf_kw_per_cms: float = SPF_KW_PER_CMS) -> pd.DataFrame:
    co2 = pd.to_numeric(df_merged.get(CO2_COL), errors="coerce").dropna()
    co2 = co2.sort_index()

    if co2.empty:
        return pd.DataFrame(columns=["avg_power_kW", "total_energy_kWh"])  # empty

    idx = co2.index
    if len(idx) < 2:
        return pd.DataFrame(columns=["avg_power_kW", "total_energy_kWh"])  # empty

    dt_h = compute_delta_t_hours(idx)
    dt_s = dt_h * 3600.0

    # Compute step-wise Q where CO2 decays
    co2_next = co2.shift(-1)
    mask_valid = (co2 > c_out + 5.0) & (co2_next > c_out + 5.0) & (co2_next < co2)
    numer = (co2 - c_out).clip(lower=1e-6)
    denom = (co2_next - c_out).clip(lower=1e-6)
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_ratio = np.log(numer / denom)
    q_cms = (volume_m3 / dt_s) * ln_ratio
    q_cms = q_cms.where(mask_valid, other=np.nan)
    q_cms = q_cms.clip(lower=0.0)

    # Power proxy kW
    p_kw = spf_kw_per_cms * q_cms

    # Daily summaries
    daily_avg_power = p_kw.groupby(p_kw.index.date).mean()
    daily_total_energy = (p_kw * dt_h).groupby(p_kw.index.date).sum()

    daily_df = pd.DataFrame({
        "avg_power_kW": daily_avg_power,
        "total_energy_kWh": daily_total_energy,
    })
    daily_df.index = pd.to_datetime(daily_df.index)
    daily_df = daily_df.sort_index()
    return daily_df


def main() -> None:
    # Load data
    merged = load_merged()
    preds = load_predictions()
    sched = load_schedule()

    # Actual indoor temperature series for test (from predictions_with_errors.csv)
    y_actual = preds["actual"].astype(float)

    # Fixed setpoint series over y_actual index
    s_fixed = pd.Series(FIXED_SETPOINT, index=y_actual.index)

    # Adaptive setpoint series (align to y_actual)
    if "T_set" in sched.columns:
        s_adapt = sched["T_set"].astype(float)
    else:
        # fallback: if schedule lacks T_set, emulate using BASE_SETPOINT
        s_adapt = pd.Series(FIXED_SETPOINT, index=y_actual.index)

    # Compute comfort and IAE metrics
    metrics_fixed = comfort_and_iae(y_actual, s_fixed, BAND_HALF)
    metrics_adapt = comfort_and_iae(y_actual, s_adapt, BAND_HALF)

    # CO2-based airflow/power proxy per day (use merged CO2 over the test window range)
    co2_slice = merged.loc[y_actual.index.min(): y_actual.index.max()]
    power_daily = estimate_airflow_power_from_co2(co2_slice)

    # Comparison table
    summary = pd.DataFrame([
        {"scenario": "fixed_22", **metrics_fixed},
        {"scenario": "adaptive", **metrics_adapt},
    ])
    summary.to_csv(SUMMARY_CSV, index=False)

    # Save daily power proxy
    power_daily.to_csv(POWER_DAILY_CSV, index_label="date")

    # Plots
    # Time-in-band vs scenario
    plt.figure(figsize=(5,4))
    scenarios = ["fixed_22", "adaptive"]
    values = [metrics_fixed["comfort_pct"], metrics_adapt["comfort_pct"]]
    plt.bar(scenarios, values, color=["#4c72b0", "#55a868"])
    plt.ylabel("Comfort in Band (%)")
    plt.title("Time in Comfort Band (±0.5°C)")
    plt.ylim(0, 100)
    plt.tight_layout(); plt.savefig(PLOT_TIME_IN_BAND, dpi=160); plt.close()

    # IAE_heat/IAE_cool bars
    plt.figure(figsize=(6,4))
    width = 0.35
    x = np.arange(2)
    heat_vals = [metrics_fixed["IAE_heat"], metrics_adapt["IAE_heat"]]
    cool_vals = [metrics_fixed["IAE_cool"], metrics_adapt["IAE_cool"]]
    plt.bar(x - width/2, heat_vals, width, label="IAE_heat", color="#c44e52")
    plt.bar(x + width/2, cool_vals, width, label="IAE_cool", color="#8172b2")
    plt.xticks(x, scenarios)
    plt.ylabel("IAE (degC·h)")
    plt.title("IAE Heat/Cool by Scenario")
    plt.legend()
    plt.tight_layout(); plt.savefig(PLOT_IAE_BARS, dpi=160); plt.close()

    # CO2-decay derived fan-power proxy by day (plot total daily energy)
    if not power_daily.empty:
        plt.figure(figsize=(10,4))
        power_daily["total_energy_kWh"].plot(kind="bar", color="#64b5cd")
        plt.ylabel("Total Energy Proxy (kWh)")
        plt.title("Daily CO2-Decay Fan Power Proxy (V=%.0f m^3, SPF=%.1f)" % (VOLUME_M3, SPF_KW_PER_CMS))
        plt.tight_layout(); plt.savefig(PLOT_FAN_POWER_DAILY, dpi=160); plt.close()

    print(f"Saved summary to: {SUMMARY_CSV}")
    print(f"Saved daily power proxy to: {POWER_DAILY_CSV}")
    print("Plots saved:")
    print(" -", PLOT_TIME_IN_BAND)
    print(" -", PLOT_IAE_BARS)
    if not power_daily.empty:
        print(" -", PLOT_FAN_POWER_DAILY)


if __name__ == "__main__":
    main()