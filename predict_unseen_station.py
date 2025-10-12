#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict 7-day water levels for an UNSEEN station using a saved multi-station model.

Key ideas:
- Input parity with training: process_features -> categorize_features -> smart_fillna
- Feature-aware scaling (rainfall log1p+Robust, humidity/pressure MinMax, others Standard, cyclical pass-through)
- Unseen-station embedding handling:
    nearest  : use the training station whose TARGET scaler stats are closest
    topk     : weighted average of K nearest stations' predictions
    average  : average predictions over all training stations

Outputs:
- predictions CSV aligned to the last observation date
- processed daily weather/water CSVs + raw API dumps (for traceability)
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta, timezone

import requests
import numpy as np
import pandas as pd
import torch

# ---------- repo paths / model ----------
REPO_DIR = os.path.abspath(".")
sys.path.insert(0, REPO_DIR)

MODEL_PATH = os.path.join(REPO_DIR, "models/transformer_h512_L2_seq40--Borup_Bygholm_Gesager_Himmelev_Hoven_Karstoft_Kirkea_Ledreborg_Mollebaek_Sengelose_train_Skjern.pth")

OUT_DIR = "./predictions"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_RAW_WEATHER = os.path.join(OUT_DIR, "raw_weather_data.json")
OUT_RAW_WATER   = os.path.join(OUT_DIR, "raw_water_data.json")
OUT_CSV_WEATHER = os.path.join(OUT_DIR, "processed_weather_data.csv")
OUT_CSV_WATER   = os.path.join(OUT_DIR, "processed_water_data.csv")

# ---------- import your training helpers ----------
sys.path.insert(0, os.path.join(REPO_DIR, "Data and code from Florian"))
from Trainer import (  # feature processing & scaling parity with training
    process_features,          # adds season_sin/cos; returns df_proc, feature_cols
    categorize_features,       # per-feature-type categorization
    smart_fillna,              # imputation consistent with feature types
    apply_station_scalers,     # apply existing per-station scalers
    apply_global_scalers,      # apply global scalers
    inverse_transform_predictions,  # invert target scaling
    scale_features_per_station      # fit NEW per-station scalers on the fly
)
from Models_Alt import get_model       # transformer/gru factory
from Models import MultiStationLSTM    # lstm architecture


# ---------- data fetch & prep ----------

def fetch_weather_daily(lat: float, lon: float, past_days: int,
                        tz: str = "Europe/Copenhagen") -> pd.DataFrame:
    """Open-Meteo hourly → daily means; columns: date, Temp, Humidity, Wind_speed, rainfall_sum."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "past_days": past_days,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": tz
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    with open(OUT_RAW_WEATHER, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    h = data["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "temperature_celsius": h["temperature_2m"],
        "relative_humidity_percent": h["relative_humidity_2m"],
        "wind_speed_kmh": h["wind_speed_10m"],
        "precipitation": h["precipitation"],
    })
    
    # Fill NaNs before aggregations
    df["precipitation"] = pd.Series(h["precipitation"]).fillna(0).values
    df[["temperature_celsius","relative_humidity_percent","wind_speed_kmh"]] = \
        df[["temperature_celsius","relative_humidity_percent","wind_speed_kmh"]].fillna(method="ffill")
    
    df["wind_speed_ms"] = df["wind_speed_kmh"] * 0.27778
    df["date"] = df["time"].dt.date

    daily = df.groupby("date", as_index=False).agg({
        "temperature_celsius": "mean",
        "relative_humidity_percent": "mean",
        "wind_speed_ms": "mean",
        "precipitation": "sum",  # Sum precipitation for daily total
    })
    daily = daily.rename(columns={
        "temperature_celsius": "Temp",
        "relative_humidity_percent": "Humidity",
        "wind_speed_ms": "Wind_speed",
        "precipitation": "rainfall_sum_1d",
    })
    daily.to_csv(OUT_CSV_WEATHER, index=False)
    return daily


def check_and_fill_missing_days(daily_df: pd.DataFrame, max_missing: int = 3, 
                                 lookback_days: int = 40) -> pd.DataFrame:
    """
    Check for missing days and fill them using smart interpolation strategies.
    Handles gaps in the beginning, middle, and end of data.
    
    Strategies:
    - Gaps in the middle: Linear interpolation
    - Gaps at the beginning: Backward fill
    - Gaps at the end: Forward fill
    
    Args:
        daily_df: DataFrame with 'date' and 'water_level_cm' columns
        max_missing: Maximum number of missing days allowed (default: 3, meaning 1, 2, or 3)
        lookback_days: Number of days to look back for checking missing data (default: 40)
    
    Returns:
        DataFrame with filled values if missing days <= max_missing
        
    Raises:
        RuntimeError: If missing days > max_missing
    """
    if daily_df.empty:
        return daily_df
    
    # Convert date column to datetime
    daily_df = daily_df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values('date').reset_index(drop=True)
    
    # Get the most recent date and calculate the start of the lookback period
    most_recent_date = daily_df['date'].max()
    lookback_start = most_recent_date - timedelta(days=lookback_days)
    
    # Filter to the lookback period
    lookback_df = daily_df[daily_df['date'] >= lookback_start].copy()
    
    if len(lookback_df) == 0:
        print(f"⚠️  No data in the last {lookback_days} days")
        return daily_df
    
    # Create a complete date range for the lookback period
    date_range_start = lookback_df['date'].min()
    date_range_end = lookback_df['date'].max()
    all_dates = pd.date_range(start=date_range_start, end=date_range_end, freq='D')
    
    # Find missing dates
    existing_dates = set(lookback_df['date'].dt.date)
    all_dates_set = set(all_dates.date)
    missing_dates = sorted(all_dates_set - existing_dates)
    
    num_missing = len(missing_dates)
    
    if num_missing == 0:
        print(f"✅ No missing days in the last {lookback_days} days")
        return daily_df
    
    print(f"📊 Found {num_missing} missing days in the last {lookback_days} days")
    
    if num_missing > max_missing:
        raise RuntimeError(
            f"❌ Too many missing days in the last {lookback_days} days: {num_missing} missing days found.\n"
            f"   Maximum allowed: {max_missing} days (meaning 1, 2, or 3 days).\n"
            f"   Missing dates: {missing_dates}\n"
            f"   Cannot generate reliable predictions with this much missing data."
        )
    
    # Fill missing days using smart strategies
    print(f"🔧 Filling {num_missing} missing days...")
    
    # Create complete date range DataFrame
    complete_dates_df = pd.DataFrame({'date': all_dates})
    
    # Merge with existing data
    merged_df = complete_dates_df.merge(lookback_df[['date', 'water_level_cm']], 
                                         on='date', how='left')
    
    # Strategy 1: Linear interpolation for gaps in the middle
    merged_df['water_level_cm'] = merged_df['water_level_cm'].interpolate(
        method='linear', limit_direction='both'
    )
    
    # Strategy 2: Forward fill for any remaining NaN at the end
    merged_df['water_level_cm'] = merged_df['water_level_cm'].fillna(method='ffill')
    
    # Strategy 3: Backward fill for any remaining NaN at the beginning
    merged_df['water_level_cm'] = merged_df['water_level_cm'].fillna(method='bfill')
    
    # Combine: take data before lookback period + filled lookback period
    before_lookback = daily_df[daily_df['date'] < lookback_start]
    result_df = pd.concat([before_lookback, merged_df], ignore_index=True)
    result_df = result_df.sort_values('date').reset_index(drop=True)
    
    print(f"✅ Successfully filled {num_missing} missing days (interpolation + forward/backward fill)")
    
    return result_df


def fetch_water_daily(vandah_station_id: str, past_days: int) -> pd.DataFrame:
    """Vandah 15-min → daily mean (cm); columns: date, water_level_cm."""
    to_time = datetime.now(timezone.utc).replace(microsecond=0)
    from_time = to_time - timedelta(days=past_days)

    url = "https://vandah.miljoeportal.dk/api/water-levels"
    params = {
        "stationId": vandah_station_id,
        "from": from_time.strftime("%Y-%m-%dT%H:%MZ"),
        "to":   to_time.strftime("%Y-%m-%dT%H:%MZ"),
        "format": "json",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    raw = r.json()
    with open(OUT_RAW_WATER, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    if not raw or not raw[0].get("results"):
        raise RuntimeError("No water level data returned from Vandah")

    recs = raw[0]["results"]
    df = pd.DataFrame({
        "dt": pd.to_datetime([rr["measurementDateTime"] for rr in recs], utc=True),
        "water_level_cm": [rr["result"] for rr in recs],
    })
    df["date"] = df["dt"].dt.date
    daily = df.groupby("date", as_index=False)["water_level_cm"].mean()
    
    # STEP 1: Check total missing days including gap to today
    daily['date'] = pd.to_datetime(daily['date'])
    last_date = daily['date'].max()
    today = pd.Timestamp.now().normalize()
    days_to_today = (today - last_date).days
    
    # STEP 2: Check and fill missing days within the data range + extend to today
    daily_filled = check_and_fill_missing_days(daily, max_missing=3, lookback_days=40)
    
    # STEP 3: Extend to today if needed (counting this as part of total missing days)
    if 0 < days_to_today <= 3:
        # Extend to today with forward fill
        print(f"📅 Extending data from {last_date.date()} to {today.date()} ({days_to_today} days)")
        date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), end=today, freq='D')
        last_value = daily_filled['water_level_cm'].iloc[-1]
        
        extension_df = pd.DataFrame({
            'date': date_range,
            'water_level_cm': last_value
        })
        
        daily_filled = pd.concat([daily_filled, extension_df], ignore_index=True)
        print(f"✅ Extended with {len(extension_df)} days using last value: {last_value:.2f} cm")
    elif days_to_today > 3:
        raise RuntimeError(
            f"❌ Data is too old: last measurement was {days_to_today} days ago ({last_date.date()}).\n"
            f"   Maximum allowed data age: 3 days.\n"
            f"   Cannot generate reliable predictions with data this stale."
        )
    
    # Ensure date column is in the correct format (not datetime)
    daily = daily_filled
    daily['date'] = daily['date'].dt.date
    
    daily.to_csv(OUT_CSV_WATER, index=False)
    return daily


# ---------- model utilities ----------

def load_model_from_checkpoint(model_path: str, device: torch.device):
    """Load model + checkpoint dict."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_type = ckpt["type"]
    cfg = ckpt["model_config"]

    if "lstm" in model_type.lower():
        model = MultiStationLSTM(
            cfg["input_size"], cfg["hidden_size"], cfg["num_layers"], cfg["output_size"],
            cfg["num_stations"], cfg.get("station_embedding_dim", 8),
            cfg.get("use_seasons", False), cfg.get("season_embedding_dim", 4),
            cfg.get("dropout", 0.5)
        )
    else:
        model = get_model(model_name=model_type, **cfg)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model, ckpt


def build_inputs_parity(merged_daily: pd.DataFrame, ckpt: dict):
    """
    Build inputs exactly like training:
    - convert water_level cm→m
    - add Season (1..4) then process_features (adds season_sin/cos)
    - smart_fillna
    Returns (df_proc, feature_cols)
    """
    df = merged_daily.copy()
    if "water_level_cm" not in df.columns:
        raise ValueError("merged_daily must have 'water_level_cm'")

    df["water_level"] = df["water_level_cm"] / 100.0
    
    # ensure expected weather cols exist (NaNs allowed; smart_fillna will handle)
    for col in ["Temp", "Humidity", "Wind_speed"]:
        if col not in df.columns:
            df[col] = np.nan
    
    # Build any rainfall_sum_{Nd} features the checkpoint needs
    import re
    if "rainfall_sum_1d" in df.columns:
        wanted_sums = set()
        for f in ckpt.get("features", []):
            m = re.match(r"rainfall_sum_(\d+)d", f)
            if m:
                wanted_sums.add(int(m.group(1)))
        for N in sorted(wanted_sums):
            col = f"rainfall_sum_{N}d"
            if col not in df.columns:
                df[col] = (
                    df["rainfall_sum_1d"]
                    .rolling(window=N, min_periods=1).sum()
                )
    
    # Add lagged features only if requested by checkpoint
    feature_set = ckpt.get("features", [])
    if "water_level_lag_1d" in feature_set:
        df["water_level_lag_1d"] = df["water_level"].shift(1)
    if "Temp_4day_lagged" in feature_set:
        df["Temp_4day_lagged"] = df["Temp"].rolling(4, min_periods=1).mean().shift(1)
    
    # Calculate API only if requested by checkpoint
    if "API" in feature_set and "rainfall_sum_1d" in df.columns:
        api_values = []
        for i in range(len(df)):
            if i == 0:
                api_values.append(df["rainfall_sum_1d"].iloc[i])
            else:
                # API = previous API * 0.9 + current rainfall
                prev_api = api_values[-1] if api_values else 0
                current_rain = df["rainfall_sum_1d"].iloc[i]
                api_values.append(prev_api * 0.9 + current_rain)
        df["API"] = api_values
    elif "API" in feature_set:
        df["API"] = 0.0

    dates = pd.to_datetime(df["date"])
    # 1..4 (training uses Season numeric, then process_features -> sin/cos)
    df["Season"] = ((dates.dt.month % 12 + 3) // 3).astype(int)

    feature_set = ckpt.get("features", [])
    print(f"Checkpoint feature set: {feature_set}")
    
    # Map rainfall aliases to the names the checkpoint might use
    if "Rainfall_day" in feature_set and "rainfall_sum_1d" in df.columns:
        df["Rainfall_day"] = df["rainfall_sum_1d"]
    if "Precipitation" in feature_set and "rainfall_sum_1d" in df.columns:
        df["Precipitation"] = df["rainfall_sum_1d"]
    
    # Ensure every requested feature column exists (filled with NaN if we didn't fetch it)
    for f in feature_set:
        if f == "Season":
            continue
        if f not in df.columns:
            df[f] = np.nan
    
    df_proc, feature_cols = process_features(df, feature_set)   # adds season_sin/cos
    print(f"Final feature columns: {feature_cols}")

    feat_cats = categorize_features(feature_cols)               # feature-aware types
    df_proc = smart_fillna(df_proc, feat_cats)                  # training-like fill
    return df_proc, feature_cols, feat_cats


def fit_new_station_feature_scalers(df_proc: pd.DataFrame, feature_cols, feat_cats):
    """
    For unseen stations, fit NEW per-station feature scalers on the recent window
    using the same per-type logic as training.
    Returns (scaled_np, station_scalers_dict)
    """
    scaled_np, station_scalers = scale_features_per_station(df_proc, feature_cols, feat_cats)
    # Note: station_scalers contains dicts per feature type (e.g., rainfall, pressure, etc.)
    # We also stash the feature order for safety:
    station_scalers["feature_cols"] = list(feature_cols)
    return scaled_np, station_scalers


def choose_embedding_for_unseen(ckpt: dict, water_daily_cm: pd.Series,
                                strategy: str = "nearest", topk: int = 3):
    """
    Choose a station embedding id for an unseen station.

    Distance: compare recent water level distribution (mean/std in m) to each training
    station's target scaler (StandardScaler) mean_/scale_. Fallbacks if missing.

    Returns:
        one of:
          - single int station_id   (nearest)
          - list[int] station_ids   (topk, average)
          - weights (for topk)
    """
    station_to_id = ckpt.get("station_to_id", {})
    scalers = ckpt.get("scalers", {})
    train_stats = []

    # recent stats for new station
    recent_m = (water_daily_cm.values / 100.0)
    mu_new = float(np.nanmean(recent_m))
    sd_new = float(np.nanstd(recent_m) if np.nanstd(recent_m) > 1e-12 else 1.0)

    for st_name, st_id in station_to_id.items():
        st_scalers = scalers.get(st_name, {})
        tgt = st_scalers.get("target")  # StandardScaler usually
        if tgt is None or not hasattr(tgt, "mean_") or not hasattr(tgt, "scale_"):
            continue
        mu = float(tgt.mean_[0])
        sd = float(tgt.scale_[0] if tgt.scale_[0] > 1e-12 else 1.0)
        # simple euclidean in (mu, sd)
        dist = np.sqrt((mu - mu_new) ** 2 + (sd - sd_new) ** 2)
        train_stats.append((dist, st_id, st_name))

    if not train_stats:
        # fallback: first station id or 0
        if station_to_id:
            first_name = next(iter(station_to_id))
            return station_to_id[first_name]

        return 0

    train_stats.sort(key=lambda x: x[0])  # by distance ascending

    if strategy == "average":
        # return all station ids to average predictions equally
        return [sid for _, sid, _ in train_stats], None

    if strategy == "topk":
        k = min(topk, len(train_stats))
        top = train_stats[:k]
        dists = np.array([t[0] for t in top], dtype=float)
        # inverse-distance weights (with epsilon)
        w = 1.0 / (dists + 1e-6)
        w = (w / w.sum()).astype(float)
        ids = [t[1] for t in top]
        return ids, w

    # default: nearest
    return train_stats[0][1]


def predict_with_embedding(model, ckpt, x_np, station_id, last_date, device, target_scaler_new=None):
    """
    Predict using a single station embedding id.
    Returns (dates, preds_cm, preds_m)
    """
    seq_len = ckpt.get("sequence_length", 30)
    pred_days = ckpt.get("prediction_days", 7)
    use_seasons = ckpt["model_config"].get("use_seasons", False)

    x = torch.tensor(x_np[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0)
    sid = torch.tensor([int(station_id)], dtype=torch.long, device=device)

    season_ids = None
    if use_seasons:
        month = pd.Timestamp(last_date).month
        season_1to4 = (month % 12 + 3) // 3
        season_0to3 = int(season_1to4 - 1)
        season_ids = torch.tensor([season_0to3], dtype=torch.long, device=device)

    with torch.no_grad():
        y_scaled = model(x, sid, season_ids).cpu().numpy().reshape(-1, 1)

    # For unseen stations, use the new target scaler fitted on their history
    if target_scaler_new is not None:
        preds_m = target_scaler_new.inverse_transform(y_scaled).flatten()
    else:
        # Fallback to training station scaler (for seen stations)
        norm_per_station = ckpt.get("normalize_per_station", True)
        if not norm_per_station:
            # Global normalization
            preds_m = inverse_transform_predictions(y_scaled, ckpt["scalers"], station_name=None).flatten()
        else:
            # Per-station normalization - use the station that was chosen for embedding
            station_to_id = ckpt.get("station_to_id", {})
            station_name = None
            for name, sid_int in station_to_id.items():
                if sid_int == station_id:
                    station_name = name
                    break
            
            if station_name is None:
                # fallback to first available station
                station_name = list(station_to_id.keys())[0]
            
            preds_m = inverse_transform_predictions(
                y_scaled, ckpt["scalers"], station_name=station_name
            ).flatten()
    
    preds_cm = preds_m * 100.0

    dates = [(pd.Timestamp(last_date) + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
             for i in range(pred_days)]
    return dates, preds_cm, preds_m


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH, help="Path to .pth checkpoint")
    ap.add_argument("--vandah_id", required=True, help="Vandah station id (new/unseen)")
    ap.add_argument("--lat", type=float, required=True, help="Latitude for weather")
    ap.add_argument("--lon", type=float, required=True, help="Longitude for weather")
    ap.add_argument("--past_days", type=int, default=30, help="History days to fetch")
    ap.add_argument("--unseen_strategy", choices=["nearest", "topk", "average"],
                    default="nearest")
    ap.add_argument("--topk", type=int, default=3, help="K for topk strategy")
    ap.add_argument("--anchor", choices=["none", "replace", "adjust", "blend"],
                    default="none", help="Optional anchoring of forecast level")
    ap.add_argument("--blend_alpha", type=float, default=0.5,
                    help="Blend weight if --anchor blend")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("1) Loading model...")
    model, ckpt = load_model_from_checkpoint(args.model, device)

    # Update default past_days if not specified by user
    if args.past_days == 30:  # default value
        seq_len = ckpt.get("sequence_length", 30)
        pred_days = ckpt.get("prediction_days", 7)
        args.past_days = max(seq_len + pred_days, 60)
        print(f"Using {args.past_days} days of history (sequence_length={seq_len} + prediction_days={pred_days})")

    print("2) Fetching weather & water...")
    weather_daily = fetch_weather_daily(args.lat, args.lon, args.past_days)
    water_daily = fetch_water_daily(args.vandah_id, args.past_days)

    # Use left merge to keep only water level dates (not future weather-only dates)
    merged = pd.merge(water_daily, weather_daily, on="date", how="left")
    merged = merged.sort_values('date').reset_index(drop=True)
    
    # Fill missing weather data (if any) using forward fill, then backward fill
    weather_cols = ['Temp', 'Humidity', 'Wind_speed', 'rainfall_sum_1d']
    for col in weather_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(method='ffill').fillna(method='bfill')
    
    if merged.empty:
        raise RuntimeError("No overlapping dates between weather and water data")

    print("3) Building features (training parity) ...")
    df_proc, feature_cols, feat_cats = build_inputs_parity(merged, ckpt)

    # Check normalization setting from checkpoint
    norm_per_station = ckpt.get("normalize_per_station", True)

    # 3a) Scale features based on checkpoint's normalization setting
    if norm_per_station:
        print("4) Scaling features (new per-station scalers for unseen station)...")
        x_np, new_station_scalers = fit_new_station_feature_scalers(df_proc, feature_cols, feat_cats)
    else:
        print("4) Scaling features using GLOBAL scalers from checkpoint...")
        global_scalers = ckpt["scalers"]["global"]
        feature_cols_saved = global_scalers.get("feature_cols", feature_cols)
        feat_cats = global_scalers.get("feature_categories", feat_cats)
        x_np = apply_global_scalers(df_proc, feature_cols_saved, global_scalers, feat_cats)
        feature_cols = feature_cols_saved  # preserve exact order used in training

    # sanity vs model input size
    in_size = ckpt["model_config"]["input_size"]
    if x_np.shape[1] != in_size:
        raise ValueError(f"Feature width mismatch: got {x_np.shape[1]}, expected {in_size}")
    
    # Sequence length guard
    seq_len = ckpt.get("sequence_length", 30)
    if len(x_np) < seq_len:
        raise RuntimeError(f"Not enough rows to form a {seq_len}-day sequence (have {len(x_np)}).")

    last_date = pd.to_datetime(merged["date"]).max()
    current_level_cm = float(merged.sort_values("date")["water_level_cm"].iloc[-1])

    # Target scaler setup based on normalization setting
    from sklearn.preprocessing import StandardScaler
    target_scaler_new = None
    if norm_per_station:
        # Fit NEW target scaler on unseen station's history (meters)
        target_scaler_new = StandardScaler()
        hist_m = (merged["water_level_cm"].values / 100.0).reshape(-1, 1)
        target_scaler_new.fit(hist_m)

    print("5) Choosing embedding for unseen station...")
    emb_choice = choose_embedding_for_unseen(ckpt, merged["water_level_cm"],
                                             strategy=args.unseen_strategy,
                                             topk=args.topk)

    # 6) Predict according to strategy
    print("6) Predicting...")
    if isinstance(emb_choice, tuple) or isinstance(emb_choice, list):
        # topk or average
        if isinstance(emb_choice, tuple):
            ids, weights = emb_choice
        else:
            ids, weights = emb_choice, None  # average

        preds_list_cm, preds_list_m = [], []
        dates_ref = None
        for i, sid in enumerate(ids):
            dates_i, cm, m = predict_with_embedding(model, ckpt, x_np, sid, last_date, device, target_scaler_new)
            if i == 0: dates_ref = dates_i
            preds_list_cm.append(np.array(cm))
            preds_list_m.append(np.array(m))

        preds_arr_cm = np.stack(preds_list_cm, axis=0)
        preds_arr_m  = np.stack(preds_list_m, axis=0)

        if weights is None:
            # simple average
            preds_cm = preds_arr_cm.mean(axis=0)
            preds_m  = preds_arr_m.mean(axis=0)
        else:
            w = np.array(weights).reshape(-1, 1)
            preds_cm = (preds_arr_cm * w).sum(axis=0)
            preds_m  = (preds_arr_m  * w).sum(axis=0)
        
        dates = dates_ref
    else:
        # single nearest id
        dates, preds_cm, preds_m = predict_with_embedding(model, ckpt, x_np, emb_choice, last_date, device, target_scaler_new)

    # anchoring options (align forecast level to last observation)
    if args.anchor != "none":
        if args.anchor == "replace":
            # force day-1 = current level shift for entire horizon
            delta = current_level_cm - preds_cm[0]
            preds_cm = preds_cm + delta
            preds_m  = preds_cm / 100.0
        elif args.anchor == "adjust":
            # add constant offset so day-1 hits current level; do not exceed ±1 std (light guard)
            delta = current_level_cm - preds_cm[0]
            preds_cm = preds_cm + np.clip(delta, -50.0, 50.0)
            preds_m  = preds_cm / 100.0
        elif args.anchor == "blend":
            # blend day-1 toward current obs, decay over horizon
            alpha = float(np.clip(args.blend_alpha, 0.0, 1.0))
            deltas = (current_level_cm - preds_cm[0]) * np.exp(-np.linspace(0, 2.0, len(preds_cm)))
            preds_cm = preds_cm + alpha * deltas
            preds_m  = preds_cm / 100.0

    # save CSV
    out_path = os.path.join(OUT_DIR, f"predictions_{args.vandah_id}_unseen.csv")
    out_df = pd.DataFrame({
        "date": dates,
        "predicted_water_level_cm": preds_cm,
        "predicted_water_level_m": preds_m,
        "change_from_last_daily_mean_cm": preds_cm - current_level_cm
    })
    out_df.to_csv(out_path, index=False)

    # print summary
    print("\n=== Forecast (unseen station) ===")
    for d, cm, m in zip(dates, preds_cm, preds_m):
        sign = "+" if (cm - current_level_cm) >= 0 else ""
        print(f"{d}: {cm:.2f} cm ({m:.3f} m)  ({sign}{cm - current_level_cm:.2f} cm vs last daily mean)")

    print(f"\nPredictions saved → {out_path}")
    print(f"Processed weather  → {OUT_CSV_WEATHER}")
    print(f"Processed water    → {OUT_CSV_WATER}")
    print(f"Raw dumps          → {OUT_RAW_WEATHER}, {OUT_RAW_WATER}")


if __name__ == "__main__":
    main() 