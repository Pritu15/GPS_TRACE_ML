r"""
Standalone sanity check for inference_bundle.joblib -- Dhaka travel-time prediction model.

Nothing else from the project is needed: just this script + inference_bundle.joblib
in the same folder (or pass the path as an argument).

Install dependencies first:
    pip install pandas numpy scikit-learn xgboost joblib requests

Run:
    python test_inference_bundle.py
    python test_inference_bundle.py path\to\inference_bundle.joblib

What it does:
    1. Loads the bundle (both the XGBoost and Random Forest models, plus everything
       needed to turn a raw (src, dst, time) query into the exact feature row they expect).
    2. Predicts travel time for one example Dhaka trip (Zigatola -> Dhanmondi).
    3. Prints both models' predictions in minutes.

Note on OSRM: the real pipeline calls a local OSRM routing server (bundle["osrm_base"],
normally http://localhost:5050) to get real road distance/duration. Since you probably
don't have that server running, this script auto-detects it and, if unreachable, falls
back to a rough straight-line-based estimate instead -- so it still runs, but predictions
in that mode are noticeably less accurate than the real pipeline's. That's expected and
fine for "does the model load and run" testing.
"""
import sys
import math
import numpy as np
import pandas as pd
import requests
import joblib

BUNDLE_PATH = sys.argv[1] if len(sys.argv) > 1 else "inference_bundle.joblib"

print(f"Loading bundle from: {BUNDLE_PATH}")
bundle = joblib.load(BUNDLE_PATH)
print("Loaded OK. Bundle contains:", list(bundle.keys()))


# ---- geometry helpers ----
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dlmb) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def traffic_bucket(h):
    if h < 6: return "night"
    elif h < 8: return "morning_offpeak"
    elif h < 10: return "morning_rush"
    elif h < 17: return "midday"
    elif h < 21: return "evening_rush"
    else: return "evening_winddown"


FIXED_HOLIDAYS_MD = {(2, 21), (3, 26), (4, 14), (5, 1), (8, 15), (12, 16), (12, 25)}


def grid_cell_from_bundle(lat, lon, b):
    gp = b["grid_params"]
    r = min(int((lat - gp["min_lat"]) / gp["lat_step"]), gp["n_rows"] - 1)
    c = min(int((lon - gp["min_lon"]) / gp["lon_step"]), gp["n_cols"] - 1)
    return r * gp["n_cols"] + c


def get_route(src_lat, src_lon, dst_lat, dst_lon, b):
    """Real OSRM route if the server is reachable, else a rough straight-line fallback."""
    url = f"{b['osrm_base']}/route/v1/driving/{src_lon},{src_lat};{dst_lon},{dst_lat}?overview=false"
    try:
        route = requests.get(url, timeout=3).json()["routes"][0]
        return route["distance"] / 1000.0, route["duration"], True
    except Exception:
        haversine = haversine_km(src_lat, src_lon, dst_lat, dst_lon)
        approx_distance_km = haversine * 1.35       # typical Dhaka road detour factor
        approx_duration_sec = approx_distance_km / 20.0 * 3600.0  # assume ~20 km/h avg
        return approx_distance_km, approx_duration_sec, False


def build_feature_row(src_lat, src_lon, dst_lat, dst_lon, query_time, b):
    dt = pd.to_datetime(query_time)

    haversine_distance_km = haversine_km(src_lat, src_lon, dst_lat, dst_lon)
    bearing_degrees = bearing_deg(src_lat, src_lon, dst_lat, dst_lon)

    osrm_route_distance_km, osrm_free_flow_duration_sec, used_real_osrm = get_route(
        src_lat, src_lon, dst_lat, dst_lon, b
    )
    route_directness_ratio = (
        haversine_distance_km / osrm_route_distance_km if osrm_route_distance_km > 0 else np.nan
    )

    src_zone_id = grid_cell_from_bundle(src_lat, src_lon, b)
    dst_zone_id = grid_cell_from_bundle(dst_lat, dst_lon, b)
    od_zone_pair_id = f"{src_zone_id}_{dst_zone_id}"

    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    day_of_week = dt.day_name()
    is_friday, is_saturday = day_of_week == "Friday", day_of_week == "Saturday"
    is_weekend = is_friday or is_saturday
    rush_hour_flag = (8 <= hour < 10) or (17 <= hour < 21)
    bucket = traffic_bucket(hour)
    is_holiday = (dt.month, dt.day) in FIXED_HOLIDAYS_MD

    lvl1, lvl2, lvl3 = b["lvl1"], b["lvl2"], b["lvl3"]
    key1 = (od_zone_pair_id, bucket)
    od_pair_seen = od_zone_pair_id in lvl2.index
    if key1 in lvl1.index and lvl1.loc[key1, "n_trips"] >= b["min_support"]:
        historical_avg_speed_kmh, level = lvl1.loc[key1, "speed"], "od_pair+period"
    elif od_pair_seen and lvl2.loc[od_zone_pair_id, "n_trips"] >= b["min_support"]:
        supported = bucket in lvl3.index and lvl3.loc[bucket, "n_trips"] >= b["min_support"]
        factor = lvl3.loc[bucket, "speed"] / b["global_speed"] if supported else 1.0
        historical_avg_speed_kmh, level = lvl2.loc[od_zone_pair_id, "speed"] * factor, "od_pair_only"
    elif bucket in lvl3.index and lvl3.loc[bucket, "n_trips"] >= b["min_support"]:
        historical_avg_speed_kmh, level = lvl3.loc[bucket, "speed"], "period_only"
    else:
        historical_avg_speed_kmh, level = b["global_speed"], "global_fallback"

    row = {
        "src_zone_id": src_zone_id, "dst_zone_id": dst_zone_id, "od_zone_pair_id": od_zone_pair_id,
        "day_of_week": day_of_week, "traffic_period_bucket": bucket,
        "haversine_distance_km": haversine_distance_km, "bearing_degrees": bearing_degrees,
        "osrm_route_distance_km": osrm_route_distance_km,
        "osrm_free_flow_duration_sec": osrm_free_flow_duration_sec,
        "route_directness_ratio": route_directness_ratio,
        "hour_sin": np.sin(2 * np.pi * hour / 24.0), "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "is_friday": int(is_friday), "is_saturday": int(is_saturday), "is_weekend": int(is_weekend),
        "is_holiday": int(is_holiday), "rush_hour_flag": int(rush_hour_flag),
        "historical_avg_speed_kmh": historical_avg_speed_kmh,
    }
    return row, level, od_pair_seen, used_real_osrm


def predict_travel_time(src_lat, src_lon, dst_lat, dst_lon, query_time, b):
    """Returns predicted duration in MINUTES for both models."""
    row, level, od_pair_seen, used_real_osrm = build_feature_row(
        src_lat, src_lon, dst_lat, dst_lon, query_time, b
    )

    xrow = pd.DataFrame([row])
    for c in b["categorical_cols"]:
        xrow[c] = pd.Categorical([str(row[c])], categories=b["cat_categories"][c])
    for c in ["is_friday", "is_saturday", "is_weekend", "is_holiday", "rush_hour_flag"]:
        xrow[c] = xrow[c].astype(int)
    xrow = xrow[b["feature_cols"]]
    xgb_pred_sec = float(np.exp(b["xgb_model"].predict(xrow)[0]))

    orow = pd.get_dummies(pd.DataFrame([row]), columns=b["categorical_cols"])
    orow = orow.reindex(columns=b["onehot_columns"], fill_value=0)
    rf_pred_sec = float(np.exp(b["rf_model"].predict(orow)[0]))

    return (
        {"xgboost": xgb_pred_sec / 60.0, "random_forest": rf_pred_sec / 60.0},
        level, od_pair_seen, used_real_osrm,
    )


if __name__ == "__main__":
    # Example: Zigatola -> Dhanmondi, a Tuesday morning
    SRC_LAT, SRC_LON = 23.7388773, 90.3756685
    DST_LAT, DST_LON = 23.7564776, 90.3622345
    QUERY_TIME = "2026-09-16T09:30:00"

    print(f"\nOSRM server configured at: {bundle['osrm_base']}")
    preds, level, seen, used_real_osrm = predict_travel_time(
        SRC_LAT, SRC_LON, DST_LAT, DST_LON, QUERY_TIME, bundle
    )

    if not used_real_osrm:
        print("!! OSRM server not reachable -- used a rough straight-line fallback for road "
              "distance/duration. Predictions below are a basic sanity check only, not the "
              "real pipeline's accuracy.")

    print(f"\nQuery: ({SRC_LAT}, {SRC_LON}) -> ({DST_LAT}, {DST_LON}) at {QUERY_TIME}")
    print(f"OD zone pair: {level} | seen in training: {seen}")
    print(f"Predicted travel time -- XGBoost: {preds['xgboost']:.1f} min, "
          f"Random Forest: {preds['random_forest']:.1f} min")
    print("\nIf you see numbers printed above with no errors, the model loaded and ran "
          "successfully on this machine.")
