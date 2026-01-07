import io
import math
import datetime as dt
import pandas as pd
import numpy as np

import streamlit as st
import plotly.express as px
import requests   # 👈 ADD THIS


# ---------------------
# Caching wrappers
# ---------------------

@st.cache_data(show_spinner=False)
def load_csv_cached(file) -> pd.DataFrame:
    """
    Robust CSV loader: read columns as text first (prevents pandas from
    guessing wrong types); we normalize/convert in parse_activities().
    """
    # read as strings to keep commas/locale intact; low_memory avoids dtype flips
    df = pd.read_csv(
        file,
        dtype=str,
        low_memory=False,
        encoding="utf-8"
    )
    return df

def _robust_parse_datetime(val: str) -> pd.Timestamp:
    """Try multiple datetime formats, then fall back to dayfirst=True."""
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()

    # fast path
    try:
        return pd.to_datetime(s, errors="raise", infer_datetime_format=True, utc=False)
    except Exception:
        pass

    # explicit formats seen in Strava exports / Excel re-saves
    for fmt in (
        "%b %d, %Y, %I:%M:%S %p",   # Sep 24, 2017, 12:31:42 AM
        "%b %d, %Y %I:%M:%S %p",    # Sep 24, 2017 12:31:42 AM
        "%Y-%m-%d %H:%M:%S",        # 2017-09-24 00:31:42
        "%d-%m-%Y %H:%M",           # 27-10-2002 23:00  (day-first)
    ):
        try:
            return pd.to_datetime(s, format=fmt, errors="raise", utc=False)
        except Exception:
            continue

    # last resort: generic parse with dayfirst
    return pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    """
    Make numbers reliable: drop thin spaces/commas, then to_numeric.
    Handles values like '27,981.3' or '6 780.5' (unicode thin space).
    """
    return pd.to_numeric(
        series.astype(str)
              .str.replace("\u2009", "", regex=False)   # thin space
              .str.replace(",", "", regex=False)
              .str.strip(),
        errors="coerce"
    )


@st.cache_data(show_spinner=False)
def parse_activities_cached(df: pd.DataFrame, v: int = 3) -> pd.DataFrame:
    return parse_activities(df)


# ---------------------
# Strava API helpers
# ---------------------

@st.cache_data(show_spinner=False, ttl=900)
def get_strava_access_token() -> str:
    """
    Use long-lived refresh token (in .streamlit/secrets.toml)
    to get a short-lived access token.
    """
    cfg = st.secrets["strava"]
    resp = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "grant_type": "refresh_token",
            "refresh_token": cfg["refresh_token"],
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


@st.cache_data(show_spinner=True)
def fetch_strava_activities_json(
    before: int | None = None,
    after: int | None = None,
    max_pages: int = 10,
    per_page: int = 200,
) -> list[dict]:
    """
    Call /athlete/activities and collect pages of activities.
    """
    access_token = get_strava_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}

    all_acts: list[dict] = []
    page = 1
    while page <= max_pages:
        params = {"page": page, "per_page": per_page}
        if before is not None:
            params["before"] = before
        if after is not None:
            params["after"] = after

        r = requests.get(
            "https://www.strava.com/api/v3/athlete/activities",
            headers=headers,
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        acts = r.json()
        if not acts:
            break

        all_acts.extend(acts)
        if len(acts) < per_page:
            break
        page += 1

    return all_acts


def strava_json_to_dataframe(activities: list[dict]) -> pd.DataFrame:
    """
    Convert Strava API JSON to the same schema as your CSV-based pipeline.
    Units: distance in km, time in seconds, speed in km/h.
    """
    if not activities:
        return pd.DataFrame(columns=[
            "Activity ID","Activity Date","Activity Name","Activity Type",
            "Distance","Moving Time","Elapsed Time",
            "Average Speed","Max Speed","Elevation Gain",
            "Calories","Commute","Carbon Saved"
        ])

    rows = []
    for a in activities:
        aid   = a.get("id")
        name  = a.get("name")
        a_type = a.get("sport_type") or a.get("type")

        dist_km = (a.get("distance") or 0.0) / 1000.0
        mov_s   = a.get("moving_time") or 0
        elap_s  = a.get("elapsed_time") or 0

        avg_kmh = (a.get("average_speed") or 0.0) * 3.6
        max_kmh = (a.get("max_speed") or 0.0) * 3.6

        elev_m  = a.get("total_elevation_gain") or 0.0

        calories = a.get("calories")
        if calories is None and a.get("kilojoules") is not None:
            calories = a["kilojoules"] / 4.184  # rough kJ→kcal

        commute = bool(a.get("commute", False))
        start_date = a.get("start_date_local") or a.get("start_date")

        carbon_saved = dist_km * 0.21 if commute else 0.0

        rows.append({
            "Activity ID": aid,
            "Activity Date": start_date,
            "Activity Name": name,
            "Activity Type": a_type,
            "Distance": dist_km,
            "Moving Time": mov_s,
            "Elapsed Time": elap_s,
            "Average Speed": avg_kmh,
            "Max Speed": max_kmh,
            "Elevation Gain": elev_m,
            "Calories": calories,
            "Commute": commute,
            "Carbon Saved": carbon_saved,
        })

    return pd.DataFrame(rows)


# ---------------------
# App Config
# ---------------------
st.set_page_config(page_title="Strava Insights – Multi‑Sport", layout="wide")

# --- light CSS polish: smaller labels, tighter metric spacing ---
st.markdown("""
<style>
/* compact metric labels */
[data-testid="stMetricLabel"] > div {
  font-size: 0.83rem;
  color: #6b7280; /* gray-500 */
  margin-bottom: -6px;
}
/* pull metric numbers up a bit */
[data-testid="stMetricValue"] {
  margin-top: -10px !important;
}

/* optional: slightly reduce top padding under the page title */
section.main > div:first-child { padding-top: 0.4rem; }

/* thinner section divider */
hr {
  margin-top: 0.6rem !important;
  margin-bottom: 0.6rem !important;
}
</style>
""", unsafe_allow_html=True)



st.title("Prasad's multi-sports (🚴‍♂️,🏃,🚶,🏊) stats insights dashboard from Strava")
#st.caption("Start with Cycling KPIs, easily extendable to Running, Walking, Swimming, Yoga, and more.")

# ---------------------
# Helper functions
# ---------------------
SPORT_ALIASES = {
    "Cycling": [
        "ride", "virtual ride", "gravel ride", "mountain bike ride",
        "e-bike ride", "velomobile", "handcycle", "indoor cycling"
    ],
    "Running": [
        "run", "trail run"
    ],
    "Walking": [
        "walk", "hike", "trek"
    ],
    "Swimming": [
        "swim"
    ],
    "Workout": [
        "workout", "yoga", "weight training", "crossfit"
    ],
    "Other": []
}


def canonical_sport(activity_type: str) -> str:
    """Map Strava 'Activity Type' into canonical sport buckets."""
    if not isinstance(activity_type, str):
        return "Other"
    t = activity_type.strip().lower()
    for sport, aliases in SPORT_ALIASES.items():
        for alias in aliases:
            if alias in t:
                return sport
    # Fallback guesses
    if "bike" in t:
        return "Cycling"
    if "run" in t:
        return "Running"
    if "walk" in t or "hike" in t or "trek" in t:
        return "Walking"
    if "swim" in t:
        return "Swimming"
    if "yoga" in t or "workout" in t or "training" in t:
        return "Workout"
    return "Other"

########
def parse_activities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- Ensure columns exist (create empty when missing) ----
    expected = [
        "Activity ID","Activity Date","Activity Name","Activity Type",
        "Distance","Moving Time","Elapsed Time",
        "Average Speed","Max Speed","Elevation Gain","Calories","Commute","Carbon Saved"
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    # ---- Robust datetime parsing for “Activity Date” ----
    # (uses your helper _robust_parse_datetime)
    df["Activity Date"] = df["Activity Date"].apply(_robust_parse_datetime)

    # ---- Numeric coercion for key metrics ----
    for c in ["Distance", "Average Speed", "Max Speed", "Elevation Gain", "Calories", "Carbon Saved"]:
        df[c] = _to_numeric_clean(df[c])

    # Times are in seconds in the export; coerce to numeric
    for c in ["Moving Time", "Elapsed Time"]:
        df[c] = _to_numeric_clean(df[c])

    # ---- Safety net: back-fill Distance from a duplicate column (often meters) ----
    alt_distance_col = None
    for c in df.columns:
        if c.lower().startswith("distance") and c != "Distance":
            alt_distance_col = c
            break
    if alt_distance_col is not None:
        alt_num = _to_numeric_clean(df[alt_distance_col])
        # If the other distance looks like meters, convert to km
        if alt_num.dropna().median() > 1000:
            alt_num = alt_num / 1000.0
        df["Distance"] = df["Distance"].fillna(alt_num)

    # ---- Normalize Commute to True/False ----
    if df["Commute"].dtype != bool:
        df["Commute"] = (
            df["Commute"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        )

    # ---- Canonical sport mapping ----
    df["Sport"] = df["Activity Type"].astype(str).apply(canonical_sport)

    # ---- Derived periods / helpers (now safe to use .dt) ----
    df["Year"]         = df["Activity Date"].dt.year
    df["Month"]        = df["Activity Date"].dt.to_period("M").astype(str)
    df["MonthPeriod"]  = df["Activity Date"].dt.to_period("M")
    df["Week"]         = df["Activity Date"].dt.to_period("W").astype(str)

    return df


#############


def kpi_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)

def cycling_kpis_layout(df: pd.DataFrame):
    """Structured 5-column KPI layout for Cycling tab (matches desired layout)."""
    if df.empty:
        st.info("No cycling activities found.")
        return

    # Basic aggregates
    num_rides         = len(df)
    total_distance    = pd.to_numeric(df["Distance"], errors="coerce").sum()
    avg_distance      = pd.to_numeric(df["Distance"], errors="coerce").mean()
    total_elev        = pd.to_numeric(df["Elevation Gain"], errors="coerce").sum()
    total_time_hours  = pd.to_numeric(df["Moving Time"], errors="coerce").sum() / 3600.0

    rides_50k         = (pd.to_numeric(df["Distance"], errors="coerce") >= 50).sum()
    rides_100k        = (pd.to_numeric(df["Distance"], errors="coerce") >= 100).sum()

    # Avg rides per calendar month across the selected span (inclusive)
    start_m = df["Activity Date"].min().to_period("M")
    end_m   = df["Activity Date"].max().to_period("M")
    months_span = int((end_m - start_m).n) + 1
    avg_rides_month = (num_rides / months_span) if months_span > 0 else 0.0

    longest_ride_km   = pd.to_numeric(df["Distance"], errors="coerce").max()

    num_commutes      = df["Commute"].sum()
    carbon_saved_kg   = pd.to_numeric(df.get("Carbon Saved", 0), errors="coerce").sum()
    avg_commute_km    = pd.to_numeric(df.loc[df["Commute"], "Distance"], errors="coerce").mean()

    # ---- Row 1
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1], gap="large")
    with c1: st.metric("No. of Rides", f"{num_rides:,}")
    with c2: st.metric("Total Distance (km)", f"{total_distance:,.2f}")
    with c3: st.metric("Avg Distance/Ride (km)", f"{avg_distance:,.2f}")
    with c4: st.metric("Total Elevation (m)", f"{total_elev:,.0f}")
    with c5: st.metric("Total Time (hrs)", f"{total_time_hours:,.1f}")

    # ---- Row 2
    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([1,1,1,1,1], gap="large")
    with r2c1: st.metric("50K+ Rides", f"{rides_50k:,}")
    with r2c2: st.metric("100K+ Rides", f"{rides_100k:,}")
    with r2c3: st.metric("Avg Rides/Month", f"{avg_rides_month:,.1f}")
    with r2c4: st.metric("Longest Ride (km)", f"{longest_ride_km:,.2f}")
    with r2c5: st.write("")  # keep this empty per your mock


    st.write("")   # small spacer
    
    # ===== Commute KPIs (full set) =====
    st.markdown("### Exclusive commute insights")

    comm = df[df["Commute"] == True].copy()
    if comm.empty:
        st.info("No commute rides in the current filters.")
    else:
        # numeric series with coercion
        dist_c = pd.to_numeric(comm["Distance"], errors="coerce")
        time_c = pd.to_numeric(comm["Moving Time"], errors="coerce")          # seconds
        elev_c = pd.to_numeric(comm["Elevation Gain"], errors="coerce")

        # counts & calendar span (same definition used for avg rides/month)
        start_m = df["Activity Date"].min().to_period("M")
        end_m   = df["Activity Date"].max().to_period("M")
        months_span = int((end_m - start_m).n) + 1 if pd.notna(start_m) and pd.notna(end_m) else 0

        num_commutes                 = int(comm.shape[0])
        total_commute_km             = float(dist_c.sum())
        avg_dist_per_commute_km      = float(dist_c.mean())
        avg_commutes_per_month       = (num_commutes / months_span) if months_span > 0 else 0.0
        avg_commute_km_per_month     = (total_commute_km / months_span) if months_span > 0 else 0.0
        longest_commute_km           = float(dist_c.max())
        total_commute_hours          = float(time_c.sum()) / 3600.0
        total_elev_commute_m         = float(elev_c.sum())
        avg_commute_speed_kmh        = (total_commute_km / (time_c.sum() / 3600.0)) if time_c.sum() > 0 else float("nan")  # distance–time weighted

        # Row 1
        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1], gap="large")
        with c1: st.metric("No. of Commutes", f"{num_commutes:,}")
        with c2: st.metric("Total Commute Distance (km)", f"{total_commute_km:,.2f}")
        with c3: st.metric("Avg Distance / Commute (km)", f"{avg_dist_per_commute_km:,.2f}")
        with c4: st.metric("Avg Commutes / Month", f"{avg_commutes_per_month:,.1f}")
        with c5: st.metric("Avg Commute Distance / Month (km)", f"{avg_commute_km_per_month:,.1f}")

        # Row 2
        r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([1,1,1,1,1], gap="large")
        with r2c1: st.metric("Avg Speed of a Commute (km/h)", f"{avg_commute_speed_kmh:,.2f}" if not np.isnan(avg_commute_speed_kmh) else "—")
        with r2c2: st.metric("Longest Commute Distance (km)", f"{longest_commute_km:,.2f}")
        with r2c3: st.metric("Total Saddle Time in Commutes (hrs)", f"{total_commute_hours:,.1f}")
        with r2c4: st.metric("Total Elevation in Commutes (m)", f"{total_elev_commute_m:,.0f}")
        with r2c5: st.metric("Total Carbon Saved (kg)", f"{carbon_saved_kg:,.2f}")
        # r2c5 left free (or reuse for commute-only carbon if ever available)



    st.write("")  # tiny spacer
    st.divider()
###################################################
# ---------- Running helpers ----------
def _pace_min_per_km(total_seconds: float, total_km: float) -> float:
    """Distance-weighted pace (min/km)."""
    if not total_km or np.isnan(total_km) or total_km <= 0:
        return np.nan
    return (total_seconds / 60.0) / total_km

def _fmt_pace(min_per_km: float) -> str:
    if np.isnan(min_per_km):
        return "—"
    mins = int(min_per_km)
    secs = int(round((min_per_km - mins) * 60))
    if secs == 60:
        mins += 1; secs = 0
    return f"{mins}:{secs:02d}"

def running_bucket_stats(df: pd.DataFrame,
                         min_km: float,
                         max_km: float,
                         include_max: bool = False) -> dict:
    """
    Bucket stats between [min_km, max_km) by default.
    If include_max=True, use [min_km, max_km].
    Returns: {count, avg_pace (min/km), best_time (sec)}.
    """
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    time_s = pd.to_numeric(df["Moving Time"], errors="coerce")

    if include_max:
        mask = (dist >= min_km) & (dist <= max_km)
    else:
        mask = (dist >= min_km) & (dist < max_km)

    d_sel = dist[mask]
    t_sel = time_s[mask]

    if d_sel.empty or d_sel.sum() <= 0:
        return {"count": 0, "avg_pace": np.nan, "best_time": np.nan}

    avg_pace = _pace_min_per_km(t_sel.sum(), d_sel.sum())
    best_time = float(t_sel.min())

    return {
        "count": int(mask.sum()),
        "avg_pace": avg_pace,
        "best_time": best_time,
    }




def _fmt_time_hms(seconds: float) -> str:
    if np.isnan(seconds) or seconds <= 0:
        return "—"
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def running_kpis_layout(df: pd.DataFrame):
    """Quick view + bucket KPIs + commute insights for Running."""
    if df.empty:
        st.info("No Running activities found.")
        return

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    time_s = pd.to_numeric(df["Moving Time"], errors="coerce")
    elev = pd.to_numeric(df["Elevation Gain"], errors="coerce")

    total_runs   = int(df.shape[0])
    total_km     = float(dist.sum())
    total_time_h = float(time_s.sum()) / 3600.0
    total_elev   = float(elev.sum())
    longest_km   = float(dist.max())
    avg_dist_run = total_km / total_runs if total_runs > 0 else 0.0
    avg_pace_all = _pace_min_per_km(time_s.sum(), dist.sum())

    # ----- Distance buckets -----
    buckets = {
        "<5K":  running_bucket_stats(df, 0.0, 5.0),                # [0, 5)
        "5K":   running_bucket_stats(df, 5.0, 10.0),               # [5, 10)
        "10K":  running_bucket_stats(df, 10.0, 15.0),              # [10, 15)
        "15K":  running_bucket_stats(df, 15.0, 20.0),              # [15, 20)
        "HM":   running_bucket_stats(df, 20.0, 23.0, include_max=True),  # [20, 23]
    }

    # Quick view – Row 1: overall stats
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1], gap="large")
    with c1: st.metric("Runs", f"{total_runs:,}")
    with c2: st.metric("Total Distance (km)", f"{total_km:,.2f}")
    with c3: st.metric("Avg Distance / Run (km)", f"{avg_dist_run:,.2f}")
    with c4: st.metric("Avg Pace (min/km)", _fmt_pace(avg_pace_all))
    with c5: st.metric("Total Elevation (m)", f"{total_elev:,.0f}")

    # Quick view – Row 2: bucket counts (moved here)
    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([1, 1, 1, 1, 1], gap="large")
    with r2c1: st.metric("<5K – runs", f"{buckets['<5K']['count']:,}")
    with r2c2: st.metric("5K – runs", f"{buckets['5K']['count']:,}")
    with r2c3: st.metric("10K – runs", f"{buckets['10K']['count']:,}")
    with r2c4: st.metric("15K – runs", f"{buckets['15K']['count']:,}")
    with r2c5: st.metric("Half Marathon (21.1K) – runs", f"{buckets['HM']['count']:,}")

    # Quick view – Row 3: <5K avg pace + longest run
    r3c1, r3c2, r3c3, r3c4, r3c5 = st.columns([1, 1, 1, 1, 1], gap="large")
    with r3c1: st.metric("<5K – Avg Pace", _fmt_pace(buckets["<5K"]["avg_pace"]))
    with r3c2: st.metric("Longest Run (km)", f"{longest_km:,.2f}")
    # r3c3–r3c5 left free for future metrics

    st.write("")
    st.markdown("### Overall Running insights")

    # Overall section now ONLY shows Avg Pace & Best Time (no counts)
    cols = st.columns(4, gap="large")
    labels = [("5K", "5K"), ("10K", "10K"), ("15K", "15K"), ("HM", "Half Marathon (21.1K)")]

    for (key, label), col in zip(labels, cols):
        stats = buckets[key]
        with col:
            st.metric(f"{label} – Avg Pace", _fmt_pace(stats["avg_pace"]))
            st.metric(f"{label} – Best Time", _fmt_time_hms(stats["best_time"]))

    st.write("")
    st.markdown("### Exclusive running commute insights")

    # -------- Commute section unchanged --------
    comm = df[df["Commute"] == True].copy()
    if comm.empty:
        st.info("No commute runs in the current filters.")
    else:
        cd = pd.to_numeric(comm["Distance"], errors="coerce")
        ct = pd.to_numeric(comm["Moving Time"], errors="coerce")
        ce = pd.to_numeric(comm["Elevation Gain"], errors="coerce")

        start_m = df["Activity Date"].min().to_period("M")
        end_m   = df["Activity Date"].max().to_period("M")
        months_span = int((end_m - start_m).n) + 1 if pd.notna(start_m) and pd.notna(end_m) else 0

        num_comm          = int(comm.shape[0])
        total_comm_km     = float(cd.sum())
        avg_dist_comm     = float(cd.mean())
        avg_comm_per_month = (num_comm / months_span) if months_span > 0 else 0.0
        avg_pace_comm     = _pace_min_per_km(ct.sum(), cd.sum())
        longest_comm_km   = float(cd.max())
        total_comm_hours  = float(ct.sum()) / 3600.0
        total_comm_elev   = float(ce.sum())

        r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns([1,1,1,1,1], gap="large")
        with r1c1: st.metric("No. of Commute Runs", f"{num_comm:,}")
        with r1c2: st.metric("Total Commute Distance (km)", f"{total_comm_km:,.2f}")
        with r1c3: st.metric("Longest Commute Run (km)", f"{longest_comm_km:,.2f}")
        with r1c4: st.metric("Avg Pace of Commute Run (min/km)", _fmt_pace(avg_pace_comm))
        with r1c5: st.metric("Avg Commute Runs / Month", f"{avg_comm_per_month:,.1f}")

        r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([1,1,1,1,1], gap="large")
        with r2c1: st.metric("Avg Distance / Commute Run (km)", f"{avg_dist_comm:,.2f}")
        with r2c2: st.metric("Total Commute Time (hrs)", f"{total_comm_hours:,.1f}")
        with r2c3: st.metric("Total Elevation in Commutes (m)", f"{total_comm_elev:,.0f}")
        with r2c4: st.write("")
        with r2c5: st.write("")

    st.write("")
    st.divider()



###################################################
# ---------- Walking helpers ----------
###################################################
# ---------- Walking helpers ----------
def walking_kpis_layout(df: pd.DataFrame):
    """Quick view + bucket KPIs + commute insights for Walking."""
    if df.empty:
        st.info("No Walking activities found.")
        return

    dist   = pd.to_numeric(df["Distance"], errors="coerce")
    time_s = pd.to_numeric(df["Moving Time"], errors="coerce")
    elev   = pd.to_numeric(df["Elevation Gain"], errors="coerce")

    total_walks   = int(df.shape[0])
    total_km      = float(dist.sum())
    total_time_h  = float(time_s.sum()) / 3600.0
    total_elev    = float(elev.sum())
    longest_km    = float(dist.max())
    avg_pace_all  = _pace_min_per_km(time_s.sum(), dist.sum())

    # ----- Distance buckets -----
    buckets = {
        "5K":  running_bucket_stats(df, 5.0, 10.0),        # [5, 10)
        "10K": running_bucket_stats(df, 10.0, 15.0),       # [10, 15)
        "15K": running_bucket_stats(df, 15.0, 20.0),       # [15, 20)
    }

    # ===== Walking – Quick View =====
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1], gap="large")
    with c1: st.metric("Walks", f"{total_walks:,}")
    with c2: st.metric("Total Distance (km)", f"{total_km:,.2f}")
    with c3: st.metric("Avg Pace (min/km)", _fmt_pace(avg_pace_all))
    with c4: st.metric("Total Elevation (m)", f"{total_elev:,.0f}")
    with c5: st.metric("Longest Walk (km)", f"{longest_km:,.2f}")

    # Quick view – bucket counts
    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([1, 1, 1, 1, 1], gap="large")
    with r2c1: st.metric("5K – walks",  f"{buckets['5K']['count']:,}")
    with r2c2: st.metric("10K – walks", f"{buckets['10K']['count']:,}")
    with r2c3: st.metric("15K – walks", f"{buckets['15K']['count']:,}")
    with r2c4: st.write("")
    with r2c5: st.write("")

    st.write("")
    st.markdown("### Overall Walking insights")

    # Overall section: Avg Pace & Best Time for each bucket
    cols = st.columns(3, gap="large")
    for (key, label), col in zip(
        [("5K", "5K"), ("10K", "10K"), ("15K", "15K")],
        cols
    ):
        stats = buckets[key]
        with col:
            st.metric(f"{label} – Avg Pace", _fmt_pace(stats["avg_pace"]))
            st.metric(f"{label} – Best Time", _fmt_time_hms(stats["best_time"]))

    # ===== Exclusive walking commute insights =====
    st.write("")
    st.markdown("### Exclusive walking commute insights")

    comm = df[df["Commute"] == True].copy()
    if comm.empty:
        st.info("No commute walks in the current filters.")
    else:
        cd = pd.to_numeric(comm["Distance"], errors="coerce")
        ct = pd.to_numeric(comm["Moving Time"], errors="coerce")
        ce = pd.to_numeric(comm["Elevation Gain"], errors="coerce")

        # same month-span logic as Running/Cycling
        start_m = df["Activity Date"].min().to_period("M")
        end_m   = df["Activity Date"].max().to_period("M")
        months_span = int((end_m - start_m).n) + 1 if pd.notna(start_m) and pd.notna(end_m) else 0

        num_comm          = int(comm.shape[0])
        total_comm_km     = float(cd.sum())
        avg_dist_comm     = float(cd.mean())
        avg_comm_per_month = (num_comm / months_span) if months_span > 0 else 0.0
        avg_pace_comm     = _pace_min_per_km(ct.sum(), cd.sum())
        longest_comm_km   = float(cd.max())
        total_comm_hours  = float(ct.sum()) / 3600.0
        total_comm_elev   = float(ce.sum())

        r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns([1,1,1,1,1], gap="large")
        with r1c1: st.metric("No. of Commute Walks", f"{num_comm:,}")
        with r1c2: st.metric("Total Commute Distance (km)", f"{total_comm_km:,.2f}")
        with r1c3: st.metric("Longest Commute Walk (km)", f"{longest_comm_km:,.2f}")
        with r1c4: st.metric("Avg Pace of Commute Walk (min/km)", _fmt_pace(avg_pace_comm))
        with r1c5: st.metric("Avg Commute Walks / Month", f"{avg_comm_per_month:,.1f}")

        r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([1,1,1,1,1], gap="large")
        with r2c1: st.metric("Avg Distance / Commute Walk (km)", f"{avg_dist_comm:,.2f}")
        with r2c2: st.metric("Total Commute Time (hrs)", f"{total_comm_hours:,.1f}")
        with r2c3: st.metric("Total Elevation in Commutes (m)", f"{total_comm_elev:,.0f}")
        with r2c4: st.write("")  # spare for future metrics
        with r2c5: st.write("")

    st.write("")
    st.divider()




###################################################
def generic_time_series(df: pd.DataFrame, sport_name: str):
    """Monthly trends for count, distance, elevation with proper period sorting."""
    if df.empty:
        st.warning(f"No data to plot for {sport_name}.")
        return

    # Self-heal: add MonthPeriod if missing (e.g., old cache)
    if "MonthPeriod" not in df.columns:
        df = df.copy()
        df["MonthPeriod"] = pd.to_datetime(df["Activity Date"], errors="coerce").dt.to_period("M")


    monthly = (
        df.groupby("MonthPeriod")
          .agg(Rides=("Activity ID", "count"),
               Distance_km=("Distance", "sum"),
               Elevation_m=("Elevation Gain", "sum"))
          .reset_index()
          .sort_values("MonthPeriod")
    )
    monthly["Month"] = monthly["MonthPeriod"].astype(str)

    fig1 = px.bar(monthly, x="Month", y="Rides",
                  title=f"{sport_name} – Monthly Rides Count",
                  labels={"Month":"Month", "Rides":"Count"})
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(monthly, x="Month", y="Distance_km",
                   title=f"{sport_name} – Monthly Distance (km)",
                   labels={"Month":"Month", "Distance_km":"Distance (km)"})
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(monthly, x="Month", y="Elevation_m",
                   title=f"{sport_name} – Monthly Elevation Gain (m)",
                   labels={"Month":"Month", "Elevation_m":"Elevation (m)"})
    st.plotly_chart(fig3, use_container_width=True)



def distance_distribution(df: pd.DataFrame, sport_name: str):
    if df.empty:
        return
    fig = px.histogram(df, x="Distance", nbins=40, title=f"{sport_name} – Distance Distribution (km)",
                       labels={"Distance":"Distance (km)"})
    st.plotly_chart(fig, use_container_width=True)


# def duration_vs_distance(df: pd.DataFrame, sport_name: str):
#     if df.empty:
#         return
#     plot_df = df.assign(Moving_Hours=df["Moving Time"]/3600.0)
#     try:
#         fig = px.scatter(plot_df, x="Distance", y="Moving_Hours",
#                          title=f"{sport_name} – Distance vs Time",
#                          labels={"Distance":"Distance (km)", "Moving_Hours":"Time (hrs)"},
#                          trendline="ols")
#     except Exception:
#         fig = px.scatter(plot_df, x="Distance", y="Moving_Hours",
#                          title=f"{sport_name} – Distance vs Time",
#                          labels={"Distance":"Distance (km)", "Moving_Hours":"Time (hrs)"})
#     st.plotly_chart(fig, use_container_width=True)



def commute_split(df: pd.DataFrame, sport_name: str):
    if df.empty:
        return
    agg = df.groupby("Commute").agg(Count=("Activity ID","count"), Distance_km=("Distance","sum")).reset_index()
    agg["Type"] = np.where(agg["Commute"], "Commute", "Non-Commute")
    fig = px.bar(agg, x="Type", y="Count", title=f"{sport_name} – Commute vs Non‑Commute (Count)")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.bar(agg, x="Type", y="Distance_km", title=f"{sport_name} – Commute vs Non‑Commute (Distance km)")
    st.plotly_chart(fig2, use_container_width=True)


def download_button(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")

##################################################################################################
# ---------------------
# Sidebar – Data Source
# ---------------------
st.sidebar.header("📥 Data source")

source = st.sidebar.radio(
    "Choose source",
    ["Strava API (live)", "CSV upload", "Sample demo"],
    index=0,
)

raw = None
####################################################
# ---- 1) Live Strava API ----
if source == "Strava API (live)":
    years_back = st.sidebar.slider("Years to fetch", 1, 10, value=3)

    today = dt.date.today()
    after_date = today.replace(year=today.year - years_back)
    after_ts = int(dt.datetime.combine(after_date, dt.time.min).timestamp())

    # Fetch if:
    #  - we never fetched before, OR
    #  - user explicitly clicks refresh button
    refresh_clicked = st.sidebar.button("🔄 Fetch latest from Strava")

    if "api_activities" not in st.session_state or refresh_clicked:
        with st.spinner("Fetching activities from Strava…"):
            acts_json = fetch_strava_activities_json(after=after_ts, max_pages=20)
            st.session_state["api_activities"] = acts_json

    acts_json = st.session_state.get("api_activities", [])
    raw = strava_json_to_dataframe(acts_json)

    if raw is None or raw.empty:
        st.warning("No activities returned from Strava. Try increasing years or check API.")
        st.stop()


# ---- 3) Sample demo ----
elif source == "Sample demo":
    raw = pd.DataFrame({
        "Activity ID": [1,2,3,4],
        "Activity Date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05"),
                          pd.Timestamp("2024-02-10"), pd.Timestamp("2024-02-12")],
        "Activity Name": ["Morning Ride","Evening Run","Commute Ride","Park Walk"],
        "Activity Type": ["Ride","Run","Ride","Walk"],
        "Distance": [35.2, 5.1, 12.3, 2.4],
        "Moving Time": [5400, 1800, 2400, 1800],
        "Elevation Gain": [420, 20, 110, 5],
        "Average Speed": [23.4, 10.2, 18.2, 4.0],
        "Commute": [False, False, True, False],
        "Carbon Saved": [0.0, 0.0, 2.1, 0.0]
    })

# ✅ Use the same normalization pipeline as before
data = parse_activities_cached(raw)


###########################################
# ---------------------
# Sidebar – Filters (Year overrides Date Range)
# ---------------------
st.sidebar.header("🔎 Filters")

# 1) Sports
available_sports = sorted(data["Sport"].unique().tolist())
sport_selection = st.sidebar.multiselect("Select Sports", available_sports, default=available_sports)

# 2) Year & Date range
date_min = pd.to_datetime(data["Activity Date"].min())
date_max = pd.to_datetime(data["Activity Date"].max())

years = sorted(data["Year"].dropna().unique().astype(int))
year_selection = st.sidebar.selectbox("Year", ["All"] + [str(y) for y in years], index=0)

date_range = st.sidebar.date_input(
    "Date Range",
    [date_min.date(), date_max.date()],
    disabled=(year_selection != "All")   # disable when a Year is selected
)

# 3) Commute
commute_filter = st.sidebar.selectbox("Commute Filter",
                                      ["All", "Commute Only", "Non-Commute Only"],
                                      index=0)


#########################################################
# ---------------------
# Apply filters in logical order
# ---------------------
df = data[data["Sport"].isin(sport_selection)].copy()

# Year overrides Date Range – use direct year match (safer)
if year_selection != "All":
    yr = int(year_selection)
    # robust: handles tz/format quirks and excludes NaT rows automatically
    df = df[pd.to_datetime(df["Activity Date"], errors="coerce").dt.year == yr]
else:
    if isinstance(date_range, list) and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date   = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        df = df[(df["Activity Date"] >= start_date) & (df["Activity Date"] < end_date)]

# Commute last
if commute_filter == "Commute Only":
    df = df[df["Commute"] == True]
elif commute_filter == "Non-Commute Only":
    df = df[df["Commute"] == False]

with st.expander("🔍 Debug: data sanity (toggle)"):
    # 2.1 how many rows would match the selected year directly on the raw data?
    if year_selection != "All":
        yr = int(year_selection)
        raw_year_mask = pd.to_datetime(data["Activity Date"], errors="coerce").dt.year == yr
        st.write(f"Raw rows in {yr} (all sports, before sidebar filters):",
                 int(raw_year_mask.sum()))

    # 2.2 did any dates fail to parse?
    nat_count = pd.to_datetime(data["Activity Date"], errors="coerce").isna().sum()
    st.write("Rows with unparseable 'Activity Date':", int(nat_count))

    # 2.3 which 'Activity Type' labels really exist for the period in view?
    in_view = df.copy()
    st.write("Top 'Activity Type' labels in current view:")
    st.dataframe(
        in_view["Activity Type"].str.lower().value_counts().head(20),
        use_container_width=True
    )

#########################################################


# ---- Effective date range for header display ----
if year_selection != "All":
    hdr_start = pd.Timestamp(f"{year_selection}-01-01").date()
    hdr_end   = pd.Timestamp(f"{year_selection}-12-31").date()
else:
    if isinstance(date_range, list) and len(date_range) == 2:
        hdr_start = pd.to_datetime(date_range[0]).date()
        hdr_end   = pd.to_datetime(date_range[1]).date()
    else:
        # Fallback to the filtered data span (rare)
        hdr_start = pd.to_datetime(df["Activity Date"]).min().date() if len(df) else None
        hdr_end   = pd.to_datetime(df["Activity Date"]).max().date() if len(df) else None


###########################################

st.success(f"Loaded **{len(df):,}** activities from **{len(data):,}** total after filters.")



with st.container():
    st.markdown("##### Current Selection")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.write(f"**Sports:** {', '.join(sport_selection) if sport_selection else '—'}")
    with c2: st.write(f"**Year:** {year_selection}")
    with c3: st.write(f"**Date:** {hdr_start} → {hdr_end}" if hdr_start and hdr_end else "**Date:** —")
    with c4: st.write(f"**Commute:** {commute_filter}")


# ---------------------
# Tabs
# ---------------------
tab_all, tab_cyc, tab_run, tab_walk, tab_swim, tab_other = st.tabs(
    ["📊 All Sports", "🚴 Cycling", "🏃 Running", "🚶 Walking", "🏊 Swimming", "➕ Yoga/Strength training"]
)

with tab_all:
    st.subheader("All Sports – Overview")
    # Generic monthly charts across all selected sports
    generic_time_series(df, "All Sports")
    distance_distribution(df, "All Sports")

    # Export filtered data
    st.divider()
    st.caption("Export the currently filtered dataset:")
    download_button(df, "⬇️ Download CSV", "strava_filtered.csv")

with tab_cyc:
    st.subheader("Overall cycling insights")
    cyc = df[df["Sport"] == "Cycling"].copy()
    cycling_kpis_layout(cyc)
    #st.caption("Using new layout ✅ cycling_kpis_layout()")


    st.markdown("#### Cycling – Trends & Distributions")
    generic_time_series(cyc, "Cycling")
    distance_distribution(cyc, "Cycling")
    #duration_vs_distance(cyc, "Cycling")
    commute_split(cyc, "Cycling")
    st.markdown("#### Cycling – Monthly KPIs")
    if not cyc.empty:
        cyc_monthly = (
            cyc.groupby("MonthPeriod", as_index=False)
               .agg(Rides=("Activity ID", "count"),
                    Distance_km=("Distance", "sum"),
                    Elevation_m=("Elevation Gain", "sum"),
                    Time_hours=("Moving Time", lambda s: s.sum() / 3600.0))
               .sort_values("MonthPeriod")
        )
        cyc_monthly["Month"] = cyc_monthly["MonthPeriod"].astype(str)
        cyc_monthly = cyc_monthly[["Month","Rides","Distance_km","Elevation_m","Time_hours"]]

        totals = pd.DataFrame([{
            "Month": "TOTAL",
            "Rides": int(cyc_monthly["Rides"].sum()),
            "Distance_km": round(cyc_monthly["Distance_km"].sum(), 2),
            "Elevation_m": round(cyc_monthly["Elevation_m"].sum(), 0),
            "Time_hours": round(cyc_monthly["Time_hours"].sum(), 1),
        }])

        st.dataframe(pd.concat([cyc_monthly, totals], ignore_index=True), use_container_width=True)
        download_button(cyc_monthly, "⬇️ Download cycling-monthly.csv", "cycling_monthly.csv")
    else:
        st.info("No cycling data in current filters.")

    

with tab_run:
    st.subheader("Running – Quick View")
    run = df[df["Sport"] == "Running"].copy()

    if run.empty:
        st.info("No Running activities in current filters.")
    else:
        # KPIs + distance buckets + commute insights
        running_kpis_layout(run)

        # Trends/Distributions (kept same as Cycling)
        st.markdown("#### Running – Trends & Distributions")
        generic_time_series(run, "Running")
        distance_distribution(run, "Running")


with tab_walk:
    st.subheader("Walking – Quick View")
    walk = df[df["Sport"] == "Walking"].copy()
    if walk.empty:
        st.info("No Walking activities in current filters.")
    else:
        walking_kpis_layout(walk)

        # Trends/Distributions (same as other tabs)
        st.markdown("#### Walking – Trends & Distributions")
        generic_time_series(walk, "Walking")
        distance_distribution(walk, "Walking")


with tab_swim:
    st.subheader("Swimming – Quick View")
    swim = df[df["Sport"] == "Swimming"].copy()
    if swim.empty:
        st.info("No Swimming activities in current filters.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("Swims", f"{len(swim):,}")
        with c2: kpi_card("Total Distance (km)", f"{swim['Distance'].sum():,.2f}")
        with c3: kpi_card("Total Time (hrs)", f"{(swim['Moving Time'].sum()/3600):,.1f}")
        generic_time_series(swim, "Swimming")
        distance_distribution(swim, "Swimming")

with tab_other:
    st.subheader("Yoga/Strength training – Quick View")
    oth = df[df["Sport"].isin(["Workout","Other"])].copy()
    if oth.empty:
        st.info("No Other/Workout activities in current filters.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("Sessions", f"{len(oth):,}")
        with c2: kpi_card("Total Time (hrs)", f"{(oth['Moving Time'].sum()/3600):,.1f}")
        with c3: kpi_card("Calories", f"{oth['Calories'].sum():,.0f}")
        generic_time_series(oth, "Other/Workout")
        distance_distribution(oth, "Other/Workout")

st.divider()
st.caption("Built with ❤️ in Streamlit. This project is intentionally modular to grow feature‑by‑feature.")
