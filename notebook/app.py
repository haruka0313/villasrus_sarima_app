"""
Sistem Prediksi Tingkat Hunian Vila — PT Bali Cipta Vila Mandiri
SARIMA Forecasting + Integrated Price-Occupancy Analysis

Versi: Supabase (PostgreSQL) — menggantikan SQLite

Instalasi:
    pip install streamlit pandas numpy statsmodels pmdarima scikit-learn scipy plotly supabase python-dotenv
    streamlit run app.py
"""

# ══════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════

import hashlib
import io
import json
import os
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from scipy import stats
from scipy.signal import periodogram as scipy_periodogram
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf as sm_pacf
from supabase import create_client, Client

try:
    from pmdarima import auto_arima
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pmdarima", "-q"])
    from pmdarima import auto_arima

warnings.filterwarnings("ignore")
load_dotenv()


# ══════════════════════════════════════════════════════════════
# KONSTANTA & KONFIGURASI
# ══════════════════════════════════════════════════════════════

FORECAST_STEPS         = 6
MIN_SEASONAL_TRAIN     = 30
FLAT_STD_THRESH        = 1.5
MIN_CYCLES             = 2
ACF_ALPHA              = 0.10
FALLBACK_M             = 12
SESSION_DURATION_HOURS = 24 * 30

SESSION_KEY = "_sess_token"

LOGO_URL = "https://storage.googleapis.com/villasrus/public/images/logo/villasrus-311x256.webp"

VILLA_COLORS = [
    "#2563EB", "#7C3AED", "#059669", "#DB2777",
    "#D97706", "#B45309", "#0891B2", "#DC2626",
]

DEFAULT_VILLAS = [
    ("briana_villas",   "canggu",   "#2563EB"),
    ("castello_villas", "canggu",   "#7C3AED"),
    ("elina_villas",    "canggu",   "#059669"),
    ("isola_villas",    "canggu",   "#DB2777"),
    ("eindra_villas",   "seminyak", "#D97706"),
    ("esha_villas",     "seminyak", "#B45309"),
    ("ozamiz_villas",   "seminyak", "#0891B2"),
]

OCCUPANCY_ATTRS = [
    "date", "arrivals", "arriving_guests", "departures", "departing_guests",
    "stay_through", "staying_guests", "booked", "booked_guests",
    "available", "black", "occupancy_total"
]
FINANCIAL_ATTRS = [
    "room_revenue", "daily_revenue", "average_daily_rate",
    "revpar", "revenue_per_guest"
]

MODEL_BUCKET = "sarima-models"  # Supabase Storage bucket untuk model .pkl


# ══════════════════════════════════════════════════════════════
# SUPABASE CLIENT
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        st.error("❌ SUPABASE_URL dan SUPABASE_KEY belum diset di file .env")
        st.stop()
    return create_client(url, key)


# ══════════════════════════════════════════════════════════════
# UTILITAS WARNA
# ══════════════════════════════════════════════════════════════

def hex_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ══════════════════════════════════════════════════════════════
# DATABASE — INIT & SEED
# ══════════════════════════════════════════════════════════════

def init_db():
    """Seed data awal ke Supabase jika belum ada."""
    sb = get_supabase()

    # Seed admin jika users kosong
    users = sb.table("users").select("id").limit(1).execute()
    if not users.data:
        sb.table("users").insert({
            "username": "admin",
            "password": _hash_pw("admin123"),
            "role": "admin"
        }).execute()

    # Seed villa_config jika kosong
    villas = sb.table("villa_config").select("villa").limit(1).execute()
    if not villas.data:
        sb.table("villa_config").insert([
            {"villa": v, "area": a, "color": c}
            for v, a, c in DEFAULT_VILLAS
        ]).execute()


# ══════════════════════════════════════════════════════════════
# DATABASE — HELPERS
# ══════════════════════════════════════════════════════════════

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def _gen_token(username: str) -> str:
    raw = f"{username}{time.time()}{os.urandom(16).hex()}"
    return hashlib.sha256(raw.encode()).hexdigest()

# ── Auth ──
def db_auth(username: str, password: str) -> dict | None:
    sb = get_supabase()
    res = sb.table("users").select("*").eq("username", username).eq("password", _hash_pw(password)).execute()
    return res.data[0] if res.data else None

def db_register(username: str, password: str, role: str = "user") -> tuple[bool, str]:
    sb = get_supabase()
    try:
        existing = sb.table("users").select("id").eq("username", username).execute()
        if existing.data:
            return False, "Username sudah terdaftar."
        sb.table("users").insert({
            "username": username,
            "password": _hash_pw(password),
            "role": role
        }).execute()
        return True, "Registrasi berhasil!"
    except Exception as e:
        return False, str(e)

def db_get_users() -> list[dict]:
    sb = get_supabase()
    res = sb.table("users").select("id, username, role, created").execute()
    return res.data or []

# ── Session ──
def session_save(token: str, username: str):
    sb = get_supabase()
    expires = time.time() + SESSION_DURATION_HOURS * 3600
    # Hapus session lama
    sb.table("sessions").delete().eq("username", username).execute()
    sb.table("sessions").insert({
        "token": token,
        "username": username,
        "expires": expires
    }).execute()
    st.session_state[SESSION_KEY] = token
    try:
        st.query_params[SESSION_KEY] = token
    except Exception:
        pass

def session_load() -> dict | None:
    sb = get_supabase()
    token = st.session_state.get(SESSION_KEY)
    if not token:
        try:
            token = st.query_params.get(SESSION_KEY)
        except Exception:
            pass
    if not token:
        return None
    now = time.time()
    # Join sessions + users
    res = sb.table("sessions").select("username, expires").eq("token", token).gt("expires", now).execute()
    if not res.data:
        _session_del(token)
        return None
    username = res.data[0]["username"]
    user_res = sb.table("users").select("role").eq("username", username).execute()
    if not user_res.data:
        return None
    st.session_state[SESSION_KEY] = token
    return {"username": username, "role": user_res.data[0]["role"]}

def session_logout():
    token = st.session_state.get(SESSION_KEY)
    _session_del(token)
    try:
        st.query_params.pop(SESSION_KEY, None)
    except Exception:
        pass
    st.session_state.pop(SESSION_KEY, None)

def _session_del(token: str | None):
    sb = get_supabase()
    if token:
        sb.table("sessions").delete().eq("token", token).execute()
    # Hapus expired sessions
    sb.table("sessions").delete().lt("expires", time.time()).execute()

# ── Villa Config ──
def db_load_villas() -> dict:
    sb = get_supabase()
    res = sb.table("villa_config").select("villa, area, color").execute()
    return {r["villa"]: {"area": r["area"], "color": r["color"]} for r in (res.data or [])}

def db_save_villa(villa: str, area: str, color: str):
    sb = get_supabase()
    existing = sb.table("villa_config").select("villa").eq("villa", villa).execute()
    if existing.data:
        sb.table("villa_config").update({"area": area, "color": color}).eq("villa", villa).execute()
    else:
        sb.table("villa_config").insert({"villa": villa, "area": area, "color": color}).execute()

def db_delete_villa(villa: str):
    sb = get_supabase()
    sb.table("villa_config").delete().eq("villa", villa).execute()
    sb.table("raw_data").delete().eq("villa", villa).execute()
    sb.table("models").delete().eq("villa", villa).execute()

# ── Raw Data ──
def db_save_data(villa: str, dtype: str, filename: str, content: str, rows: int) -> bool:
    """Simpan atau gabung data. Return True jika merged."""
    sb = get_supabase()
    existing = sb.table("raw_data").select("content").eq("villa", villa).eq("data_type", dtype).execute()

    if existing.data:
        try:
            df_old = _parse_csv(existing.data[0]["content"])
            df_new = _parse_csv(content)
            date_col = df_old.columns[0]
            df_merged = pd.concat([df_old, df_new]).drop_duplicates(subset=[date_col])
            merged_content = df_merged.to_csv(index=False)
            total_rows = len(df_merged)
        except Exception:
            merged_content = existing.data[0]["content"]
            total_rows = rows
        sb.table("raw_data").update({
            "content": merged_content,
            "rows": total_rows,
            "filename": filename,
            "uploaded": datetime.utcnow().isoformat()
        }).eq("villa", villa).eq("data_type", dtype).execute()
        return True
    else:
        sb.table("raw_data").insert({
            "villa": villa,
            "data_type": dtype,
            "filename": filename,
            "content": content,
            "rows": rows
        }).execute()
        return False

def db_load_data(villa: str, dtype: str) -> pd.DataFrame | None:
    sb = get_supabase()
    res = sb.table("raw_data").select("content").eq("villa", villa).eq("data_type", dtype).execute()
    if not res.data:
        return None
    try:
        return _parse_csv(res.data[0]["content"])
    except Exception:
        return None

def db_delete_data(villa: str, dtype: str):
    sb = get_supabase()
    sb.table("raw_data").delete().eq("villa", villa).eq("data_type", dtype).execute()

def db_list_data() -> list[dict]:
    sb = get_supabase()
    res = sb.table("raw_data").select("villa, data_type, filename, rows, uploaded").order("uploaded", desc=True).execute()
    return res.data or []

def db_data_info(villa: str, dtype: str) -> dict | None:
    sb = get_supabase()
    res = sb.table("raw_data").select("filename, rows, uploaded").eq("villa", villa).eq("data_type", dtype).execute()
    return res.data[0] if res.data else None

def _parse_csv(content: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(content), sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

# ── Model — disimpan ke Supabase Storage ──
def db_save_model(villa: str, info: dict):
    sb = get_supabase()
    saveable = {k: v for k, v in info.items() if k not in ("model",)}
    pkl_bytes = pickle.dumps(saveable)
    order_str = f"SARIMA{info['order']}x{info['seasonal_order']}"
    try:
        aic = round(info["model"].aic, 2)
    except Exception:
        aic = None

    # Upload pkl ke Supabase Storage
    file_path = f"{villa}.pkl"
    try:
        # Hapus file lama jika ada
        sb.storage.from_(MODEL_BUCKET).remove([file_path])
    except Exception:
        pass
    sb.storage.from_(MODEL_BUCKET).upload(
        file_path,
        pkl_bytes,
        {"content-type": "application/octet-stream"}
    )

    # Simpan metadata ke tabel models
    existing = sb.table("models").select("villa").eq("villa", villa).execute()
    model_data = {
        "villa": villa,
        "order_str": order_str,
        "aic": aic,
        "rmse": round(info.get("rmse", 0), 4),
        "mape": round(info.get("mape", 0), 4),
        "trained_at": datetime.utcnow().isoformat()
    }
    if existing.data:
        sb.table("models").update(model_data).eq("villa", villa).execute()
    else:
        sb.table("models").insert(model_data).execute()

def db_load_model(villa: str) -> dict | None:
    sb = get_supabase()
    # Cek metadata dulu
    res = sb.table("models").select("villa").eq("villa", villa).execute()
    if not res.data:
        return None
    try:
        # Download pkl dari Storage
        file_path = f"{villa}.pkl"
        pkl_bytes = sb.storage.from_(MODEL_BUCKET).download(file_path)
        return pickle.loads(pkl_bytes)
    except Exception:
        return None

def db_model_exists(villa: str) -> bool:
    sb = get_supabase()
    res = sb.table("models").select("villa").eq("villa", villa).execute()
    return bool(res.data)

def db_model_trained_at(villa: str) -> str:
    sb = get_supabase()
    res = sb.table("models").select("trained_at").eq("villa", villa).execute()
    return res.data[0]["trained_at"] if res.data else "—"

def db_list_models() -> list[dict]:
    sb = get_supabase()
    res = sb.table("models").select("villa, order_str, aic, rmse, mape, trained_at").order("trained_at", desc=True).execute()
    return res.data or []

# ── Logs ──
def log_upload(username: str, villa: str, dtype: str, filename: str, rows: int):
    sb = get_supabase()
    sb.table("upload_log").insert({
        "username": username,
        "villa": villa,
        "data_type": dtype,
        "filename": filename,
        "rows": rows
    }).execute()

def log_model(username: str, villa: str, order_str: str, aic: float, rmse: float, mape: float):
    sb = get_supabase()
    sb.table("model_log").insert({
        "username": username,
        "villa": villa,
        "order_str": order_str,
        "aic": aic,
        "rmse": rmse,
        "mape": mape
    }).execute()

def get_upload_log() -> list[dict]:
    sb = get_supabase()
    res = sb.table("upload_log").select("*").order("uploaded", desc=True).limit(100).execute()
    return res.data or []

def get_model_log() -> list[dict]:
    sb = get_supabase()
    res = sb.table("model_log").select("*").order("trained", desc=True).limit(100).execute()
    return res.data or []


# ══════════════════════════════════════════════════════════════
# DATA CLEANING & PARSING
# (Tidak berubah — sama dengan versi SQLite)
# ══════════════════════════════════════════════════════════════

def _find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col
    return None

def _parse_dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if parsed.isna().mean() > 0.3:
        s = s.str.replace(r"^(Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s+", "", regex=True)
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return parsed

def _parse_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[Rp$€IDR\s,]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def _parse_occupancy(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    if s.str.contains("%").any():
        result = pd.to_numeric(s.str.replace("%", "", regex=False), errors="coerce")
    else:
        result = pd.to_numeric(s, errors="coerce")
    if result.dropna().max() <= 1.0:
        result = result * 100.0
    return result.clip(0, 100)

def clean_occupancy(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    date_col = _find_col(df, ["date", "tanggal", "tgl"]) or df.columns[0]
    df[date_col] = _parse_dates(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    occ_col = _find_col(df, ["occupancy", "occ", "hunian", "occupied"])
    if occ_col:
        series = _parse_occupancy(df[occ_col]).rename("occupancy")
    else:
        booked_col = _find_col(df, ["booked"])
        avail_col  = _find_col(df, ["available"])
        if booked_col and avail_col:
            total  = df[booked_col].add(df[avail_col])
            series = (df[booked_col] / total.replace(0, np.nan) * 100).rename("occupancy")
        else:
            num_cols = df.select_dtypes(include=np.number).columns
            series = _parse_occupancy(df[num_cols[0]]).rename("occupancy")
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series   = (series.reindex(full_idx)
                .interpolate(method="time", limit=7)
                .ffill().bfill()
                .clip(0, 100))
    return series

def clean_revenue(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    date_col = _find_col(df, ["date", "tanggal", "tgl"]) or df.columns[0]
    df[date_col] = _parse_dates(df[date_col])
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    rev_col = _find_col(df, [
        "revenue", "pendapatan", "income", "total", "daily_revenue",
        "room_revenue", "gross", "amount", "jumlah", "adr"
    ])
    if rev_col:
        df["revenue"] = _parse_numeric(df[rev_col])
        monthly = df["revenue"].resample("MS").sum().rename("revenue")
        return monthly[monthly > 0]
    return pd.Series(dtype=float, name="revenue")


# ══════════════════════════════════════════════════════════════
# LOAD DATA (cached)
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_all_data(_villa_cfg_json: str) -> tuple[dict, dict]:
    villa_cfg = json.loads(_villa_cfg_json)
    occ_dict, fin_dict = {}, {}
    for villa in villa_cfg:
        for dtype in ("financial", "occupancy"):
            df = db_load_data(villa, dtype)
            if df is not None:
                try:
                    occ_dict[villa] = clean_occupancy(df)
                    break
                except Exception:
                    continue
        df_fin = db_load_data(villa, "financial")
        if df_fin is not None:
            try:
                rev = clean_revenue(df_fin)
                if len(rev) > 0:
                    fin_dict[villa] = rev
            except Exception:
                pass
    return occ_dict, fin_dict


# ══════════════════════════════════════════════════════════════
# ANALISIS STATISTIK
# (Tidak berubah — sama dengan versi SQLite)
# ══════════════════════════════════════════════════════════════

def adf_test(series: pd.Series) -> dict:
    res = adfuller(series.dropna(), autolag="AIC")
    stationary = res[1] < 0.05
    return {
        "statistic": round(res[0], 4),
        "pvalue":    round(res[1], 4),
        "stationary": stationary,
        "critical":  {k: round(v, 3) for k, v in res[4].items()},
    }

def detect_m(monthly: pd.Series) -> tuple[int, str]:
    n = len(monthly)
    freqs, power = scipy_periodogram(monthly.values)
    valid = freqs[1:] > 0
    periods = np.round(1.0 / freqs[1:][valid]).astype(int)
    pwr     = power[1:][valid]
    pp = {}
    for p, pw in zip(periods, pwr):
        if p not in pp or pw > pp[p]:
            pp[p] = pw
    pgram_m = next(
        (int(p) for p in sorted(pp, key=pp.get, reverse=True) if 4 <= p <= n // MIN_CYCLES),
        None
    )
    if pgram_m is None:
        nlags = min(n // 2, 36)
        try:
            acf_vals, ci = sm_acf(monthly, nlags=nlags, alpha=ACF_ALPHA, fft=True)
            for lag in range(4, nlags + 1):
                if lag <= n // MIN_CYCLES:
                    if abs(acf_vals[lag]) > abs(ci[lag][1] - acf_vals[lag]):
                        return lag, "acf"
        except Exception:
            pass
        return FALLBACK_M, "fallback"
    return pgram_m, "periodogram"

def compute_rmse(actual, predicted) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))

def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.abs(actual) > 5
    if mask.sum() < 2:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

def run_adf_all(clean_occ: dict, villa_cfg: dict) -> tuple[dict, pd.DataFrame]:
    villa_d, rows = {}, []
    for villa, series in clean_occ.items():
        monthly = series.resample("MS").mean().dropna()
        res0 = adf_test(monthly)
        if res0["stationary"]:
            d, note = 0, "✅ Stasioner pada level (d=0)"
        else:
            diff1 = monthly.diff().dropna()
            res1 = adf_test(diff1)
            d    = 1
            note = "🔄 Stasioner setelah diff(1) → d=1" if res1["stationary"] \
                   else f"⚠️ Masih belum stasioner (p={res1['pvalue']:.3f}), d=1 digunakan"
        villa_d[villa] = d
        rows.append({
            "Vila":         villa.replace("_villas", "").title(),
            "Area":         villa_cfg.get(villa, {}).get("area", "").title(),
            "N (bln)":      len(monthly),
            "ADF Stat":     res0["statistic"],
            "p-value":      res0["pvalue"],
            "Stasioner?":   "✅ Ya" if res0["stationary"] else "❌ Tidak",
            "d digunakan":  d,
            "Keterangan":   note,
        })
    return villa_d, pd.DataFrame(rows)

def run_detect_m_all(clean_occ: dict, villa_cfg: dict) -> tuple[dict, pd.DataFrame]:
    villa_m, rows = {}, []
    for villa, series in clean_occ.items():
        monthly = series.resample("MS").mean().dropna()
        m, method = detect_m(monthly)
        villa_m[villa] = m
        rows.append({
            "Vila":    villa.replace("_villas", "").title(),
            "Area":    villa_cfg.get(villa, {}).get("area", "").title(),
            "N (bln)": len(monthly),
            "m":       m,
            "Metode":  method,
            "Status":  "✅" if method != "fallback" else "⚠️ Fallback m=12",
        })
    return villa_m, pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# SARIMA — TRAINING & FORECAST
# (Tidak berubah — sama dengan versi SQLite)
# ══════════════════════════════════════════════════════════════

def train_sarima(series: pd.Series, d: int, m: int, color: str, title: str) -> dict:
    monthly      = series.resample("MS").mean().dropna()
    use_seasonal = len(monthly) >= MIN_SEASONAL_TRAIN
    max_pq       = 2 if d >= 1 else 3
    max_PQ       = 1 if use_seasonal else 0
    split_idx    = max(int(len(monthly) * 0.85), len(monthly) - 6)
    train, test  = monthly.iloc[:split_idx], monthly.iloc[split_idx:]
    try:
        auto = auto_arima(
            train,
            seasonal=use_seasonal, m=m if use_seasonal else 1,
            d=d, D=None if use_seasonal else 0,
            start_p=0, max_p=max_pq, start_q=0, max_q=max_pq,
            start_P=0, max_P=max_PQ, start_Q=0, max_Q=max_PQ,
            information_criterion="aic", stepwise=True,
            error_action="ignore", suppress_warnings=True, trace=False,
        )
        order, seasonal_order = auto.order, auto.seasonal_order
    except Exception:
        order, seasonal_order = (1, d, 1), (0, 0, 0, 0)
    if order == (0, 0, 0) and seasonal_order[:3] == (0, 0, 0):
        order = (1, d, 0)
    try:
        model = SARIMAX(
            train, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    except Exception:
        order, seasonal_order = (1, d, 1), (0, 0, 0, 0)
        model = SARIMAX(
            train, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    pred_obj   = model.get_forecast(steps=len(test))
    pred_mean  = pred_obj.predicted_mean.clip(0, 100)
    pred_ci    = pred_obj.conf_int(alpha=0.10)
    rmse_val = compute_rmse(test.values, pred_mean.values)
    mape_val = compute_mape(test.values, pred_mean.values)
    return {
        "model": model,
        "order": order,
        "seasonal_order": seasonal_order,
        "train": train, "test": test, "monthly": monthly,
        "d": d, "m": m, "use_seasonal": use_seasonal,
        "pred_mean": pred_mean, "pred_ci": pred_ci,
        "rmse": rmse_val, "mape": mape_val,
        "color": color, "title": title,
    }

def make_forecast(info: dict) -> dict:
    monthly = info["monthly"]
    order, s_order, d = info["order"], info["seasonal_order"], info["d"]
    m = info.get("m", FALLBACK_M)
    try:
        model_full = SARIMAX(
            monthly, order=order, seasonal_order=s_order,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    except Exception:
        model_full = info["model"]
    fore_obj  = model_full.get_forecast(steps=FORECAST_STEPS)
    fore_mean = fore_obj.predicted_mean.clip(0, 100)
    fore_ci   = fore_obj.conf_int(alpha=0.10).clip(0, 100)
    is_flat = fore_mean.std() < FLAT_STD_THRESH
    if is_flat:
        for s_cand in [(1, 0, 0, m), (0, 0, 1, m), (1, 0, 1, m)]:
            try:
                mc = SARIMAX(
                    monthly, order=(order[0], d, order[2]), seasonal_order=s_cand,
                    enforce_stationarity=False, enforce_invertibility=False
                ).fit(disp=False)
                fc = mc.get_forecast(steps=FORECAST_STEPS)
                fc_mean = fc.predicted_mean.clip(0, 100)
                if fc_mean.std() > FLAT_STD_THRESH and mc.aic < model_full.aic + 10:
                    fore_mean = fc_mean
                    fore_ci   = fc.conf_int(alpha=0.10).clip(0, 100)
                    s_order   = s_cand
                    is_flat   = False
                    break
            except Exception:
                continue
    return {
        "fore_mean":    fore_mean,
        "fore_ci":      fore_ci,
        "used_s_order": s_order,
        "is_flat":      is_flat,
    }


# ══════════════════════════════════════════════════════════════
# PLOTLY — BASE LAYOUT & CHART FUNCTIONS
# (Tidak berubah — sama dengan versi SQLite)
# ══════════════════════════════════════════════════════════════

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,252,1)",
    font=dict(family="DM Sans, sans-serif", size=11, color="#374151"),
    xaxis=dict(gridcolor="#F3F4F6", zeroline=False),
    yaxis=dict(gridcolor="#F3F4F6", zeroline=False),
    margin=dict(l=60, r=30, t=70, b=50),
    hovermode="x unified",
)

def apply_base(fig: go.Figure, **extra) -> go.Figure:
    layout = dict(BASE_LAYOUT)
    for key in ("xaxis", "yaxis"):
        if key in extra and key in layout:
            merged = dict(layout[key])
            merged.update(extra.pop(key))
            layout[key] = merged
    layout.update(extra)
    fig.update_layout(**layout)
    return fig

def chart_trend_all(clean_occ: dict, villa_cfg: dict) -> go.Figure:
    fig = go.Figure()
    for villa, series in clean_occ.items():
        color   = villa_cfg.get(villa, {}).get("color", "#2563EB")
        monthly = series.resample("MS").mean()
        fig.add_trace(go.Scatter(
            x=monthly.index, y=monthly.values,
            line=dict(color=color, width=2),
            mode="lines+markers", marker=dict(size=4),
            name=villa.replace("_villas", "").title(),
            hovertemplate="<b>%{fullData.name}</b><br>%{x|%b %Y}: %{y:.1f}%<extra></extra>",
        ))
    apply_base(fig,
        title=dict(text="Tren Okupansi Bulanan — Semua Vila", font=dict(size=14, color="#1E3A5F")),
        yaxis=dict(title="Okupansi (%)", range=[0, 110], ticksuffix="%", gridcolor="#F3F4F6"),
        height=380,
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E5E7EB", borderwidth=1),
    )
    return fig

def chart_bar_mean(clean_occ: dict, villa_cfg: dict) -> go.Figure:
    data = sorted([
        (villa.replace("_villas", "").title(),
         round(series.resample("MS").mean().mean(), 1),
         villa_cfg.get(villa, {}).get("color", "#2563EB"))
        for villa, series in clean_occ.items()
    ], key=lambda x: x[1], reverse=True)
    global_mean = np.mean([d[1] for d in data])
    fig = go.Figure()
    for title, mean_val, color in data:
        fig.add_trace(go.Bar(
            x=[title], y=[mean_val],
            marker_color=color,
            text=[f"{mean_val:.1f}%"], textposition="outside",
            name=title,
            hovertemplate=f"<b>{title}</b><br>Mean: {mean_val:.1f}%<extra></extra>",
        ))
    fig.add_hline(y=global_mean, line_dash="dash", line_color="#EF4444",
        annotation_text=f"Avg: {global_mean:.1f}%", annotation_position="top right",
        annotation_font=dict(color="#EF4444", size=11))
    apply_base(fig,
        title=dict(text="Rata-rata Okupansi per Vila", font=dict(size=14, color="#1E3A5F")),
        yaxis=dict(title="Okupansi (%)", range=[0, 120], ticksuffix="%", gridcolor="#F3F4F6"),
        showlegend=False, height=340,
    )
    return fig

def chart_decomposition(monthly: pd.Series, m: int, color: str, title: str) -> go.Figure | None:
    if len(monthly) < 24:
        return None
    decomp = seasonal_decompose(monthly, model="additive", period=m, extrapolate_trend="freq")
    components = [
        (decomp.observed, "Observed",  color,    "lines"),
        (decomp.trend,    "Trend",     "#1D4ED8", "lines"),
        (decomp.seasonal, "Seasonal",  "#059669", "lines"),
        (decomp.resid,    "Residual",  "#6B7280", "bar"),
    ]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
        subplot_titles=["Observed (Aktual)", "Trend", f"Seasonal (m={m})", "Residual"],
        vertical_spacing=0.07)
    for i, (comp, label, c, mode) in enumerate(components, start=1):
        if mode == "bar":
            fig.add_trace(go.Bar(x=monthly.index, y=comp.values, marker_color=c,
                opacity=0.65, name=label,
                hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{label}: %{{y:.3f}}<extra></extra>"), row=i, col=1)
            fig.add_hline(y=0, line_color="#9CA3AF", line_width=0.8, row=i, col=1)
        else:
            fig.add_trace(go.Scatter(x=monthly.index, y=comp.values,
                line=dict(color=c, width=1.8), name=label,
                hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{label}: %{{y:.2f}}<extra></extra>"), row=i, col=1)
        fig.update_yaxes(title_text=label, gridcolor="#F3F4F6", row=i, col=1)
        fig.update_xaxes(gridcolor="#F3F4F6", row=i, col=1)
    fig.update_layout(height=620,
        title=dict(text=f"{title} — Dekomposisi Aditif | m={m}", font=dict(size=13, color="#1E3A5F")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
        font=dict(family="DM Sans, sans-serif", size=11),
        showlegend=False, margin=dict(l=60, r=30, t=80, b=40))
    return fig

def chart_acf_pacf(monthly: pd.Series, m: int, color: str, title: str) -> go.Figure:
    n = len(monthly)
    conf = 1.96 / np.sqrt(n)
    acf_vals,  _ = sm_acf(monthly,  nlags=min(36, n - 1),     alpha=0.05, fft=True)
    pacf_vals, _ = sm_pacf(monthly, nlags=min(36, n // 2 - 1), alpha=0.05)
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=[f"ACF — {title}", f"PACF — {title}"],
        horizontal_spacing=0.1)
    for col_i, (vals, lbl) in enumerate([(acf_vals, "ACF"), (pacf_vals, "PACF")], start=1):
        lags = list(range(len(vals)))
        bar_colors = [color if abs(v) > conf else "#D1D5DB" for v in vals]
        fig.add_trace(go.Bar(x=lags, y=vals, marker_color=bar_colors, name=lbl,
            hovertemplate=f"Lag %{{x}}<br>{lbl}: %{{y:.3f}}<extra></extra>"), row=1, col=col_i)
        fig.add_hline(y= conf, line_dash="dash", line_color="#9CA3AF", line_width=1, row=1, col=col_i)
        fig.add_hline(y=-conf, line_dash="dash", line_color="#9CA3AF", line_width=1, row=1, col=col_i)
        for lag in range(m, len(vals), m):
            fig.add_vline(x=lag, line_dash="dot", line_color=color,
                line_width=0.8, opacity=0.4, row=1, col=col_i)
        fig.update_xaxes(gridcolor="#F3F4F6", row=1, col=col_i)
        fig.update_yaxes(gridcolor="#F3F4F6", row=1, col=col_i)
    apply_base(fig, height=300, showlegend=False)
    return fig

def chart_model_fit(info: dict) -> go.Figure:
    train, test = info["train"], info["test"]
    pred_mean, pred_ci = info["pred_mean"], info["pred_ci"]
    color, title = info["color"], info["title"]
    rmse, mape   = info["rmse"], info.get("mape", float("nan"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(pred_ci.index) + list(pred_ci.index[::-1]),
        y=list(pred_ci.iloc[:, 1].clip(0, 100)) + list(pred_ci.iloc[:, 0].clip(0, 100)),
        fill="toself", fillcolor=hex_rgba(color, 0.12),
        line=dict(color="rgba(0,0,0,0)"), name="CI 90%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=train.index, y=train.values,
        line=dict(color="#9CA3AF", width=1.5), name="Data Latih",
        hovertemplate="<b>%{x|%b %Y}</b><br>Latih: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=test.index, y=test.values,
        line=dict(color="#111827", width=2.2), mode="lines+markers",
        marker=dict(size=7), name="Aktual (Test)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Aktual: %{y:.1f}%<extra></extra>"))
    mape_label = f"{mape:.1f}%" if not np.isnan(mape) else "—"
    fig.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean.values,
        line=dict(color=color, width=2.2, dash="dash"),
        mode="lines+markers", marker=dict(size=6, symbol="square"),
        name=f"Prediksi | RMSE={rmse:.1f}% | MAPE={mape_label}",
        hovertemplate="<b>%{x|%b %Y}</b><br>Prediksi: %{y:.1f}%<extra></extra>"))
    if len(test) > 0:
        fig.add_vline(x=test.index[0].isoformat(), line_color="#9CA3AF",
            line_width=1.2, line_dash="dot")
    order_str = f"SARIMA{info['order']}×{info['seasonal_order']}"
    apply_base(fig,
        title=dict(text=f"{title} — {order_str} | d={info['d']} m={info['m']} | AIC={info['model'].aic:.1f}",
            font=dict(size=13, color="#1E3A5F")),
        yaxis=dict(title="Okupansi (%)", range=[-5, 115], ticksuffix="%", gridcolor="#F3F4F6"),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#E5E7EB", borderwidth=1))
    return fig

def chart_forecast(info: dict, fore: dict) -> go.Figure:
    monthly = info["monthly"]
    color, title = info["color"], info["title"]
    fore_mean, fore_ci = fore["fore_mean"], fore["fore_ci"]
    is_flat = fore["is_flat"]
    fore_color = "#EF4444" if is_flat else color
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(fore_ci.index) + list(fore_ci.index[::-1]),
        y=list(fore_ci.iloc[:, 1]) + list(fore_ci.iloc[:, 0]),
        fill="toself", fillcolor=hex_rgba(fore_color, 0.10),
        line=dict(color="rgba(0,0,0,0)"), name="CI 90%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly.values,
        line=dict(color="#9CA3AF", width=1.5), name="Historis",
        hovertemplate="<b>%{x|%b %Y}</b><br>Historis: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=info["test"].index, y=info["test"].values,
        line=dict(color="#111827", width=2), mode="lines+markers",
        marker=dict(size=5), name="Aktual (Test)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Aktual: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=info["pred_mean"].index, y=info["pred_mean"].values,
        line=dict(color=color, width=1.5, dash="dot"), name="Fitted", opacity=0.7,
        hovertemplate="<b>%{x|%b %Y}</b><br>Fitted: %{y:.1f}%<extra></extra>"))
    flat_label = " ⚠️ Flat" if is_flat else f" (σ={fore_mean.std():.1f}%)"
    fig.add_trace(go.Scatter(
        x=fore_mean.index, y=fore_mean.values,
        line=dict(color=fore_color, width=3),
        mode="lines+markers+text",
        marker=dict(size=9, symbol="diamond", color=fore_color),
        text=[f"{v:.0f}%" for v in fore_mean.values],
        textposition="top center", textfont=dict(size=10, color=fore_color),
        name=f"Forecast +{FORECAST_STEPS}M{flat_label}",
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: %{y:.1f}%<extra></extra>"))
    end_x = monthly.index[-1].isoformat()
    fig.add_vline(x=end_x, line_color="#EF4444", line_width=1.3, line_dash="dash")
    fig.add_annotation(x=end_x, y=0.96, xref="x", yref="paper",
        text="← Historis | Forecast →", showarrow=False,
        font=dict(color="#EF4444", size=9),
        bgcolor="rgba(255,255,255,0.8)", borderpad=3, xanchor="center")
    if is_flat:
        fig.add_annotation(x=0.5, y=0.92, xref="paper", yref="paper",
            text="⚠️ Forecast flat — model tidak menangkap pola musiman",
            showarrow=False, font=dict(size=11, color="white"),
            bgcolor="#EF4444", borderpad=6)
    apply_base(fig,
        title=dict(text=f"{title} — Forecast +{FORECAST_STEPS} Bulan | SARIMA{info['order']}×{fore['used_s_order']}",
            font=dict(size=13, color="#1E3A5F")),
        yaxis=dict(title="Okupansi (%)", range=[0, 125], ticksuffix="%", gridcolor="#F3F4F6"),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#E5E7EB", borderwidth=1))
    return fig

def chart_scatter_occ_rev(clean_occ: dict, clean_fin: dict, villa_cfg: dict) -> go.Figure | None:
    avail = [v for v in clean_occ if v in clean_fin]
    if not avail:
        return None
    n_cols = min(3, len(avail))
    n_rows = (len(avail) + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols,
        subplot_titles=[v.replace("_villas", "").title() for v in avail],
        horizontal_spacing=0.10, vertical_spacing=0.14)
    for idx, villa in enumerate(avail):
        row, col = idx // n_cols + 1, idx % n_cols + 1
        color = villa_cfg.get(villa, {}).get("color", "#2563EB")
        occ_m = clean_occ[villa].resample("MS").mean().rename("occ")
        comb  = pd.concat([occ_m, clean_fin[villa]], axis=1).dropna()
        if len(comb) < 5:
            continue
        rev_m = comb["revenue"] / 1e6
        r, p  = stats.pearsonr(comb["occ"], rev_m)
        m_, b_ = np.polyfit(comb["occ"], rev_m, 1)
        x_line = np.linspace(comb["occ"].min(), comb["occ"].max(), 100)
        fig.add_trace(go.Scatter(x=comb["occ"], y=rev_m, mode="markers",
            marker=dict(color=color, size=8, opacity=0.75, line=dict(color="white", width=1)),
            name=villa.replace("_villas", "").title(),
            hovertemplate="<b>%{customdata}</b><br>Occ: %{x:.1f}%<br>Rev: %{y:.1f}M IDR<extra></extra>",
            customdata=comb.index.strftime("%b %Y")), row=row, col=col)
        fig.add_trace(go.Scatter(x=x_line, y=m_ * x_line + b_,
            line=dict(color="#EF4444", width=2, dash="dash"), showlegend=False), row=row, col=col)
        fig.add_annotation(x=comb["occ"].quantile(0.05), y=rev_m.max() * 0.95,
            text=f"r = {r:.3f}<br>p = {p:.3f}", showarrow=False,
            font=dict(size=10, color="#374151"),
            bgcolor="rgba(255,255,255,0.88)", borderpad=4, row=row, col=col)
        fig.update_yaxes(title_text="Revenue (Juta IDR)", gridcolor="#F3F4F6", row=row, col=col)
        fig.update_xaxes(title_text="Okupansi (%)", gridcolor="#F3F4F6", row=row, col=col)
    fig.update_layout(height=380 * n_rows,
        title=dict(text="Keterkaitan Okupansi ↔ Revenue", font=dict(size=14, color="#1E3A5F")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
        font=dict(family="DM Sans, sans-serif", size=11),
        showlegend=False, margin=dict(l=60, r=30, t=80, b=40))
    return fig

def chart_residual(info: dict) -> go.Figure:
    resid = info["model"].resid.dropna()
    color = info["color"]
    title = info["title"]
    lb     = acorr_ljungbox(resid, lags=[12, 24], return_df=True)
    p12    = lb.loc[12, "lb_pvalue"]
    _, p_n = stats.shapiro(resid[:50])
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=["Residual atas Waktu", "Distribusi Residual", "Q-Q Plot"],
        horizontal_spacing=0.09)
    fig.add_trace(go.Scatter(x=resid.index, y=resid.values,
        line=dict(color=color, width=1.5), name="Residual"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#9CA3AF", line_width=0.8, row=1, col=1)
    mu, sd = resid.mean(), resid.std()
    x_fit  = np.linspace(resid.min(), resid.max(), 200)
    fig.add_trace(go.Histogram(x=resid.values, nbinsx=20, marker_color=color, opacity=0.65,
        histnorm="probability density", name="Hist"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_fit, y=stats.norm.pdf(x_fit, mu, sd),
        line=dict(color="#111827", width=1.5, dash="dash"), name="Normal"), row=1, col=2)
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
        marker=dict(color=color, size=5, opacity=0.7), name="Q-Q"), row=1, col=3)
    x_line = np.array([min(osm), max(osm)])
    fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept,
        line=dict(color="#EF4444", width=1.5, dash="dash"), showlegend=False), row=1, col=3)
    for ci in range(1, 4):
        fig.update_yaxes(gridcolor="#F3F4F6", row=1, col=ci)
        fig.update_xaxes(gridcolor="#F3F4F6", row=1, col=ci)
    lb_lbl = "✅ LB OK" if p12 > 0.05 else "⚠️ Autokorelasi"
    n_lbl  = "✅ Normal" if p_n > 0.05 else "⚠️ Non-normal"
    fig.update_layout(
        title=dict(text=f"{title} — Diagnostik Residual | {lb_lbl} (p12={p12:.3f}) | Shapiro p={p_n:.3f} {n_lbl}",
            font=dict(size=12, color="#1E3A5F")),
        height=310,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
        font=dict(family="DM Sans, sans-serif", size=11),
        showlegend=False, margin=dict(l=60, r=30, t=70, b=40))
    return fig


# ══════════════════════════════════════════════════════════════
# KOMPONEN UI BERSAMA
# (Tidak berubah — sama dengan versi SQLite)
# ══════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, sub: str = "", color: str = "#2563EB"):
    st.markdown(f"""
        <div style="
            background: white;
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 14px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            margin-bottom: 4px;
        ">
            <div style="font-size:11px;color:#6B7280;font-weight:500;text-transform:uppercase;
                        letter-spacing:0.05em;margin-bottom:2px;">{label}</div>
            <div style="font-size:22px;font-weight:700;color:#111827;">{value}</div>
            {f'<div style="font-size:11px;color:#6B7280;margin-top:2px;">{sub}</div>' if sub else ''}
        </div>
    """, unsafe_allow_html=True)

def section_header(text: str, icon: str = ""):
    st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin:24px 0 12px;">
            <div style="width:4px;height:20px;background:#2563EB;border-radius:2px;"></div>
            <span style="font-size:16px;font-weight:700;color:#1E3A5F;">{icon} {text}</span>
        </div>
    """, unsafe_allow_html=True)

def status_badge(value: float, thresholds: tuple = (75, 50)) -> tuple[str, str, str]:
    if value >= thresholds[0]:
        return "🟢", "Tinggi", "#DCFCE7"
    elif value >= thresholds[1]:
        return "🟡", "Sedang", "#FEF9C3"
    else:
        return "🔴", "Rendah", "#FEE2E2"

def model_quality_badge(mape: float) -> tuple[str, str]:
    if np.isnan(mape):
        return "⚪", "Belum dievaluasi"
    if mape < 10:
        return "🟢", f"Sangat Baik (MAPE={mape:.1f}%)"
    elif mape < 20:
        return "🟡", f"Cukup Baik (MAPE={mape:.1f}%)"
    else:
        return "🔴", f"Perlu Review (MAPE={mape:.1f}%)"

def period_filter(clean_occ: dict, key: str) -> tuple:
    all_dates = []
    for s in clean_occ.values():
        m = s.resample("MS").mean().dropna()
        if len(m):
            all_dates += [m.index.min(), m.index.max()]
    if not all_dates:
        return None, None
    g_min, g_max = min(all_dates), max(all_dates)
    PRESETS = {"Semua Data": None, "6 Bln Terakhir": 6, "1 Thn Terakhir": 12,
               "2 Thn Terakhir": 24, "Custom": "custom"}
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        preset = st.selectbox("Periode", list(PRESETS.keys()), index=0, key=f"{key}_preset")
    if preset == "Custom":
        months_list = pd.date_range(g_min, g_max, freq="MS")
        labels      = [d.strftime("%b %Y") for d in months_list]
        with c2:
            sl = st.selectbox("Dari", labels, index=0, key=f"{key}_start")
        with c3:
            el = st.selectbox("Sampai", labels, index=len(labels) - 1, key=f"{key}_end")
        ds = pd.Timestamp(datetime.strptime(sl, "%b %Y"))
        de = pd.Timestamp(datetime.strptime(el, "%b %Y"))
        if ds > de:
            ds, de = de, ds
    elif PRESETS[preset] is None:
        ds, de = g_min, g_max
    else:
        n   = PRESETS[preset]
        de  = g_max
        ds  = max(g_max - pd.DateOffset(months=n - 1), g_min)
        with c2:
            st.info(f"📅 {ds.strftime('%b %Y')} — {de.strftime('%b %Y')}")
    n_months = len(pd.date_range(ds, de, freq="MS"))
    st.caption(f"🗓️ **{ds.strftime('%B %Y')} — {de.strftime('%B %Y')}** ({n_months} bulan)")
    return ds, de

def filter_occ(clean_occ: dict, ds, de) -> dict:
    if ds is None or de is None:
        return clean_occ
    end_ = de + pd.DateOffset(months=1)
    return {v: s[(s.index >= ds) & (s.index <= end_)]
            for v, s in clean_occ.items()
            if len(s[(s.index >= ds) & (s.index <= end_)]) > 0}


# ══════════════════════════════════════════════════════════════
# HALAMAN — LOGIN / DASHBOARD / DATA / STRATEGI
# (Tidak berubah dari versi SQLite — fungsi page_* sama persis)
# ══════════════════════════════════════════════════════════════

def page_login():
    st.markdown("""
    <style>
    .block-container { max-width: 520px; padding-top: 4rem; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center;margin-bottom:16px;'>"
        f"<img src='{LOGO_URL}' style='height:80px;object-fit:contain;'></div>",
        unsafe_allow_html=True)
    st.markdown(
        "<h2 style='text-align:center;color:#1E3A5F;margin:0 0 4px;'>"
        "Sistem Prediksi Tingkat Hunian</h2>"
        "<p style='text-align:center;color:#6B7280;font-size:14px;margin-bottom:24px;'>"
        "PT Bali Cipta Vila Mandiri</p>",
        unsafe_allow_html=True)
    tab_login, tab_reg = st.tabs(["🔐 Masuk", "📝 Daftar Akun"])
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Masukkan username")
            password = st.text_input("Password", type="password", placeholder="Masukkan password")
            submitted = st.form_submit_button("Masuk", use_container_width=True, type="primary")
        if submitted:
            user = db_auth(username, password)
            if user:
                token = _gen_token(username)
                session_save(token, username)
                st.session_state.user      = user
                st.session_state.logged_in = True
                st.success(f"Selamat datang, **{username}**!")
                time.sleep(0.4)
                st.rerun()
            else:
                st.error("Username atau password salah.")
        st.caption("👤 Admin default: `admin` / `admin123`")
    with tab_reg:
        with st.form("reg_form"):
            nu  = st.text_input("Username baru")
            np_ = st.text_input("Password", type="password")
            np2 = st.text_input("Konfirmasi Password", type="password")
            reg = st.form_submit_button("Daftar", use_container_width=True)
        if reg:
            if np_ != np2:
                st.error("Password tidak cocok.")
            elif len(nu) < 3:
                st.error("Username minimal 3 karakter.")
            elif len(np_) < 6:
                st.error("Password minimal 6 karakter.")
            else:
                ok, msg = db_register(nu, np_)
                st.success(msg + " Silakan login.") if ok else st.error(msg)


def page_dashboard(clean_occ: dict, clean_fin: dict, villa_cfg: dict):
    is_admin = st.session_state.user.get("role") == "admin"
    st.markdown(
        f"<h1 style='color:#1E3A5F;margin-bottom:4px;'>🏠 Dashboard Utama</h1>"
        f"<p style='color:#6B7280;margin-bottom:20px;'>"
        f"Ringkasan performa seluruh unit vila · {datetime.now().strftime('%d %B %Y, %H:%M')}</p>",
        unsafe_allow_html=True)
    if not clean_occ:
        st.warning("⚠️ Belum ada data vila. Upload data melalui menu **Manajemen Data**.")
        return
    section_header("Ringkasan Keseluruhan", "📊")
    all_means   = [s.resample("MS").mean().mean() for s in clean_occ.values()]
    global_mean = np.mean(all_means) if all_means else 0
    n_with_model = sum(1 for v in villa_cfg if db_model_exists(v))
    n_with_data  = len(clean_occ)
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Vila Terdaftar", str(len(villa_cfg)), "unit", "#2563EB")
    with c2: kpi_card("Vila dengan Data", str(n_with_data), "unit aktif", "#059669")
    with c3: kpi_card("Model Prediksi Siap", str(n_with_model), "villa terlatih",
                 "#7C3AED" if n_with_model == len(villa_cfg) else "#D97706")
    with c4:
        icon, _, _ = status_badge(global_mean)
        kpi_card("Rata-rata Okupansi", f"{global_mean:.1f}%", f"{icon} keseluruhan", "#DB2777")
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Performa per Area", "📍")
    area_stats: dict = {}
    for villa, series in clean_occ.items():
        area = villa_cfg.get(villa, {}).get("area", "lainnya").title()
        area_stats.setdefault(area, []).append(series.resample("MS").mean().mean())
    area_cols = st.columns(max(1, len(area_stats)))
    for i, (area, means) in enumerate(sorted(area_stats.items())):
        with area_cols[i]:
            icon, label, bg = status_badge(np.mean(means))
            st.markdown(f"""
                <div style="background:{bg};border-radius:10px;padding:16px;text-align:center;">
                    <div style="font-size:20px;">{icon}</div>
                    <div style="font-weight:700;font-size:18px;color:#111827;">{np.mean(means):.1f}%</div>
                    <div style="color:#374151;font-weight:600;">{area}</div>
                    <div style="font-size:12px;color:#6B7280;">{len(means)} vila</div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Visualisasi Performa", "📈")
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.plotly_chart(chart_trend_all(clean_occ, villa_cfg), use_container_width=True)
    with col_right:
        st.plotly_chart(chart_bar_mean(clean_occ, villa_cfg), use_container_width=True)
    section_header("Status Vila", "🏡")
    rows = []
    for villa, cfg in villa_cfg.items():
        has_data  = villa in clean_occ
        has_model = db_model_exists(villa)
        occ_mean  = clean_occ[villa].resample("MS").mean().mean() if has_data else 0
        icon, lbl, _ = status_badge(occ_mean)
        row = {
            "Vila":       villa.replace("_villas", "").title(),
            "Area":       cfg["area"].title(),
            "Data":       "✅" if has_data else "❌",
            "Model":      "✅" if has_model else "❌",
            "Okupansi":   f"{occ_mean:.1f}%" if has_data else "—",
            "Status":     f"{icon} {lbl}" if has_data else "—",
            "Diperbarui": db_model_trained_at(villa) if has_model else "—",
        }
        if is_admin:
            mi = db_load_model(villa)
            if mi:
                row["RMSE (%)"] = round(mi.get("rmse", 0), 2)
                row["MAPE (%)"] = round(mi.get("mape", 0), 2)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Sistem Prediksi Hunian — PT Bali Cipta Vila Mandiri",
        page_icon="🏖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
    .block-container { padding-top: 2rem; }
    h1 { color: #1E3A5F; font-weight: 700; }
    h2 { color: #2563EB; border-bottom: 2px solid #EFF6FF; padding-bottom: 6px; font-weight: 600; }
    h3 { color: #374151; font-weight: 600; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EFF6FF 0%, #F0F7FF 100%);
        border-right: 1px solid #DBEAFE;
    }
    .stExpander { border: 1px solid #E5E7EB !important; border-radius: 10px !important; }
    div[data-testid="stMetric"] {
        background: white; border-radius: 8px; padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    </style>
    """, unsafe_allow_html=True)

    # Init Supabase & seed data
    if "db_initialized" not in st.session_state:
        init_db()
        st.session_state["db_initialized"] = True

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user      = {}

    if not st.session_state.logged_in:
        saved = session_load()
        if saved:
            st.session_state.user      = saved
            st.session_state.logged_in = True

    if not st.session_state.logged_in:
        page_login()
        return

    if "villa_config" not in st.session_state:
        st.session_state.villa_config = db_load_villas()
    villa_cfg = st.session_state.villa_config
    is_admin  = st.session_state.user.get("role") == "admin"
    username  = st.session_state.user.get("username", "")

    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center;padding:12px 0 8px;'>"
            f"<img src='{LOGO_URL}' style='height:60px;object-fit:contain;'></div>",
            unsafe_allow_html=True)
        role_color = "#DC2626" if is_admin else "#2563EB"
        role_label = "Admin" if is_admin else "User"
        st.markdown(
            f"<div style='text-align:center;padding:4px 0 12px;'>"
            f"<span style='font-size:13px;color:#374151;'>{username}</span>&nbsp;"
            f"<span style='background:{role_color};color:white;font-size:11px;"
            f"font-weight:600;padding:2px 8px;border-radius:12px;'>{role_label}</span>"
            f"</div>", unsafe_allow_html=True)
        st.divider()
        if is_admin:
            nav = st.radio("", [
                "🏠 Dashboard Utama",
                "📊 Strategi Hunian & Harga",
                "📂 Manajemen Data",
            ], label_visibility="collapsed")
        else:
            nav = st.radio("", [
                "🏠 Dashboard Utama",
                "📊 Strategi Hunian & Harga",
            ], label_visibility="collapsed")
        if nav in ["🏠 Dashboard Utama", "📊 Strategi Hunian & Harga"]:
            st.divider()
            st.markdown(
                "<div style='font-size:12px;font-weight:600;color:#374151;"
                "text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;'>Filter Vila</div>",
                unsafe_allow_html=True)
            selected_villas = st.multiselect(
                "", options=list(villa_cfg.keys()),
                default=list(villa_cfg.keys()),
                format_func=lambda v: f"{v.replace('_villas','').title()} · {villa_cfg[v]['area'].title()}",
                label_visibility="collapsed")
            if not selected_villas:
                st.warning("Pilih minimal 1 vila.")
                selected_villas = list(villa_cfg.keys())
        else:
            selected_villas = list(villa_cfg.keys())
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            session_logout()
            st.session_state.logged_in = False
            st.session_state.user      = {}
            st.rerun()
        st.markdown(
            "<div style='font-size:10px;color:#9CA3AF;text-align:center;padding-top:8px;'>"
            "SARIMA Forecasting System<br>PT Bali Cipta Vila Mandiri<br>© 2025</div>",
            unsafe_allow_html=True)

    if nav == "📂 Manajemen Data":
        if not is_admin:
            st.error("🔒 Akses ditolak.")
            return
        # Halaman manajemen data sama persis seperti sebelumnya
        # (tidak diulang agar file tidak terlalu panjang)
        st.info("Halaman Manajemen Data — sambungkan dari versi app.py original.")
        return

    with st.spinner("Memuat data..."):
        clean_occ_all, clean_fin_all = load_all_data(json.dumps(villa_cfg, sort_keys=True))

    if nav == "🏠 Dashboard Utama":
        page_dashboard(clean_occ_all, clean_fin_all, villa_cfg)
        return

    clean_occ = {v: s for v, s in clean_occ_all.items() if v in selected_villas}
    clean_fin = {v: s for v, s in clean_fin_all.items() if v in selected_villas}
    if not clean_occ:
        st.error("❌ Tidak ada data untuk vila yang dipilih.")
        return

    with st.spinner("Analisis ADF & deteksi siklus..."):
        villa_d, _ = run_adf_all(clean_occ, villa_cfg)
        villa_m, _ = run_detect_m_all(clean_occ, villa_cfg)

    sarima_models: dict = {}
    cache = st.session_state.get("sarima_cache", {})
    for villa in selected_villas:
        if villa in cache and villa in clean_occ:
            sarima_models[villa] = cache[villa]

    for villa in [v for v in selected_villas if v not in sarima_models and v in clean_occ]:
        mi = db_load_model(villa)
        if not mi:
            continue
        try:
            monthly = clean_occ[villa].resample("MS").mean().dropna()
            order   = mi.get("order",   (1, 1, 1))
            s_order = mi.get("seasonal_order", (0, 0, 0, 0))
            split_  = max(int(len(monthly) * 0.85), len(monthly) - 6)
            train_, test_ = monthly.iloc[:split_], monthly.iloc[split_:]
            fitted_ = SARIMAX(
                train_, order=order, seasonal_order=s_order,
                enforce_stationarity=False, enforce_invertibility=False
            ).fit(disp=False)
            pred_  = fitted_.get_forecast(steps=len(test_))
            pm_    = pred_.predicted_mean.clip(0, 100)
            pc_    = pred_.conf_int(alpha=0.10)
            sarima_models[villa] = {
                **mi, "model": fitted_,
                "train": train_, "test": test_, "monthly": monthly,
                "pred_mean": pm_, "pred_ci": pc_,
                "rmse": compute_rmse(test_.values, pm_.values),
                "mape": compute_mape(test_.values, pm_.values),
                "color": villa_cfg.get(villa, {}).get("color", "#2563EB"),
                "title": villa.replace("_villas", "").title(),
            }
        except Exception:
            pass

    fore_info_all = {}
    for villa, info in sarima_models.items():
        try:
            fore_info_all[villa] = make_forecast(info)
        except Exception:
            pass

    st.info("📊 Halaman Strategi — sambungkan page_strategi() dari app.py original.")


if __name__ == "__main__":
    main()