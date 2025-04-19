# ----------------------------------------------
# Streamlit one‑page “Live Financial Sentiment” 
# ----------------------------------------------
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
from streamlit_autorefresh import st_autorefresh

# ──────────────────────────────────────────────
# 0 · ENV / constants
# ──────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("FMP_API_KEY", "")
if not API_KEY:
    st.stop()  # hard‑fail early; user must set key

MODEL_PATH = "best_model"          # weights + copied FinBERT tokenizer files
REQUEST_LIMIT_PER_HOUR = 150       # stay below free‑tier 250
DEFAULT_REFRESH_SEC = 25           # ~144 calls / h
ROLLING_WINDOW_MIN = 60            # keep last 60 min
SENT_POS_TH = 0.40
SENT_NEG_TH = -0.40

# ──────────────────────────────────────────────
# 1 · Model loader
# ──────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    """Load fine‑tuned BERT model & tokenizer once."""
    try:
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception:
        # Fallback to FinBERT tokenizer folder if user mis‑copied
        tok = AutoTokenizer.from_pretrained("finbert_sentiment_model",
                                            local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device).eval()
    return tok, mdl, device


tokenizer, model, device = load_model(MODEL_PATH)

# ──────────────────────────────────────────────
# 2 · Helpers
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_headlines(limit: int = 50) -> pd.DataFrame:
    """GET latest headlines from FMP."""
    url = (
        "https://financialmodelingprep.com/api/v3/stock_news"
        f"?limit={limit}&apikey={API_KEY}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["publishedDate"] = pd.to_datetime(df["publishedDate"])
    return df[["symbol", "publishedDate", "title", "url"]]


def score_batch(texts: list[str]) -> np.ndarray:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    # probs[:, 0]=neg, 1=neu, 2=pos
    return probs


def score_headlines(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    probs = score_batch(df["title"].tolist())
    df["p_neg"], df["p_neu"], df["p_pos"] = probs.T
    df["sentiment"] = df["p_pos"] - df["p_neg"]
    return df


def prune_history(df: pd.DataFrame, window_min: int) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_min)
    return df[df["publishedDate"] >= cutoff].copy()


# ──────────────────────────────────────────────
# 3 · Streamlit layout
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Live Financial Sentiment",
    page_icon="📈",
    layout="wide",
)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    refresh_sec = st.slider(
        "Refresh interval (sec)",
        min_value=15,
        max_value=60,
        value=DEFAULT_REFRESH_SEC,
        step=5,
    )
    window_min = st.slider(
        "Rolling window (minutes)",
        min_value=15,
        max_value=120,
        value=ROLLING_WINDOW_MIN,
        step=15,
    )
    st.markdown(
        f"*API calls/hour* ≈ `{int(3600 / refresh_sec)}` "
        f"  (≤ {REQUEST_LIMIT_PER_HOUR} target)"
    )

# Auto‑refresh
st_autorefresh(interval=refresh_sec * 1000, key="tick")

# Initialise session‑level history store
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# 3.1 · Fetch & score new headlines
try:
    latest = fetch_headlines(limit=50)
except requests.HTTPError as e:
    st.warning(f"📛 API error: {e}. Pausing for 2 min.")
    time.sleep(120)
    st.experimental_rerun()

scored = score_headlines(latest)

# 3.2 · Merge into history & prune
hist = pd.concat([scored, st.session_state["history"]])
hist.drop_duplicates(subset=["url"], inplace=True)
hist.sort_values("publishedDate", ascending=False, inplace=True)
hist = prune_history(hist, window_min)
st.session_state["history"] = hist

# ──────────────────────────────────────────────
# 4 · Dashboard UI
# ──────────────────────────────────────────────
st.title("📈 Live Financial News Sentiment")
st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')} (local)")

# 4.1 · KPI strip
avg_sent = hist["sentiment"].mean() if not hist.empty else 0.0
n_headlines = len(hist)
pct_pos = (hist["sentiment"] > SENT_POS_TH).mean() * 100 if n_headlines else 0.0
pct_neg = (hist["sentiment"] < SENT_NEG_TH).mean() * 100 if n_headlines else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Average sentiment", f"{avg_sent:+.2f}")
col2.metric("Headlines (rolling)", str(n_headlines))
col3.metric("% Pos / Neg", f"{pct_pos:.0f}% / {pct_neg:.0f}%")

# 4.2 · Sparkline
if not hist.empty:
    spark_df = (
        hist.set_index("publishedDate")
        .sort_index()
        .rolling("10min")["sentiment"]
        .mean()
        .dropna()
    )
    st.line_chart(spark_df, height=200, use_container_width=True)

# 4.3 · Top movers tables
def movers(df: pd.DataFrame, bullish: bool = True):
    mask = df["sentiment"] >= SENT_POS_TH if bullish else df["sentiment"] <= SENT_NEG_TH
    tbl = (
        df[mask]
        .groupby("symbol")["sentiment"]
        .mean()
        .sort_values(ascending=not bullish)
        .head(5)
        .reset_index()
        .rename(columns={"sentiment": "avg_score"})
    )
    return tbl


bullish_tbl = movers(hist, bullish=True)
bearish_tbl = movers(hist, bullish=False)

col4, col5 = st.columns(2)
col4.subheader("🚀 Top 5 bullish tickers")
col4.dataframe(bullish_tbl, use_container_width=True, hide_index=True)
col5.subheader("🔻 Top 5 bearish tickers")
col5.dataframe(bearish_tbl, use_container_width=True, hide_index=True)

# 4.4 · Raw feed
with st.expander("Raw headline feed"):
    st.dataframe(
        hist[["publishedDate", "symbol", "sentiment", "title"]],
        use_container_width=True,
        hide_index=True,
    )
