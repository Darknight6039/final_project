# ----------------------------------------------
# Streamlit one‑page “Live Financial News Sentiment”
# ----------------------------------------------
import os
import time
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from streamlit_autorefresh import st_autorefresh
from urllib.parse import quote_plus
from datetime import datetime

# ──────────────────────────────────────────────
# 0 · Page config & ENV / constants
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Live Financial Sentiment",
    page_icon="📈",
    layout="wide",
)

load_dotenv()
NEWS_KEY = os.getenv("NEWSAPI_KEY", "")
if not NEWS_KEY:
    st.error("❌ NEWSAPI_KEY not set in .env — please add and restart.")
    st.stop()

# confirm key loaded
st.write("🔑 Loaded NewsAPI key:", NEWS_KEY[:4] + "…")

MODEL_PATH = "best_model"
DEFAULT_REFRESH_SEC = 600    # 10 min → stays under 100/day
ROLLING_WINDOW_MIN = None    # not used any more
SENT_POS_TH = 0.40           # still used for KPI % pos/neg
SENT_NEG_TH = -0.40

# ──────────────────────────────────────────────
# 1 · Model loader
# ──────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    try:
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(
            "finbert_sentiment_model", local_files_only=True
        )
    mdl = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device).eval()
    return tok, mdl, device

tokenizer, model, device = load_model(MODEL_PATH)

# ──────────────────────────────────────────────
# 2 · Headline fetcher (NewsAPI)
# ──────────────────────────────────────────────
COMMON_WORDS = {
    "THE","AND","FOR","WITH","FROM","THIS","THAT",
    "WILL","HAVE","YOU","YOUR","SAID"
}
def _extract_ticker(title: str) -> str:
    for token in re.findall(r"\b[A-Z]{2,5}\b", title):
        if token not in COMMON_WORDS:
            return token
    return "NEWS"

def fetch_headlines(
    query: str = "stocks OR earnings",
    page_size: int = 100,
    page: int = 1,
) -> pd.DataFrame:
    if page_size > 100:
        page_size = 100
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={quote_plus(query)}"
        "&language=en"
        "&sortBy=publishedAt"
        f"&pageSize={page_size}"
        f"&page={page}"
        f"&apiKey={NEWS_KEY}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    raw = resp.json().get("articles", [])
    st.write(f"🔄 Fetched {len(raw)} articles from NewsAPI")  # debug
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["publishedDate"] = pd.to_datetime(df["publishedAt"], utc=True)
    df["symbol"] = df["title"].apply(_extract_ticker)
    return df[["symbol","publishedDate","title","url"]]

# ──────────────────────────────────────────────
# 3 · Scoring helpers
# ──────────────────────────────────────────────
def score_batch(texts: list[str]) -> np.ndarray:
    enc = tokenizer(
        texts,padding=True,truncation=True,
        max_length=64,return_tensors="pt"
    )
    enc = {k:v.to(device) for k,v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits,dim=-1).cpu().numpy()
    return probs

def score_headlines(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    probs = score_batch(df["title"].tolist())
    df["p_neg"],df["p_neu"],df["p_pos"] = probs.T
    df["sentiment"] = df["p_pos"] - df["p_neg"]
    return df

# ──────────────────────────────────────────────
# 4 · Streamlit layout & state
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    query = st.text_input("NewsAPI query", "stocks OR earnings")
    refresh_sec = st.slider(
        "Refresh interval (sec)",
        min_value=300, max_value=1800,
        value=DEFAULT_REFRESH_SEC, step=300
    )
    st.markdown(f"*Calls/day* ≈ `{int(86400/refresh_sec)}` ≤ 100")

st_autorefresh(interval=refresh_sec*1000, key="tick")

# history store (no pruning)
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# fetch & score
try:
    latest = fetch_headlines(query=query)
except requests.HTTPError as e:
    st.warning(f"📛 API error: {e}. Pausing 30 min.")
    time.sleep(1800)
    st.experimental_rerun()

if latest.empty:
    st.warning(f"No articles for query “{query}”. Try a broader term.")

scored = score_headlines(latest)

# accumulate history
hist = pd.concat([scored, st.session_state["history"]], ignore_index=True)
hist.drop_duplicates(subset=["url"], inplace=True)
hist.sort_values("publishedDate", ascending=False, inplace=True)
st.session_state["history"] = hist

# ──────────────────────────────────────────────
# 5 · Dashboard UI
# ──────────────────────────────────────────────
st.title("📈 Live Financial News Sentiment")
st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

# 5.1: KPIs
avg_sent = hist["sentiment"].mean() if not hist.empty else 0.0
n_head = len(hist)
pct_pos = (hist["sentiment"]>SENT_POS_TH).mean()*100 if n_head else 0.0
pct_neg = (hist["sentiment"]<SENT_NEG_TH).mean()*100 if n_head else 0.0
c1,c2,c3 = st.columns(3)
c1.metric("Average sentiment", f"{avg_sent:+.2f}")
c2.metric("Headlines (total)", str(n_head))
c3.metric("% Pos/Neg", f"{pct_pos:.0f}% / {pct_neg:.0f}%")

# 5.2: Sparkline (over entire history)
if not hist.empty:
    spark = (
        hist.set_index("publishedDate")
            .sort_index()["sentiment"]
            .expanding()
            .mean()
    )
    st.line_chart(spark, height=200, use_container_width=True)

# 5.3: Top movers (unconditional)
def movers(df, bullish=True):
    grp = df.groupby("symbol")["sentiment"].mean()
    sorted_ = grp.sort_values(ascending=not bullish).head(5)
    return sorted_.reset_index().rename(columns={"sentiment":"avg_score"})

bull = movers(hist, True)
bear = movers(hist, False)

col4,col5 = st.columns(2)
col4.subheader("🚀 Top 5 bullish symbols")
col4.dataframe(bull, use_container_width=True, hide_index=True)
col5.subheader("🔻 Top 5 bearish symbols")
col5.dataframe(bear, use_container_width=True, hide_index=True)

# 5.4: Raw feed
with st.expander("Raw headline feed"):
    st.dataframe(
        hist[["publishedDate","symbol","sentiment","title"]],
        use_container_width=True, hide_index=True
    )
