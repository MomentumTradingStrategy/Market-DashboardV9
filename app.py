from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Market Overview Dashboard", layout="wide")

BENCHMARK = "SPY"
PRICE_HISTORY_PERIOD = "2y"


def _asof_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# =========================
# CSS
# =========================
CSS = """
<style>
.block-container {max-width: 1750px; padding-top: 1.0rem; padding-bottom: 2rem;}
.section-title {font-weight: 900; font-size: 1.15rem; margin: 0.65rem 0 0.4rem 0;}
.small-muted {opacity: 0.75; font-size: 0.9rem;}
.hr {border-top: 1px solid rgba(255,255,255,0.12); margin: 14px 0;}
.card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
  padding: 12px 14px;
  margin-bottom: 12px;
}
.card h3{margin:0 0 8px 0; font-size: 1.02rem; font-weight: 950;}
.card .hint{opacity:0.72; font-size:0.88rem; margin-top:-2px; margin-bottom:10px;}

.badge{
  display:inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 0.78rem;
  letter-spacing: 0.2px;
  border: 1px solid rgba(255,255,255,0.12);
}
.badge-yes{background: rgba(124,252,154,0.15); color:#7CFC9A;}
.badge-no{background: rgba(255,107,107,0.12); color:#FF6B6B;}
.badge-neutral{background: rgba(255,200,60,0.12); color: rgba(255,200,60,0.98);}

.pill{
  display:inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-weight: 950;
  font-size: 0.82rem;
  border: 1px solid rgba(255,255,255,0.12);
}
.pill-red{background: rgba(255,80,80,0.16); color:#FF6B6B;}
.pill-amber{background: rgba(255,200,60,0.16); color: rgba(255,200,60,0.98);}
.pill-green{background: rgba(80,255,120,0.16); color:#7CFC9A;}

.metric-grid{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px 14px;
}
.metric-row{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 10px;
  padding: 6px 10px;
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.02);
}
.metric-row .k{opacity:0.85; font-weight:800;}
.metric-row .v{font-weight:950;}

.pl-table-wrap {border-radius: 10px; overflow: hidden; border: 1px solid rgba(255,255,255,0.10);}
table.pl-table {border-collapse: collapse; width: 100%; font-size: 13px;}
table.pl-table thead th {
  position: sticky; top: 0;
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.92);
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.12);
  font-weight: 900;
}
table.pl-table tbody td{
  padding: 7px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  vertical-align: middle;
}
td.mono {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
td.ticker {font-weight: 900;}
td.name {white-space: normal; line-height: 1.15;}
tr.group-row td{
  background: rgba(0,0,0,0.70) !important;
  border-bottom: 1px solid rgba(255,255,255,0.10);
}
tr.group-row td:not(.name){
  color: rgba(0,0,0,0) !important;
}
tr.group-row td.name{
  color: #FFFFFF !important;
  font-weight: 950 !important;
  letter-spacing: 0.2px;
}
.kv{
  display:flex;
  align-items:baseline;
  justify-content:space-between;
  gap: 10px;
  padding: 7px 10px;
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.02);
  margin-bottom: 8px;
}
.kv .k{opacity:0.82; font-weight:850;}
.kv .v{font-weight:950;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# ETF / MACRO TICKERS (1 per line)
# ============================================================
TICKERS_RAW = r"""
SPY
QQQ
DIA
IWM
RSP
QQQE
EDOW
MDY
IWN
IWO
XLC
XLY
XLP
XLE
XLF
XLV
XLI
XLB
XLRE
XLK
XLU
SOXX
SMH
XSD
IGV
XSW
IGM
VGT
XT
CIBR
BOTZ
AIQ
XTL
VOX
FCOM
FDN
SOCL
XRT
IBUY
CARZ
IDRV
ITB
XHB
PEJ
VDC
FSTA
KXI
PBJ
VPU
FUTY
IDU
IYE
VDE
XOP
IEO
OIH
IXC
IBB
XBI
PBE
IDNA
IHI
XHE
XHS
XPH
FHLC
PINK
KBE
KRE
IAT
KIE
IAI
KCE
IYG
VFH
ITA
PPA
XAR
IYT
XTN
VIS
FIDU
XME
GDX
SIL
SLX
PICK
VAW
VNQ
IYR
REET
SRVR
HOMZ
SCHH
NETL
GLD
SLV
UNG
USO
DBA
CORN
DBB
PALL
URA
UGA
CPER
COW
SOYB
WEAT
DBC
IEMG
VGK
FEZ
EWQ
DAX
FXI
EEM
EWJ
EWU
EWZ
EWG
EWT
EWH
EWI
EWW
PIN
IDX
EWY
EWA
EWM
EWS
EWC
EWP
EZA
EWL
UUP
FXE
FXY
FXB
FXA
FXF
FXC
IBIT
ETHA
XRP
SOLZ
GLNK
TLT
BND
SHY
IEF
SGOV
IEI
TLH
AGG
MUB
GOVT
IGSB
USHY
IGIB
""".strip()


def parse_ticker_list(raw: str) -> list[str]:
    out = []
    for ln in raw.splitlines():
        t = ln.strip().upper()
        if t:
            out.append(t)
    seen = set()
    uniq = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq


ALL_TICKERS = parse_ticker_list(TICKERS_RAW)
ALL_TICKERS_SET = set(ALL_TICKERS)

MAJOR = ALL_TICKERS[:10]
SECTORS = ALL_TICKERS[10:21]

# ===========================
# SUB-SECTOR / INDUSTRY GROUP MAP
# ===========================
SUBSECTOR_LEFT = {
    "Semiconductors": ["SOXX", "SMH", "XSD"],
    "Software / Cloud / Broad Tech": ["IGV", "XSW", "IGM", "VGT", "XT"],
    "Cyber Security": ["CIBR"],
    "AI / Robotics / Automation": ["BOTZ", "AIQ"],
    "Telecom & Communication": ["XTL", "VOX", "FCOM"],
    "Internet / Media / Social": ["FDN", "SOCL"],
    "Retail": ["XRT", "IBUY"],
    "Autos / EV": ["IDRV", "CARZ"],
    "Homebuilders / Construction": ["ITB", "XHB"],
    "Leisure & Entertainment": ["PEJ"],
    "Consumer Staples": ["VDC", "FSTA", "KXI", "PBJ"],
    "Utilities": ["VPU", "FUTY", "IDU"],
    "Energy": ["IYE", "VDE"],
    "Exploration & Production": ["XOP", "IEO"],
    "Oil Services": ["OIH"],
    "Global Energy": ["IXC"],
}
SUBSECTOR_RIGHT = {
    "Biotechnology / Genomics": ["IBB", "XBI", "PBE", "IDNA"],
    "Medical Equipment": ["IHI", "XHE"],
    "Health Care Providers / Services": ["XHS"],
    "Pharmaceuticals": ["XPH"],
    "Broad / Alternative Health": ["FHLC", "PINK"],
    "Banks": ["KBE", "KRE", "IAT"],
    "Insurance": ["KIE"],
    "Capital Markets / Brokerage": ["IAI", "KCE"],
    "Diversified Financial Services": ["IYG"],
    "Broad Financials": ["VFH"],
    "Aerospace & Defense": ["ITA", "PPA", "XAR"],
    "Transportation": ["IYT", "XTN"],
    "Broad Industrials": ["VIS", "FIDU"],
    "Materials": ["XME", "GDX", "SIL", "SLX", "PICK", "VAW"],
    "Real Estate": ["VNQ", "IYR", "REET"],
    "Specialty REITs": ["SRVR", "HOMZ", "SCHH", "NETL"],
}
SUBSECTOR_ALL = {}
SUBSECTOR_ALL.update(SUBSECTOR_LEFT)
SUBSECTOR_ALL.update(SUBSECTOR_RIGHT)

# ===========================
# Macro Assets Map
# ===========================
MACRO_ASSETS = {
    "Commodities": ["GLD", "SLV", "UNG", "USO", "DBA", "CORN", "DBB", "PALL", "URA", "UGA", "CPER", "COW", "SOYB", "WEAT", "DBC"],
    "Foreign Markets": ["IEMG", "VGK", "FEZ", "EWQ", "DAX", "EWU", "FXI", "EEM", "EWJ", "EWZ", "EWG", "EWT", "EWH", "EWI", "EWW", "PIN", "IDX", "EWY", "EWA", "EWM", "EWS", "EWC", "EWP", "EZA", "EWL"],
    "Currencies": ["UUP", "FXE", "FXY", "FXB", "FXA", "FXF", "FXC"],
    "Crypto": ["IBIT", "ETHA", "XRP", "SOLZ", "GLNK"],
    "Treasuries / Bonds": ["TLT", "BND", "SHY", "IEF", "SGOV", "IEI", "TLH", "AGG", "MUB", "GOVT", "IGSB", "USHY", "IGIB"],
}

# ===========================
# Stocks (LOCKED) + Watch List (EDITABLE)
# ===========================
STOCKS_LOCKED_RAW = r"""
NVDA, AAPL, GOOG, MSFT, AMZN, META, AVGO, TSLA, TSM, BRKA.A, LLY, WMT, JPM, V, ORCL, MA, XOM, JNJ, PLTR, BAC, NFLX, ABBV, COST, MU, BABA, AMD, HD, GE, PG, UNH, CVX, WFC, MS, CSCO, KO, CAT, GS, TM, IBM, MRK, AXP, LRCX, RTX, NVO, CRM, PM, AMAT, TMO, ABT, TMUS, SHOP, C, MCD, SHEL, ISRG, APP, DIS, LIN, BX, QCOM, INTC, PEP, GEV, SCHW, KLAC, BLK, INTU, BA, AMGN, UBER, TXN, APH, T, BKNG, TJX, VZ, ACN, NEE, DHR, ANET, COF, SPGI, SONY, NOW, GILD, BSX, GILD, PFE, ADI, ADBE, SYK, UNP, UL, LOW, TTE, DE, SCCO, HON, PANW, PGR, IBKR, BN, CB, ETN, MDT, LMT, COP, NEM, VRTX, SPOT, PH, CRWDMELI, BMY, CEG, HCA, HOOD, ADP, CVS, MCK, SBUX
""".strip()

WATCHLIST_DEFAULT = r"""
SNDK
WDC
STX
HL
LSCC
LRCX
ADI
AEO
CIEN
AU
""".strip()

TICKER_ALIASES = {
    "BRKA.A": "BRK-A",
    "BRKB.B": "BRK-B",
    "BRK.A": "BRK-A",
    "BRK.B": "BRK-B",
}


def parse_csv_or_lines(raw: str) -> list[str]:
    raw = (raw or "").upper()
    raw = raw.replace("CRWDMELI", "CRWD, MELI")

    parts = []
    for chunk in raw.replace("\n", ",").split(","):
        t = chunk.strip().upper()
        if not t:
            continue
        t = TICKER_ALIASES.get(t, t)
        parts.append(t)

    seen = set()
    out = []
    for t in parts:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


STOCKS_LOCKED = parse_csv_or_lines(STOCKS_LOCKED_RAW)


def init_watchlist_state():
    if "watchlist_text" not in st.session_state:
        st.session_state.watchlist_text = WATCHLIST_DEFAULT


def get_watchlist_tickers() -> list[str]:
    init_watchlist_state()
    return parse_csv_or_lines(st.session_state.watchlist_text)


# -----------------------------
# Data pulls
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_prices(tickers, period=PRICE_HISTORY_PERIOD):
    df = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError("No data returned from price source.")

    if isinstance(df.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            if (t, "Close") in df.columns:
                closes[t] = df[(t, "Close")]
        close_df = pd.DataFrame(closes)
    else:
        close_df = pd.DataFrame({tickers[0]: df["Close"]})

    return close_df.dropna(how="all").ffill()


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def fetch_names(tickers: list[str]) -> dict[str, str]:
    names = {t: t for t in tickers}
    for t in tickers:
        try:
            inf = yf.Ticker(t).info
            n = inf.get("shortName") or inf.get("longName")
            if n:
                names[t] = str(n)
        except Exception:
            pass
    names["SPY"] = "S&P 500"
    names["QQQ"] = "Nasdaq-100"
    names["DIA"] = "Dow"
    names["IWM"] = "Russell 2000"
    names["RSP"] = "S&P 500 EW"
    return names


# -----------------------------
# Sparkline
# -----------------------------
SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline_from_series(s: pd.Series, n=26):
    s = s.dropna().tail(n)
    if s.empty:
        return "", []
    if s.nunique() == 1:
        mid = len(SPARK_CHARS) // 2
        return (SPARK_CHARS[mid] * len(s)), ([mid] * len(s))

    lo, hi = float(s.min()), float(s.max())
    if hi - lo <= 1e-12:
        return "", []
    scaled = (s - lo) / (hi - lo)
    idx = (scaled * (len(SPARK_CHARS) - 1)).round().astype(int).clip(0, len(SPARK_CHARS) - 1)
    levels = idx.tolist()
    spark = "".join(SPARK_CHARS[i] for i in levels)
    return spark, levels


def _ret(close: pd.Series, periods: int):
    return close.pct_change(periods=periods)


def _ratio_rs(close_t: pd.Series, close_b: pd.Series, periods: int):
    t = close_t / close_t.shift(periods)
    b = close_b / close_b.shift(periods)
    return (t / b) - 1


# -----------------------------
# Metrics / Table
# -----------------------------
def build_table(p: pd.DataFrame, tickers: list[str], name_map: dict[str, str]) -> pd.DataFrame:
    horizons_ret = {"% 1D": 1, "% 1W": 5, "% 1M": 21, "% 3M": 63, "% 6M": 126, "% 1Y": 252}
    horizons_rs = {"RS 1W": 5, "RS 1M": 21, "RS 3M": 63, "RS 6M": 126, "RS 1Y": 252}

    b = p[BENCHMARK]
    rows = []

    for t in tickers:
        if t not in p.columns:
            continue

        close = p[t]
        last_price = float(close.dropna().iloc[-1]) if close.dropna().shape[0] else np.nan

        rs_ratio_series = (close / close.shift(21)) / (b / b.shift(21))
        spark, levels = sparkline_from_series(rs_ratio_series, n=26)

        rec = {
            "Ticker": t,
            "Name": name_map.get(t, t),
            "Price": last_price,
            "Relative Strength 1M": spark,
            "__spark_levels": levels,
            "__is_header": False,
        }

        for col, n in horizons_rs.items():
            rr = _ratio_rs(close, b, n)
            rec[col] = float(rr.dropna().iloc[-1]) if rr.dropna().shape[0] else np.nan

        for col, n in horizons_ret.items():
            r = _ret(close, n)
            rec[col] = float(r.dropna().iloc[-1]) if r.dropna().shape[0] else np.nan

        rows.append(rec)

    df = pd.DataFrame(rows)

    for col in horizons_rs.keys():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            df[col] = (s.rank(pct=True) * 99).round().clip(1, 99)

    return df


def grouped_block(groups: dict[str, list[str]], df_by_ticker: dict[str, dict]) -> pd.DataFrame:
    out_rows = []
    for group_name, ticks in groups.items():
        out_rows.append(
            {
                "Ticker": "",
                "Name": group_name,
                "Price": "",
                "Relative Strength 1M": "",
                "RS 1W": "",
                "RS 1M": "",
                "RS 3M": "",
                "RS 6M": "",
                "RS 1Y": "",
                "% 1D": "",
                "% 1W": "",
                "% 1M": "",
                "% 3M": "",
                "% 6M": "",
                "% 1Y": "",
                "__spark_levels": [],
                "__is_header": True,
            }
        )
        for t in ticks:
            if t in df_by_ticker:
                out_rows.append(df_by_ticker[t])
    return pd.DataFrame(out_rows)


# -----------------------------
# HTML rendering
# -----------------------------
def rs_bg(v):
    try:
        v = float(v)
    except:
        return ""
    if np.isnan(v):
        return ""
    x = (v - 1) / 98.0
    if x < 0.5:
        r = 255
        g = int(80 + (x / 0.5) * (180 - 80))
    else:
        r = int(255 - ((x - 0.5) / 0.5) * (255 - 40))
        g = 200
    b = 60
    return f"background-color: rgb({r},{g},{b}); color:#0B0B0B; font-weight:900; border-radius:6px; padding:2px 6px; display:inline-block; min-width:32px; text-align:center;"


def pct_style(v):
    try:
        v = float(v)
    except:
        return ""
    if np.isnan(v):
        return ""
    if v > 0:
        return "color:#7CFC9A; font-weight:800;"
    if v < 0:
        return "color:#FF6B6B; font-weight:800;"
    return "opacity:0.9; font-weight:700;"


def fmt_price(v):
    if v == "" or v is None:
        return ""
    try:
        return f"${float(v):,.2f}"
    except:
        return ""


def fmt_pct(v):
    if v == "" or v is None:
        return ""
    try:
        return f"{float(v):.2%}"
    except:
        return ""


def fmt_rs(v):
    if v == "" or v is None:
        return ""
    try:
        return f"{float(v):.0f}"
    except:
        return ""


def spark_html(spark: str, levels: list[int]):
    if not spark or not levels or len(spark) != len(levels):
        return ""

    def level_to_rgb(lv: int):
        t = lv / 7.0
        if t <= 0.5:
            k = t / 0.5
            r1, g1, b1 = 255, 80, 80
            r2, g2, b2 = 255, 200, 60
            r = int(r1 + (r2 - r1) * k)
            g = int(g1 + (g2 - g1) * k)
            b = int(b1 + (b2 - b1) * k)
        else:
            k = (t - 0.5) / 0.5
            r1, g1, b1 = 255, 200, 60
            r2, g2, b2 = 80, 255, 120
            r = int(r1 + (r2 - r1) * k)
            g = int(g1 + (g2 - g1) * k)
            b = int(b1 + (b2 - b1) * k)
        return r, g, b

    spans = []
    for ch, lv in zip(spark, levels):
        r, g, b = level_to_rgb(int(lv))
        spans.append(f'<span style="color: rgb({r},{g},{b}); font-weight:900;">{ch}</span>')
    return "".join(spans)


def render_table_html(df: pd.DataFrame, columns: list[str], height_px: int = 360):
    th = "".join([f"<th>{c}</th>" for c in columns])

    trs = []
    for _, row in df.iterrows():
        is_header = bool(row.get("__is_header", False))
        tr_class = "group-row" if is_header else ""

        tds = []
        for c in columns:
            val = row.get(c, "")
            td_class = ""
            if c == "Ticker":
                td_class = "ticker"
            elif c == "Name":
                td_class = "name"
            elif c == "Relative Strength 1M":
                td_class = "mono"

            if is_header:
                cell_html = str(val) if c == "Name" else ""
            else:
                if c == "Price":
                    cell_html = fmt_price(val)
                elif c.startswith("% "):
                    txt = fmt_pct(val)
                    stl = pct_style(val)
                    cell_html = f'<span style="{stl}">{txt}</span>' if stl and txt != "" else txt
                elif c.startswith("RS "):
                    txt = fmt_rs(val)
                    stl = rs_bg(val)
                    cell_html = f'<span style="{stl}">{txt}</span>' if stl and txt != "" else txt
                elif c == "Relative Strength 1M":
                    cell_html = spark_html(str(val), row.get("__spark_levels", []))
                else:
                    cell_html = "" if (val is None or (isinstance(val, float) and np.isnan(val))) else str(val)

            tds.append(f'<td class="{td_class}">{cell_html}</td>')

        trs.append(f'<tr class="{tr_class}">' + "".join(tds) + "</tr>")

    table = f"""
    <div class="pl-table-wrap" style="max-height:{height_px}px; overflow:auto;">
      <table class="pl-table">
        <thead><tr>{th}</tr></thead>
        <tbody>
          {''.join(trs)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(table, unsafe_allow_html=True)


# ============================================================
# VERSION 9: LOCKED MANUAL INPUTS (EDIT WEEKLY IN CODE ONLY)
# ============================================================
# ✅ Search for: "EDIT WEEKLY HERE"
# ============================================================

# --- EDIT WEEKLY HERE ---
MANUAL_ASOF_LABEL = "Manual inputs updated: 2026-01-09"  # <-- change this weekly if you want

MANUAL_INPUTS = {
    "Stock Market Exposure": {"Exposure": "40-60%"},
    "Market Type": {"Type": "Bull Quiet"},
    "Trend Condition (QQQ)": {
        "Above 5DMA": "Yes",
        "Above 10DMA": "Yes",
        "Above 20DMA": "Yes",
        "Above 50DMA": "Yes",
        "Above 200DMA": "No",
    },
    "Nasdaq Net 52-Week New High/Low": {"Daily": 231, "Weekly": 811, "Monthly": -828},
    "Market Indicators": {
        "VIX": 16.34,
        "PCC": 0.67,
        "Credit (IEI vs HYG)": "Aligned",
        "U.S. Dollar": "Downtrend",
        "DXY Price": 103.25,
        "Distribution Days": 2,
        "Up/Down Volume (Daily)": 2.36,
        "Up/Down Volume (Weekly)": 2.10,
        "Up/Down Volume (Monthly)": 1.80,
        "A/D Ratio (Daily)": 2.20,
        "A/D Ratio (Weekly)": 1.95,
        "A/D Ratio (Monthly)": 1.70,
    },
    "Macro": {"Fed Funds": 4.09, "M2 Money": 22.2, "10yr": 4.02},
    "Breadth & Participation": {
        "% Price Above 10DMA": 56,
        "% Price Above 20DMA": 49,
        "% Price Above 50DMA": 58,
        "% Price Above 200DMA": 68,
    },
    "Composite Model": {"Monetary Policy": 1.0, "Liquidity Flow": 2.0, "Rates & Credit": 2.0, "Tape Strength": 2.0, "Sentiment": 1.0},
    "Hot Sectors / Industry Groups": {"Notes": ""},  # <- removed "Type here..."
    "Market Correlations": {"Correlated": "Dow, Nasdaq", "Uncorrelated": "Dollar, Bonds"},
}
# --- EDIT WEEKLY HERE ---


EXPOSURE_PILL = {
    "0-20%": "pill pill-red",
    "20-40%": "pill pill-amber",
    "40-60%": "pill pill-green",
    "60-80%": "pill pill-green",
    "80-100%": "pill pill-green",
}


def _yesno_badge(v: str) -> str:
    v = str(v).strip().lower()
    if v in ["yes", "y", "true", "1"]:
        return '<span class="badge badge-yes">YES</span>'
    if v in ["no", "n", "false", "0"]:
        return '<span class="badge badge-no">NO</span>'
    return '<span class="badge badge-neutral">—</span>'


def _num_color(v):
    try:
        v = float(v)
    except:
        return "opacity:0.85; font-weight:950;"
    if v > 0:
        return "color:#7CFC9A; font-weight:950;"
    if v < 0:
        return "color:#FF6B6B; font-weight:950;"
    return "opacity:0.85; font-weight:950;"


def _score_to_label(score: float):
    try:
        score = float(score)
    except:
        score = 1.0
    if score <= 0.5:
        return ("Bad", "badge badge-no")
    if score < 1.5:
        return ("Neutral", "badge badge-neutral")
    return ("Good", "badge badge-yes")


def _total_score_pill(total: float) -> str:
    try:
        total = float(total)
    except:
        total = 0.0
    if total >= 7.0:
        return "pill pill-green"
    if total >= 5.0:
        return "pill pill-amber"
    return "pill pill-red"


def _kv(label: str, value_html: str):
    st.markdown(f'<div class="kv"><div class="k">{label}</div><div class="v">{value_html}</div></div>', unsafe_allow_html=True)


def _pill_for_market_type(mt: str) -> str:
    mt = (mt or "").strip().lower()
    if mt in ["bull quiet", "bull volatile"]:
        return "pill pill-green"
    if mt in ["bear quiet", "bear volatile"]:
        return "pill pill-red"
    if mt in ["sideways quiet", "sideways volatile"]:
        return "pill pill-amber"
    return "pill pill-amber"


def _pill_for_credit(val: str) -> str:
    v = (val or "").strip().lower()
    if v == "aligned":
        return "pill pill-green"
    if v in ["divergent", "divergence"]:
        return "pill pill-red"
    return "pill pill-amber"


def _pill_for_dollar(val: str) -> str:
    v = (val or "").strip().lower()
    if v == "downtrend":
        return "pill pill-green"
    if v == "uptrend":
        return "pill pill-red"
    if v == "sideways":
        return "pill pill-amber"
    return "pill pill-amber"


def render_manual_inputs_locked(mi: dict):
    # Title change + remove "Manual Inputs / Locked..." text
    st.markdown('<div class="card"><h3>Big Picture Market Pulse Dashboard</h3></div>', unsafe_allow_html=True)
    st.caption(MANUAL_ASOF_LABEL)

    c1, c2, c3 = st.columns([1.05, 1.15, 1.15])

    with c1:
        st.markdown('<div class="card"><h3>Stock Market Exposure</h3>', unsafe_allow_html=True)
        ex = str(mi.get("Stock Market Exposure", {}).get("Exposure", "")).strip()
        pill_class = EXPOSURE_PILL.get(ex, "pill pill-amber")
        st.markdown(f'Current: <span class="{pill_class}">{ex or "—"}</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>Market Type</h3>', unsafe_allow_html=True)
        mt = str(mi.get("Market Type", {}).get("Type", "")).strip()
        _kv("Type", f'<span class="{_pill_for_market_type(mt)}">{mt or "—"}</span>')
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>Trend Condition (QQQ)</h3><div class="hint">Simple yes/no filters.</div>', unsafe_allow_html=True)
        tc = mi.get("Trend Condition (QQQ)", {}) or {}
        keys = ["Above 5DMA", "Above 10DMA", "Above 20DMA", "Above 50DMA", "Above 200DMA"]
        for k in keys:
            _kv(k, _yesno_badge(tc.get(k, "—")))
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><h3>Nasdaq Net 52-Week New High/Low</h3><div class="hint">Positive = green, negative = red.</div>', unsafe_allow_html=True)
        hl = mi.get("Nasdaq Net 52-Week New High/Low", {}) or {}
        d = hl.get("Daily", "")
        w = hl.get("Weekly", "")
        m = hl.get("Monthly", "")
        st.markdown(
            f"""
            <div class="metric-grid">
              <div class="metric-row"><div class="k">Daily</div><div class="v" style="{_num_color(d)}">{d}</div></div>
              <div class="metric-row"><div class="k">Weekly</div><div class="v" style="{_num_color(w)}">{w}</div></div>
              <div class="metric-row"><div class="k">Monthly</div><div class="v" style="{_num_color(m)}">{m}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>Market Indicators</h3><div class="hint">Your tape / sentiment inputs.</div>', unsafe_allow_html=True)
        ind = mi.get("Market Indicators", {}) or {}

        _kv("VIX", f'{ind.get("VIX","—")}')
        _kv("Put/Call (PCC)", f'{ind.get("PCC","—")}')

        credit_val = str(ind.get("Credit (IEI vs HYG)", "—"))
        _kv("Credit (IEI vs HYG)", f'<span class="{_pill_for_credit(credit_val)}">{credit_val}</span>')

        dollar_val = str(ind.get("U.S. Dollar", "—"))
        _kv("U.S. Dollar", f'<span class="{_pill_for_dollar(dollar_val)}">{dollar_val}</span>')

        _kv("DXY Price", f'{ind.get("DXY Price","—")}')
        _kv("Distribution Days", f'{ind.get("Distribution Days","—")}')

        st.markdown('<div class="small-muted" style="margin: 8px 0 6px 0;"><b>Up/Down Volume Ratio</b></div>', unsafe_allow_html=True)
        _kv("Daily", f'{ind.get("Up/Down Volume (Daily)","—")}')
        _kv("Weekly", f'{ind.get("Up/Down Volume (Weekly)","—")}')
        _kv("Monthly", f'{ind.get("Up/Down Volume (Monthly)","—")}')

        st.markdown('<div class="small-muted" style="margin: 10px 0 6px 0;"><b>Advance/Decline Ratio</b></div>', unsafe_allow_html=True)
        _kv("Daily", f'{ind.get("A/D Ratio (Daily)","—")}')
        _kv("Weekly", f'{ind.get("A/D Ratio (Weekly)","—")}')
        _kv("Monthly", f'{ind.get("A/D Ratio (Monthly)","—")}')

        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card"><h3>Macro</h3><div class="hint">High-level backdrop.</div>', unsafe_allow_html=True)
        mac = mi.get("Macro", {}) or {}
        _kv("Fed Funds", f'{mac.get("Fed Funds","—")}')
        _kv("M2 Money", f'{mac.get("M2 Money","—")}')
        _kv("10yr", f'{mac.get("10yr","—")}')
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>Breadth & Participation</h3><div class="hint">Percent of names above key moving averages.</div>', unsafe_allow_html=True)
        br = mi.get("Breadth & Participation", {}) or {}
        _kv("% Price Above 10DMA", f'{br.get("% Price Above 10DMA","—")}%')
        _kv("% Price Above 20DMA", f'{br.get("% Price Above 20DMA","—")}%')
        _kv("% Price Above 50DMA", f'{br.get("% Price Above 50DMA","—")}%')
        _kv("% Price Above 200DMA", f'{br.get("% Price Above 200DMA","—")}%')
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>Composite Model</h3><div class="hint">0.0–2.0 each • Total out of 10.</div>', unsafe_allow_html=True)
        cm = mi.get("Composite Model", {}) or {}
        components = ["Monetary Policy", "Liquidity Flow", "Rates & Credit", "Tape Strength", "Sentiment"]

        total = 0.0
        for comp in components:
            v = float(cm.get(comp, 0.0) or 0.0)
            total += v
            lbl, cls = _score_to_label(v)
            if v <= 0.5:
                score_pill = "pill pill-red"
            elif v < 1.5:
                score_pill = "pill pill-amber"
            else:
                score_pill = "pill pill-green"
            _kv(comp, f'<span class="{cls}">{lbl.upper()}</span> <span class="{score_pill}">{v:.1f}</span>')

        total_pill = _total_score_pill(total)
        st.markdown(
            f'<div style="margin-top:10px;"><b>Total Score:</b> <span class="{total_pill}">{total:.1f}</span> <span class="small-muted">/ 10.0</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Hot sectors: remove hint line + remove "bar"
        st.markdown('<div class="card"><h3>Hot Sectors / Industry Groups</h3>', unsafe_allow_html=True)
        hs = mi.get("Hot Sectors / Industry Groups", {}) or {}
        notes = str(hs.get("Notes", "") or "").strip()
        st.markdown(notes if notes else "_(none)_")
        st.markdown("</div>", unsafe_allow_html=True)

        # Market correlations: remove hint line
        st.markdown('<div class="card"><h3>Market Correlations</h3>', unsafe_allow_html=True)
        mc = mi.get("Market Correlations", {}) or {}
        _kv("Correlated", f'{mc.get("Correlated","—")}')
        _kv("Uncorrelated", f'{mc.get("Uncorrelated","—")}')
        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# UI
# =========================
st.title("Market Overview Dashboard")
st.caption(f"As of: {_asof_ts()} • RS Benchmark: {BENCHMARK}")

with st.sidebar:
    st.subheader("Controls")
    if st.button("Refresh Data"):
        fetch_prices.clear()
        fetch_names.clear()
        st.rerun()

# Build the full pull list (ETFs + STOCKS + WATCHLIST + BENCHMARK)
watchlist_ticks = get_watchlist_tickers()
pull_list = list(dict.fromkeys(ALL_TICKERS + STOCKS_LOCKED + watchlist_ticks + [BENCHMARK]))

try:
    price_df = fetch_prices(pull_list, period=PRICE_HISTORY_PERIOD)
except Exception as e:
    st.error(f"Data pull failed: {e}")
    st.stop()

name_map = fetch_names(pull_list)

show_cols = [
    "Ticker",
    "Name",
    "Price",
    "Relative Strength 1M",
    "RS 1W",
    "RS 1M",
    "RS 3M",
    "RS 6M",
    "RS 1Y",
    "% 1D",
    "% 1W",
    "% 1M",
    "% 3M",
    "% 6M",
    "% 1Y",
]

# Major / Sectors / Sub-sectors
df_major = build_table(price_df, MAJOR, name_map)
df_sectors = build_table(price_df, SECTORS, name_map)

all_sub_ticks = []
for v in list(SUBSECTOR_ALL.values()):
    all_sub_ticks.extend(v)
all_sub_ticks = [t for t in all_sub_ticks if t in ALL_TICKERS_SET]

df_sub_master = build_table(price_df, all_sub_ticks, name_map)
df_by_ticker = {r["Ticker"]: r.to_dict() for _, r in df_sub_master.iterrows()}
df_sub_all = grouped_block(SUBSECTOR_ALL, df_by_ticker)

st.markdown('<div class="section-title">Major U.S. Indexes</div>', unsafe_allow_html=True)
render_table_html(df_major, show_cols, height_px=330)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">U.S. Sectors</div>', unsafe_allow_html=True)
render_table_html(df_sectors, show_cols, height_px=360)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">U.S. Sub-Sectors / Industry Groups</div>', unsafe_allow_html=True)
render_table_html(df_sub_all, show_cols, height_px=1100)

# Macro
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Commodities, Global Markets, Currencies, Crypto, Treasuries & Bonds</div>', unsafe_allow_html=True)

macro_ticks = []
for v in list(MACRO_ASSETS.values()):
    macro_ticks.extend(v)
macro_ticks = [t for t in macro_ticks if t in ALL_TICKERS_SET]

df_macro_master = build_table(price_df, macro_ticks, name_map)
df_macro_by_ticker = {r["Ticker"]: r.to_dict() for _, r in df_macro_master.iterrows()}
df_macro_all = grouped_block(MACRO_ASSETS, df_macro_by_ticker)

render_table_html(df_macro_all, show_cols, height_px=900)

# Stocks (LOCKED)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Mega & Large Cap Stocks ($100B+)</div>', unsafe_allow_html=True)
df_stocks = build_table(price_df, STOCKS_LOCKED, name_map)
render_table_html(df_stocks, show_cols, height_px=900)

# Watch List (EDITABLE)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Watch List</div>', unsafe_allow_html=True)

df_watch = build_table(price_df, watchlist_ticks, name_map)
render_table_html(df_watch, show_cols, height_px=420)

st.markdown(
    '<div class="card"><h3>Watch List (editable)</h3><div class="hint">Edit tickers (one per line). Click Apply to update the table.</div>',
    unsafe_allow_html=True,
)

init_watchlist_state()
new_text = st.text_area("Tickers", value=st.session_state.watchlist_text, height=140, label_visibility="collapsed")

c1, c2, c3 = st.columns([1, 1, 6])
with c1:
    if st.button("Apply", use_container_width=True):
        st.session_state.watchlist_text = new_text
        fetch_prices.clear()
        fetch_names.clear()
        st.rerun()
with c2:
    if st.button("Reset", use_container_width=True):
        st.session_state.watchlist_text = WATCHLIST_DEFAULT
        fetch_prices.clear()
        fetch_names.clear()
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# Big Picture Market Pulse Dashboard (Manual Inputs)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
render_manual_inputs_locked(MANUAL_INPUTS)
# Bottom explanation (UPDATED per your instructions)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

st.markdown(
    """
**How RS is Calculated:** RS = % change in (Symbol ÷ Benchmark), then percentile-ranked (1–99)

- For each symbol, we divide its price by the benchmark’s price to create a daily ratio for the lookback period.
- We then measure the percentage change in that ratio from the start of the period to the end.
- This shows how much the symbol outperformed or underperformed the benchmark over that window.
- Finally, all RS scores are ranked as a percentile (1–99) compared to all symbols shown in the dashboard.

**Big Picture Market Pulse Dashboard** is part of our analysis and is updated weekly.
"""
)


