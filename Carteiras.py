# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date

st.set_page_config(page_title="Carteira: Atual vs Proposta", layout="wide")


# =========================================================
# LINKS (mantenha genéricos por enquanto; você troca depois)
# =========================================================
# IMPORTANTE: pode ser link "blob" do GitHub — o app converte para "raw" automaticamente
TICKERS_URL_BLOB = "https://github.com/SEUUSER/SEUREPO/blob/main/tickers/tickers_ibra.csv"
PROPOSTA_URL_BLOB = "https://github.com/SEUUSER/SEUREPO/blob/main/carteiras/carteira_proposta.csv"

IBOV_TICKER = "^BVSP"


# =========================
# Helpers: GitHub + Tickers
# =========================
def github_blob_to_raw(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return url
    url = url.strip()
    if "raw.githubusercontent.com" in url:
        return url
    # Converte:
    # https://github.com/user/repo/blob/main/path.csv
    # -> https://raw.githubusercontent.com/user/repo/main/path.csv
    return (
        url.replace("https://github.com/", "https://raw.githubusercontent.com/")
           .replace("/blob/", "/")
    )

def normalize_yf_ticker(t: str) -> str:
    t = str(t).strip().upper()
    if not t or t.lower() == "nan":
        return ""
    if t.startswith("^"):    # índices tipo ^BVSP
        return t
    if "." in t:             # já tem sufixo (.SA, .US etc.)
        return t
    # default: Brasil
    return f"{t}.SA"

@st.cache_data(ttl=60 * 60)
def load_tickers_single_column(url_blob: str) -> list[str]:
    url = github_blob_to_raw(url_blob)
    df = pd.read_csv(url)

    # tenta achar coluna com nome comum
    series = None
    for col in df.columns:
        c = str(col).strip().lower()
        if c in ["ticker", "tickers", "ativo", "ativos", "symbol", "symbols"]:
            series = df[col]
            break

    if series is None:
        series = df.iloc[:, 0]

    tickers = (
        series.astype(str)
              .str.strip()
              .replace({"nan": "", "None": ""})
    )
    tickers = [t for t in tickers.tolist() if t]

    tickers_norm = [normalize_yf_ticker(t) for t in tickers]
    tickers_norm = [t for t in tickers_norm if t]  # remove vazios
    tickers_norm = sorted(set(tickers_norm))
    return tickers_norm

@st.cache_data(ttl=60 * 60)
def load_proposta(url_blob: str) -> pd.DataFrame:
    url = github_blob_to_raw(url_blob)
    df = pd.read_csv(url)

    # Esperado: algo como (Ticker, Peso)
    # tenta normalizar nomes
    cols = {c: str(c).strip().lower() for c in df.columns}
    inv = {v: k for k, v in cols.items()}

    # descobre coluna ticker
    ticker_col = None
    for candidate in ["ticker", "ativo", "symbol"]:
        if candidate in inv:
            ticker_col = inv[candidate]
            break
    if ticker_col is None:
        ticker_col = df.columns[0]  # fallback

    # descobre coluna peso
    weight_col = None
    for candidate in ["peso", "weight", "weights", "allocation", "alocacao", "alocação"]:
        if candidate in inv:
            weight_col = inv[candidate]
            break
    if weight_col is None:
        weight_col = df.columns[1] if len(df.columns) > 1 else None

    if weight_col is None:
        raise ValueError("CSV da proposta precisa ter pelo menos 2 colunas: Ticker e Peso.")

    out = df[[ticker_col, weight_col]].copy()
    out.columns = ["Ticker", "Peso"]
    out["Ticker"] = out["Ticker"].astype(str).str.strip().apply(normalize_yf_ticker)
    out["Peso"] = pd.to_numeric(out["Peso"], errors="coerce").fillna(0.0)

    out = out[out["Ticker"].ne("")]
    s = out["Peso"].sum()
    if s > 0:
        out["Peso"] = out["Peso"] / s
    else:
        out["Peso"] = 0.0

    # agrega duplicados
    out = out.groupby("Ticker", as_index=False)["Peso"].sum()
    s = out["Peso"].sum()
    if s > 0:
        out["Peso"] = out["Peso"] / s
    return out


# =========================
# Helpers: preços, setores
# =========================
@st.cache_data(ttl=60 * 60)
def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # inclui ajustes (dividendos/splits) no preço
        progress=False,
        group_by="column",
    )

    # MultiIndex quando tem vários tickers
    if isinstance(df.columns, pd.MultiIndex):
        # geralmente vem: ('Close', ticker)
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"].copy()
        else:
            close = df.xs("Close", axis=1, level=0, drop_level=True)
    else:
        # um ticker só
        close = df["Close"].to_frame()
        close.columns = tickers[:1]

    close = close.dropna(how="all")
    return close

@st.cache_data(ttl=60 * 60 * 24)
def fetch_sectors(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        sector = "Unknown"
        try:
            info = yf.Ticker(t).info
            sector = info.get("sector") or "Unknown"
        except Exception:
            pass
        rows.append({"Ticker": t, "Setor": sector})
    return pd.DataFrame(rows)

def adequacy_score(w_current: pd.Series, w_target: pd.Series) -> dict:
    all_t = sorted(set(w_current.index) | set(w_target.index))
    wc = w_current.reindex(all_t).fillna(0.0)
    wt = w_target.reindex(all_t).fillna(0.0)

    l1 = float(np.abs(wc - wt).sum())               # 0..2
    adherence = (1 - l1 / 2) * 100                  # 0..100
    overlap = float(np.minimum(wc, wt).sum()) * 100 # 0..100

    gaps = (wt - wc).sort_values(ascending=False)
    add_more = gaps[gaps > 0.02].head(10)           # gaps > 2%
    reduce   = gaps[gaps < -0.02].head(10)          # gaps < -2%

    return {
        "adherence_pct": adherence,
        "overlap_pct": overlap,
        "l1_distance": l1,
        "add_more": add_more,
        "reduce": reduce,
        "wc": wc,
        "wt": wt,
    }

def portfolio_cum_return(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    rets = prices.pct_change().dropna(how="all")
    w = weights.reindex(rets.columns).fillna(0.0)
    s = w.sum()
    if s > 0:
        w = w / s
    port = (rets * w).sum(axis=1)
    return (1 + port).cumprod() - 1

def format_pct(x: float) -> str:
    return f"{x:.1f}%"


# =========================
# Sidebar: Período
# =========================
st.sidebar.header("Período de análise")
colA, colB = st.sidebar.columns(2)
start_date = colA.date_input("Data inicial", value=date(2025, 1, 1))
end_date = colB.date_input("Data final", value=date.today())

if start_date >= end_date:
    st.sidebar.error("A data inicial precisa ser menor que a data final.")
    st.stop()

st.sidebar.divider()
st.sidebar.caption("Links do GitHub (você troca depois)")
st.sidebar.text("Tickers (universo):")
st.sidebar.code(TICKERS_URL_BLOB, language="text")
st.sidebar.text("Carteira proposta:")
st.sidebar.code(PROPOSTA_URL_BLOB, language="text")


# =========================
# Carrega universo + proposta
# =========================
st.title("Comparador: Carteira Atual (cliente) vs Carteira Proposta (padrão)")

with st.spinner("Carregando tickers e carteira proposta do GitHub..."):
    try:
        universo = load_tickers_single_column(TICKERS_URL_BLOB)
    except Exception as e:
        st.error("Não consegui ler o arquivo de tickers do GitHub. Verifique o link e o CSV.")
        st.exception(e)
        st.stop()

    try:
        proposta = load_proposta(PROPOSTA_URL_BLOB)
    except Exception as e:
        st.error("Não consegui ler a carteira proposta do GitHub. Verifique o link e o CSV (Ticker + Peso).")
        st.exception(e)
        st.stop()

w_target = proposta.set_index("Ticker")["Peso"]


# =========================
# 1) Carteira atual (input)
# =========================
st.subheader("1) Carteira atual (cliente)")
st.caption("O usuário pode digitar os tickers sem “.SA” (ex: PETR4). Você pode informar Peso (%) ou Financeiro (R$).")

default_current = pd.DataFrame({
    "Ticker": ["PETR4", "VALE3", "ITUB4"],
    "Tipo": ["Peso", "Peso", "Peso"],   # Peso ou Financeiro
    "Valor": [30, 40, 30],              # se Peso: em %; se Financeiro: R$
})

current_input = st.data_editor(
    default_current,
    num_rows="dynamic",
    use_container_width=True
)

cur = current_input.copy()
cur["Ticker"] = cur["Ticker"].astype(str).str.strip().apply(normalize_yf_ticker)
cur["Tipo"] = cur["Tipo"].astype(str).str.strip().str.lower()
cur["Valor"] = pd.to_numeric(cur["Valor"], errors="coerce").fillna(0.0)

cur = cur[cur["Ticker"].ne("")].copy()

if cur.empty:
    st.warning("Preencha ao menos 1 linha na carteira atual.")
    st.stop()

# valida: ticker deve estar no universo (quando aplicável)
# (Não barra índices ou tickers com sufixo diferente — mas para BR padrão, isso ajuda)
invalid = sorted(set(cur["Ticker"]) - set(universo))
if invalid:
    st.warning(
        "Alguns tickers da carteira atual não estão no universo (lista do GitHub). "
        "Vou manter mesmo assim, mas pode falhar no download de preços:\n\n"
        + ", ".join(invalid)
    )

is_fin = cur["Tipo"].eq("financeiro")
is_wgt = cur["Tipo"].eq("peso")

if not (is_fin.any() or is_wgt.any()):
    st.error("A coluna 'Tipo' precisa ser 'Peso' ou 'Financeiro'.")
    st.stop()

w_from_weight = cur[is_wgt].groupby("Ticker")["Valor"].sum() / 100.0
w_from_fin = cur[is_fin].groupby("Ticker")["Valor"].sum()

if w_from_fin.sum() > 0:
    w_from_fin = w_from_fin / w_from_fin.sum()
else:
    w_from_fin = w_from_fin * 0.0

w_current = w_from_weight.add(w_from_fin, fill_value=0.0)
if w_current.sum() > 0:
    w_current = w_current / w_current.sum()


# =========================
# 2) Adequação (primeiro)
# ======================
