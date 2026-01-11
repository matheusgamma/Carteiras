# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
import requests

st.title("Comparador de Carteiras - Renova Invest")

st.markdown("---")


st.sidebar.image(
    "https://raw.githubusercontent.com/matheusgamma/Carteiras/master/avatar-renova-twitter.png",
    width=300
)


st.set_page_config(page_title="Comparador de Carteiras", layout="wide")

PROPOSTA_URL_BLOB = "https://github.com/matheusgamma/Carteiras/blob/master/Carteiras/carteira_proposta.csv"
IBOV_TICKER = "^BVSP"
SGS_CDI_CODE = 12  # CDI (BCB/SGS)


# =========================
# Helpers
# =========================


def compute_similarity_leigo(w_current: pd.Series, w_target: pd.Series, gap_threshold: float = 0.02):
    """
    Retorna métricas leigas e também wc/wt alinhados.
    gap_threshold: 0.02 = 2% (em peso) para entrar nas listas de comprar/vender.
    """
    all_t = sorted(set(w_current.index) | set(w_target.index))
    wc = w_current.reindex(all_t).fillna(0.0)
    wt = w_target.reindex(all_t).fillna(0.0)

    dist = float(np.abs(wc - wt).sum())          # 0..2
    similarity = (1 - dist / 2) * 100            # 0..100

    common_assets = int(((wc > 0) & (wt > 0)).sum())
    outside_pct = float(wc[wt == 0].sum()) * 100

    gaps = (wt - wc).sort_values(ascending=False)
    add_more = gaps[gaps > gap_threshold].head(5)
    reduce = gaps[gaps < -gap_threshold].sort_values().head(5)

    return {
        "similarity_pct": similarity,
        "outside_pct": outside_pct,
        "common_assets": common_assets,
        "wc": wc,
        "wt": wt,
        "add_more": add_more,
        "reduce": reduce,
    }
@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def fetch_sectors(tickers: tuple) -> pd.DataFrame:
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

def github_blob_to_raw(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return url
    url = url.strip()
    if "raw.githubusercontent.com" in url:
        return url
    return url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")

def normalize_yf_ticker(t: str) -> str:
    t = str(t).strip().upper()
    if not t or t.lower() == "nan":
        return ""
    if t.startswith("^"):
        return t
    if "." in t:
        return t
    return f"{t}.SA"

def display_ticker(t: str) -> str:
    """Somente para exibição: remove .SA."""
    t = str(t).strip().upper()
    return t[:-3] if t.endswith(".SA") else t

def _normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().apply(normalize_yf_ticker)
    df["Peso"] = pd.to_numeric(df["Peso"], errors="coerce").fillna(0.0)
    df = df[df["Ticker"].ne("")]
    df = df.groupby("Ticker", as_index=False)["Peso"].sum()
    s = df["Peso"].sum()
    df["Peso"] = (df["Peso"] / s) if s > 0 else 0.0
    return df.sort_values("Peso", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_proposta(url_blob: str) -> pd.DataFrame:
    """
    Aceita:
    - Com cabeçalho: Ticker,Peso
    - Sem cabeçalho: 0,PETR4,0.10 (id, ticker, peso)
    - Sem cabeçalho: PETR4,0.10
    """
    url = github_blob_to_raw(url_blob)
    try:
        df = pd.read_csv(url)
    except Exception:
        df = pd.read_csv(url, header=None)

    cols_lower = [str(c).strip().lower() for c in df.columns]
    if "ticker" in cols_lower and any(c in cols_lower for c in ["peso", "weight", "weights"]):
        ticker_col = df.columns[cols_lower.index("ticker")]
        peso_col = df.columns[cols_lower.index("peso")] if "peso" in cols_lower else df.columns[cols_lower.index("weight")]
        out = df[[ticker_col, peso_col]].copy()
        out.columns = ["Ticker", "Peso"]
        return _normalize_weights(out)

    if df.shape[1] >= 3:
        out = df.iloc[:, [1, 2]].copy()
        out.columns = ["Ticker", "Peso"]
        return _normalize_weights(out)

    if df.shape[1] == 2:
        out = df.iloc[:, [0, 1]].copy()
        out.columns = ["Ticker", "Peso"]
        return _normalize_weights(out)

    raise ValueError("Formato do CSV da proposta inválido. Esperado 2 ou 3 colunas (Ticker, Peso).")

def proposta_fallback() -> pd.DataFrame:
    df = pd.DataFrame({"Ticker": ["PETR4", "VALE3", "ITUB4", "WEGE3"], "Peso": [0.25, 0.25, 0.25, 0.25]})
    return _normalize_weights(df)

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    tickers = [t for t in tickers if t]
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].copy() if "Close" in df.columns.get_level_values(0) else df.xs("Close", axis=1, level=0)
    else:
        close = df["Close"].to_frame()
        close.columns = tickers[:1]

    return close.dropna(how="all")

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_cdi_sgs(start_date: date, end_date: date) -> pd.Series:
    """
    Retorna CDI diário (taxa % ao dia) do SGS via API BCB.
    Endpoint: /dados/serie/bcdata.sgs.{codigo}/dados?formato=json&dataInicial=dd/mm/aaaa&dataFinal=dd/mm/aaaa
    """
    di = start_date.strftime("%d/%m/%Y")
    df = end_date.strftime("%d/%m/%Y")
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{SGS_CDI_CODE}/dados?formato=json&dataInicial={di}&dataFinal={df}"

    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    if not data:
        return pd.Series(dtype=float)

    tmp = pd.DataFrame(data)
    tmp["data"] = pd.to_datetime(tmp["data"], format="%d/%m/%Y")
    tmp["valor"] = pd.to_numeric(tmp["valor"], errors="coerce")
    tmp = tmp.dropna(subset=["valor"]).set_index("data").sort_index()

    # CDI vem como % ao dia
    return tmp["valor"]

def portfolio_cum_return(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    rets = prices.pct_change().dropna(how="all")
    w = weights.reindex(rets.columns).fillna(0.0)
    s = w.sum()
    if s > 0:
        w = w / s
    port = (rets * w).sum(axis=1)
    return (1 + port).cumprod() - 1

def portfolio_daily_returns(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    rets = prices.pct_change().dropna(how="all")
    w = weights.reindex(rets.columns).fillna(0.0)
    s = w.sum()
    if s > 0:
        w = w / s
    return (rets * w).sum(axis=1)

def annualized_vol(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return np.nan
    return float(daily_returns.std(ddof=1) * np.sqrt(252))

def beta_to_benchmark(port_daily: pd.Series, bench_daily: pd.Series) -> float:
    df = pd.concat([port_daily, bench_daily], axis=1).dropna()
    if df.shape[0] < 10:
        return np.nan
    p = df.iloc[:, 0].values
    b = df.iloc[:, 1].values
    var_b = np.var(b, ddof=1)
    if var_b == 0:
        return np.nan
    return float(np.cov(p, b, ddof=1)[0, 1] / var_b)

def weights_from_user_inputs(rows: list[dict], modo: str) -> pd.Series:
    df = pd.DataFrame(rows).groupby("Ticker", as_index=False)["Valor"].sum()
    if modo == "Peso (%)":
        w = df.set_index("Ticker")["Valor"] / 100.0
        s = w.sum()
        return (w / s) if s > 0 else w * 0.0
    else:
        fin = df.set_index("Ticker")["Valor"]
        return (fin / fin.sum()) if fin.sum() > 0 else fin * 0.0


# =========================
# Session state
# =========================
if "carteira_atual" not in st.session_state:
    st.session_state.carteira_atual = []


# =========================
# Sidebar
# =========================
st.sidebar.header("Período de análise")
colA, colB = st.sidebar.columns(2)
start_date = colA.date_input("Data inicial", value=date(2025, 1, 1))
end_date = colB.date_input("Data final", value=date.today())
if start_date >= end_date:
    st.sidebar.error("A data inicial precisa ser menor que a data final.")
    st.stop()

st.sidebar.divider()
modo = st.sidebar.radio("Como o cliente informa a carteira atual?", ["Peso (%)", "Financeiro (R$)"])
calcular = st.sidebar.button("✅ Calcular", use_container_width=True)


# =========================
# UI: Input
# =========================
st.title("Comparador de Carteiras")
st.caption("Monte a carteira atual do cliente e clique em **Calcular** para ver performance e comparação com a carteira modelo.")

st.markdown("## 1) Carteira atual (cliente)")

with st.container(border=True):
    st.caption("Digite o ticker sem .SA (ex: PETR4) e clique em **Adicionar**.")
    c1, c2, c3 = st.columns([2, 2, 1])
    ticker_in = c1.text_input("Ticker do ativo", placeholder="Ex: PETR4, VALE3, ITUB4")

    if modo == "Peso (%)":
        valor_in = c2.number_input("Peso (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
    else:
        valor_in = c2.number_input("Valor (R$)", min_value=0.0, value=0.0, step=1000.0)

    add_clicked = c3.button("➕ Adicionar", use_container_width=True)

if add_clicked:
    t = normalize_yf_ticker(ticker_in)
    if not t:
        st.warning("Digite um ticker válido.")
    elif valor_in <= 0:
        st.warning("Digite um valor maior que zero.")
    else:
        updated = False
        for row in st.session_state.carteira_atual:
            if row["Ticker"] == t:
                row["Valor"] += float(valor_in)
                updated = True
                break
        if not updated:
            st.session_state.carteira_atual.append({"Ticker": t, "Valor": float(valor_in)})

if st.session_state.carteira_atual:
    df_atual_val = pd.DataFrame(st.session_state.carteira_atual).groupby("Ticker", as_index=False)["Valor"].sum()
    df_atual_val = df_atual_val.sort_values("Valor", ascending=False).reset_index(drop=True)

    with st.container(border=True):
        st.markdown("### ✅ Ativos adicionados")
        show_df = df_atual_val.copy()
        show_df["Ticker"] = show_df["Ticker"].apply(display_ticker)
        st.dataframe(show_df, hide_index=True, use_container_width=True)

        colr1, colr2, colr3 = st.columns([2, 1, 1])
        to_remove = colr1.selectbox("Remover ativo", [""] + df_atual_val["Ticker"].tolist(), format_func=display_ticker)
        if colr2.button("🗑️ Remover", use_container_width=True) and to_remove:
            st.session_state.carteira_atual = [r for r in st.session_state.carteira_atual if r["Ticker"] != to_remove]
            st.rerun()
        if colr3.button("🧹 Limpar tudo", use_container_width=True):
            st.session_state.carteira_atual = []
            st.rerun()
else:
    st.info("Adicione ao menos 1 ativo para calcular.")

if not calcular:
    st.stop()

if not st.session_state.carteira_atual:
    st.warning("Adicione ao menos 1 ativo antes de calcular.")
    st.stop()


# =========================
# Build weights + load model
# =========================
w_current = weights_from_user_inputs(st.session_state.carteira_atual, modo)

with st.spinner("Carregando carteira modelo..."):
    try:
        proposta = load_proposta(PROPOSTA_URL_BLOB)
    except Exception:
        proposta = proposta_fallback()

w_model = proposta.set_index("Ticker")["Peso"]

# =========================
# 2) Performance primeiro (com CDI)
# =========================
st.markdown("## 2) Rentabilidade no período")

universe_compare = sorted(set(w_current.index) | set(w_model.index))
tickers_prices = sorted(set(universe_compare + [IBOV_TICKER]))

with st.spinner("Baixando preços (ações + Ibovespa)..."):
    prices = fetch_prices(tuple(tickers_prices), str(start_date), str(end_date))

if prices.empty:
    st.error("Não consegui baixar preços para o período selecionado. Verifique tickers e conexão.")
    st.stop()

# separa ativos e ibov
asset_cols = [c for c in prices.columns if c != IBOV_TICKER]
prices_assets = prices[asset_cols].dropna(how="all")

# IBOV retorno diário
if IBOV_TICKER in prices.columns:
    ibov_daily = prices[IBOV_TICKER].pct_change().dropna()
    ibov_cum = (prices[IBOV_TICKER] / prices[IBOV_TICKER].dropna().iloc[0]) - 1
else:
    ibov_daily = pd.Series(dtype=float)
    ibov_cum = None

# carteiras
cur_cum = portfolio_cum_return(prices_assets, w_current)
mod_cum = portfolio_cum_return(prices_assets, w_model)

cur_daily = portfolio_daily_returns(prices_assets, w_current)
mod_daily = portfolio_daily_returns(prices_assets, w_model)

# CDI (SGS)
with st.spinner("Buscando CDI (Banco Central)..."):
    cdi_daily_pct = fetch_cdi_sgs(start_date, end_date)  # % ao dia

# transforma CDI em série de retorno acumulado no mesmo index do gráfico
# CDI diário: (1 + pct/100).cumprod()-1
if not cdi_daily_pct.empty:
    cdi_cum_raw = (1 + (cdi_daily_pct / 100.0)).cumprod() - 1
    # alinha no index do chart (datas de mercado)
    cdi_cum = cdi_cum_raw.reindex(cur_cum.index).ffill()
else:
    cdi_cum = None

# monta chart
chart_df = pd.DataFrame({
    "Carteira Atual": cur_cum,
    "Carteira Modelo": mod_cum,
})
if ibov_cum is not None:
    chart_df["Ibovespa"] = ibov_cum.reindex(chart_df.index).ffill()
if cdi_cum is not None:
    chart_df["CDI"] = cdi_cum  # label

with st.container(border=True):
    st.markdown("### 📈 Retorno acumulado")
    st.line_chart(chart_df)

    # retorno final
    last = (chart_df.iloc[-1] * 100).round(2)

    # volatilidade e beta (anualizados)
    vol_cur = annualized_vol(cur_daily) * 100
    vol_mod = annualized_vol(mod_daily) * 100

    beta_cur = beta_to_benchmark(cur_daily, ibov_daily) if not ibov_daily.empty else np.nan
    beta_mod = beta_to_benchmark(mod_daily, ibov_daily) if not ibov_daily.empty else np.nan

    st.markdown("### 📌 Resumo do período")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retorno Carteira Atual", f"{last.get('Carteira Atual', np.nan):.2f}%")
    c2.metric("Retorno Carteira Modelo", f"{last.get('Carteira Modelo', np.nan):.2f}%")
    c3.metric("Retorno Ibovespa", f"{last.get('Ibovespa', np.nan):.2f}%" if "Ibovespa" in last.index else "—")
    c4.metric("Retorno CDI", f"{last.get('CDI', np.nan):.2f}%" if "CDI" in last.index else "—")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Volatilidade (Atual)", f"{vol_cur:.1f}% a.a." if np.isfinite(vol_cur) else "—")
    d2.metric("Volatilidade (Modelo)", f"{vol_mod:.1f}% a.a." if np.isfinite(vol_mod) else "—")
    d3.metric("Beta (Atual vs IBOV)", f"{beta_cur:.2f}" if np.isfinite(beta_cur) else "—")
    d4.metric("Beta (Modelo vs IBOV)", f"{beta_mod:.2f}" if np.isfinite(beta_mod) else "—")





# =========================
# 2.5) Carteira modelo (apenas informativo)
# =========================
st.markdown("## Carteira modelo (proposta) — composição")

# proposta já é DataFrame com colunas: Ticker, Peso (0..1)
modelo_df = proposta.copy()
modelo_df["Ativo"] = modelo_df["Ticker"].apply(display_ticker)
modelo_df["Peso (%)"] = (modelo_df["Peso"] * 100).round(2)

# busca setores só para os tickers do modelo
with st.spinner("Carregando setores da carteira modelo..."):
    sectors_modelo = fetch_sectors(tuple(modelo_df["Ticker"].tolist())).set_index("Ticker")

modelo_df["Setor"] = modelo_df["Ticker"].map(sectors_modelo["Setor"]).fillna("Unknown")

# deixa com cara mais “clean”: sem ticker .SA e sem coluna crua "Ticker"
show_modelo = modelo_df[["Ativo", "Peso (%)", "Setor"]].copy()

with st.container(border=True):
    st.caption("Abaixo está a carteira proposta usada como referência na comparação.")
    st.dataframe(show_modelo, use_container_width=True, hide_index=True)




# =========================
# =========================
# 3) Comparação leiga: o que comprar/vender + aderência
# =========================
st.markdown("## 3) Comparação com a carteira proposta (bem simples)")

cmp = compute_similarity_leigo(w_current, w_model, gap_threshold=0.02)

# Métricas e contagens leigas
wc = pd.Series(cmp["wc"])
wt = pd.Series(cmp["wt"])

ativos_atual = set(wc[wc > 0].index)
ativos_modelo = set(wt[wt > 0].index)

faltam_comprar = sorted([t for t in ativos_modelo if wc.get(t, 0) == 0])
devem_vender_total = sorted([t for t in ativos_atual if wt.get(t, 0) == 0])

# Tabela comparativa por ativo (Atual vs Proposta)
all_assets = sorted(ativos_atual | ativos_modelo)
rows = []
for t in all_assets:
    a = float(wc.get(t, 0.0))
    p = float(wt.get(t, 0.0))
    diff = p - a  # positivo = falta aumentar (comprar); negativo = reduzir (vender)

    # Ação leiga (sem termos técnicos)
    if abs(diff) < 0.01:  # < 1pp: manter
        acao = "Manter"
    elif diff > 0:
        acao = "Comprar / Aumentar"
    else:
        acao = "Vender / Reduzir"

    rows.append({
        "Ativo": display_ticker(t),
        "Atual (%)": round(a * 100, 2),
        "Proposta (%)": round(p * 100, 2),
        "Diferença (pp)": round((diff) * 100, 2),
        "Ação sugerida": acao
    })

df_comp = pd.DataFrame(rows)

# Ordena pela diferença (maior necessidade de compra no topo)
df_comp = df_comp.sort_values("Diferença (pp)", ascending=False).reset_index(drop=True)

# Listas de ações (top)
df_comprar = df_comp[df_comp["Ação sugerida"] == "Comprar / Aumentar"].copy()
df_vender  = df_comp[df_comp["Ação sugerida"] == "Vender / Reduzir"].copy()

# dá prioridade para o que é mais relevante (maior diferença em pp)
df_comprar = df_comprar.sort_values("Diferença (pp)", ascending=False)
df_vender  = df_vender.sort_values("Diferença (pp)", ascending=True)  # mais negativo primeiro

with st.container(border=True):
    st.markdown("### ✅ Resumo ")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aderência ao modelo", f"{cmp['similarity_pct']:.0f}%")
    c2.metric("Ativos em comum", f"{cmp['common_assets']}")
    c3.metric("Ativos que faltam comprar", f"{len(faltam_comprar)}")
    c4.metric("Ativos fora do modelo", f"{len(devem_vender_total)}")

    st.caption(
        "• **Aderência**: quanto a carteira atual está parecida com a carteira proposta.\n"
        "• **Faltam comprar**: ativos que existem no modelo e não existem na carteira atual.\n"
        "• **Fora do modelo**: ativos que estão na carteira atual mas não existem no modelo."
    )

with st.container(border=True):
    st.markdown("### 🛒 O que comprar / aumentar (para ficar mais parecido)")
    if df_comprar.empty:
        st.write("Nenhuma compra relevante. Você já está bem próximo do modelo nesses pontos.")
    else:
        st.dataframe(
            df_comprar[["Ativo", "Atual (%)", "Proposta (%)", "Diferença (pp)"]].head(10),
            use_container_width=True,
            hide_index=True
        )

with st.container(border=True):
    st.markdown("### 🧹 O que vender / reduzir (está acima do modelo ou fora dele)")
    if df_vender.empty:
        st.write("Nenhuma redução relevante.")
    else:
        st.dataframe(
            df_vender[["Ativo", "Atual (%)", "Proposta (%)", "Diferença (pp)"]].head(10),
            use_container_width=True,
            hide_index=True
        )

with st.container(border=True):
    st.markdown("### 📌 Comparação completa por ativo (Atual vs Proposta)")
    st.dataframe(
        df_comp[["Ativo", "Atual (%)", "Proposta (%)", "Diferença (pp)", "Ação sugerida"]],
        use_container_width=True,
        hide_index=True
    )


# =========================
# 4) Setores: pizza/donut (Atual vs Proposta)
# =========================
st.markdown("## 4) Diversificação por setores (pizza)")

# Import aqui para não obrigar quem não quer plotly a instalar manualmente
import plotly.express as px

# (1) Busca setores só dos tickers relevantes (união)
universe_compare = sorted(set(w_current.index) | set(w_model.index))

with st.spinner("Buscando setores (yfinance)..."):
    sectors_df = fetch_sectors(tuple(universe_compare)).set_index("Ticker")  # col: Setor

# (2) Constrói pesos por setor
def sector_weights(weights: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({"Ticker": weights.index, "Peso": weights.values})
    tmp = tmp[tmp["Peso"] > 0].copy()
    tmp = tmp.join(sectors_df, on="Ticker")
    tmp["Setor"] = tmp["Setor"].fillna("Unknown")
    out = tmp.groupby("Setor")["Peso"].sum().sort_values(ascending=False)
    return out

sec_cur = sector_weights(w_current)
sec_mod = sector_weights(w_model)

# (3) Donuts lado a lado
colA, colB = st.columns(2)

with colA:
    st.markdown("### Carteira Atual")
    dfp = sec_cur.reset_index()
    dfp.columns = ["Setor", "Peso"]
    dfp["Peso (%)"] = (dfp["Peso"] * 100).round(2)
    fig = px.pie(dfp, names="Setor", values="Peso (%)", hole=0.55)
    st.plotly_chart(fig, use_container_width=True)

with colB:
    st.markdown("### Carteira Proposta")
    dfp = sec_mod.reset_index()
    dfp.columns = ["Setor", "Peso"]
    dfp["Peso (%)"] = (dfp["Peso"] * 100).round(2)
    fig = px.pie(dfp, names="Setor", values="Peso (%)", hole=0.55)
    st.plotly_chart(fig, use_container_width=True)

