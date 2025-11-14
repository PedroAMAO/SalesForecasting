# ===============================================
# 📈 Previsão com Decomposição + ARIMA (versão enxuta e robusta)
# ===============================================
import os, warnings, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import base64
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from xgboost import XGBRegressor





# 🔒 Limita threads (evita travamentos em alguns ambientes)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

# ===============================
# UI / CSS
# ===============================
st.markdown("""
    <style>
        .main { padding: 2rem 5rem; }
        .block-container { max-width: 100%; min-height: 90vh; }
        html, body { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)
st.title("📈 Previsão com Decomposição Log + Tendência + Sazonalidade")

# ===============================
# Utils
# ===============================
def prever_full_classico(modelo_classico, df_filial):
    """Gera previsão clássica (tend + saz + IC) somente no histórico."""
    prevs = []
    for _, row in df_filial.iterrows():
        t_i = int(row["t"])
        m_i = int(row["mes_num"])
        y, ic_inf, ic_sup, _ = modelo_classico.prever_nivel_com_ic(t_i, m_i)
        prevs.append({
            "data": row["data"],
            "previsao": y,
            "ic_inf": ic_inf,
            "ic_sup": ic_sup,
        })
    return pd.DataFrame(prevs)

def prever_full_arima(modelo_classico, modelo_arima, df_filial):
    """Reconstrói previsão ARIMA (A+B) somente no histórico observado."""
    serie_residuo = modelo_arima.modelo.fittedvalues  # só A
    n_hist = len(df_filial)

    # Como ARIMA só tem previsões do período A, usamos apenas fitted
    # e completamos com forecast até completar o histórico (B)
    passos_faltando = n_hist - len(serie_residuo)

    if passos_faltando > 0:
        futuros_B = modelo_arima.modelo.forecast(steps=passos_faltando)
        ruido_total = pd.concat([serie_residuo, futuros_B])
    else:
        ruido_total = serie_residuo

    ruido_total = ruido_total.reset_index(drop=True)

    prevs = []
    for idx, row in df_filial.iterrows():
        t_i = int(row["t"])
        m_i = int(row["mes_num"])

        tend_saz_log = modelo_classico.prever_log(t_i, m_i)
        ruido_i = float(ruido_total.iloc[idx])

        yhat_log = tend_saz_log + ruido_i

        prevs.append({
            "data": row["data"],
            "previsao": np.exp(yhat_log) - 1.0,
            "yhat_log": yhat_log
        })

    return pd.DataFrame(prevs)


def rolling_ml(df_filial, tipo_tendencia, arima_order, lag_window=6, janela_minima=12):
    resultados = []
    datas = df_filial['data'].unique()

    for i in range(janela_minima, len(datas)-1):
        data_corte = datas[i]
        data_target = data_corte + pd.DateOffset(months=1)

        if data_target not in df_filial['data'].values:
            continue

        # 1) clássico até o corte
        modelo_classico, df_treino, _ = treinar_modelo_classico(
            df_filial, data_corte, tipo_tendencia
        )

        # 2) ARIMA nos resíduos até o corte
        modelo_arima = treinar_arima_ruido(df_treino, arima_order)

        # 3) treinar ML até o corte (idêntico ao gráfico)
        modelo_ml_obj = treinar_modelo_ml(
            df_filial=df_filial,
            df_prev_classico_full=prever_full_classico(modelo_classico, df_filial),
            df_prev_arima_completo=prever_full_arima(modelo_classico, modelo_arima, df_filial),
            data_corte=data_corte,
            lag_window=lag_window
        )

        # 4) prever o mês seguinte
        df_prev_ml_full = prever_ml(
            modelo_ml_obj,
            df_filial,
            prever_full_classico(modelo_classico, df_filial),
            prever_full_arima(modelo_classico, modelo_arima, df_filial),
            data_corte,
            meses_a_frente=1
        )

        y_pred = df_prev_ml_full.loc[
            df_prev_ml_full["data"] == data_target, "previsao"
        ].values[0]

        y_real = df_filial.loc[df_filial['data'] == data_target, 'alvo'].values[0]

        erro_abs = abs(y_real - y_pred)
        erro_pct = erro_abs / max(1e-6, y_real)

        resultados.append({
            "data_corte": data_corte,
            "data_prev": data_target,
            "real": y_real,
            "prev": y_pred,
            "erro_abs": erro_abs,
            "erro_pct": erro_pct,
        })

    return pd.DataFrame(resultados)

def rolling_arima(df_filial, tipo_tendencia, arima_order, janela_minima=12):
    resultados = []
    datas = df_filial['data'].unique()

    for i in range(janela_minima, len(datas)-1):
        data_corte = datas[i]
        data_target = data_corte + pd.DateOffset(months=1)

        if data_target not in df_filial['data'].values:
            continue

        # treina modelo clássico
        modelo_classico, df_treino, _ = treinar_modelo_classico(
            df_filial, data_corte, tipo_tendencia
        )

        # treina ARIMA nos resíduos — igual ao gráfico
        modelo_arima = treinar_arima_ruido(df_treino, arima_order)

        # t/mês do mês seguinte
        t_next = int(df_filial.loc[df_filial['data'] == data_target, 't'].values[0])
        mes_next = int(data_target.month)

        # previsão estrutural (tend + saz) do próximo mês
        yhat_log = modelo_classico.prever_log(t_next, mes_next)

        # previsão ARIMA h=1
        ruido_prev = modelo_arima.modelo.forecast(steps=1).iloc[0]

        y_pred = np.exp(yhat_log + ruido_prev) - 1.0
        y_real = df_filial.loc[df_filial['data'] == data_target, 'alvo'].values[0]

        erro_abs = abs(y_real - y_pred)
        erro_pct = erro_abs / max(1e-6, y_real)

        resultados.append({
            "data_corte": data_corte,
            "data_prev": data_target,
            "real": y_real,
            "prev": y_pred,
            "erro_abs": erro_abs,
            "erro_pct": erro_pct,
        })

    return pd.DataFrame(resultados)
def rolling_classico(df_filial, tipo_tendencia, janela_minima=12):
    resultados = []
    datas = df_filial['data'].unique()

    for i in range(janela_minima, len(datas)-1):
        data_corte = datas[i]
        data_target = data_corte + pd.DateOffset(months=1)

        # não existe realizado → pula
        if data_target not in df_filial['data'].values:
            continue

        # treina o modelo clássico exatamente como no gráfico
        modelo_classico, df_treino, _ = treinar_modelo_classico(
            df_filial, data_corte, tipo_tendencia
        )

        t_next = int(df_filial.loc[df_filial['data'] == data_target, 't'].values[0])
        mes_next = int(data_target.month)

        y_pred, _, _, _ = modelo_classico.prever_nivel_com_ic(t_next, mes_next)
        y_real = df_filial.loc[df_filial['data'] == data_target, 'alvo'].values[0]

        erro_abs = abs(y_real - y_pred)
        erro_pct = erro_abs / max(1e-6, y_real)

        resultados.append({
            "data_corte": data_corte,
            "data_prev": data_target,
            "real": y_real,
            "prev": y_pred,
            "erro_abs": erro_abs,
            "erro_pct": erro_pct,
        })

    return pd.DataFrame(resultados)


def rolling_eval(df_filial, modelo, meses_a_frente=1, janela_minima=12):
    """
    Avalia um modelo prevendo 1 mês à frente em vários cortes.
    df_filial      -> DataFrame da filial (data, alvo)
    modelo         -> função que recebe (df, data_corte) e devolve previsão do próximo mês
    """
    resultados = []
    datas = df_filial['data'].unique()

    for i in range(janela_minima, len(datas) - meses_a_frente):
        data_corte = datas[i]

        # data do target (mês seguinte)
        data_target = data_corte + pd.DateOffset(months=meses_a_frente)

        # Se não existe realizado do target, pula
        if data_target not in df_filial['data'].values:
            continue

        y_real = df_filial.loc[df_filial['data'] == data_target, 'alvo'].values[0]

        # chama o modelo (Clássico, ARIMA ou ML)
        try:
            y_pred = modelo(df_filial, data_corte)
        except:
            continue

        erro_abs = abs(y_real - y_pred)
        erro_pct = erro_abs / max(1e-6, y_real)

        resultados.append({
            "data_corte": data_corte,
            "data_prev": data_target,
            "real": y_real,
            "prev": y_pred,
            "erro_abs": erro_abs,
            "erro_pct": erro_pct,
        })

    return pd.DataFrame(resultados)



def gerar_pdf_completo(
    filial,
    m_class_total, m_arima_total, m_ml_total,
    m_class_pos, m_arima_pos, m_ml_pos,
    relatorio_llm,
    grafico_png
):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    story = []

    # =============================
    # TÍTULO
    # =============================
    story.append(Paragraph(f"<b>Relatório de Previsão — Filial {filial}</b>", styles["Title"]))
    story.append(Spacer(1, 0.4*cm))

    # =============================
    # IMAGEM DO GRÁFICO
    # =============================
    img_buf = io.BytesIO(grafico_png)
    story.append(Image(img_buf, width=16*cm, height=7*cm))
    story.append(Spacer(1, 0.6*cm))

    # =============================
    # MÉTRICAS (Tabela)
    # =============================
    dados_tabela = [
        ["Modelo", "MAPE (%)", "R²", "RMSE"],
        ["Clássico", f"{m_class_total['MAPE (%)']:.2f}", f"{m_class_total['R²']:.3f}", f"{m_class_total['RMSE']:.2f}"],
        ["ARIMA",   f"{m_arima_total['MAPE (%)']:.2f}", f"{m_arima_total['R²']:.3f}", f"{m_arima_total['RMSE']:.2f}"],
    ]

    if m_ml_total:
        dados_tabela.append(
            ["ML", f"{m_ml_total['MAPE (%)']:.2f}", f"{m_ml_total['R²']:.3f}", f"{m_ml_total['RMSE']:.2f}"]
        )

    tabela = Table(dados_tabela)
    tabela.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('ALIGN', (1,1), (-1,-1), 'CENTER')
    ]))
    story.append(tabela)
    story.append(Spacer(1, 0.8*cm))

    # =============================
    # RELATÓRIO DA LLM
    # =============================
    story.append(Paragraph("<b>Interpretação Automática (LLM)</b>", styles["Heading2"]))
    for par in relatorio_llm.split("\n"):
        if par.strip():
            story.append(Paragraph(par.strip(), styles["BodyText"]))
            story.append(Spacer(1, 0.25*cm))

    doc.build(story)
    pdf_value = buffer.getvalue()
    buffer.close()
    return pdf_value



def extract_context_text(context_file):
    import pandas as pd
    import io
    

    filename = context_file.name.lower()

    # TXT
    if filename.endswith(".txt"):
        return context_file.read().decode("utf-8")

    # PDF
    if filename.endswith(".pdf"):
        import fitz   # PyMuPDF
        doc = fitz.open(stream=context_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    
    # CSV
    if filename.endswith(".csv"):
        df = pd.read_csv(context_file)
        return df.to_string()

    # XLSX
    if filename.endswith(".xlsx"):
        xls = pd.ExcelFile(context_file)
        df = xls.parse(xls.sheet_names[0])
        return df.to_string()

    return ""


def calc_metrics(y_true, y_pred):
    # garante arrays 1D lisinhos
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    r2   = r2_score(y_true, y_pred)

    # ⚠️ sklearn antigo NÃO aceita squared=False → calculamos RMSE “na mão”
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        'MAPE (%)': mape,
        'R²': r2,
        'RMSE': rmse
    }


def safe_sheet_read(file):
    """Tenta ler 'Planilha1'; se não existir, pega a primeira aba."""
    try:
        xl = pd.ExcelFile(file)
        sheet = 'Planilha1' if 'Planilha1' in xl.sheet_names else xl.sheet_names[0]
        df = xl.parse(sheet)
    except Exception:
        # fallback direto
        df = pd.read_excel(file)
    return df

def normalize_columns(df):
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)
    # suporta 'mês' ou 'mes'
    if 'mês' in df.columns and 'mes' not in df.columns:
        df = df.rename(columns={'mês': 'mes'})
    # padroniza nomes esperados
    rename_map = {}
    if 'ano' not in df.columns:
        # tenta achar algo parecido
        for c in df.columns:
            if c.startswith('ano'):
                rename_map[c] = 'ano'
    if 'filial' not in df.columns:
        for c in df.columns:
            if 'filial' in c:
                rename_map[c] = 'filial'
    if 'realizado' not in df.columns:
        for c in df.columns:
            if 'realiz' in c:
                rename_map[c] = 'realizado'
    df = df.rename(columns=rename_map)
    missing = [c for c in ['ano','mes','filial','realizado'] if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando: {missing}. Esperado: ano, mês/mes, Filial, Realizado")
    return df

def to_month_start_date(ano, mes):
    try:
        ano = int(ano)
        mes = int(mes)
        return pd.to_datetime(f"{ano}-{mes:02d}-01")
    except Exception:
        return pd.NaT

def clip_predictions(df, is_share):
    if is_share:
        return df.assign(
            previsao=np.clip(df['previsao'], 0.0, 1.0),
            ic_inf=np.clip(df['ic_inf'], 0.0, 1.0),
            ic_sup=np.clip(df['ic_sup'], 0.0, 1.0)
        )
    else:
        return df.assign(
            previsao=np.clip(df['previsao'], 0.0, None),
            ic_inf=np.clip(df['ic_inf'], 0.0, None),
            ic_sup=np.clip(df['ic_sup'], 0.0, None)
        )

def parse_arima_order(s):
    s = s.strip()
    m = re.match(r'^\s*\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?\s*$', s)
    if not m:
        raise ValueError("Formato inválido. Use p,d,q (ex: 1,0,0).")
    return tuple(map(int, m.groups()))

# ============================================================
# 🎯 MODELO CLÁSSICO (Tendência + Sazonalidade Média)
# ============================================================
class ClassicalTrendSeasonalModel:
    def __init__(self, tipo_tendencia, media_log, modelo_tend, saz_media, sigma_ruido):
        self.tipo_tendencia = tipo_tendencia
        self.media_log = media_log
        self.modelo_tend = modelo_tend  # pode ser None se "Média"
        self.saz_media = saz_media      # Series indexada por mês
        self.sigma_ruido = sigma_ruido  # desvio do ruído no log

    def prever_log(self, t_i, mes_num):
        """Retorna previsão em log (tendência + sazonalidade)."""
        # Tendência
        if self.tipo_tendencia == "Média":
            tend_i = self.media_log
        else:
            if self.tipo_tendencia == "Quadrática":
                x = np.array([[t_i, t_i**2]])
            else:
                x = np.array([[t_i]])
            tend_i = float(self.modelo_tend.predict(x)[0])

        # Sazonalidade
        saz_i = float(self.saz_media.get(mes_num, 0.0))
        return tend_i + saz_i

    def prever_nivel_com_ic(self, t_i, mes_num):
        """
        Retorna: previsão no nível, ic_inf, ic_sup, yhat_log.
        """
        yhat_log = self.prever_log(t_i, mes_num)
        y = np.exp(yhat_log) - 1.0
        ic_inf = np.exp(yhat_log - 2.57 * self.sigma_ruido) - 1.0
        ic_sup = np.exp(yhat_log + 2.57 * self.sigma_ruido) - 1.0
        return y, ic_inf, ic_sup, yhat_log


def treinar_modelo_classico(df_filial, data_corte, tipo_tendencia):
    """
    Treina o modelo clássico (tendência + sazonalidade) até a data de corte.
    Retorna:
        modelo_classico  -> objeto ClassicalTrendSeasonalModel
        df_treino        -> base até o corte (já com residuo/ruido)
        df_real          -> base completa (se quiser usar depois)
    """
    df_treino = df_filial[df_filial['data'] <= data_corte].copy()
    df_real = df_filial.copy()

    # log sempre positivo (protege zero)
    df_treino['log_venda'] = np.log(df_treino['alvo'] + 1.0)
    df_real['log_venda'] = np.log(df_real['alvo'] + 1.0)

    media_log = df_treino['log_venda'].mean()

    # ---------------------------
    # Tendência
    # ---------------------------
    modelo_tend = None
    if tipo_tendencia == "Média":
        tendencia_train = np.full(len(df_treino), media_log)
    else:
        X_tr = pd.DataFrame({'t': df_treino['t'].values})
        if tipo_tendencia == "Quadrática":
            X_tr['t2'] = X_tr['t']**2

        modelo_tend = LinearRegression().fit(X_tr, df_treino['log_venda'])
        tendencia_train = modelo_tend.predict(X_tr)

    # ---------------------------
    # Sazonalidade média
    # ---------------------------
    df_treino['residuo'] = df_treino['log_venda'] - tendencia_train

    if tipo_tendencia != "Média":
        saz_media = df_treino.groupby(df_treino['data'].dt.month)['residuo'].mean()
    else:
        saz_media = pd.Series(0.0, index=range(1, 13))

    # ruído (residuo - sazonalidade)
    df_treino['ruido'] = df_treino['residuo'] - df_treino['data'].dt.month.map(saz_media)
    sigma_ruido = float(df_treino['ruido'].std())
    if not np.isfinite(sigma_ruido) or sigma_ruido == 0:
        sigma_ruido = 1e-6

    modelo_classico = ClassicalTrendSeasonalModel(
        tipo_tendencia=tipo_tendencia,
        media_log=media_log,
        modelo_tend=modelo_tend,
        saz_media=saz_media,
        sigma_ruido=sigma_ruido
    )

    return modelo_classico, df_treino, df_real


# ============================================================
# 🔧 Helper — Modelo ARIMA encapsulado para o ruído
# ============================================================

class ArimaRuidoModel:
    """
    Wrap simples para manter o ARIMA ajustado nos resíduos.
    """
    def __init__(self, modelo, order, n_treino):
        self.modelo = modelo     # modelo ARIMA do statsmodels já ajustado
        self.order = order       # tupla (p,d,q)
        self.n_treino = n_treino # quantidade de pontos no treino (útil em rolling)


def treinar_arima_ruido(df_treino, arima_order):
    """
    Ajusta um ARIMA nos resíduos do modelo clássico.

    df_treino deve conter:
        data, ruido

    arima_order é uma tupla (p,d,q)

    Retorna:
        ArimaRuidoModel
    """

    serie = df_treino.set_index("data")["ruido"].dropna()

    # Segurança mínima: impede ARIMA impossível
    if len(serie) < sum(arima_order) + 3:
        raise ValueError(
            f"Poucos pontos ({len(serie)}) para treinar ARIMA com ordem {arima_order}"
        )

    modelo = ARIMA(serie, order=arima_order).fit()

    return ArimaRuidoModel(
        modelo=modelo,
        order=arima_order,
        n_treino=len(serie)
    )


# ============================================================
# 🔍 Otimização opcional do ARIMA via Rolling-Origin (cross-val)
# ============================================================

from itertools import product

def otimizar_arima(df_treino,
                   grid_p=range(0,3),
                   grid_d=range(0,3),
                   grid_q=range(0,3),
                   metrica='mape',
                   min_janela=12):
    """
    Rolling-Origin Cross-Validation 100% honesta.
    Avalia previsões one-step-ahead da série de ruído do df_treino.

    Retorna:
        melhor -> dict {p,d,q,erro}
        resultados -> lista de tentativas
    """

    serie = df_treino.set_index("data")["ruido"].dropna()
    resultados = []

    for p, d, q in product(grid_p, grid_d, grid_q):
        try:
            # precisa de pelo menos (p+d+q) + janela mínima
            if len(serie) < max(p, d, q) + min_janela:
                continue

            erros = []

            # Rolling-origin
            for corte_idx in range(min_janela, len(serie) - 1):
                serie_treino = serie.iloc[:corte_idx]
                verdadeiro = serie.iloc[corte_idx + 1]

                modelo_tmp = ARIMA(serie_treino, order=(p, d, q)).fit()
                prev = modelo_tmp.forecast(steps=1).iloc[0]

                if metrica == 'rmse':
                    erro = (verdadeiro - prev) ** 2
                else:  # mape-like: erro absoluto
                    erro = abs(verdadeiro - prev)

                erros.append(erro)

            if erros:
                resultados.append({
                    'p': p,
                    'd': d,
                    'q': q,
                    'erro': float(np.mean(erros))
                })

        except Exception:
            continue

    if not resultados:
        raise ValueError("Nenhum ARIMA válido encontrado na validação rolling-origin.")

    melhor = min(resultados, key=lambda x: x['erro'])
    return melhor, resultados

# ============================================================
#  MACHINE LEARNING
# ============================================================


# ============================================================
#  FUNÇÃO 1 — Build Lags (histórico até o corte)
# ============================================================

def build_lags(df_base, lag_window=6):
    """
    df_base deve conter:
        data, alvo, prev_cl, prev_ar, t, mes_num
    lag_window: Nº de lags (N)
    Retorna:
        df_feat  -> base com todas as features históricas
        feature_cols -> lista das features para treinar o modelo ML
    """

    df = df_base.copy().sort_values("data").reset_index(drop=True)
    feature_cols = []

    # -------------------------
    # 1. Lags de Y, Clássico, ARIMA
    # -------------------------
    for k in range(1, lag_window+1):
        df[f"lag_y_{k}"]  = df["alvo"].shift(k)
        df[f"lag_cl_{k}"] = df["prev_cl"].shift(k)
        df[f"lag_ar_{k}"] = df["prev_ar"].shift(k)

        feature_cols += [f"lag_y_{k}", f"lag_cl_{k}", f"lag_ar_{k}"]

        # ---------- Erros ----------
        df[f"err_cl_{k}"] = df[f"lag_y_{k}"] - df[f"lag_cl_{k}"]
        df[f"err_ar_{k}"] = df[f"lag_y_{k}"] - df[f"lag_ar_{k}"]

        feature_cols += [f"err_cl_{k}", f"err_ar_{k}"]

    # -------------------------
    # 2. Deltas (variações dentro da janela)
    # -------------------------
    for k in range(1, lag_window):
        df[f"delta_y_{k}"] = df[f"lag_y_{k}"] - df[f"lag_y_{k+1}"]
        df[f"delta_cl_{k}"] = df[f"lag_cl_{k}"] - df[f"lag_cl_{k+1}"]
        df[f"delta_ar_{k}"] = df[f"lag_ar_{k}"] - df[f"lag_ar_{k+1}"]

        df[f"delta_err_cl_{k}"] = df[f"err_cl_{k}"] - df[f"err_cl_{k+1}"]
        df[f"delta_err_ar_{k}"] = df[f"err_ar_{k}"] - df[f"err_ar_{k+1}"]

        feature_cols += [
            f"delta_y_{k}", f"delta_cl_{k}", f"delta_ar_{k}",
            f"delta_err_cl_{k}", f"delta_err_ar_{k}"
        ]

    # -------------------------
    # 3. Features contemporâneas
    # -------------------------
    df["feat_prev_cl_t"] = df["prev_cl"]
    df["feat_prev_ar_t"] = df["prev_ar"]

    df["feat_t"] = df["t"]
    df["feat_mes_sin"] = np.sin(2 * np.pi * df["mes_num"] / 12)
    df["feat_mes_cos"] = np.cos(2 * np.pi * df["mes_num"] / 12)

    feature_cols += ["feat_prev_cl_t", "feat_prev_ar_t", "feat_t",
                     "feat_mes_sin", "feat_mes_cos"]

    # Remover linhas incompletas
    df_feat = df.dropna(subset=feature_cols + ["alvo"]).reset_index(drop=True)

    return df_feat, feature_cols


# ============================================================
# 🔶 FUNÇÃO 2 — Next Step Predict (forecast recursivo sem vazamento)
# ============================================================

def next_step_predict(estado, prev_cl_t, prev_ar_t,
                      modelo_ml, feature_cols, lag_window):

    estado = estado.copy()

    # ----- Atualiza previsões estruturais do mês -----
    estado["feat_prev_cl_t"] = prev_cl_t
    estado["feat_prev_ar_t"] = prev_ar_t

    # ----- Previsão ML -----
    X_step = pd.DataFrame([estado[feature_cols]])
    y_ml_t = modelo_ml.predict(X_step)[0]

    # ----- Atualiza lags de Y -----
    for k in range(lag_window, 1, -1):
        estado[f"lag_y_{k}"] = estado[f"lag_y_{k-1}"]
    estado["lag_y_1"] = y_ml_t

    # ----- Atualiza lags clássico -----
    for k in range(lag_window, 1, -1):
        estado[f"lag_cl_{k}"] = estado[f"lag_cl_{k-1}"]
    estado["lag_cl_1"] = prev_cl_t

    # ----- Atualiza lags ARIMA -----
    for k in range(lag_window, 1, -1):
        estado[f"lag_ar_{k}"] = estado[f"lag_ar_{k-1}"]
    estado["lag_ar_1"] = prev_ar_t

    # ----- Calcula novos erros -----
    err_cl_t = y_ml_t - prev_cl_t
    err_ar_t = y_ml_t - prev_ar_t

    for k in range(lag_window, 1, -1):
        estado[f"err_cl_{k}"] = estado[f"err_cl_{k-1}"]
        estado[f"err_ar_{k}"] = estado[f"err_ar_{k-1}"]

    estado["err_cl_1"] = err_cl_t
    estado["err_ar_1"] = err_ar_t

    # ----- Atualiza deltas -----
    for k in range(1, lag_window):
        estado[f"delta_y_{k}"] = estado[f"lag_y_{k}"] - estado[f"lag_y_{k+1}"]
        estado[f"delta_cl_{k}"] = estado[f"lag_cl_{k}"] - estado[f"lag_cl_{k+1}"]
        estado[f"delta_ar_{k}"] = estado[f"lag_ar_{k}"] - estado[f"lag_ar_{k+1}"]

        estado[f"delta_err_cl_{k}"] = estado[f"err_cl_{k}"] - estado[f"err_cl_{k+1}"]
        estado[f"delta_err_ar_{k}"] = estado[f"err_ar_{k}"] - estado[f"err_ar_{k+1}"]

    # ----- Avança no tempo -----
    estado["t"] += 1
    estado["mes_num"] = (estado["mes_num"] % 12) + 1

    return estado, y_ml_t

def treinar_modelo_ml(df_filial, df_prev_classico_full, df_prev_arima_completo,
                      data_corte, lag_window=6):

    # monta df_base com prev_cl + prev_ar
    df_base = (
        df_filial[['data', 'alvo', 't', 'mes_num']]
        .merge(df_prev_classico_full[['data','previsao']].rename(columns={'previsao':'prev_cl'}),
               on='data', how='left')
        .merge(df_prev_arima_completo[['data','previsao']].rename(columns={'previsao':'prev_ar'}),
               on='data', how='left')
        .sort_values('data')
        .reset_index(drop=True)
    )

    # cria features
    df_feat, feature_cols = build_lags(df_base, lag_window)

    # corta para treino
    df_train = df_feat[df_feat['data'] <= data_corte]

    # treina XGB
    modelo_ml = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    modelo_ml.fit(df_train[feature_cols], df_train['alvo'])

    # ⚠️ PREVISÕES HISTÓRICAS DO ML — ESSENCIAL PARA NÃO QUEBRAR NO PREVER!
    df_feat["prev_ml_hist"] = modelo_ml.predict(df_feat[feature_cols])

    # monta objeto
    return {
        "modelo": modelo_ml,
        "feature_cols": feature_cols,
        "lag_window": lag_window,
        "df_feat": df_feat,
        "df_base": df_base
    }


def prever_ml(modelo_ml_obj, df_filial, df_prev_classico_full, df_prev_arima_completo,
              data_corte, meses_a_frente, is_share=False):

    modelo_ml = modelo_ml_obj["modelo"]
    feature_cols = modelo_ml_obj["feature_cols"]
    lag_window = modelo_ml_obj["lag_window"]
    df_feat = modelo_ml_obj["df_feat"]

    # Histórico até corte
    df_train = df_feat[df_feat['data'] <= data_corte]
    if df_train.empty:
        raise ValueError("Não há base histórica suficiente para ML até a data de corte!")

    estado = df_train.iloc[-1].copy()

    # Descobre quantos steps são necessários (histórico + futuro)
    ult_data_hist = df_filial['data'].max()
    gap_meses = max(0, (ult_data_hist.year - data_corte.year)*12 +
                       (ult_data_hist.month - data_corte.month))
    total_steps = gap_meses + meses_a_frente

    futuros = []

    for i in range(1, total_steps+1):
        data_fut = data_corte + pd.DateOffset(months=i)

        # ------- Segurança obrigatória -------  
        # tenta buscar prev clássico e ARIMA — se não existir, pula
        row_cl = df_prev_classico_full.loc[df_prev_classico_full['data'] == data_fut]
        row_ar = df_prev_arima_completo.loc[df_prev_arima_completo['data'] == data_fut]

        if row_cl.empty or row_ar.empty:
            # Se a data ainda não existe nas previsões clássica/ARIMA, não dá pra prever ML
            continue

        prev_cl_t = float(row_cl["previsao"].values[0])
        prev_ar_t = float(row_ar["previsao"].values[0])

        # ------- previsão recursiva -------  
        estado, y_ml_t = next_step_predict(
            estado,
            prev_cl_t,
            prev_ar_t,
            modelo_ml,
            feature_cols,
            lag_window
        )

        futuros.append({
            "data": data_fut,
            "previsao": float(y_ml_t)
        })

    # Histórico já previsto pelo ML
    df_prev_ml_hist = (
        df_feat[df_feat['data'] <= data_corte][["data", "prev_ml_hist"]]
        .rename(columns={"prev_ml_hist": "previsao"})
        .dropna(subset=["previsao"])
    )

    df_prev_ml_fut = pd.DataFrame(futuros)

    # Junta hist + futuro
    df_prev_ml_full = pd.concat([df_prev_ml_hist, df_prev_ml_fut], ignore_index=True)

    # Ordena e limpa
    df_prev_ml_full = (
        df_prev_ml_full
        .dropna(subset=["data", "previsao"])
        .sort_values("data")
        .reset_index(drop=True)
    )

    # ICs triviais (ML ainda não usa sigma)
    df_prev_ml_full["ic_inf"] = df_prev_ml_full["previsao"]
    df_prev_ml_full["ic_sup"] = df_prev_ml_full["previsao"]

    # Clip (para Share %)
    df_prev_ml_full = clip_predictions(df_prev_ml_full, is_share)

    return df_prev_ml_full


# ===============================
# Upload
# ===============================
arquivo = st.file_uploader("📤 Faça upload do Excel de vendas", type=["xlsx"])

if not arquivo:
    st.markdown("""
    <div style='margin-top: 2rem; font-size: 18px;'>
      🔍 <b>Instruções:</b><br>
      • Envie um .xlsx com a aba <code>Planilha1</code> (ou usaremos a primeira).<br>
      • Colunas: <code>ano</code>, <code>mês</code>, <code>Filial</code>, <code>Realizado</code>.<br>
      • Depois, escolha filial e data de corte. 🎯
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ===============================
# Carregamento e preparo base
# ===============================
with st.spinner("Lendo e preparando os dados..."):
    base = safe_sheet_read(arquivo)
    base = normalize_columns(base)

    # garante tipos
    base['ano'] = pd.to_numeric(base['ano'], errors='coerce').astype('Int64')
    base['mes'] = pd.to_numeric(base['mes'], errors='coerce').astype('Int64')
    base['realizado'] = pd.to_numeric(base['realizado'], errors='coerce')
    base['filial'] = base['filial'].astype(str).str.strip()

    # data mensal
    base['data'] = [to_month_start_date(a, m) for a, m in zip(base['ano'], base['mes'])]
    base = base.dropna(subset=['data']).sort_values('data')

    # total por data e linha "Total"
    df_total = base.groupby('data', as_index=False)['realizado'].sum()
    df_total['filial'] = 'Total'
    df_total['ano'] = df_total['data'].dt.year
    df_total['mes'] = df_total['data'].dt.month

    df_com_total = pd.concat([base, df_total], ignore_index=True)
    # total de filiais (exclui 'Total' pra evitar dobrar a soma)
    soma_filiais = (
        df_com_total[df_com_total['filial'] != 'Total']
        .groupby('data')['realizado']
        .transform('sum')
    )
    df_com_total['total_por_data'] = soma_filiais
    df_com_total['pct_filial'] = df_com_total['realizado'] / df_com_total['total_por_data']

# ===============================
# Seleções
# ===============================
modelo_realizado = st.selectbox("🏁 Base do realizado", ['Real R$', 'Real Share %'])

# filiais disponíveis
if modelo_realizado == 'Real Share %':
    filiais_disponiveis = sorted(df_com_total[df_com_total['filial'] != 'Total']['filial'].unique())
else:
    filiais_disponiveis = sorted(df_com_total['filial'].unique())

filial = st.selectbox("🏬 Selecione a filial", filiais_disponiveis)

df_filial = (
    df_com_total[df_com_total['filial'] == filial]
    .sort_values('data')
    .reset_index(drop=True)
)

if df_filial.empty:
    st.error("Sem dados para a filial selecionada.")
    st.stop()

# coluna alvo e log
if modelo_realizado == 'Real Share %':
    df_filial['alvo'] = df_filial['pct_filial'].fillna(0.0)
else:
    df_filial['alvo'] = df_filial['realizado'].fillna(0.0)

# eixo temporal e mês
df_filial['t'] = np.arange(len(df_filial))
df_filial['mes_num'] = df_filial['data'].dt.month

datas = df_filial['data'].tolist()
default_corte = datas[-4] if len(datas) >= 4 else datas[-1]
data_corte = st.select_slider("📅 Data de corte para previsão", options=datas, value=default_corte)

tipo_tendencia = st.selectbox("📈 Tipo de Tendência", [ "Linear", "Quadrática","Média"])


# ===============================
# Tendência + sazonalidade (ENCAPSULADO)
# ===============================
modelo_classico, df_treino, df_real = treinar_modelo_classico(
    df_filial=df_filial,
    data_corte=data_corte,
    tipo_tendencia=tipo_tendencia
)

# pega sigma e sazonalidade se quiser usar em outros lugares (ARIMA, etc.)
sigma_ruido = modelo_classico.sigma_ruido
saz_media = modelo_classico.saz_media

# Previsão clássica (sem ARIMA) para todo histórico
prevs = []
for _, row in df_filial.iterrows():
    t_i = int(row['t'])
    m_i = int(row['mes_num'])
    y, ic_inf, ic_sup, _ = modelo_classico.prever_nivel_com_ic(t_i, m_i)
    prevs.append({
        'data': row['data'],
        'previsao': y,
        'ic_inf': ic_inf,
        'ic_sup': ic_sup
    })

df_prev_classico_hist = pd.DataFrame(prevs)

# ===============================
# ARIMA nos resíduos (ENCAPSULADO)
# ===============================
st.markdown("---")
st.subheader("🔮 Previsão de Vendas Futuras")

meses_a_frente = st.number_input("Meses à frente", min_value=1, max_value=24, value=6)

usar_otimizacao = st.checkbox("🔍 Otimizar parâmetros do ARIMA automaticamente", value=False)

if usar_otimizacao:
    # rodamos o rolling-origin para escolher o melhor ARIMA
    melhor, hist = otimizar_arima(
        df_treino=df_treino,
        metrica='mape'
    )
    p, d, q = melhor['p'], melhor['d'], melhor['q']
    st.success(f"Melhor ARIMA encontrado: ({p},{d},{q}) — erro médio = {melhor['erro']:.4f}")

else:
    ordem_arima_txt = st.text_input("Ordem do ARIMA (p,d,q)", "1,0,0")
    p, d, q = parse_arima_order(ordem_arima_txt)
    st.info(f"Usando ARIMA manual: ({p},{d},{q})")


# ---------------------------------------
# treinar modelo ARIMA encapsulado
# ---------------------------------------
modelo_arima = treinar_arima_ruido(df_treino, (p, d, q))


# ---------------------------------------
# Reconstrução dos resíduos (A + B + C)
# ---------------------------------------
serie_residuo = df_treino.set_index("data")["ruido"]

tam_A = len(df_treino)
tam_B = len(df_filial) - len(df_treino)
tam_C = meses_a_frente

# resíduos fitted (A)
residuos_A = modelo_arima.modelo.fittedvalues

# resíduos futuros (B + C)
steps_forecast = tam_B + tam_C
residuos_BC = modelo_arima.modelo.forecast(steps=steps_forecast)

# junta tudo
residuos_todos = pd.concat([residuos_A, residuos_BC])

datas_todas = pd.date_range(
    start=df_filial['data'].iloc[0],
    periods=len(residuos_todos),
    freq='MS'
)

df_all = pd.DataFrame({
    'data': datas_todas,
    't': np.arange(len(residuos_todos)),
    'mes_num': datas_todas.month,
    'ruido_prev': residuos_todos.values
})

# reconstruir previsões completas
prevs_arima_full = []
for _, row in df_all.iterrows():
    t_i = int(row['t'])
    m_i = int(row['mes_num'])

    tend_saz_log = modelo_classico.prever_log(t_i, m_i)
    ruido_i = float(row['ruido_prev'])

    yhat_log = tend_saz_log + ruido_i

    prevs_arima_full.append({
        'data': row['data'],
        'yhat_log': yhat_log,
        'previsao': np.exp(yhat_log) - 1.0
    })

df_prev_arima_completo = pd.DataFrame(prevs_arima_full)
# ===============================
# Sigma do ARIMA (somente período A, sem vazamento)
# ===============================

# 1) Reconstruir tendência + sazonalidade no período A
df_A = df_treino.copy()

df_A["tend_saz_log"] = [
    modelo_classico.prever_log(int(row.t), int(row.data.month))
    for _, row in df_A.iterrows()
]

# 2) Adiciona o ruído fitted (ARIMA) — alinhado pelo índice
df_A["ruido_fitted"] = residuos_A.values

# 3) Reconstrói o yhat no log
df_A["yhat_log"] = df_A["tend_saz_log"] + df_A["ruido_fitted"]

# 4) Cálculo do erro no log somente na área de treino
df_A["erro_log"] = df_A["log_venda"] - df_A["yhat_log"]

# 5) Sigma baseado somente no treino
sigma_arima = float(df_A["erro_log"].std())
if not np.isfinite(sigma_arima) or sigma_arima == 0:
    sigma_arima = 1e-6
if not np.isfinite(sigma_arima) or sigma_arima == 0:
    sigma_arima = 1e-6

# ===============================
# Aplicar Intervalo de Confiança no forecast completo (A + B + C)
# ===============================
df_prev_arima_completo["ic_inf"] = (
    np.exp(df_prev_arima_completo["yhat_log"] - 2.57 * sigma_arima) - 1.0
)

df_prev_arima_completo["ic_sup"] = (
    np.exp(df_prev_arima_completo["yhat_log"] + 2.57 * sigma_arima) - 1.0
)


# Junta clássico (histórico) + futuro clássico sem ARIMA
# (futuro clássico reaproveita sigma_ruido e tendência+sazonal)
ult_data = df_filial['data'].max()
datas_fut = pd.date_range(start=ult_data + pd.DateOffset(months=1), periods=meses_a_frente, freq='MS')
fut = []
start_t = int(df_filial['t'].max()) + 1
for i, dta in enumerate(datas_fut):
    t_i = start_t + i
    m_i = int(dta.month)
    yhat_log = modelo_classico.prever_log(t_i, m_i)

    fut.append({
        'data': dta,
        'previsao': np.exp(yhat_log) - 1.0,
        'ic_inf': np.exp(yhat_log - 2.57*sigma_ruido) - 1.0,
        'ic_sup': np.exp(yhat_log + 2.57*sigma_ruido) - 1.0
    })
df_prev_futuro_classico = pd.DataFrame(fut)

df_prev_classico_hist = df_prev_classico_hist.sort_values('data')
df_prev_classico_full = pd.concat([df_prev_classico_hist, df_prev_futuro_classico], ignore_index=True)

# Clip de previsões conforme o modo (R$ vs Share)
is_share = (modelo_realizado == 'Real Share %')
df_prev_classico_full = clip_predictions(df_prev_classico_full, is_share)
df_prev_arima_completo = clip_predictions(df_prev_arima_completo, is_share)

# ============================================================
# 🔶 APLICAÇÃO MACHINE LEARNING
# ============================================================
# ============================================================
# 🔶 PIPELINE ML COMPLETO (treino + histórico + forecast futuro)
# ============================================================

# Ativa o modelo ML se o usuário quiser
usar_ml = st.checkbox("🤖 Ativar previsão com Machine Learning (ML)", value=False)

if usar_ml:
    st.subheader("🔮 Previsão via Modelo ML (Meta-Model)")
    lag_window = st.slider("Janela de lags (N)", min_value=3, max_value=12, value=6)

    # 1) Treina ML
    modelo_ml_obj = treinar_modelo_ml(
        df_filial=df_filial,
        df_prev_classico_full=df_prev_classico_full,
        df_prev_arima_completo=df_prev_arima_completo,
        data_corte=data_corte,
        lag_window=lag_window
    )

    # 2) Faz toda previsao (histórico + futuro)
    df_prev_ml_full = prever_ml(
        modelo_ml_obj,
        df_filial,
        df_prev_classico_full,
        df_prev_arima_completo,
        data_corte,
        meses_a_frente
    )

    # 3) Clipa se share
    df_prev_ml_full = clip_predictions(df_prev_ml_full, is_share)

  
# ======================================================
# 🎨 GRÁFICO MASTER — Realizado vs Clássico vs ARIMA vs ML
# ======================================================
st.markdown("## 📊 Comparativo de Modelos — Filial " + filial)

fig, ax = plt.subplots(figsize=(16, 6))

# ------------------------------------------------------
# BASE REAL - sempre atrás
# ------------------------------------------------------
ax.bar(df_filial['data'], 
       df_filial['alvo'], 
       width=20, 
       color='gray', 
       alpha=0.35, 
       label='Realizado', 
       zorder=1)

# ------------------------------------------------------
# CLÁSSICO (linha + IC FUTURO apenas)
# ------------------------------------------------------
ax.plot(df_prev_classico_full['data'],
        df_prev_classico_full['previsao'],
        'o--', color='black', label='Clássico', zorder=4)

mask_future = df_prev_classico_full['data'] >= data_corte
ax.fill_between(df_prev_classico_full.loc[mask_future, 'data'],
                df_prev_classico_full.loc[mask_future, 'ic_inf'],
                df_prev_classico_full.loc[mask_future, 'ic_sup'],
                alpha=0.12, color='black', zorder=2)

# ------------------------------------------------------
# ARIMA (linha + IC FUTURO apenas)
# ------------------------------------------------------
ax.plot(df_prev_arima_completo['data'],
        df_prev_arima_completo['previsao'],
        'o-', color='royalblue', linewidth=2,
        label='ARIMA (Resíduos)', zorder=5)

mask_future_arima = df_prev_arima_completo['data'] >= data_corte
ax.fill_between(df_prev_arima_completo.loc[mask_future_arima, 'data'],
                df_prev_arima_completo.loc[mask_future_arima, 'ic_inf'],
                df_prev_arima_completo.loc[mask_future_arima, 'ic_sup'],
                alpha=0.18, color='royalblue', zorder=2)

# ------------------------------------------------------
# ML (linha apenas, sem banda)
# ------------------------------------------------------
if usar_ml:
    ax.plot(df_prev_ml_full['data'],
            df_prev_ml_full['previsao'],
            'o-', color='purple', linewidth=2,
            label='ML Meta-Model', zorder=6)

# ------------------------------------------------------
# CORTE
# ------------------------------------------------------
ax.axvline(data_corte, linestyle='--', color='red', linewidth=2,
           label='Data de Corte', zorder=10)

# ------------------------------------------------------
# 📌 FORMATAÇÃO PRELIMINAR
# ------------------------------------------------------
ax.set_title(f"📈 Comparativo de Modelos — Filial {filial}", fontsize=16)
ax.set_xlabel("Data")
ax.set_ylabel("Vendas" if not is_share else "Participação (%)")
ax.legend()

# =========================================================
# 🔧 AJUSTE AUTOMÁTICO DO EIXO Y (40% ABAIXO DO MENOR VALOR)
# =========================================================
valores_min = []

# previsões
valores_min.append(df_prev_classico_full['previsao'].min())
valores_min.append(df_prev_arima_completo['previsao'].min())

if usar_ml:
    valores_min.append(df_prev_ml_full['previsao'].min())

# histórico (alvo real)
valores_min.append(df_filial['alvo'].min())

# menor valor geral
vmin = min(valores_min)

# aplica o desconto de 40%
limite_inferior = vmin * 0.95
ax.set_ylim(bottom=limite_inferior)

# =========================================================
# SALVAR EM PNG (IMPORTANTE: *DEPOIS* DO AJUSTE DE EIXO)
# =========================================================
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
buf.seek(0)

# renderiza no streamlit
st.pyplot(fig)

# ======================================================
# 📐 PREPARAÇÃO DAS SÉRIES PARA MÉTRICAS
# ======================================================

# 1) Blindagem — padroniza coluna alvo
if "valor" in df_filial.columns and "alvo" not in df_filial.columns:
    df_filial = df_filial.rename(columns={"valor": "alvo"})

# 2) Montar dataframes alinhados (somente datas em comum)
df_classico_valid = (
    df_filial[['data', 'alvo']]
    .merge(df_prev_classico_full[['data', 'previsao']], on='data', how='inner')
    .dropna(subset=['alvo', 'previsao'])
    .rename(columns={'previsao': 'prev_class'})
)

df_arima_valid = (
    df_filial[['data', 'alvo']]
    .merge(df_prev_arima_completo[['data', 'previsao']], on='data', how='inner')
    .dropna(subset=['alvo', 'previsao'])
    .rename(columns={'previsao': 'prev_arima'})
)

df_ml_valid = None
if usar_ml:
    df_ml_valid = (
        df_filial[['data', 'alvo']]
        .merge(df_prev_ml_full[['data', 'previsao']], on='data', how='inner')
        .dropna(subset=['alvo', 'previsao'])
        .rename(columns={'previsao': 'prev_ml'})
    )

# 3) Máscaras de período
mask_obs = df_classico_valid['alvo'].notnull()
mask_pos = df_classico_valid['data'] > data_corte

# ======================================================
# 📐 FUNÇÃO AUXILIAR PARA MÉTRICAS
# ======================================================
def metricas(df, col_pred, mask_total, mask_pos):
    if df is None:
        return None, None
    m_total = calc_metrics(
        df.loc[mask_total, 'alvo'],
        df.loc[mask_total, col_pred],
    )
    m_pos = (
        calc_metrics(
            df.loc[mask_pos, 'alvo'],
            df.loc[mask_pos, col_pred]
        )
        if mask_pos.any()
        else {'MAPE (%)': np.nan, 'R²': np.nan, 'RMSE': np.nan}
    )
    return m_total, m_pos

# ======================================================
# 📐 MÉTRICAS FINAIS
# ======================================================
m_class_total, m_class_pos = metricas(df_classico_valid, "prev_class", mask_obs, mask_pos)
m_arima_total, m_arima_pos = metricas(df_arima_valid, "prev_arima", mask_obs, mask_pos)

if usar_ml:
    mask_obs_ml = df_ml_valid['alvo'].notnull()
    mask_pos_ml = df_ml_valid['data'] > data_corte
    m_ml_total, m_ml_pos = metricas(df_ml_valid, "prev_ml", mask_obs_ml, mask_pos_ml)
else:
    m_ml_total, m_ml_pos = None, None

# ======================================================
# 🖥️ EXIBIÇÃO
# ======================================================
st.markdown("## 📐 Métricas Comparativas dos Modelos")

# ----- TOTAL -----
st.markdown("### 🟢 Período Completo")

c1, c2, c3 = st.columns(3)
c1.metric("Clássico — MAPE (%)", f"{m_class_total['MAPE (%)']:.2f}")
c2.metric("Clássico — R²", f"{m_class_total['R²']:.3f}")
c3.metric("Clássico — RMSE", f"{m_class_total['RMSE']:,.2f}")

c1, c2, c3 = st.columns(3)
c1.metric("ARIMA — MAPE (%)", f"{m_arima_total['MAPE (%)']:.2f}")
c2.metric("ARIMA — R²", f"{m_arima_total['R²']:.3f}")
c3.metric("ARIMA — RMSE", f"{m_arima_total['RMSE']:,.2f}")

if usar_ml:
    c1, c2, c3 = st.columns(3)
    c1.metric("ML — MAPE (%)", f"{m_ml_total['MAPE (%)']:.2f}")
    c2.metric("ML — R²", f"{m_ml_total['R²']:.3f}")
    c3.metric("ML — RMSE", f"{m_ml_total['RMSE']:,.2f}")

# ----- PÓS-CORTE -----
st.markdown("### 🔴 Após o Corte")

c1, c2, c3 = st.columns(3)
c1.metric("Clássico — MAPE (%)", f"{m_class_pos['MAPE (%)']:.2f}" if pd.notna(m_class_pos['MAPE (%)']) else "—")
c2.metric("Clássico — R²", f"{m_class_pos['R²']:.3f}" if pd.notna(m_class_pos['R²']) else "—")
c3.metric("Clássico — RMSE", f"{m_class_pos['RMSE']:,.2f}" if pd.notna(m_class_pos['RMSE']) else "—")

c1, c2, c3 = st.columns(3)
c1.metric("ARIMA — MAPE (%)", f"{m_arima_pos['MAPE (%)']:.2f}" if pd.notna(m_arima_pos['MAPE (%)']) else "—")
c2.metric("ARIMA — R²", f"{m_arima_pos['R²']:.3f}" if pd.notna(m_arima_pos['R²']) else "—")
c3.metric("ARIMA — RMSE", f"{m_arima_pos['RMSE']:,.2f}" if pd.notna(m_arima_pos['RMSE']) else "—")

if usar_ml:
    c1, c2, c3 = st.columns(3)
    c1.metric("ML — MAPE (%)", f"{m_ml_pos['MAPE (%)']:.2f}" if pd.notna(m_ml_pos['MAPE (%)']) else "—")
    c2.metric("ML — R²", f"{m_ml_pos['R²']:.3f}" if pd.notna(m_ml_pos['R²']) else "—")
    c3.metric("ML — RMSE", f"{m_ml_pos['RMSE']:,.2f}" if pd.notna(m_ml_pos['RMSE']) else "—")


# ===============================
# Logs (debug)
# ===============================
with st.expander("🛠️ Logs (debug)"):
    st.write(f"Sigma clássico (ruído): {sigma_ruido:.6f}")
    st.write(f"Sigma ARIMA (erro_log): {sigma_arima:.6f}")
    st.write(f"ARIMA final: ({p},{d},{q}) — treino em {len(serie_residuo)} pontos")
    st.write(f"Dados disponíveis: {df_filial['data'].min().date()} → {df_filial['data'].max().date()} | corte: {data_corte.date()}")



# ======================================================
# 🔁 ROLLING EVALUATION — 3 MODELOS (re-treinado a cada corte)
# ======================================================
st.markdown("## 🔁 Avaliação Rolling — 1 passo à frente (Realista)")

rodar_rolling = st.button("📉 Rodar Avaliação Rolling (Realista)")

if rodar_rolling:
    st.warning("Executando rolling realista... isso pode demorar...")

    # -----------------------
    # CLÁSSICO
    # -----------------------
    df_roll_class = rolling_classico(
        df_filial=df_filial,
        tipo_tendencia=tipo_tendencia,
        janela_minima=12
    )

    # -----------------------
    # ARIMA
    # -----------------------
    df_roll_arima = rolling_arima(
        df_filial=df_filial,
        tipo_tendencia=tipo_tendencia,
        arima_order=(p, d, q),
        janela_minima=12
    )

    # -----------------------
    # ML
    # -----------------------
    if usar_ml:
        df_roll_ml = rolling_ml(
            df_filial=df_filial,
            tipo_tendencia=tipo_tendencia,
            arima_order=(p, d, q),
            lag_window=lag_window,
            janela_minima=12
        )
    else:
        df_roll_ml = None

    # -----------------------
    # RESUMO MÉTRICAS
    # -----------------------
    st.subheader("📊 Resultado Rolling Realista (MAPE Médio)")

    resumo = []
    resumo.append({
        "Modelo": "Clássico",
        "MAPE (%)": df_roll_class["erro_pct"].mean()*100,
        "Erro Abs Médio": df_roll_class["erro_abs"].mean()
    })
    resumo.append({
        "Modelo": "ARIMA",
        "MAPE (%)": df_roll_arima["erro_pct"].mean()*100,
        "Erro Abs Médio": df_roll_arima["erro_abs"].mean()
    })

    if usar_ml:
        resumo.append({
            "Modelo": "ML",
            "MAPE (%)": df_roll_ml["erro_pct"].mean()*100,
            "Erro Abs Médio": df_roll_ml["erro_abs"].mean()
        })

    st.dataframe(pd.DataFrame(resumo))

    # -----------------------
    # DETALHES
    # -----------------------
    st.markdown("### 📘 Rolling — Clássico")
    st.dataframe(df_roll_class)

    st.markdown("### 📘 Rolling — ARIMA")
    st.dataframe(df_roll_arima)

    if usar_ml:
        st.markdown("### 📘 Rolling — ML")
        st.dataframe(df_roll_ml)


    fig2, ax2 = plt.subplots(figsize=(16,6))

    ax2.plot(df_roll_class["data_prev"], df_roll_class["erro_pct"]*100,
            '-o', label="Clássico", color='black')

    ax2.plot(df_roll_arima["data_prev"], df_roll_arima["erro_pct"]*100,
            '-o', label="ARIMA", color='royalblue')

    if usar_ml:
        ax2.plot(df_roll_ml["data_prev"], df_roll_ml["erro_pct"]*100,
                '-o', label="ML", color='purple')

    ax2.set_title("Erro (%) por Corte — Rolling Evaluation")
    ax2.set_xlabel("Data prevista")
    ax2.set_ylabel("MAPE (%)")
    ax2.legend()

    st.pyplot(fig2)



# ===============================
# Diagnóstico LLM — Interpretação Automática
# ===============================
from openai import OpenAI

st.markdown("---")
st.header("🧩 Interpretação Automática das Previsões")

context_file = st.file_uploader(
    "📎 Anexar arquivo de contexto (opcional)",
    type=["pdf"]
)

if st.button("🧠 Gerar Interpretação com LLM"):

    with st.spinner("Analisando resultados e gerando interpretação..."):
        try:
            client = OpenAI(api_key=st.secrets["openai_api_key"])
            context_text = ""

            # ======================================================
            # 🔧 BLOCO OPCIONAL — Rolling
            # ======================================================
            rolling_text = ""
            rolling_explicacao = """
A Avaliação Rolling simula como cada modelo teria performado caso estivéssemos,
historicamente, em “tempo real”. Ou seja: em cada mês t, o modelo é treinado
usando apenas os dados disponíveis até t, e faz uma previsão para t+1.

Esse procedimento evita absolutamente qualquer vazamento e mede:
• estabilidade temporal do modelo,
• robustez da tendência e da sazonalidade,
• sensibilidade a mudanças de regime,
• consistência real do modelo ao longo da série.

É um teste muito mais rigoroso e próximo da vida real do que avaliar apenas o
período pós-corte atual.
"""

            # ======================================================
            # MÉTRICAS DO ROLLING (se existirem)
            # ======================================================
            if "df_roll_class" in locals():

                mape_class_roll = df_roll_class["erro_pct"].mean() * 100
                err_class_roll = df_roll_class["erro_abs"].mean()

                mape_arima_roll = df_roll_arima["erro_pct"].mean() * 100
                err_arima_roll = df_roll_arima["erro_abs"].mean()

                rolling_text += f"""
### 📉 Avaliação Rolling (1 passo à frente — realista)

- **Modelo Clássico**
    • MAPE médio rolling = {mape_class_roll:.2f}%  
    • Erro Absoluto Médio = {err_class_roll:.2f}

- **Modelo ARIMA**
    • MAPE médio rolling = {mape_arima_roll:.2f}%  
    • Erro Absoluto Médio = {err_arima_roll:.2f}
"""

                if usar_ml and "df_roll_ml" in locals():
                    mape_ml_roll = df_roll_ml["erro_pct"].mean() * 100
                    err_ml_roll = df_roll_ml["erro_abs"].mean()

                    rolling_text += f"""
- **Modelo ML**
    • MAPE médio rolling = {mape_ml_roll:.2f}%  
    • Erro Absoluto Médio = {err_ml_roll:.2f}
"""
            else:
                rolling_text = "Nenhuma avaliação rolling foi executada."

            # ======================================================
            # BLOCO ML
            # ======================================================
            if usar_ml:
                bloco_ml = f"""
- Modelo de Machine Learning (XGBoost):
    • Lags utilizados: {lag_window}
    • MAPE = {m_ml_total['MAPE (%)']:.2f}%
    • R² = {m_ml_total['R²']:.3f}
    • RMSE = {m_ml_total['RMSE']:.2f}
"""
            else:
                bloco_ml = ""

            # ======================================================
            # PROMPT
            # ======================================================
            prompt = f"""
Analise tecnicamente os resultados da previsão de vendas da filial "{filial}".

Use as métricas abaixo:

- Modelo Clássico: MAPE = {m_class_total['MAPE (%)']:.2f}%, R² = {m_class_total['R²']:.3f}, RMSE = {m_class_total['RMSE']:.2f}
- Modelo ARIMA:    MAPE = {m_arima_total['MAPE (%)']:.2f}%, R² = {m_arima_total['R²']:.3f}, RMSE = {m_arima_total['RMSE']:.2f}
{bloco_ml}

### 🧠 O que analisar
Produza uma interpretação clara e objetiva abordando:
- Qual modelo performa melhor e por quais motivos;
- Comportamento pós-corte: onde cada modelo acerta/erra;
- Confiabilidade dos intervalos de confiança;
- Possíveis causas para desvios (mudança de regime, sazonalidade atípica, rupturas);
- Sugestões de melhoria (variáveis externas, sazonalidade dinâmica, janela de lags, ML etc.).

### 🌀 Explicação da Avaliação Rolling
{rolling_explicacao}

### 📉 Análise Rolling (resultado realista)
{rolling_text}

Se houver arquivo de contexto, integre os fatos relevantes e gere insights práticos.
Mantenha tom consultivo, direto e técnico.
"""

            # contexto
            if context_file:
                context_text = extract_context_text(context_file)

            prompt += context_text[:15000] + "\n---\n"

            # imagem
            img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um consultor sênior especializado em séries temporais e gestão comercial."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }}
                        ]
                    }
                ],
                temperature=0.4,
                max_tokens=900
            )

            relatorio = response.choices[0].message.content

            st.markdown("### 📊 Interpretação Gerada:")
            st.markdown(relatorio)

            st.session_state['relatorio_llm'] = relatorio

        except Exception as e:
            st.error(f"Erro ao gerar interpretação: {e}")


# ===============================
# 📄 PDF FINAL (sempre aparece depois de já existir relatório)
# ===============================
if 'relatorio_llm' in st.session_state:
    st.markdown("---")
    st.subheader("📄 Gerar PDF do Relatório Completo")

    if st.button("📥 Baixar PDF"):
        try:
            pdf_bytes = gerar_pdf_completo(
                filial,
                m_class_total, m_arima_total, m_ml_total,
                m_class_pos, m_arima_pos, m_ml_pos,
                st.session_state['relatorio_llm'],
                buf.getvalue()
            )

            st.download_button(
                label="⬇️ Clique aqui para baixar",
                data=pdf_bytes,
                file_name=f"relatorio_previsao_{filial}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Erro ao gerar PDF: {e}")






