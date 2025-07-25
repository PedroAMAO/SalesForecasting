import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
import xgboost as xgb  
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from itertools import product
import warnings
from statsmodels.tsa.stattools import adfuller
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

os.environ["OMP_NUM_THREADS"] = "1"  # Para OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"





warnings.filterwarnings("ignore")


#dicionario de hiperparametros
param_grids = {
    "Random Forest": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "Bayesian Ridge": {
        "alpha_1": [1e-6, 1e-5, 1e-4],
        "lambda_1": [1e-6, 1e-5, 1e-4]
    },
    "MLP": {
        "hidden_layer_sizes": [(4,), (8,), (16,), (8,8) ,(16,4), (32,16,8), (16,8,4,2),  (64,32,16,8), (128,64,32,16,8)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "activation": ["relu", "tanh", "logistic", "identity"],  # <-- incluído
        "max_iter": [10000]
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200,500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.1, 0.3]
    },
    "Decision Tree": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "K-Nearest Neighbors": {
        "n_neighbors": [3, 5, 7, 9, 15],
        "weights": ["uniform", "distance"]
    }
}



# Configuração inicial do Streamlit
st.set_page_config(layout="wide")

# CSS personalizado para melhor visualização
st.markdown("""
    <style>
        .main { padding: 2rem 5rem; }
        .block-container { max-width: 100%; min-height: 90vh; }
        html, body { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title("📈 Previsão com Decomposição Log + Tendência + Sazonalidade")

# Upload do Excel
arquivo = st.file_uploader("📤 Faça upload do Excel de vendas", type="xlsx")

# Instruções detalhadas
if not arquivo:
    st.markdown("""
        <div style='margin-top: 2rem; font-size: 18px;'>
            🔍 <b>Instruções:</b><br>
            - Faça upload de um arquivo Excel (.xlsx) com a aba <code>Planilha1</code><br>
            - Colunas necessárias: <code>ano</code>, <code>mês</code>, <code>Filial</code>, <code>Realizado</code><br>
            - Após o upload, selecione a filial e a data de corte 🎯
        </div>
    """, unsafe_allow_html=True)

# Executar após upload
# Executar após upload
if arquivo:
    # Carregar dados
    df = pd.read_excel(arquivo, sheet_name='Planilha1')
    
    df = df.iloc[:, :4]  # mantém só as 4 primeiras colunas
    
    df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mês'].astype(str) + '-01')
    #print(df.head())

    # Agrupa por ano e mês e soma o Realizado
    df_total = df.groupby(['data'], as_index=False)['Realizado'].sum()
    

    # Marca como "Total" na coluna Filial
    df_total['Filial'] = 'Total'

    # Garante a mesma ordem das colunas
    #df_total = df_total[['ano', 'mês', 'Filial', 'Realizado']]

    # Junta com o DataFrame original
    df_com_total = pd.concat([df, df_total], ignore_index=True)

    #Calculo do Share por filial

    #adiciona a Coluna Share por filial por data
    #df_com_total['total_por_data'] = df_com_total.groupby('data')['Realizado'].transform('sum')
    df_com_total['total_por_data'] = (
        df_com_total[df_com_total['Filial'] != 'Total']
        .groupby('data')['Realizado']
        .transform('sum')
    )   
    df_com_total['pct_filial'] = df_com_total['Realizado'] / df_com_total['total_por_data']


    # Selecionar Reais ou Share
    modelo_realizado = st.selectbox("🏬 Selecione o realizado", ['Real R$', 'Real Share %'])


    # Selecionar filial
    
    # Filtra lista de filiais conforme o modelo selecionado
    if modelo_realizado == 'Real Share %':
        filiais_disponiveis = sorted(df_com_total[df_com_total['Filial'] != 'Total']['Filial'].unique())
    else:
        filiais_disponiveis = sorted(df_com_total['Filial'].unique())

    # Selecionar filial
    filial = st.selectbox("🏬 Selecione a filial", filiais_disponiveis)
    df_filial = df_com_total[df_com_total['Filial'] == filial].sort_values('data').reset_index(drop=True)
    #print(df_filial.head())
    # Escolher data de corte
    datas = df_filial['data'].tolist()
    data_corte = st.select_slider("📅 Data de corte para previsão", options=datas, value=datas[-4])

    # Variáveis auxiliares
    df_filial['t'] = np.arange(len(df_filial))
    df_filial['mes'] = df_filial['data'].dt.month

    #Incluir Total (Somatorio filiais)
    


    # Dados de treino até data de corte
    df_treino = df_filial[df_filial['data'] <= data_corte].copy()
    df_treino['log_venda'] = np.log(df_treino['Realizado'] + 1)

    df_real = df_filial[df_filial['Realizado'].notnull()].copy()
    
    df_real['log_venda'] = np.log(df_real['Realizado'] + 1)



    if modelo_realizado =='Real Share %':
        df_treino['log_venda'] = np.log(df_treino['pct_filial']+1)
        df_real['log_venda'] = np.log(df_real['pct_filial']+1)
        df_filial['log_venda'] = np.log(df_filial['pct_filial']+1)
        df_treino['Realizado'] = df_treino['pct_filial']
        df_real['Realizado'] = df_real['pct_filial']
        df_filial['Realizado'] = df_filial['pct_filial']
    
    
    #cálculo da media
    media_realizado =  df_treino['log_venda'].mean()
    
    


    # Seleção do modelo de tendência
    tipo_tendencia = st.selectbox("📈 Tipo de Tendência", ["Linear", "Quadrática", "Média"])

    
    def prever_tendencia_individual(t_i):
        if tipo_tendencia == "Média":
            return 0  # sem tendência, a média já vai entrar no lugar da estrutura
        elif "Quadrática" in tipo_tendencia:
            x_input = np.array([[t_i, t_i**2]])
        else:
            x_input = np.array([[t_i]])
        return modelo_tendencia.predict(x_input)[0]

    # Y: sempre log
    y_treino = df_treino['log_venda']

   # Novo cálculo com base em TODO o período realizado
    


    
    



    # Exibição do histograma
    


    
    
    # Modelo de tendência (Regressão Linear)

    # Prepara X de acordo com o tipo
    if tipo_tendencia == "Média":
        tendencia = media_realizado 
        tendencia_real =  media_realizado
    else:
        X_treino = df_treino[['t']].copy()


        if tipo_tendencia == "Quadrática":
            X_treino['t2'] = X_treino['t'] ** 2

       
        # Ajusta modelo
        X_treino_split, X_teste_split, y_treino_split, y_teste_split = train_test_split(X_treino, y_treino, test_size=0.3, random_state=42)

        modelo_tendencia = LinearRegression().fit(X_treino_split, y_treino_split)
        tendencia = modelo_tendencia.predict(X_treino)

        
     

        # Reaplica o modelo de tendência (mesmo modelo treinado antes)
        X_real = df_real[['t']].copy()
        if "Quadrática" in tipo_tendencia:
            X_real['t2'] = X_real['t'] ** 2

        tendencia_real = modelo_tendencia.predict(X_real)


        

    # Salva resíduos
    df_treino['residuo'] = y_treino - tendencia
    df_real['residuo'] = df_real['log_venda'] - tendencia_real
    
    # Sazonalidade média
    
    if tipo_tendencia != "Média":
        media_sazonal_treino = df_treino.groupby('mes')['residuo'].mean()
    else:
        media_sazonal_treino = pd.Series(0, index=range(1,13))

    df_real['ruido'] = df_real['residuo'] - df_real['mes'].map(media_sazonal_treino)
    df_treino['ruido'] = df_treino['residuo'] - df_treino['mes'].map(media_sazonal_treino)
    # Calcula o desvio padrão do ruído no total
    sigma_ruido = df_treino['ruido'].std()

    # Proteção anti-NaN
    if pd.isna(sigma_ruido) or sigma_ruido == 0:
        sigma_ruido = 1e-6


    # Previsões históricas (dentro da base)
    previsoes = []
    for idx, row in df_filial.iterrows():
        t_i = row['t']
        mes_i = row['mes']
        if tipo_tendencia == "Média":
           tendencia_i = media_realizado
        else:
             
            tendencia_i =  prever_tendencia_individual(t_i)

        sazonal_i = media_sazonal_treino.get(mes_i, 0)
        yhat_log = tendencia_i + sazonal_i
        previsoes.append({
            'data': row['data'],
            'previsao': np.exp(yhat_log) - 1,
            'ic_sup': np.exp(yhat_log + 2.57 * sigma_ruido) - 1,
            'ic_inf': np.exp(yhat_log - 2.57 * sigma_ruido) - 1
        })
    df_prev = pd.DataFrame(previsoes)

    # Avaliação do modelo após o corte
    df_merged = pd.merge(df_filial, df_prev, on='data')
    df_teste = df_merged[df_merged['Realizado'].notnull()].copy()
   
    

    #aplicar ARIMA nos Residuos
    # Série temporal dos resíduos (apenas dados até a data de corte)
    #serie_residuo = df_treino.set_index('data')['residuo']

    # Define ordem do ARIMA (ajuste fino depois)
    
    
    # ⬛️ Checkbox para otimizar automaticamente
    usar_otimizacao = st.checkbox("🔍 Otimizar parâmetros do ARIMA automaticamente", value=False)

    # Se ativar, escolha a métrica
    if usar_otimizacao:
        metrica_label = st.selectbox("🎯 Métrica de otimização", ["R²", "MAPE"])
        metrica_arima = 'r2' if metrica_label == "R²" else 'mape'

    else:
        ordem_arima = st.text_input("Ordem do ARIMA (p,d,q)", "1,0,0")
        p, d, q = map(int, ordem_arima.split(','))


    def otimizar_arima(df_treino, df_filial, data_corte, media_sazonal_treino, prever_tendencia_individual, metrica='r2', grid_p=(0,1,2), grid_d=(0,1), grid_q=(0,1,2)):
        melhores_resultados = []

        serie_residuo = df_treino.set_index('data')['residuo']
        print(f"Série de resíduos: {len(serie_residuo)} pontos")

        for p, d, q in product(grid_p, grid_d, grid_q):
            try:
                modelo = ARIMA(serie_residuo, order=(p,d,q)).fit()

                if modelo.nobs < max(p, d, q) + 1:
                    print(f"❌ ARIMA({p},{d},{q}) descartado: poucos dados ({modelo.nobs})")
                    continue

                n_total = len(df_filial)
                residuos_previstos = modelo.predict(start=0, end=n_total - 1)

                if residuos_previstos.isna().any():
                    print(f"❌ ARIMA({p},{d},{q}) descartado: previsão com NaNs")
                    continue

                datas = pd.date_range(start=df_filial['data'].iloc[0], periods=n_total, freq='MS')
                previsoes = []
                for t_i, data_i in enumerate(datas):
                    mes_i = data_i.month
                    if tipo_tendencia == "Média":
                        tendencia_i = media_realizado
                    else:
             
                        tendencia_i =  prever_tendencia_individual(t_i)
                    #tendencia_i = prever_tendencia_individual(t_i)
                    sazonal_i = media_sazonal_treino.get(mes_i, 0)
                    ruido_i = residuos_previstos.iloc[t_i]
                    yhat_log = tendencia_i + sazonal_i + ruido_i
                    previsoes.append({'data': data_i, 'previsao': np.exp(yhat_log) - 1})

                df_prev = pd.DataFrame(previsoes)
                df_merge = pd.merge(df_filial, df_prev, on='data')
                df_valid = df_merge.dropna(subset=['Realizado', 'previsao'])

                if df_valid.empty:
                    print(f"❌ ARIMA({p},{d},{q}) descartado: df_valid vazio após merge")
                    continue

                y_true = df_valid['Realizado']
                y_pred = df_valid['previsao']

                if metrica == 'r2':
                    score = r2_score(y_true, y_pred)
                elif metrica == 'mape':
                    score = mean_absolute_percentage_error(y_true, y_pred)
                else:
                    raise ValueError("Métrica inválida")

                print(f"✅ ARIMA({p},{d},{q}) score: {score:.4f}")
                melhores_resultados.append({'p': p, 'd': d, 'q': q, 'score': score})

            except Exception as e:
                print(f"💥 Erro com ARIMA({p},{d},{q}): {e}")
                continue

        if not melhores_resultados:
            raise ValueError("Nenhum modelo ARIMA gerou previsões válidas.")

        if metrica == 'r2':
            melhor = max(melhores_resultados, key=lambda x: x['score'])
        else:
            melhor = min(melhores_resultados, key=lambda x: x['score'])

        return melhor, melhores_resultados





    # Nova funcionalidade: Previsão além dos dados realizados
    st.markdown("---")
    st.subheader("🔮 Previsão de Vendas Futuras")

    meses_a_frente = st.number_input(
        "Quantos meses deseja prever além do período disponível?", 
        min_value=1, max_value=24, value=6
    )


   

    if usar_otimizacao:
        print("🚨 Shape treino:", df_treino.shape)
        print("🚨 Últimos resíduos:", df_treino['residuo'].tail())
        print("🚨 Série ARIMA vai ter", len(df_treino.set_index('data')['residuo']), "pontos")

        melhor_modelo, historico = otimizar_arima(
            df_treino=df_treino,
            df_filial=df_filial,
            data_corte=data_corte,
            media_sazonal_treino=media_sazonal_treino,
            prever_tendencia_individual=prever_tendencia_individual,
            metrica=metrica_arima.lower(),  # transforma 'R²' em 'r2'
            grid_p=range(0, 4),
            grid_d=range(0, 3),
            grid_q=range(0, 4)
        )
        p, d, q = melhor_modelo['p'], melhor_modelo['d'], melhor_modelo['q']
        st.success(f"Melhor ordem ARIMA encontrada: ({p},{d},{q}) com {metrica_arima} = {melhor_modelo['score']:.4f}")
    else:
        st.info(f"Usando ARIMA manual: ({p},{d},{q})")

    # Treina o modelo com os parâmetros definidos
    print("DF TREINO")
    print(df_treino.head())
    print(df_treino.tail())
    
    serie_residuo = df_treino.set_index('data')['residuo']
    modelo_arima = ARIMA(serie_residuo, order=(p,d,q))
    modelo_ajustado = modelo_arima.fit()



    

    # Número total de períodos (histórico + futuro)
    n_total = len(df_filial) + meses_a_frente

    # Previsão ARIMA para todo o período
    residuos_completos = modelo_ajustado.predict(start=0, end=n_total - 1)

    # Construir DataFrame com todas as datas
    datas_completas = pd.date_range(start=df_filial['data'].iloc[0], periods=n_total, freq='MS')
    df_arima_completo = pd.DataFrame({
        'data': datas_completas,
        'ruido_previsto': residuos_completos
    })

    # Adiciona t e mês
    df_arima_completo['t'] = np.arange(n_total)
    df_arima_completo['mes'] = df_arima_completo['data'].dt.month

    
    # Index que representa o período com dados realizados
    n_realizados = df_filial['Realizado'].notnull().sum()

    # Resíduos do ARIMA apenas no período com dados observados
    residuos_estimados = modelo_ajustado.resid[:n_realizados]

        
    

    previsoes_arima_full = []
    for idx, row in df_arima_completo.iterrows():
        t_i = row['t']
        mes_i = row['mes']
        if tipo_tendencia == "Média":
           tendencia_i = media_realizado
        else:
             
            tendencia_i =  prever_tendencia_individual(t_i)
        sazonal_i = media_sazonal_treino.get(mes_i, 0)
        ruido_i = row['ruido_previsto']
        yhat_log = tendencia_i + sazonal_i + ruido_i
        previsoes_arima_full.append({
            'data': row['data'],
            'previsao': np.exp(yhat_log) - 1,
            'yhat_log': yhat_log
            #'ic_inf': np.exp(yhat_log - 1.96 * sigma_arima) - 1,
            #'ic_sup': np.exp(yhat_log + 1.96 * sigma_arima) - 1
        })

    df_prev_arima_completo = pd.DataFrame(previsoes_arima_full)
    print(df_prev_arima_completo.head())
    print(df_prev_arima_completo.tail())


    # Junta as previsões em log com os dados reais em log
    df_arima_residuos_log = pd.merge(
        df_real[['data', 'log_venda']],
        df_prev_arima_completo[['data', 'yhat_log']],
        on='data',
        how='inner'
    )

    # Calcula o resíduo em log
    # Filtra apenas até a data de corte
    df_residuos_treino = df_arima_residuos_log[df_arima_residuos_log['data'] <= data_corte]


    df_residuos_treino['erro_log'] = df_residuos_treino['log_venda'] - df_residuos_treino['yhat_log']





    # Desvio padrão final
    sigma_arima = df_residuos_treino['erro_log'].std()
    if pd.isna(sigma_arima) or sigma_arima == 0:
        sigma_arima = 1e-6

    df_prev_arima_completo['ic_inf'] = np.exp(df_prev_arima_completo['yhat_log'] - 2.57 * sigma_arima) - 1
    df_prev_arima_completo['ic_sup'] = np.exp(df_prev_arima_completo['yhat_log'] + 2.57 * sigma_arima) - 1


    #Checagem dos Desvios Padrao    
    print(f"📉 Sigma (Modelo Clássico - sigma_ruido): {sigma_ruido:.4f}")
    print(f"📈 Sigma (ARIMA nos Resíduos - sigma_arima): {sigma_arima:.4f}")

    if sigma_arima > sigma_ruido:
        print("⚠️ O desvio padrão do ARIMA é maior que o do modelo clássico. Isso pode deixar o IC mais largo.")
    elif sigma_arima < sigma_ruido:
        print("✅ O ARIMA reduziu a variabilidade dos resíduos. Ponto positivo!")
    else:
        print("ℹ️ Os desvios padrão dos dois modelos são iguais.")
    
    
    
    # Previsão para os próximos N períodos (mesmos meses_a_frente)
    residuos_previstos = modelo_ajustado.forecast(steps=meses_a_frente)

    


    # Criar novos períodos futuros
    ultima_data = df_filial['data'].max()
    novas_datas = pd.date_range(start=ultima_data + pd.DateOffset(months=1), periods=meses_a_frente, freq='MS')
    df_futuro = pd.DataFrame({
        'data': novas_datas,
        't': np.arange(len(df_filial), len(df_filial) + meses_a_frente),
    })
    df_futuro['mes'] = df_futuro['data'].dt.month

    # Gerar previsões futuras
    previsoes_futuras = []
    for idx, row in df_futuro.iterrows():
        t_i = row['t']
        mes_i = row['mes']
        if tipo_tendencia == "Média":
           tendencia_i = media_realizado
        else:
             
            tendencia_i =  prever_tendencia_individual(t_i)
        sazonal_i = media_sazonal_treino.get(mes_i, 0)
        yhat_log = tendencia_i + sazonal_i
        previsoes_futuras.append({
            'data': row['data'],
            'previsao': np.exp(yhat_log) - 1,
            'ic_sup': np.exp(yhat_log + 2.57 * sigma_ruido) - 1,
            'ic_inf': np.exp(yhat_log - 2.57 * sigma_ruido) - 1
        })
    df_prev_futuro = pd.DataFrame(previsoes_futuras)

    # Atualiza previsão final adicionando os resíduos previstos
    previsoes_ajustadas = []
    for idx, row in df_futuro.iterrows():
        t_i = row['t']
        mes_i = row['mes']
        if tipo_tendencia == "Média":
           tendencia_i = media_realizado
        else: 
            tendencia_i =  prever_tendencia_individual(t_i)
        sazonal_i = media_sazonal_treino.get(mes_i, 0)
        ruido_i = residuos_previstos.iloc[idx] if idx < len(residuos_previstos) else 0
        yhat_log = tendencia_i + sazonal_i + ruido_i
        previsoes_ajustadas.append({
            'data': row['data'],
            'previsao': np.exp(yhat_log) - 1
    })
    df_prev_futuro_arima = pd.DataFrame(previsoes_ajustadas)

    





    # Gráfico unificado com histórico + futuro
    st.markdown("### 📅 Previsão Histórica + Futura")
    fig, ax = plt.subplots(figsize=(14, 5))

    # Barras (Realizado)
    ax.bar(df_filial['data'], df_filial['Realizado'], color='lightgray', width=20, label='Realizado')

    #
    # Concatena os dois blocos
    df_completo = pd.concat([df_prev, df_prev_futuro])

    # Linha única e IC contínuo (azul, tudo igual)
    ax.plot(df_completo['data'], df_completo['previsao'], 'o-', color='blue', label='Previsão')
    ax.fill_between(df_completo['data'], df_completo['ic_inf'], df_completo['ic_sup'], color='blue', alpha=0.2, label='IC 95%')

    # Linha do corte
    ax.axvline(data_corte, color='red', linestyle='--', label='Corte')

    # Formatação do eixo X
    todas_datas = pd.concat([df_filial['data'], df_prev_futuro['data']])
    ax.set_xticks(todas_datas[::2])
    ax.set_xticklabels(todas_datas[::2].dt.strftime('%Y-%m'), rotation=45)

    # Títulos e legenda
    ax.set_title(f"Previsão vs Realizado (Histórico + Futuro) – Filial {filial}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Vendas")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    st.pyplot(fig)




    st.markdown("### 🔁 Comparativo: Previsão com e sem ARIMA nos Resíduos")

    fig, ax = plt.subplots(figsize=(14, 5))

    # Linha clássica
    ax.plot(df_completo['data'], df_completo['previsao'], 'o--', label='Clássica', color='blue')
    ax.fill_between(df_completo['data'], df_completo['ic_inf'], df_completo['ic_sup'], color='blue', alpha=0.15, label='IC Clássico')

    # Linha com ARIMA
    ax.plot(df_prev_arima_completo['data'], df_prev_arima_completo['previsao'], 'o-', label='Com ARIMA nos Resíduos', color='green')
    ax.fill_between(df_prev_arima_completo['data'], df_prev_arima_completo['ic_inf'], df_prev_arima_completo['ic_sup'], color='green', alpha=0.15, label='IC ARIMA')

    # Barras do realizado
    ax.bar(df_filial['data'], df_filial['Realizado'], color='lightgray', width=20, label='Realizado')

    # Corte
    ax.axvline(data_corte, color='red', linestyle='--', label='Corte')

    # Eixos e legendas
    ax.set_title(f"📊 Previsão Comparativa – Filial {filial}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Vendas")
    ax.legend()
    st.pyplot(fig)

   

    st.subheader("🧪 Teste de Estacionaridade (ADF) nos Resíduos do ARIMA")

    # Aplica o teste
    serie_residuos = df_residuos_treino['erro_log'].dropna()
    resultado_adf = adfuller(serie_residuos)

    # Mostra resultados
    st.write(f"**Estatística ADF:** {resultado_adf[0]:.4f}")
    st.write(f"**Valor-p:** {resultado_adf[1]:.4f}")
    st.write(f"**Nº de lags usados:** {resultado_adf[2]}")
    st.write(f"**Nº de observações:** {resultado_adf[3]}")
    st.markdown("**Valores Críticos:**")
    for key, value in resultado_adf[4].items():
        st.write(f"{key}: {value:.4f}")

    # Interpretação
    if resultado_adf[1] < 0.05:
        st.success("✅ Resíduos são estacionários (rejeita H₀ com 95% de confiança).")
    else:
        st.warning("⚠️ Resíduos NÃO são estacionários (não rejeita H₀). Considere aumentar o parâmetro d no ARIMA.")

    #teste sazonalidade dos residuos
    residuos = df_residuos_treino['erro_log'].values


    # Remove média dos resíduos para análise mais pura
    residuos_centralizados = residuos - np.mean(residuos)

    # FFT
    fft_resultado = fft(residuos_centralizados)
    n = len(residuos)
    frequencias = np.abs(fft_resultado[:n // 2])  # metade inferior é espelho da superior

    # Frequências correspondentes a períodos
    periodos = np.arange(1, n // 2 + 1)

    # Pega o período dominante (excluindo a componente zero)
    periodo_dominante = periodos[np.argmax(frequencias[1:])]  # ignorando índice 0

    # Exibe resultado
    print(f"📊 Ciclo dominante nos resíduos do ARIMA: {periodo_dominante} períodos")

    plt.figure(figsize=(10, 4))
    plt.plot(periodos[1:], frequencias[1:], label='Magnitude da Frequência')
    plt.axvline(periodo_dominante, color='red', linestyle='--', label=f'Período dominante: {periodo_dominante}')
    plt.xlabel("Período")
    plt.ylabel("Intensidade")
    plt.title("Análise de Frequência dos Resíduos do ARIMA")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)





    # Período com dados realizados
    df_real_valid = df_filial[df_filial['Realizado'].notnull()].copy()
    # Junta Realizado + Previsão Clássica
    df_classico_valid = df_prev[df_prev['data'].isin(df_filial['data'])]
    df_comp_classico = pd.merge(df_filial, df_classico_valid, on='data')

    # Junta Realizado + Previsão com ARIMA
    df_arima_valid = df_prev_arima_completo[df_prev_arima_completo['data'].isin(df_filial['data'])]
    df_comp_arima = pd.merge(df_filial, df_arima_valid, on='data', suffixes=('', '_arima'))

    # Cria máscaras para períodos
    mask_realizado = df_comp_classico['Realizado'].notnull()
    mask_pos_corte = df_comp_classico['data'] > data_corte

    def calcular_metricas(y_true, y_pred):
        return {
            'MAPE (%)': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'R²': r2_score(y_true, y_pred),
            'RMSE': mean_squared_error(y_true, y_pred, squared=False)
        }

    # Total
    m_classico_total = calcular_metricas(df_comp_classico[mask_realizado]['Realizado'],df_comp_classico[mask_realizado]['previsao'])
    m_arima_total = calcular_metricas(df_comp_arima[mask_realizado]['Realizado'],df_comp_arima[mask_realizado]['previsao'])
    # Pós-corte
    m_classico_pos = calcular_metricas(df_comp_classico[mask_realizado & mask_pos_corte]['Realizado'],df_comp_classico[mask_realizado & mask_pos_corte]['previsao'])
    m_arima_pos = calcular_metricas(df_comp_arima[mask_realizado & mask_pos_corte]['Realizado'],df_comp_arima[mask_realizado & mask_pos_corte]['previsao'])
    
    st.markdown("### 🧪 Comparativo de Métricas – Clássico vs ARIMA nos Resíduos")

    # Total
    st.subheader("🟢 Período Completo")

    st.write("#### Modelo Clássico")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAPE (%)", f"{m_classico_total['MAPE (%)']:.2f}")
    col2.metric("R²", f"{m_classico_total['R²']:.3f}")
    col3.metric("RMSE", f"{m_classico_total['RMSE']:,.2f}")

    st.write("#### Modelo com ARIMA nos Resíduos")
    col4, col5, col6 = st.columns(3)
    col4.metric("MAPE (%)", f"{m_arima_total['MAPE (%)']:.2f}")
    col5.metric("R²", f"{m_arima_total['R²']:.3f}")
    col6.metric("RMSE", f"{m_arima_total['RMSE']:,.2f}")

    # Pós-Corte
    st.subheader("🔴 Período Pós-Corte")

    st.write("#### Modelo Clássico")
    col7, col8, col9 = st.columns(3)
    col7.metric("MAPE (%)", f"{m_classico_pos['MAPE (%)']:.2f}")
    col8.metric("R²", f"{m_classico_pos['R²']:.3f}")
    col9.metric("RMSE", f"{m_classico_pos['RMSE']:,.2f}")

    st.write("#### Modelo com ARIMA nos Resíduos")
    col10, col11, col12 = st.columns(3)
    col10.metric("MAPE (%)", f"{m_arima_pos['MAPE (%)']:.2f}")
    col11.metric("R²", f"{m_arima_pos['R²']:.3f}")
    col12.metric("RMSE", f"{m_arima_pos['RMSE']:,.2f}")

    



    ########################################3
    
  # aqui vamos incluir um botao "Aplimorar com Machine Learning" e vamos aplicar o do radom forest
  # gerar df com a media movel de Y com 4 meses ( t-1 t-2 t-3 t-4 t-5)

  # treinar um modelo de RF Regressor 
  #####Features: previsao do modelo sazonal + log. (gerado neste codigo) + mes + media movel de 4 meses ( t-1 t-2 t-3 t-4 t-5)
  ##### treino usar dados antes da linha de corte


    #projetar dados a partir da linha de corte 
    # criar limites de cofianca com as novas projecoes em todo intervalo com realizado
    
    #  onde nao tiver mais realziado, estimar de forma recorrente no periodo futuro estabelecido neste codigo
    # gerar grafico similiar ao inicial mas incluindo os novos limites de confianca e a nova previsa com RF regregssor

    # Checkbox para otimizar
    otimizar = st.checkbox("🔧 Otimizar hiperparâmetros com RandomizedSearchCV?")   
    #Escolhe o modelo de ML
    modelo_escolhido = st.selectbox(
        "🤖 Escolha o modelo de Machine Learning",
        ["Random Forest", "Bayesian Ridge", "MLP", "XGBoost", "Decision Tree", "K-Nearest Neighbors", "Ensemble"]
    )



    # Botão de ativação
    janela_mm = st.number_input(
    "📏 Tamanho da janela para Média Móvel (n períodos anteriores)",
    min_value=2, max_value=12, value=4, step=1
)

    def gerar_dados_classificacao_binaria(df, janela_mm, data_corte):
        df = df.copy()
        df = df.sort_values(['Filial', 'data']).reset_index(drop=True)

        df['mes'] = df['data'].dt.month
        # Geração dos lags
        for i in range(1, janela_mm + 1):
            df[f'lag_{i}'] = df.groupby('Filial')['Realizado'].shift(i)

        # Delta total (lag_1 - lag_n)
        df['delta_total'] = df[f'lag_1'] - df[f'lag_{janela_mm}']
        df['delta_total_bin'] = df['delta_total'].apply(lambda x: 1 if x > 0 else 0)
        df['var_pct_total'] = df['delta_total'] / (df[f'lag_{janela_mm}'] + 1e-6)

        # Média e desvio padrão dos lags
        lags_cols = [f'lag_{i}' for i in range(1, janela_mm + 1)]
        df['media_lags'] = df[lags_cols].mean(axis=1)
        df['std_lags'] = df[lags_cols].std(axis=1)

        # Variações contínuas + binarizadas entre lags consecutivos
        for i in range(1, janela_mm):
            col = f'delta_{i}_{i+1}'
            df[col] = df[f'lag_{i}'] - df[f'lag_{i+1}']
            df[f'{col}_bin'] = df[col].apply(lambda x: 1 if x > 0 else 0)
            df[f'{col}_var_pct'] = df[col] / (df[f'lag_{i+1}'] + 1e-6)

        # Target: subiu ou não (pode trocar por threshold depois)
        df['Realizado_next'] = df.groupby('Filial')['Realizado'].shift(-1)
        df['target'] = (df['Realizado_next'] > df['Realizado']).astype(int)

        # Seleciona as colunas
        col_bin = [col for col in df.columns if '_bin' in col]
        col_cont = [col for col in df.columns if col in [
            'delta_total', 'var_pct_total', 'media_lags', 'std_lags'
        ] or '_var_pct' in col or col.startswith('delta_') and '_bin' not in col]

        col_features = col_bin + col_cont + ['mes']


        df_final = df[['data', 'Filial', 'Realizado', 'target'] + col_features].dropna()

        # Separa treino e teste
        df_treino = df_final[df_final['data'] <= data_corte].copy()
        df_teste = df_final[df_final['data'] > data_corte].copy()

        return df_treino, df_teste, col_features

#     threshold = st.slider("🎚️ Ajuste o Threshold de Classificação", min_value=0.0, max_value=1.0, value=0.5, step=0.01)


#     if st.button("🔥 Gerar Modelo de Classificação"):
    
#         #global clf, scaler2
#         # Geração dos dados
        
#         df_treino, df_teste, col_binarias = gerar_dados_classificacao_binaria(df_com_total, janela_mm, data_corte)

#         X_train = df_treino[col_binarias]
#         y_train = df_treino['target']
#         X_test = df_teste[col_binarias]
#         y_test = df_teste['target']

#         X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
#         X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
# )


#        # import seaborn as sns
#         #import matplotlib.pyplot as plt

#     #     # Calcular correlação
#     #     corr = pd.DataFrame(X_train, columns=col_binarias).corr()

#     #     # Criar o gráfico
#     #     fig, ax = plt.subplots(figsize=(10, 8))
#     #     sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
#     #     st.pyplot(fig)
#     #  #-------------------SVM
        
        
#         # from sklearn.model_selection import GridSearchCV
     

#         # param_svc_grid = [
#         #     {
#         #         'kernel': ['rbf'],
#         #         'C': [0.1, 1, 10, 100, 200],
#         #         'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
#         #         'class_weight': ['balanced'],
#         #         'probability': [True]
#         #     },
#         #     {
#         #         'kernel': ['poly'],
#         #         'C': [0.1, 1, 10, 20],
#         #         'gamma': ['scale', 0.01, 0.1],
#         #         'degree': [2, 3],
#         #         'class_weight': ['balanced'],
#         #         'probability': [True]
#         #     }
#         # ]

#         # svc_search = GridSearchCV(
#         #     SVC(random_state=42),
#         #     param_grid=param_svc_grid,
#         #     cv=10,
#         #     scoring='f1',
#         #     n_jobs=1,
#         #     verbose=2
#         # )


#         scaler2 = StandardScaler()
#         X_train_split_scaled = scaler2.fit_transform(X_train_split)
#         X_test_scaled = scaler2.transform(X_test)   
#         X_val_split_scaled  = scaler2.transform(X_val_split)

#         # svc_search.fit(X_train_scaled, y_train)
#         # best_svc = svc_search.best_estimator_
 
        
        
#         # Modelo
#         #clf = MLPClassifier(hidden_layer_sizes=(64,32,8), max_iter=5000, random_state=42)
#         #clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05)
#         clf = SVC(kernel='rbf', class_weight='balanced', probability=True)
#         #clf = xgb.XGBClassifier(n_estimators=600,learning_rate=0.05,max_depth=10,scale_pos_weight=4)  # bom pra balancear manualmente a classe minoritáriause_label_encoder=False,eval_metric='logloss',random_state=42)
#         #clf = RandomForestClassifier(n_estimators=500,class_weight="balanced", random_state=42)
        
         
#         #base_models = [
#           #  ('rf', best_rf),  # <- otimizado!
#            # ('xgb', best_xgb),  # <- otimizado! ,
#            # ('lr', LogisticRegression(class_weight='balanced'))
#          #]



#         #meta_model = LogisticRegression()
#         #clf_stacking = StackingClassifier(
#         #    estimators=base_models,
#          #   final_estimator=meta_model,
#           #  passthrough=True  # passa features originais pro meta-modelo também
#         #)

      
#         #clf = VotingClassifier(
#          #   estimators=[
#           #      ('stacking', clf_stacking),
#                 #('knn',KNeighborsClassifier(n_neighbors=5, weights='distance', p=2)),  # algo diferente
#                 #('gb',GradientBoostingClassifier(n_estimators=200, learning_rate=0.05)),
#            #     ('mlp',best_mlp),
#                 #('svm', SVC(kernel='rbf', class_weight='balanced', probability=True))
                
#             #],
#             #voting='soft'
#        # )


#         clf.fit(X_train_split_scaled, y_train_split)


#         y_val_prob = clf.predict_proba(X_val_split_scaled)[:, 1]
#         y_val_pred = (y_val_prob >= threshold).astype(int)  

#         # Métricas de avaliação (validação)
#         report_val = classification_report(y_val_split, y_val_pred, target_names=["Não Subiu", "Subiu"], output_dict=True)
#         conf_val = confusion_matrix(y_val_split, y_val_pred)

#         # Exibição
#         st.markdown("### 📊 Validação do Modelo (Holdout 30%)")
#         df_report = pd.DataFrame(report_val).T
#         df_report[['precision', 'recall', 'f1-score']] *= 100

#         st.dataframe(df_report.style.format({
#             'precision': '{:.1f}%',
#             'recall': '{:.1f}%',
#             'f1-score': '{:.1f}',
#             'support': '{:.0f}'
#         }).background_gradient(cmap='Blues', axis=0))

#         st.markdown("### 🧮 Matriz de Confusão (Validação)")
#         st.dataframe(pd.DataFrame(conf_val, 
#             index=["Real: Não Subiu", "Real: Subiu"], 
#             columns=["Pred: Não Subiu", "Pred: Subiu"]
#         ).style.background_gradient(cmap='Oranges'))


#         # Previsões
#         y_prob = clf.predict_proba(X_test_scaled)[:, 1]
#         y_pred = (y_prob >= threshold).astype(int)

#         #y_pred = clf.predict(X_test)

#         # Resultados
#        # Relatório de Classificação - formatado bonitão

#         report_dict = classification_report(y_test, y_pred, target_names=["Não Subiu", "Subiu"], output_dict=True)
#         report_df = pd.DataFrame(report_dict).T

#         # Converte para porcentagem onde faz sentido
#         report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']] * 100

#         # Formata visual bonitão
#         st.markdown("### 📊 Relatório de Classificação")
#         st.dataframe(report_df.style.format({
#             'precision': '{:.1f}%',
#             'recall': '{:.1f}%',
#             'f1-score': '{:.1f}',
#             'support': '{:.0f}'
#         }).background_gradient(cmap='Blues', axis=0))

#         # Matriz de confusão formatada
#         conf_mat = confusion_matrix(y_test, y_pred)
#         conf_df = pd.DataFrame(
#             conf_mat,
#             index=["Real: 0 (Não Subiu)", "Real: 1 (Subiu)"],
#             columns=["Pred: 0", "Pred: 1"]
#         )

#         st.markdown("### 🧮 Matriz de Confusão")
#         st.dataframe(conf_df.style.background_gradient(cmap='Oranges'))
#         print(X_train.head(25))

#         # Métricas de validação
#         metricas_val = {
#             'Accuracy': accuracy_score(y_val_split, y_val_pred),
#             'Precision': precision_score(y_val_split, y_val_pred),
#             'Recall': recall_score(y_val_split, y_val_pred),
#             'F1': f1_score(y_val_split, y_val_pred)
#         }

#         # Métricas do teste real
#         metricas_teste = {
#             'Accuracy': accuracy_score(y_test, y_pred),
#             'Precision': precision_score(y_test, y_pred),
#             'Recall': recall_score(y_test, y_pred),
#             'F1': f1_score(y_test, y_pred)
#         }    

#         labels = ['Accuracy', 'Precision', 'Recall', 'F1']
#         valores_val = [metricas_val['Accuracy'], metricas_val['Precision'], metricas_val['Recall'], metricas_val['F1']]
#         valores_test = [metricas_teste['Accuracy'], metricas_teste['Precision'], metricas_teste['Recall'], metricas_teste['F1']]

#         x = range(len(labels))
#         width = 0.35

#         fig, ax = plt.subplots(figsize=(10, 4)) 
#         ax.bar(x, valores_val, width, label='Validação (Holdout)', color='blue')
#         ax.bar([i + width for i in x], valores_test, width, label='Teste Pós-Corte', color='orange')

#         ax.set_xticks([i + width / 2 for i in x])
#         ax.set_xticklabels(labels)
#         ax.set_ylabel('Score')
#         ax.set_title('Comparativo de Métricas')
#         ax.legend()

#         st.pyplot(fig)
#         from sklearn.metrics import precision_recall_curve

#         # Calcula precision, recall e thresholds
#         precisions, recalls, limiares = precision_recall_curve(y_test, y_prob)

#         # Plota
#         fig, ax = plt.subplots(figsize=(10, 3))
#         ax.plot(limiares, precisions[:-1], label='Precision', color='blue')
#         ax.plot(limiares, recalls[:-1], label='Recall', color='orange')
#         ax.set_title("🎯 Precision vs Recall por Threshold")
#         ax.set_xlabel("Threshold")
#         ax.set_ylabel("Score")
#         ax.set_ylim([0, 1])
#         ax.grid(True)
#         ax.legend()
#         st.pyplot(fig)

#         try:
#             df_aplicacao = df_com_total.copy()
#             df_aplicacao = df_aplicacao.sort_values(['Filial', 'data']).reset_index(drop=True)

#             # Recria as features binárias (com a mesma função usada no treino)
#             df_treino_aplic, df_teste_aplic, col_binarias = gerar_dados_classificacao_binaria(df_aplicacao, janela_mm, data_corte)
#             df_aplicacao_final = pd.concat([df_treino_aplic, df_teste_aplic]).copy()

#             # Aplica transformação e previsão
#             X_aplic = df_aplicacao_final[col_binarias]
#             X_aplic_scaled = scaler2.transform(X_aplic)
#             prob_clf = clf.predict_proba(X_aplic_scaled)[:, 1]
#             pred_clf = (prob_clf >= threshold).astype(int)

#             # Adiciona previsões
#             df_aplicacao_final['prob_clf'] = prob_clf
#             df_aplicacao_final['pred_clf'] = pred_clf

#             # Salva no session_state e mostra preview
#             st.session_state.df_classificacao_resultado = df_aplicacao_final
#             st.success("✅ Modelo aplicado ao dataframe completo!")

#             st.markdown("### 🔍 Amostra das previsões aplicadas")
#             st.dataframe(df_aplicacao_final[['data', 'Filial', 'prob_clf', 'pred_clf'] + col_binarias].tail(10))
#             df_com_total = df_aplicacao_final

#         except Exception as e:
#             st.error(f"Erro ao aplicar o modelo no dataframe completo: {e}")



    

    if st.button("🔥 Aprimorar com Machine Learning"):
        st.markdown(f"### 🧠 Previsão com {modelo_escolhido}")

        


        

        # Prepara scaler e modelo base
        scaler = None
        model = None

        def adicionar_sazonalidade_fft(df, periodo, col_prefix=''):
            idx = np.arange(len(df))
            df[f'{col_prefix}seno_{periodo}'] = np.sin(2 * np.pi * idx / periodo)
            df[f'{col_prefix}cosseno_{periodo}'] = np.cos(2 * np.pi * idx / periodo)
            return df

       
        

        #features
        df_prev_arima_completo_2 = df_prev_arima_completo.rename(columns={'previsao': 'prev_arima'})


        df_ml = df_merged.copy()
        df_ml['mm'] = df_ml['Realizado'].shift(1).rolling(window= janela_mm).mean()
        df_ml['std_mm'] = df_ml['Realizado'].shift(1).rolling(window=janela_mm).std()
        df_ml['mes'] = df_ml['data'].dt.month
        df_ml = pd.merge(df_ml, df_prev_arima_completo_2[['data', 'prev_arima']], on='data', how='left')
        df_ml['erro_c_a'] = (df_ml['prev_arima']/df_ml['previsao'])-1
        
        #teste com os logs
        #df_ml['prev_arima'] = np.log(df_ml['prev_arima'] + 1)
        #df_ml['previsao'] = np.log(df_ml['previsao'] + 1)/np.log(df_ml['prev_arima'] + 1)
       

       
       # Lags conforme janela selecionada
        for i in range(1, janela_mm + 1):
            df_ml[f'lag_{i}'] = df_ml['Realizado'].shift(i)

        df_ml['delta_lags'] = df_ml[f'lag_1'] - df_ml[f'lag_{janela_mm}']
        df_ml['min_lags'] = df_ml[[f'lag_{i}' for i in range(1, janela_mm + 1)]].min(axis=1)
        df_ml['max_lags'] = df_ml[[f'lag_{i}' for i in range(1, janela_mm + 1)]].max(axis=1)

        # Variações percentuais entre os lags (lag_i vs lag_{i+1})
        for i in range(1, janela_mm):
            df_ml[f'var_pct_lag_{i}_{i+1}'] = (df_ml[f'lag_{i}'] - df_ml[f'lag_{i+1}']) / (df_ml[f'lag_{i+1}'] + 1e-6)



        df_ml = adicionar_sazonalidade_fft(df_ml,periodo_dominante)
        print(df_ml.head())


        #colunas_features = ['previsao', 'mes', 'mm', 'std_mm','delta_lags', 'min_lags', 'max_lags'] + [f'lag_{i}' for i in range(1, janela_mm + 1)]
        #colunas_features = ['previsao', 'prev_arima','erro_c_a', 'mes', 'mm', 'std_mm', 'delta_lags', 'min_lags', 'max_lags',  f'seno_{periodo_dominante}', f'cosseno_{periodo_dominante}'] + [f'lag_{i}' for i in range(1, janela_mm + 1)]


        colunas_var_pct = [f'var_pct_lag_{i}_{i+1}' for i in range(1, janela_mm)]
        colunas_features = ['previsao', 'prev_arima','erro_c_a', 'mes', 'mm', 'std_mm','delta_lags', 'min_lags', 'max_lags',f'seno_{periodo_dominante}', f'cosseno_{periodo_dominante}'] + [f'lag_{i}' for i in range(1, janela_mm + 1)] + colunas_var_pct



        
        def construir_X_fut(prev_classica, prev_arima_i, erro_c_a_i, mes_i, mm_i, std_i,
                            ultimos_valores, janela_mm, colunas_features,
                            indice_futuro, periodo_dominante):

            # Pega os últimos valores defasados
            lags_valores = list(reversed(ultimos_valores[-janela_mm:]))
            while len(lags_valores) < janela_mm:
                lags_valores.insert(0, np.nan)

            if len(lags_valores) == janela_mm:
                delta_lags = lags_valores[-1] - lags_valores[0]
                min_lags = min(lags_valores)
                max_lags = max(lags_valores)
            else:
                delta_lags = 0
                min_lags = 0
                max_lags = 0

            # Seno e cosseno para componente sazonal
            seno = np.sin(2 * np.pi * indice_futuro / periodo_dominante)
            cosseno = np.cos(2 * np.pi * indice_futuro / periodo_dominante)
            

            # Calcula as variações percentuais entre os lags
            var_pct_lags = []
            for i in range(janela_mm - 1):
                atual = lags_valores[i]
                prox = lags_valores[i + 1]
                if prox == 0 or np.isnan(prox) or np.isnan(atual):
                    var_pct = 0
                else:
                    var_pct = (atual - prox) / (prox + 1e-6)
                var_pct_lags.append(var_pct)

            # Junta tudo
            linha = [prev_classica, prev_arima_i, erro_c_a_i, mes_i, mm_i, std_i,
                    delta_lags, min_lags, max_lags, seno, cosseno] + lags_valores + var_pct_lags


            
            return pd.DataFrame([linha], columns=colunas_features)



        df_ml = df_ml.dropna(subset=colunas_features)



        df_treino_ml = df_ml[df_ml['data'] <= data_corte].copy()
        X_train = df_treino_ml[colunas_features]
        y_train = df_treino_ml['Realizado']




        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train, y_train, test_size=0.5, random_state=42
        )




        #X_train_split = X_train
        #y_train_split = y_train
        #X_test_split = X_train
        #y_test_split = y_train


        # Modelagem
        scaler = None
        if modelo_escolhido == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif modelo_escolhido == "Bayesian Ridge":
            model = BayesianRidge()
        elif modelo_escolhido == "MLP":
            scaler = StandardScaler()
            X_train_split = scaler.fit_transform(X_train_split)
            X_test_split = scaler.transform(X_test_split)
            model = MLPRegressor(hidden_layer_sizes=(8), max_iter=10000, random_state=42)
        elif modelo_escolhido == "XGBoost":
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif modelo_escolhido == "Decision Tree":
            model = DecisionTreeRegressor(max_depth=10, random_state=42)
        elif modelo_escolhido == "K-Nearest Neighbors":     
             model = KNeighborsRegressor(n_neighbors=3)
        elif modelo_escolhido == "Ensemble":
                        
           # Modelos
            model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
            model_bayes = BayesianRidge()
            model_linear = LinearRegression()
            model_xgb = xgb.XGBRegressor(n_estimators=100, random_state=42)
            model_dt = DecisionTreeRegressor(max_depth=10, random_state=42)
            model_kn = KNeighborsRegressor(n_neighbors=3)
            # Ensemble
            model = VotingRegressor(estimators=[
                ('rf', model_rf),
                ('bayes', model_bayes),
               # ('linear', model_linear),
                ('xgb', model_xgb),
                ('dt',model_dt),
                ('kn',model_kn),
            ])

        base_model = model               
        
        # Aplica otimização
        if otimizar and modelo_escolhido in param_grids:
            st.info("⏳ Buscando melhores hiperparâmetros...")
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grids[modelo_escolhido],
                n_iter=10,
                cv=3,
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train_split, y_train_split)
            model = search.best_estimator_
            st.success(f"✅ Melhor modelo encontrado: {search.best_params_}")
        else:
            model = base_model


        model.fit(X_train_split, y_train_split)

        # Verifica se o modelo tem o atributo de importância
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            df_importancia = pd.DataFrame({
                'Feature': colunas_features,
                'Importância': importances
            }).sort_values(by='Importância', ascending=False)

            st.markdown("### 🌟 Importância das Features")
            st.dataframe(df_importancia.style.format({"Importância": "{:.2%}"}))

            # Gráfico
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(df_importancia['Feature'], df_importancia['Importância'])
            ax.set_xlabel("Importância")
            ax.set_title(f"Importância das Features – {modelo_escolhido}")
            plt.gca().invert_yaxis()
            st.pyplot(fig)

        X_pred = df_ml[colunas_features]
        if scaler:
            X_pred = scaler.transform(X_pred)

        df_ml['prev_ml'] = model.predict(X_pred)
        df_ml['residuo_ml'] = df_ml['Realizado'] - df_ml['prev_ml']
        sigma = df_ml['residuo_ml'].std()
        df_ml['ic_inf_ml'] = df_ml['prev_ml'] - 2.57 * sigma
        df_ml['ic_sup_ml'] = df_ml['prev_ml'] + 2.57 * sigma

        # Previsões futuras
        ultimos_valores = df_ml.set_index('data')[['Realizado']].dropna().tail(4)['Realizado'].tolist()
        previsoes_futuras = []
        indice_futuro = len(df_ml)  # começa logo após o último ponto de treino

        for i, row in df_futuro.iterrows():
            mes_i = row['mes']
            prev_classica = df_prev_futuro.loc[df_prev_futuro['data'] == row['data'], 'previsao'].values[0]
            mm_i = np.mean(ultimos_valores[-janela_mm:]) if len(ultimos_valores) >= janela_mm else np.nan
            std_i = np.std(ultimos_valores[-janela_mm:]) if len(ultimos_valores) >= janela_mm else np.nan

            if np.isnan(mm_i):
                continue

            prev_arima_i = df_prev_arima_completo_2.loc[df_prev_arima_completo_2['data'] == row['data'], 'prev_arima'].values[0]
            erro_c_a_i = (prev_arima_i / prev_classica) - 1

            X_fut = construir_X_fut(
                prev_classica, prev_arima_i, erro_c_a_i, mes_i, mm_i, std_i,
                ultimos_valores, janela_mm, colunas_features,
                indice_futuro, periodo_dominante
            )

           

            if scaler:
                X_fut = scaler.transform(X_fut)

            pred = model.predict(X_fut)[0]
            ultimos_valores.append(pred)

            previsoes_futuras.append({
                'data': row['data'],
                'prev_ml': pred,
                'ic_inf_ml': pred - 2.57 * sigma,
                'ic_sup_ml': pred + 2.57 * sigma
            })

            indice_futuro += 1  # avança o tempo


        df_ml_futuro = pd.DataFrame(previsoes_futuras)
        df_ml_plot = pd.concat([df_ml[['data', 'prev_ml', 'ic_inf_ml', 'ic_sup_ml']].dropna(), df_ml_futuro])
        print(df_ml_plot.tail(10))
        print(df_filial.tail(10))
        # Gráfico
        st.markdown(f"### 🔍 Comparação: Previsão Clássica vs {modelo_escolhido}")
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.bar(df_filial['data'], df_filial['Realizado'], color='lightgray', label='Realizado', width=20)
        df_completo_classico = pd.concat([df_prev, df_prev_futuro])
        ax.plot(df_completo_classico['data'], df_completo_classico['previsao'], 'o-', color='blue', label='Previsão Clássica')
        ax.fill_between(df_completo_classico['data'], df_completo_classico['ic_inf'], df_completo_classico['ic_sup'], color='blue', alpha=0.2)

        ax.plot(df_ml_plot['data'], df_ml_plot['prev_ml'], 'o-', label=f'Previsão {modelo_escolhido}', color='darkorange')
        ax.fill_between(df_ml_plot['data'], df_ml_plot['ic_inf_ml'], df_ml_plot['ic_sup_ml'], color='orange', alpha=0.2)

        ax.axvline(data_corte, color='red', linestyle='--', label='Corte')
        datas_completas = pd.concat([df_filial['data'], df_prev_futuro['data']])
        ax.set_xticks(datas_completas[::2])
        ax.set_xticklabels(datas_completas[::2].dt.strftime('%Y-%m'), rotation=45)
        ax.set_title(f"Comparação de Previsões – Filial {filial}")
        ax.set_xlabel("Data")
        ax.set_ylabel("Vendas")
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        fig.tight_layout(rect=[0, 0, 0.85, 1])

        st.pyplot(fig)

            # Junta os dados
        df_compara = pd.merge(df_ml[['data', 'prev_ml']], df_merged[['data', 'Realizado', 'previsao']], on='data')
        df_compara = df_compara[df_compara['Realizado'].notnull()]

        # Separa período pós-corte
        df_pos_corte = df_compara[df_compara['data'] > data_corte].copy()

        # Cálculo de erros gerais
        df_compara['erro_clas'] = df_compara['Realizado'] - df_compara['previsao']
        df_compara['erro_ml'] = df_compara['Realizado'] - df_compara['prev_ml']
        df_compara['erro_pct_clas'] = df_compara['erro_clas'].abs() / (df_compara['Realizado'] + 1e-6)
        df_compara['erro_pct_ml'] = df_compara['erro_ml'].abs() / (df_compara['Realizado'] + 1e-6)

        # Métricas gerais
        mape_clas = df_compara['erro_pct_clas'].mean()
        mape_ml = df_compara['erro_pct_ml'].mean()
        r2_clas = r2_score(df_compara['Realizado'], df_compara['previsao'])
        r2_ml = r2_score(df_compara['Realizado'], df_compara['prev_ml'])

        # Métricas pós-corte
        if not df_pos_corte.empty:
            df_pos_corte['erro_clas'] = df_pos_corte['Realizado'] - df_pos_corte['previsao']
            df_pos_corte['erro_ml'] = df_pos_corte['Realizado'] - df_pos_corte['prev_ml']
            df_pos_corte['erro_pct_clas'] = df_pos_corte['erro_clas'].abs() / (df_pos_corte['Realizado'] + 1e-6)
            df_pos_corte['erro_pct_ml'] = df_pos_corte['erro_ml'].abs() / (df_pos_corte['Realizado'] + 1e-6)

            mape_clas_pos = df_pos_corte['erro_pct_clas'].mean()
            mape_ml_pos = df_pos_corte['erro_pct_ml'].mean()
            r2_clas_pos = r2_score(df_pos_corte['Realizado'], df_pos_corte['previsao'])
            r2_ml_pos = r2_score(df_pos_corte['Realizado'], df_pos_corte['prev_ml'])
        else:
            mape_clas_pos = mape_ml_pos = r2_clas_pos = r2_ml_pos = np.nan

        # Exibição
        st.markdown(f"### 📈 Comparativo de Métricas – Clássico vs {modelo_escolhido}")

        st.markdown(f"""
        <div style='font-size: 18px'>
        🔵 <b>Modelo Clássico (Geral)</b><br>
        - MAPE: <code>{mape_clas:.2%}</code><br>
        - R²: <code>{r2_clas:.4f}</code><br><br>

        🟠 <b>{modelo_escolhido} (Geral)</b><br>
        - MAPE: <code>{mape_ml:.2%}</code><br>
        - R²: <code>{r2_ml:.4f}</code>
        </div>
        """, unsafe_allow_html=True)

        # Se houver período pós-corte, mostra também
        if not np.isnan(mape_clas_pos):
            st.markdown(f"""
            <div style='font-size: 18px; margin-top: 2rem'>
            📆 <b>Análise Pós-Corte (Data > {data_corte.date()})</b><br><br>

            🔵 <b>Modelo Clássico (Pós-Corte)</b><br>
            - MAPE: <code>{mape_clas_pos:.2%}</code><br>
            - R²: <code>{r2_clas_pos:.4f}</code><br><br>

            🟠 <b>{modelo_escolhido} (Pós-Corte)</b><br>
            - MAPE: <code>{mape_ml_pos:.2%}</code><br>
            - R²: <code>{r2_ml_pos:.4f}</code>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Não há dados com realizado após a data de corte para análise pós-corte.")


 #############################################3

    
    
    
    # Métricas do modelo histórico
    if len(df_teste) > 1:
        r2 = r2_score(df_teste['Realizado'], df_teste['previsao'])
        r2_display = f"{r2:.4f}" if r2 >= 0 else "Modelo pior que média (R² negativo)"
    else:
        r2_display = "Insuficientes dados para cálculo do R²"

    df_teste['erro'] = df_teste['Realizado'] - df_teste['previsao']
    df_teste['erro_pct'] = df_teste['erro'].abs() / (df_teste['Realizado'] + 1e-6)
    mape = df_teste['erro_pct'].mean()

    # Exibição das métricas
    st.subheader("📊 Métricas de Avaliação (Período Realizado)")
    st.markdown(f"- **MAPE**: {mape:.2%}")
    st.markdown(f"- **R²**: {r2_display}")

    # Tabela das previsões futuras
    st.markdown("### 📋 Tabela de Previsões Futuras")
    st.dataframe(df_prev_futuro[['data', 'previsao', 'ic_inf', 'ic_sup']].rename(columns={
        'data': 'Data',
        'previsao': 'Previsão',
        'ic_inf': 'Limite Inferior (IC 95%)',
        'ic_sup': 'Limite Superior (IC 95%)'
    }).set_index('Data').style.format("{:,.2f}"))

        # --- Análise da assertividade por horizonte t+ ---
    df_check = pd.merge(df_filial[['data', 'Realizado']], df_prev, on='data', how='inner')
    df_check = df_check[df_check['data'] > data_corte].copy()

    if not df_check.empty:
        df_check = df_check.sort_values('data').reset_index(drop=True)
        df_check['t+'] = np.arange(1, len(df_check) + 1)
        df_check['erro_pct'] = (df_check['Realizado'] - df_check['previsao']).abs() / (df_check['Realizado'] + 1e-6)
        df_check['dentro_ic'] = df_check.apply(lambda row: row['ic_inf'] <= row['Realizado'] <= row['ic_sup'], axis=1)

        # Gráfico de barras
        st.markdown("### 📊 Erro por Horizonte de Previsão (t+) – Gráfico de Barras")
        fig, ax = plt.subplots(figsize=(10, 5))

        cores = ['green' if v else 'red' for v in df_check['dentro_ic']]
        ax.bar(df_check['t+'], df_check['erro_pct'] * 100, color=cores)

        ax.set_xlabel("Horizonte da Previsão (t+)")
        ax.set_ylabel("Erro Percentual (MAPE %)")
        ax.set_title("Assertividade por t+ (Verde: dentro do IC | Vermelho: fora do IC)")
        ax.set_xticks(df_check['t+'])
        ax.set_ylim(0, max(df_check['erro_pct'] * 100) * 1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        st.pyplot(fig)
    else:
        st.warning("⚠️ Não há dados de realizado após a data de corte para avaliar assertividade (t+).")

