# streamlit_enem_analysis.py
# Versão otimizada para Streamlit Cloud, usando CSV e numpy para cálculos estatísticos.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import log

st.set_page_config(layout="wide", page_title="Análise ENEM (Plotly)")

st.title("📊 Análise Estatística e Modelagem — ENEM (Plotly)")
st.markdown(
    "App compatível com Streamlit Cloud Free. Implementa OLS e Logística usando apenas **numpy/pandas/plotly/streamlit**."
)

# ------------------ utilitários numéricos (numpy) ------------------

def ols_fit(X, y, add_intercept=True):
    """Ajusta o modelo de Mínimos Quadrados Ordinários (OLS) usando álgebra linear do numpy."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    if add_intercept:
        X_design = np.column_stack([np.ones(len(X)), X])
    else:
        X_design = X
    n, p = X_design.shape
    
    # Cálculos OLS e Inversa Generalizada (pinv)
    XtX = X_design.T @ X_design
    invXtX = np.linalg.pinv(XtX)
    beta = invXtX @ X_design.T @ y
    
    y_hat = (X_design @ beta).flatten()
    resid = (y.flatten() - y_hat)
    
    # Métricas e Variâncias
    RSS = float((resid**2).sum())
    df_resid = max(n - p, 1)
    sigma2 = RSS / df_resid
    cov_beta = sigma2 * invXtX
    se_beta = np.sqrt(np.diag(cov_beta))
    se_beta = np.where(se_beta == 0, 1e-12, se_beta) # Proteção contra SE zero
    t_stats = (beta.flatten() / se_beta)
    
    # R2, AIC, BIC
    y_mean = y.mean()
    TSS = float(((y - y_mean)**2).sum())
    R2 = 1 - RSS / TSS if TSS > 0 else np.nan
    aic = n * np.log(RSS / n) + 2 * p
    bic = n * np.log(RSS / n) + p * np.log(n)
    
    return {
        "beta": beta.flatten(), "se": se_beta, "t": t_stats, "y_hat": y_hat, 
        "resid": resid, "RSS": RSS, "sigma2": sigma2, "cov_beta": cov_beta, 
        "invXtX": invXtX, "n": n, "p": p, "R2": R2, "TSS": TSS, 
        "aic": aic, "bic": bic, "X_design": X_design
    }

def compute_vif(dfX):
    """Calcula o VIF (Fator de Inflação da Variância) para multicolinearidade."""
    X = np.asarray(dfX)
    n, k = X.shape
    vifs = {}
    for j in range(k):
        y = X[:, j]
        X_others = np.delete(X, j, axis=1)
        if X_others.shape[1] == 0:
            vifs[dfX.columns[j]] = np.nan
            continue
        
        # OLS para regressão de Xj nos outros X's
        XtX = X_others.T @ X_others
        beta = np.linalg.pinv(XtX) @ X_others.T @ y
        yhat = X_others @ beta
        
        ssr = ((yhat - y.mean())**2).sum()
        sst = ((y - y.mean())**2).sum()
        R2 = ssr / sst if sst > 0 else 0.0
        vif = 1.0 / (1.0 - R2) if (1.0 - R2) != 0 else np.inf
        vifs[dfX.columns[j]] = vif
    return pd.DataFrame.from_dict(vifs, orient='index', columns=['VIF'])

def durbin_watson(resid):
    """Calcula a estatística Durbin-Watson para autocorrelação."""
    r = np.asarray(resid)
    # d = sum((e_i - e_i-1)^2) / sum(e_i^2)
    return float(np.sum(np.diff(r)**2) / np.sum(r**2))

def breusch_pagan_stat(resid, X):
    """Calcula a estatística LM de Breusch-Pagan para heterocedasticidade."""
    # Regressão dos resíduos quadrados nos preditores
    y_bp = resid**2
    X_design = np.column_stack([np.ones(len(X)), np.asarray(X)])
    
    beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y_bp
    yhat = X_design @ beta
    
    ssr = ((yhat - y_bp.mean())**2).sum()
    sst = ((y_bp - y_bp.mean())**2).sum()
    R2 = ssr / sst if sst > 0 else 0.0
    LM = len(resid) * R2 # Estatística LM
    return {"LM": float(LM), "R2": float(R2)}

def jarque_bera(resid):
    """Calcula a estatística Jarque-Bera para normalidade."""
    r = np.asarray(resid)
    n = len(r)
    m2 = np.mean((r - r.mean())**2)
    m3 = np.mean((r - r.mean())**3)
    m4 = np.mean((r - r.mean())**4)
    skew = m3 / (m2**1.5) if m2>0 else 0.0
    kurt = m4 / (m2**2) if m2>0 else 0.0
    # JB = n/6 * (skew^2 + (kurt - 3)^2 / 4)
    jb = n/6.0 * (skew**2 + (kurt - 3.0)**2 / 4.0)
    return {"JB": float(jb), "skew": float(skew), "kurtosis": float(kurt)}

def influence_measures(fit):
    """Calcula Cook's Distance, DFFITS e DFBETAS."""
    X = fit["X_design"]
    invXtX = fit["invXtX"]
    resid = fit["resid"]
    MSE = fit["sigma2"]
    n, p = X.shape
    
    # Leverage (h_ii)
    X_inv = X @ invXtX
    h = np.sum(X_inv * X, axis=1)
    
    denom_safe = np.where((1 - h) == 0, 1e-12, (1 - h))
    
    # Cook's Distance
    cooks = (resid**2) / (p * MSE) * (h / denom_safe)
    
    # DFFITS
    with np.errstate(divide='ignore', invalid='ignore'):
        dffits = (resid / np.sqrt(MSE * (1 - h))) * np.sqrt(h)
    
    # DFBETAS
    invXtX_Xt = invXtX @ X.T
    delta_b = - (invXtX_Xt * resid) / denom_safe 
    se_beta = np.sqrt(np.diag(invXtX) * MSE).reshape(-1, 1)
    dfbetas = (delta_b / se_beta).T
    
    return {"leverage": h, "cooks": cooks, "dffits": dffits, "dfbetas": dfbetas}

def logistic_newton(X, y, add_intercept=True, max_iter=200, tol=1e-6):
    """Ajusta o modelo de Regressão Logística via Newton-Raphson."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    if add_intercept:
        X_design = np.column_stack([np.ones(len(X)), X])
    else:
        X_design = X
    n, p = X_design.shape
    beta = np.zeros((p,1))
    
    for _ in range(max_iter):
        z = X_design @ beta
        mu = 1.0 / (1.0 + np.exp(-z))
        W = (mu * (1 - mu)).flatten()
        W_safe = np.where(W == 0, 1e-12, W)
        
        grad = X_design.T @ (y - mu)
        XW = X_design * W_safe.reshape(-1,1)
        H = -(X_design.T @ XW) # Hessian
        
        try:
            delta = np.linalg.pinv(H) @ grad
        except Exception:
            # Adiciona pequena regularização para estabilidade se H for singular
            delta = np.linalg.pinv(H + 1e-6*np.eye(p)) @ grad
            
        beta_new = beta + delta
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
        
    probs = (1.0 / (1.0 + np.exp(-(X_design @ beta)))).flatten()
    return {"beta": beta.flatten(), "proba": probs, "X_design": X_design}

def auc_from_probs(y_true, probs):
    """Calcula a Área Sob a Curva ROC (AUC)."""
    y = np.asarray(y_true)
    p = np.asarray(probs)
    
    # Ordena pelos scores (probs)
    idx = np.argsort(-p)
    y_sorted = y[idx]
    
    # Calcula TP e FP acumulados
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    
    # Taxas: TPR (Sensibilidade) e FPR (1 - Especificidade)
    tp_rate = tp / max(tp[-1], 1)
    fp_rate = fp / max(fp[-1], 1)
    
    # Adiciona (0,0) para iniciar a curva
    x = np.concatenate([[0.0], fp_rate])
    yv = np.concatenate([[0.0], tp_rate])
    
    # Área (Método trapezoidal)
    auc = 0.0
    for i in range(1, len(x)):
        auc += (x[i] - x[i-1]) * (yv[i] + yv[i-1]) / 2.0
    return abs(auc)

# ------------------ UI / Entrada de dados (AGORA USANDO CSV) ------------------

st.sidebar.header("Dados")
use_upload = st.sidebar.checkbox("Fazer upload do arquivo (.csv)", value=False)
# Delimitador correto para o arquivo 'Enem_2024_Amostra_Perfeita.csv' é a vírgula (,)
DELIMITER = st.sidebar.text_input("Delimitador do CSV", value=",")

if use_upload:
    uploaded = st.sidebar.file_uploader("Escolha o arquivo .csv", type=["csv"])
    if uploaded is None:
        st.info("Faça upload do arquivo ou desmarque o upload para usar o arquivo padrão.")
        st.stop()
    try:
        df = pd.read_csv(uploaded, sep=DELIMITER)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo CSV: {e}")
        st.stop()
else:
    # Nome do arquivo padrão fornecido pelo usuário
    DEFAULT_FILENAME = "Enem_2024_Amostra_Perfeita.csv"
    DEFAULT_PATH = DEFAULT_FILENAME
    try:
        # Lendo o arquivo padrão com o delimitador correto (vírgula)
        df = pd.read_csv(DEFAULT_PATH, sep=',') 
    except Exception as e:
        st.warning(f"Erro ao ler o arquivo padrão '{DEFAULT_FILENAME}': {e}.")
        st.info("Por favor, faça o upload do seu arquivo CSV usando a opção acima.")
        st.stop()

st.write(f"Arquivo CSV — dimensão: {df.shape[0]} x {df.shape[1]}")
st.dataframe(df.head())

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.error("Nenhuma coluna numérica encontrada. Verifique se o delimitador está correto e se as colunas são numéricas.")
    st.stop()

st.sidebar.header("Variáveis")
# Default target é a nota média, que costuma ser a última coluna numérica
default_target_index = numeric_cols.index('NOTA_MEDIA_5_NOTAS') if 'NOTA_MEDIA_5_NOTAS' in numeric_cols else 0
target = st.sidebar.selectbox("Escolha target (Y)", numeric_cols, index=default_target_index)
predictors = st.sidebar.multiselect("Escolha preditores (X) — vazio = todas as numéricas exceto Y",
                                     [c for c in numeric_cols if c != target])
if not predictors:
    predictors = [c for c in numeric_cols if c != target]

st.write("**Target:**", target)
st.write("**Preditores:**", predictors)

# filtrar NAs
df_mod = df[[target] + predictors].dropna()
Y = df_mod[target].values
X_df = df_mod[predictors].copy()

# ------------------ 1) Correlação ------------------
st.header("1️⃣ Análise de Correlação")
st.subheader("Matriz de Correlação (Pearson)")
corr = df_mod.corr()
st.dataframe(corr.style.background_gradient(cmap="RdBu").format(precision=3))

def pearson_perm_pvals(df_numeric, n_perm=200):
    cols = df_numeric.columns
    pmat = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            x = df_numeric.iloc[:, i].values
            y = df_numeric.iloc[:, j].values
            robs = np.corrcoef(x, y)[0,1]
            count = 0
            for _ in range(n_perm):
                yperm = np.random.permutation(y)
                rperm = np.corrcoef(x, yperm)[0,1]
                if abs(rperm) >= abs(robs):
                    count += 1
            p = (count + 1) / (n_perm + 1)
            pmat.iloc[i,j] = p
            pmat.iloc[j,i] = p
    return pmat

st.subheader("P-valores (permutação, 200 permutações — reduzível)")
perm_proc = st.checkbox("Usar menos permutações (mais rápido)?", value=True)
n_perm = 100 if perm_proc else 300
with st.spinner(f"Calculando p-values por permutação ({n_perm} perm.)..."):
    pvals = pearson_perm_pvals(df_mod, n_perm=n_perm)
st.dataframe(pvals.style.format(precision=4).applymap(lambda v: 'background-color: #ffcccc' if (isinstance(v, float) and v < 0.05) else ''))

# scatter interactive with plotly
st.subheader("Gráfico de Dispersão (interativo)")
scatter_feature = st.selectbox("Escolha um preditor para scatter", predictors)
fig_scatter = px.scatter(df_mod, x=scatter_feature, y=target, trendline="ols", title=f"{target} vs {scatter_feature}")
st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------ 2) Seleção de Variáveis ------------------
st.header("2️⃣ Seleção de Variáveis (forward / backward / stepwise)")
method = st.selectbox("Método", ["backward", "forward", "stepwise"])
alpha_in = st.number_input("p-valor para entrar (aprox via |t|~2)", value=0.05, format="%.4f")
alpha_out = st.number_input("p-valor para sair (aprox via |t|~2)", value=0.05, format="%.4f")

def stepwise_selection(Xdf, y, method="stepwise", alpha_in=0.05, alpha_out=0.05, verbose=True):
    cols = list(Xdf.columns)
    included = []
    
    # Inicializa com todas as colunas para o backward, se for o caso
    if method == "backward":
        included = cols[:]
        
    while True:
        changed = False
        
        # 1. Forward Step (Adicionar)
        if method in ("forward","stepwise") or (method == "backward" and not changed):
            excluded = [c for c in cols if c not in included]
            best_col = None; best_score = 1.0
            
            for c in excluded:
                cols_try = included + [c]
                if len(y) <= len(cols_try) + 1: continue
                
                fit = ols_fit(Xdf[cols_try].values, y, add_intercept=True)
                t_last = fit["t"][-1]
                p_approx = 0.05 if abs(t_last) >= 2 else 0.32
                
                if p_approx < best_score:
                    best_score = p_approx; best_col = c
            
            if best_col is not None and best_score < alpha_in:
                included.append(best_col); changed=True
                if verbose: st.write(f"Add {best_col} (approx p {best_score:.4f})")
        
        # 2. Backward Step (Remover)
        if method in ("backward","stepwise"):
            if included:
                fit = ols_fit(Xdf[included].values, y, add_intercept=True)
                tvals = fit["t"][1:]
                
                if len(tvals) > 0:
                    p_approx_list = [0.05 if abs(t)>=2 else 0.32 for t in tvals]
                    worst_idx = int(np.argmax(p_approx_list))
                    
                    if p_approx_list[worst_idx] > alpha_out:
                        rem = included[worst_idx]
                        included.remove(rem); changed=True
                        if verbose: st.write(f"Drop {rem} (approx p {p_approx_list[worst_idx]:.4f})")
        
        if not changed:
            break
            
    return included

with st.spinner("Executando seleção..."):
    # Garante que o stepwise/backward começam com o conjunto de colunas inicial
    X_initial = X_df[predictors] 
    
    if method == "backward":
        # Para backward, começamos com todos os preditores
        selected = stepwise_selection(X_initial, Y, method=method, alpha_in=alpha_in, alpha_out=alpha_out, verbose=True)
    elif method == "forward" or method == "stepwise":
        # Para forward/stepwise, começamos do zero
        selected = stepwise_selection(X_initial, Y, method=method, alpha_in=alpha_in, alpha_out=alpha_out, verbose=True)
    else:
        selected = [] # Should not happen

st.success(f"Variáveis selecionadas: {selected if selected else 'Nenhuma'}")

# ------------------ Fit final OLS ------------------
st.header("Modelo Final (OLS — cálculos em numpy)")

# Tratamento para Intercepto apenas
if len(selected) == 0:
    st.info("Nenhuma variável foi selecionada. Ajustando modelo apenas com Intercepto.")
    # Passa array de preditores vazia, mas com intercept=True
    X_sel_values = np.zeros((len(Y), 0))
    fit = ols_fit(X_sel_values, Y, add_intercept=True) 
    coef_names = ["Intercept"]
else:
    X_sel = X_df[selected]
    fit = ols_fit(X_sel.values, Y, add_intercept=True)
    coef_names = ["Intercept"] + list(X_sel.columns)

coef_df = pd.DataFrame({
    "coef": fit["beta"],
    "se": fit["se"],
    "t-stat": fit["t"]
}, index=coef_names)
st.subheader("Coeficientes (estimativa | se | t-stat)")
st.dataframe(coef_df.style.format("{:.6g}"))

# Métricas do Modelo (Teste F, R2, RMSE)
st.subheader("5️⃣ Métricas do Modelo: Teste F, R², RMSE")
st.write(f"R² = {fit['R2']:.6f}")
RMSE = np.sqrt(fit["RSS"]/fit["n"])
st.write(f"RMSE = {RMSE:.6f}")

SSR = fit["TSS"] - fit["RSS"]
df_model = fit["p"] - 1
df_resid = fit["n"] - fit["p"]
MSR = SSR / df_model if df_model>0 else np.nan
MSE = fit["RSS"] / df_resid if df_resid>0 else np.nan
F_stat = MSR / MSE if MSE>0 else np.nan

st.write(f"F-statistic = {F_stat:.6g} (df_model={df_model}, df_resid={df_resid})")
st.write(f"AIC = {fit['aic']:.6g}  |  BIC = {fit['bic']:.6g}")

# ------------------ 3) Diagnóstico ------------------
st.header("3️⃣ Diagnóstico das Suposições")

if fit["p"] > 1:
    # 3.1) Linearidade (Resíduos vs Previstos)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fit["y_hat"], y=fit["resid"], mode="markers", name="resíduos"))
    fig.add_trace(go.Scatter(x=[min(fit["y_hat"]), max(fit["y_hat"])], y=[0,0], mode="lines", line=dict(color="red", dash="dash"), name="zero"))
    fig.update_layout(title="Resíduos vs Valores Previstos (Linearidade & Homocedasticidade)", xaxis_title="y_hat", yaxis_title="resíduos")
    st.plotly_chart(fig, use_container_width=True)

    # 3.2) Independência dos erros
    dw = durbin_watson(fit["resid"])
    st.write(f"Durbin-Watson = {dw:.4f} (≈2 sem autocorrelação)")

    # 3.3) Homocedasticidade (Breusch-Pagan)
    # A variável X_sel aqui deve ser a X_df[selected]
    X_diag = X_df[selected].values 
    bp = breusch_pagan_stat(fit["resid"], X_diag)
    st.write(f"Breusch-Pagan LM = {bp['LM']:.6g}  — interprete comparando com chi2 critério (df = k)")

    # 3.4) Normalidade dos erros (Jarque-Bera)
    jb = jarque_bera(fit["resid"])
    st.write(f"Jarque-Bera = {jb['JB']:.6g}; skew = {jb['skew']:.6g}; kurtosis = {jb['kurtosis']:.6g}")
    st.info("Interpretação rápida: JB > 5.99 ⇒ rejeita normalidade ao nível 5% (aprox).")

    # 3.5) Ausência de multicolinearidade (VIF)
    vif_df = compute_vif(X_df[selected])
    st.subheader("VIF (Multicolinearidade)")
    st.dataframe(vif_df.style.format("{:.4f}"))
else:
    st.warning("Diagnósticos das suposições não são aplicáveis quando apenas o Intercepto é ajustado (p=1).")


# ------------------ 4) Outliers / Influence ------------------
st.header("4️⃣ Outliers e Observações Influentes (DFFITS, DFBETAS e Cook's D)")
if fit["p"] > 1:
    inf = influence_measures(fit)
    cooks = inf["cooks"]
    dffits = inf["dffits"]
    dfbetas = inf["dfbetas"]
    leverage = inf["leverage"]

    cooks_df = pd.DataFrame({"index": df_mod.index, "cooks": cooks, "dffits": dffits, "leverage": leverage}).set_index("index").sort_values("cooks", ascending=False)
    st.subheader("Top 10 — Cook's distance")
    st.dataframe(cooks_df.head(10).style.format("{:.6g}"))

    st.subheader("Top 10 — |DFFITS|")
    st.dataframe(pd.DataFrame({"index": df_mod.index, "abs_dffits": np.abs(dffits)}).set_index("index").sort_values("abs_dffits", ascending=False).head(10).style.format("{:.6g}"))

    dfbetas_df = pd.DataFrame(dfbetas, index=df_mod.index, columns=coef_names)
    st.subheader("DFBETAS (máximo absoluto por coeficiente)")
    st.dataframe(dfbetas_df.abs().max().sort_values(ascending=False).to_frame("max_abs_dfbeta").style.format("{:.6g}"))

    # Cook's plot (plotly)
    fig = px.bar(x=np.arange(len(cooks)), y=cooks, labels={"x":"índice", "y":"Cook's D"}, title="Cook's Distance")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Medidas de influência não são calculadas para modelos com apenas Intercepto.")

# ------------------ 6) Comparação e Seleção de Modelos ------------------
st.header("6️⃣ Comparação e Seleção de Modelos (CV / Classificação)")

if len(predictors) > 0:
    st.subheader("K-Fold Cross-Validation (Comparação RMSE)")
    k = st.number_input("K folds (CV)", min_value=2, max_value=10, value=5)
    
    def manual_cv_rmse(Xall, yall, kfolds=5, random_state=42):
        n = len(yall)
        idx = np.arange(n)
        rng = np.random.RandomState(random_state)
        rng.shuffle(idx)
        folds = np.array_split(idx, kfolds)
        rmses = []
        
        # Check if Xall is valid for CV
        if len(Xall.shape) < 2 or Xall.shape[1] == 0:
            return [np.nan] * kfolds
            
        for i in range(kfolds):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(kfolds) if j!=i])
            Xtrain = Xall[train_idx]
            ytrain = yall[train_idx]
            Xtest = Xall[test_idx]
            ytest = yall[test_idx]
            
            if len(ytrain) <= Xtrain.shape[1] + 1:
                rmses.append(np.nan)
                continue
                
            fit_cv = ols_fit(Xtrain, ytrain, add_intercept=True)
            Xtest_design = np.column_stack([np.ones(len(Xtest)), Xtest])
            preds = (Xtest_design @ fit_cv["beta"].reshape(-1,1)).flatten()
            rmse = np.sqrt(np.mean((ytest - preds)**2))
            rmses.append(float(rmse))
        return rmses

    with st.spinner("Executando CV..."):
        rmses_all = manual_cv_rmse(X_df.values, Y, kfolds=k)
        if len(selected) > 0:
            rmses_sel = manual_cv_rmse(X_df[selected].values, Y, kfolds=k)
        else:
            rmses_sel = [np.nan] * k

    st.write("RMSE CV — Todos Preditores: mean={:.4f}, std={:.4f}".format(np.nanmean(rmses_all), np.nanstd(rmses_all)))
    st.write("RMSE CV — Preditores Selecionados: mean={:.4f}, std={:.4f}".format(np.nanmean(rmses_sel), np.nanstd(rmses_sel)))

    # Curva ROC/AUC, F1 Score, Acurácia, Precisão, Sensibilidade, Especificidade
    st.subheader("Classificação (Regressão Logística)")
    do_class = st.checkbox("Transformar target em binário e treinar logistic (Newton-Raphson)", value=False)
    
    if do_class and len(selected) > 0:
        threshold = st.number_input("Threshold para classe positiva (Y >= )", value=int(np.nanmedian(Y)))
        y_bin = (Y >= threshold).astype(int)
        
        with st.spinner("Ajustando Regressão Logística..."):
            logfit = logistic_newton(X_df[selected].values, y_bin, add_intercept=True, max_iter=200)
        
        probs = logfit["proba"]
        preds = (probs >= 0.5).astype(int)
        
        # Métricas de Classificação
        tp = int(((preds==1) & (y_bin==1)).sum())
        fp = int(((preds==1) & (y_bin==0)).sum())
        tn = int(((preds==0) & (y_bin==0)).sum())
        fn = int(((preds==0) & (y_bin==1)).sum())
        
        accuracy = (tp + tn) / max(len(y_bin), 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Sensibilidade
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Especificidade
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        auc = auc_from_probs(y_bin, probs)
        
        st.markdown(f"""
        - **Acurácia:** `{accuracy:.4f}`
        - **Precisão:** `{precision:.4f}`
        - **Sensibilidade (Recall):** `{recall:.4f}`
        - **Especificidade:** `{specificity:.4f}`
        - **F1 Score:** `{f1:.4f}`
        - **AUC:** `{auc:.4f}`
        """)

        # ROC plot
        idx = np.argsort(-probs)
        y_sorted = y_bin[idx]
        tp_roc = np.cumsum(y_sorted)
        fp_roc = np.cumsum(1 - y_sorted)
        tpr = tp_roc / max(tp_roc[-1], 1)
        fpr = fp_roc / max(fp_roc[-1], 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.concatenate([[0], fpr]), y=np.concatenate([[0], tpr]), mode='lines', name=f"AUC={auc:.4f}"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='red'), showlegend=False))
        fig.update_layout(title="Curva ROC", xaxis_title="1 - Especificidade (FPR)", yaxis_title="Sensibilidade (TPR)")
        st.plotly_chart(fig, use_container_width=True)
    elif do_class and not selected:
        st.warning("Selecione preditores (X) para rodar o modelo Logístico.")

else:
    st.info("Validação Cruzada e Classificação não são aplicáveis quando não há preditores para testar.")

st.markdown("---")
st.write("Análise concluída.")