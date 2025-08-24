
##====================================================================
# CustomLinearRegression class implementation
import numpy as np
from typing import Union
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score

import os



def plot_real_vs_pred(y_train, y_train_pred, y_test, y_test_pred, modelo_nome="Modelo"):
    """
    Plota a dispersão de valores reais vs preditos para treino e teste.
    """
    plt.figure(figsize=(16, 7))

    # Treino
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5, label="Pontos")
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--r', linewidth=2, label="Linha Ideal (y=x)")
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Predito")
    plt.title(f"Treino: Real vs. Predito ({modelo_nome})")
    plt.grid(True)
    plt.legend()

    # Teste
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5, color='orange', label="Pontos")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label="Linha Ideal (y=x)")
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Predito")
    plt.title(f"Teste: Real vs. Predito ({modelo_nome})")
    plt.grid(True)
    plt.legend()

    plt.show()


def plot_residuals(y_true, y_pred, modelo_nome="Modelo"):
    """
    Plota a distribuição dos resíduos com histograma, KDE e curva normal teórica.
    """
    residuals = y_true - y_pred
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 200)

    plt.figure(figsize=(9,6))
    sns.histplot(residuals, bins=20, stat="density", kde=True,
                 color="skyblue", edgecolor="white", alpha=0.6, label="Resíduos")
    plt.plot(x, norm.pdf(x, mu, sigma), color="red", lw=2, linestyle="--", label="Normal Esperada")
    plt.axvline(mu, color="blue", linestyle="--", lw=2, label=f"Média = {mu:.2f}")
    plt.axvline(mu + sigma, color="green", linestyle=":", lw=2, label=f"+1 Desvio = {mu+sigma:.2f}")
    plt.axvline(mu - sigma, color="green", linestyle=":", lw=2, label=f"-1 Desvio = {mu-sigma:.2f}")
    plt.title(f"Distribuição dos Resíduos ({modelo_nome})", fontsize=14)
    plt.xlabel("Erro (y_real - y_pred)", fontsize=12)
    plt.ylabel("Densidade", fontsize=12)
    plt.legend()
    sns.despine()
    plt.show()

##=================================

def avaliar_modelo(y_train, y_train_pred, y_test, y_test_pred, nome_modelo="Modelo"):
    """
    Calcula métricas de avaliação para regressão, tanto para treino quanto para teste,
    e retorna um DataFrame horizontal com todas as métricas lado a lado.

    Parâmetros:
    ----------
    y_train : array-like
        Valores reais do conjunto de treino
    y_train_pred : array-like
        Valores previstos pelo modelo no treino
    y_test : array-like
        Valores reais do conjunto de teste
    y_test_pred : array-like
        Valores previstos pelo modelo no teste
    nome_modelo : str
        Nome do modelo (opcional)

    Retorna:
    -------
    metrics_df : pd.DataFrame
        DataFrame horizontal com métricas de treino e teste
    """
    
    # Função para calcular hit rate dentro de percentual
    def hit_rate(y_true, y_pred, percent):
        return np.mean(np.abs(y_pred - y_true) <= (percent/100) * y_true)

    # Função auxiliar para métricas de um conjunto
    def calc_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        medae = median_absolute_error(y_true, y_pred)
        ev = explained_variance_score(y_true, y_pred)
        hr20 = hit_rate(y_true, y_pred, 20)
        hr10 = hit_rate(y_true, y_pred, 10)
        avg_price = np.mean(y_true)
        rmse_ratio = rmse / avg_price
        return [rmse, r2, corr, mae, mape, medae, ev, hr20, hr10, avg_price, rmse_ratio]

    # Calcula métricas
    metrics_train = calc_metrics(y_train, y_train_pred)
    metrics_test = calc_metrics(y_test, y_test_pred)

    # Criar DataFrame horizontal
    metrics_df = pd.DataFrame({
        "Modelo": [nome_modelo, nome_modelo],
        "RMSE": [metrics_train[0], metrics_test[0]],
        "R2": [metrics_train[1], metrics_test[1]],
        "Corr": [metrics_train[2], metrics_test[2]],
        "MAE": [metrics_train[3], metrics_test[3]],
        "MAPE": [metrics_train[4], metrics_test[4]],
        "MedAE": [metrics_train[5], metrics_test[5]],
        "EV": [metrics_train[6], metrics_test[6]],
        "HitRate_20%": [metrics_train[7], metrics_test[7]],
        "HitRate_10%": [metrics_train[8], metrics_test[8]],
        "Avg_price": [metrics_train[9], metrics_test[9]],
        "RMSE_over_Avg": [metrics_train[10], metrics_test[10]]
    }, index=["Treino", "Teste"])

    return metrics_df


if __name__ == '__main__':
    
        # --- 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
    # Carregar os dados
    path = r'https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx'
    df = pd.read_excel(io=path)

    # Selecionar features e target
    X = df[["X2 house age", "X3 distance to the nearest MRT station",
            "X4 number of convenience stores", "X5 latitude", "X6 longitude"]]
    y = df["Y house price of unit area"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # --- Padronizar features para redes ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # ---  Regressão Linear ---==============================================================gi
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_train_pred_lr = lr.predict(X_train)
    y_test_pred_lr = lr.predict(X_test)
    metrics_lr = avaliar_modelo(y_train, y_train_pred_lr, y_test, y_test_pred_lr, "Linear Regression")

    print(metrics_lr)

    metrics_lr.to_csv(r'data\processed\metricas_LinearRegression_funcao.csv')


    # ---  Rede Perceptron  ---==============================================================

    perceptron = SGDRegressor(max_iter=1000,
                            tol=1e-3,
                            random_state=42)
    perceptron.fit(X_train_scaled, y_train)
    y_train_pred_ps = perceptron.predict(X_train_scaled)
    y_test_pred_ps = perceptron.predict(X_test_scaled)
    metrics_ps = avaliar_modelo(y_train, y_train_pred_ps, y_test, y_test_pred_ps, "Perceptron")
    print(metrics_ps)

    metrics_ps.to_csv(r'data\processed\metricas_perceptrion_funcao.csv')
    # --- 3 MLP com 1 camada oculta ---==============================================================
    mlp_1 = MLPRegressor(hidden_layer_sizes=(10,),
                        activation='relu',
                        solver='adam',
                        max_iter=1000,
                        random_state=35)
    mlp_1.fit(X_train_scaled, y_train)
    y_train_pred_mlp1 = mlp_1.predict(X_train_scaled)
    y_test_pred_mlp1 = mlp_1.predict(X_test_scaled)
    metrics_mlp1 = avaliar_modelo(y_train, y_train_pred_mlp1, y_test, y_test_pred_mlp1, "MLP 1 camada")

    print(metrics_mlp1)

    metrics_mlp1.to_csv(r'data\processed\metricas_MLP_1hiddenLayer_funcao.csv')

    # --- 4 MLP com 2 camadas ocultas ---==============================================================
    mlp_2 = MLPRegressor(hidden_layer_sizes=(10, 5),
                        activation='relu',
                        solver='adam',
                        max_iter=1000,
                        random_state=40)
    mlp_2.fit(X_train_scaled, y_train)
    y_train_pred_mlp2 = mlp_2.predict(X_train_scaled)
    y_test_pred_mlp2 = mlp_2.predict(X_test_scaled)
    metrics_mlp2 = avaliar_modelo(y_train, y_train_pred_mlp2, y_test, y_test_pred_mlp2, "MLP 2 camadas")
    print(metrics_mlp2)
    metrics_mlp2.to_csv(r'data\processed\metricas_MLP_2hiddenLayer_funcao.csv')




    # --- SVR com kernel RBF (padrão) ---==============================================================
    svr_rbf = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr_rbf.fit(X_train_scaled, y_train)
    y_train_pred_svr = svr_rbf.predict(X_train_scaled)
    y_test_pred_svr = svr_rbf.predict(X_test_scaled)
    metrics_svr = avaliar_modelo(y_train, y_train_pred_svr, y_test, y_test_pred_svr, "SVR RBF")
    print(metrics_svr)

    metrics_svr.to_csv(r'data\processed\metricas_SVR_funcao.csv')

    # --- LSSVR Linear (aproximação) ---=-==============================================================
    lssvr = SVR(kernel='linear', C=1e6, epsilon=1e-3)
    lssvr.fit(X_train_scaled, y_train)
    y_train_pred_lssvr = lssvr.predict(X_train_scaled)
    y_test_pred_lssvr = lssvr.predict(X_test_scaled)
    metrics_lssvr = avaliar_modelo(y_train, y_train_pred_lssvr, y_test, y_test_pred_lssvr, "LSSVR Linear")

    print(metrics_lssvr)

    metrics_lssvr.to_csv(r'data\processed\metricas_LSSVR_funcao.csv')


    # --- Concatenar todos os resultados ---
    all_metrics = pd.concat([metrics_lr, metrics_ps, metrics_mlp1, metrics_mlp2, metrics_svr, metrics_lssvr])
    print(all_metrics)

    # --- Concatenar todos os resultados ---
    all_metrics = pd.concat([metrics_lr, metrics_ps, metrics_mlp1, metrics_mlp2, metrics_svr, metrics_lssvr])
    print(all_metrics)

    all_metrics.to_csv(r'data\processed\metricas.csv')



    # --- Criar pasta para salvar imagens ---
    os.makedirs("images", exist_ok=True)

    # Dicionário com todos os modelos e suas previsões
    modelos = {
        "Linear Regression": (y_train_pred_lr, y_test_pred_lr),
        "Perceptron Linear": (y_train_pred_ps, y_test_pred_ps),
        "MLP 1 camada": (y_train_pred_mlp1, y_test_pred_mlp1),
        "MLP 2 camadas": (y_train_pred_mlp2, y_test_pred_mlp2),
        "SVR RBF": (y_train_pred_svr, y_test_pred_svr),
        "LSSVR Linear": (y_train_pred_lssvr, y_test_pred_lssvr)
    }

    # --- Loop para gerar plots ---
    for nome_modelo, (y_tr_pred, y_te_pred) in modelos.items():
        # Plot Real vs Predito
        plt.figure(figsize=(16, 7))

        # Treino
        plt.subplot(1, 2, 1)
        plt.scatter(y_train, y_tr_pred, alpha=0.5, label="Pontos")
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--r', linewidth=2, label="Linha Ideal (y=x)")
        plt.xlabel("Valor Real")
        plt.ylabel("Valor Predito")
        plt.title(f"Treino: Real vs. Predito ({nome_modelo})")
        plt.grid(True)
        plt.legend()

        # Teste
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_te_pred, alpha=0.5, color='orange', label="Pontos")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label="Linha Ideal (y=x)")
        plt.xlabel("Valor Real")
        plt.ylabel("Valor Predito")
        plt.title(f"Teste: Real vs. Predito ({nome_modelo})")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"images/{nome_modelo.replace(' ', '_')}_real_vs_pred.png")
        plt.close()

        # Plot Resíduos (Treino)
        residuals = y_train - y_tr_pred
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 200)

        plt.figure(figsize=(9,6))
        sns.histplot(residuals, bins=20, stat="density", kde=True,
                    color="skyblue", edgecolor="white", alpha=0.6, label="Resíduos")
        plt.plot(x, norm.pdf(x, mu, sigma), color="red", lw=2, linestyle="--", label="Normal Esperada")
        plt.axvline(mu, color="blue", linestyle="--", lw=2, label=f"Média = {mu:.2f}")
        plt.axvline(mu + sigma, color="green", linestyle=":", lw=2, label=f"+1 Desvio = {mu+sigma:.2f}")
        plt.axvline(mu - sigma, color="green", linestyle=":", lw=2, label=f"-1 Desvio = {mu-sigma:.2f}")
        plt.title(f"Distribuição dos Resíduos ({nome_modelo})", fontsize=14)
        plt.xlabel("Erro (y_real - y_pred)", fontsize=12)
        plt.ylabel("Densidade", fontsize=12)
        plt.legend()
        sns.despine()

        plt.tight_layout()
        plt.savefig(f"images/{nome_modelo.replace(' ', '_')}_residuos.png")
        plt.close()
