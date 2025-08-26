import numpy as np
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ==============================================
# Configurações
# ==============================================
Nr = 50
Ptrain = 0.8
data_folder = 'data/processed'
os.makedirs(data_folder, exist_ok=True)

# Normalizações a testar
scalers = {
    'none': None,
    
    'zscore': StandardScaler(),
    'minmax_0_1': MinMaxScaler(feature_range=(0, 1)),
    'minmax_neg1_1': MinMaxScaler(feature_range=(-1, 1))
}

# Funções de ativação e solvers para MLP
activations = ['relu', 'tanh', 'logistic']
solvers = ['adam', 'sgd']

# ==============================================
# Carrega dataset PCA
# ==============================================
Z = np.loadtxt(os.path.join(data_folder, 'recfaces_pca.dat'))
X = Z[:, :-1]
y = Z[:, -1].astype(int)

print(f"Dimensão de X: {X.shape}, y: {y.shape}")

# ==============================================
# Inicializa dataframe para salvar resultados
# ==============================================
columns = ['Classificador', 'Normalizacao', 'Activation', 'Solver', 'Media', 'Min', 'Max', 'Mediana', 'Desvio', 'Tempo']
results_df = pd.DataFrame(columns=columns)

# ==============================================
# Loop pelas normalizações
# ==============================================
for scaler_name, scaler in scalers.items():
    if scaler is not None:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()
    
    # ---------------- MQ ----------------
    acc_list = []
    time_list = []
    for _ in range(Nr):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=Ptrain, stratify=y)
        start = time.time()
        mq = LinearRegression()
        mq.fit(X_train, y_train)
        y_pred = np.round(mq.predict(X_test)).astype(int)
        acc_list.append(accuracy_score(y_test, y_pred))
        time_list.append(time.time() - start)
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Classificador': 'MQ', 'Normalizacao': scaler_name, 'Activation': '-', 'Solver': '-',
        'Media': np.mean(acc_list)*100,
        'Min': np.min(acc_list)*100,
        'Max': np.max(acc_list)*100,
        'Mediana': np.median(acc_list)*100,
        'Desvio': np.std(acc_list)*100,
        'Tempo': np.mean(time_list)
    }])], ignore_index=True)
    
    # ---------------- PL ----------------
    acc_list = []
    time_list = []
    for _ in range(Nr):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=Ptrain, stratify=y)
        start = time.time()
        pl = LogisticRegression(max_iter=5000, multi_class='multinomial', solver='lbfgs')
        pl.fit(X_train, y_train)
        y_pred = pl.predict(X_test)
        acc_list.append(accuracy_score(y_test, y_pred))
        time_list.append(time.time() - start)
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Classificador': 'PL', 'Normalizacao': scaler_name, 'Activation': '-', 'Solver': '-',
        'Media': np.mean(acc_list)*100,
        'Min': np.min(acc_list)*100,
        'Max': np.max(acc_list)*100,
        'Mediana': np.median(acc_list)*100,
        'Desvio': np.std(acc_list)*100,
        'Tempo': np.mean(time_list)
    }])], ignore_index=True)
    
    # ---------------- MLP ----------------
    for act in activations:
        for solver in solvers:
            # MLP-1H
            acc_list = []
            time_list = []
            for _ in range(Nr):
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=Ptrain, stratify=y)
                start = time.time()
                mlp = MLPClassifier(hidden_layer_sizes=(50,), activation=act, solver=solver, max_iter=1000)
                mlp.fit(X_train, y_train)
                y_pred = mlp.predict(X_test)
                acc_list.append(accuracy_score(y_test, y_pred))
                time_list.append(time.time() - start)
            results_df = pd.concat([results_df, pd.DataFrame([{
                'Classificador': 'MLP-1H', 'Normalizacao': scaler_name, 'Activation': act, 'Solver': solver,
                'Media': np.mean(acc_list)*100,
                'Min': np.min(acc_list)*100,
                'Max': np.max(acc_list)*100,
                'Mediana': np.median(acc_list)*100,
                'Desvio': np.std(acc_list)*100,
                'Tempo': np.mean(time_list)
            }])], ignore_index=True)
            
            # MLP-2H
            acc_list = []
            time_list = []
            for _ in range(Nr):
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=Ptrain, stratify=y)
                start = time.time()
                mlp = MLPClassifier(hidden_layer_sizes=(50,30), activation=act, solver=solver, max_iter=1000)
                mlp.fit(X_train, y_train)
                y_pred = mlp.predict(X_test)
                acc_list.append(accuracy_score(y_test, y_pred))
                time_list.append(time.time() - start)
            results_df = pd.concat([results_df, pd.DataFrame([{
                'Classificador': 'MLP-2H', 'Normalizacao': scaler_name, 'Activation': act, 'Solver': solver,
                'Media': np.mean(acc_list)*100,
                'Min': np.min(acc_list)*100,
                'Max': np.max(acc_list)*100,
                'Mediana': np.median(acc_list)*100,
                'Desvio': np.std(acc_list)*100,
                'Tempo': np.mean(time_list)
            }])], ignore_index=True)

# ==============================================
# Salva CSV com melhores resultados
# ==============================================
best_results = results_df.loc[results_df.groupby('Classificador')['Media'].idxmax()]
best_results.to_csv(os.path.join(data_folder, 'tabela_resultados_pca.csv'), index=False)
print(f"Resultados salvos em '{os.path.join(data_folder, 'tabela_resultados_pca.csv')}'")
