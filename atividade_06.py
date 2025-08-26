import numpy as np
import pandas as pd
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler

# Carregar dataset
Z = np.loadtxt("data/processed/recfaces_pca.dat")
X, y = Z[:, :-1], Z[:, -1]

# Escalonamento
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# PCA com q da Questão 5
q = 3
pca = PCA(n_components=q)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Classificadores
classifiers = {
    "MQ": LinearRegression(),
    "PL": LogisticRegression(max_iter=5000),  # mais iterações
    "MLP-1H": MLPClassifier(hidden_layer_sizes=(50,), max_iter=5000, random_state=42),
    "MLP-2H": MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=5000, random_state=42)
}

# Scorer especial para MQ
def mq_accuracy(y_true, y_pred_continuous):
    y_pred = (y_pred_continuous >= 0.5).astype(int)
    return accuracy_score(y_true, y_pred)

mq_scorer = make_scorer(mq_accuracy, greater_is_better=True)

# Resultados
results = []

for name, clf in classifiers.items():
    start = time.time()

    if name == "MQ":
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        y_pred = (y_pred >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)

        scores = cross_val_score(clf, X_train_pca, y_train, cv=5, scoring=mq_scorer)
    else:
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        acc = accuracy_score(y_test, y_pred)

        scores = cross_val_score(clf, X_train_pca, y_train, cv=5, scoring="accuracy")

    exec_time = time.time() - start

    results.append({
        "Classificador": name,
        "Média": np.mean(scores),
        "Mínimo": np.min(scores),
        "Máximo": np.max(scores),
        "Mediana": np.median(scores),
        "Desvio Padrão": np.std(scores),
        "Tempo de execução": exec_time
    })

# Tabela final
df_results = pd.DataFrame(results)
print(df_results)


# Salvar em CSV/Excel se quiser
df_results.to_csv("atividade_06_pca.csv", index=False)
