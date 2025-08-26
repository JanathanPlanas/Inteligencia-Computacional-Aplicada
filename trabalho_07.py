import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# ------------------------------
# 1. Carregar dados originais
# ------------------------------
X = np.load("X_faces.npy")   # matriz de atributos
y = np.load("y_faces.npy")   # rótulos

# ------------------------------
# 2. Aplicar PCA (q escolhido na Atividade 5)
# ------------------------------
q = 120  # <-- substitua pelo valor obtido na Atividade 5
pca = PCA(n_components=q)
X_pca = pca.fit_transform(X)

# ------------------------------
# 3. Aplicar Box-Cox
# ------------------------------
# Box-Cox só funciona com valores positivos
# então vamos garantir que os dados sejam positivos
X_pca_shifted = X_pca - X_pca.min() + 1e-6  

pt = PowerTransformer(method='box-cox')  
X_boxcox = pt.fit_transform(X_pca_shifted)

# ------------------------------
# 4. Normalização Z-score
# ------------------------------
scaler = StandardScaler()
X_final = scaler.fit_transform(X_boxcox)

# ------------------------------
# 5. Definir classificadores
# ------------------------------
classifiers = {
    "MQ": Ridge(alpha=1.0),                          # Mínimos Quadrados com regularização
    "PL": LinearDiscriminantAnalysis(),              # Pseudo-LDA
    "MLP-1H": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000),
    "MLP-2H": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000)
}

# ------------------------------
# 6. Avaliar modelos com cross-validation
# ------------------------------
results = []
for name, clf in classifiers.items():
    cv = cross_validate(clf, X_final, y, cv=5, return_train_score=False,
                        scoring='accuracy', return_estimator=False)
    res = {
        "Classificador": name,
        "Média": cv['test_score'].mean(),
        "Mínimo": cv['test_score'].min(),
        "Máximo": cv['test_score'].max(),
        "Mediana": np.median(cv['test_score']),
        "Desvio Padrão": cv['test_score'].std(),
        "Tempo de execução": cv['fit_time'].sum()
    }
    results.append(res)

# ------------------------------
# 7. Mostrar tabela de resultados
# ------------------------------


df_results = pd.DataFrame(results)

df_results.to_csv(r'data\processed\trabalho_07.csv')
print(df_results)
