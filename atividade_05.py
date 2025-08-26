import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Carregar dataset
Z = np.loadtxt('data/processed/recfaces_pca.dat')  # Ajuste o caminho
X = Z[:, :-1]  # Features
Y = Z[:, -1]   # Labels

# Executar PCA completo
pca = PCA()
pca.fit(X)

# Variância explicada acumulada
VEq = np.cumsum(pca.explained_variance_ratio_)

# Plot para visualização
plt.figure(figsize=(8,5))
plt.plot(VEq, 'r-', linewidth=2)
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Explicada Acumulada')
plt.title('Variância Explicada Acumulada')
plt.grid(True)
plt.show()

# Percentual desejado
percentual = 0.98

# Indice da primeira componente que atinge ou supera 98%
q = np.argmax(VEq >= percentual) + 1  # +1 pois o índice inicia em 0
print(f"Dimensão q escolhida para preservar 98% da variância: {q}")
pca_q = PCA(n_components=q)
X_reduzido = pca_q.fit_transform(X)

print(f"Forma do dataset após redução: {X_reduzido.shape}")


'''
1️⃣ Redução de dimensionalidade

Antes do PCA: seu dataset tinha 165 amostras e 900 atributos (30×30 pixels vetorizados).

Depois do PCA com q = 3, o dataset passou a ter 165 amostras e 3 atributos.

Conclusão: houve uma redução drástica da dimensionalidade, de 900 → 3, preservando 98% da variância. Isso mostra que grande parte da informação do conjunto de dados original está concentrada em apenas 3 componentes principais.

2️⃣ Interpretação da variância explicada

O vetor VEq mostra a variância explicada acumulada por cada componente principal.

Apenas 3 componentes são suficientes para capturar praticamente toda a informação relevante dos dados.

Isso indica que as imagens dos rostos possuem muita redundância, e os principais padrões de variação (iluminação, expressão, características faciais) podem ser representados por muito poucos eixos principais.

3️⃣ Benefícios dessa redução

Descorrelação dos atributos: os 3 novos atributos (componentes principais) são ortogonais entre si, facilitando algoritmos que assumem independência ou baixa correlação.

Redução de custo computacional: treinar classificadores, redes neurais ou regressões em 3 atributos é muito mais rápido do que com 900.

Visualização: com apenas 3 dimensões, é possível plotar os dados e analisar separabilidade entre indivíduos.

Conclusão geral:

O PCA conseguiu reduzir 900 dimensões para 3, mantendo praticamente toda a informação relevante.

Visualmente, o gráfico mostra clusters distintos por indivíduo, o que indica que a classificação de rostos usando essas 3 componentes principais é viável.

A separação entre indivíduos não é perfeita, sinalizando que em algumas regiões do espaço, rostos semelhantes podem ser confundidos, especialmente se houver expressões faciais parecidas.

'''


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Supondo que X_reduzido e Y já estão definidos
# X_reduzido: (n_amostras, 3), Y: labels (identificador do indivíduo)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Cores e marcadores para diferenciar indivíduos
n_classes = len(np.unique(Y))
colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
markers = ['o', '^', 's', 'p', '*', 'X', 'D', 'v', '<', '>', 'h', '+', 'x', '|', '_']

for i, c in enumerate(np.unique(Y)):
    mask = (Y == c)
    ax.scatter(X_reduzido[mask, 0], X_reduzido[mask, 1], X_reduzido[mask, 2],
               color=colors[i % len(colors)],
               marker=markers[i % len(markers)],
               label=f'Indivíduo {c}',
               s=60, alpha=0.8, edgecolors='k')

ax.set_xlabel('Componente Principal 1', fontsize=12)
ax.set_ylabel('Componente Principal 2', fontsize=12)
ax.set_zlabel('Componente Principal 3', fontsize=12)
ax.set_title('Projeção 3D dos rostos nas 3 componentes principais (98% variância)', fontsize=14)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
ax.grid(True)

# Permite rotação interativa do gráfico
ax.view_init(elev=30, azim=120)

plt.tight_layout()
plt.show()
