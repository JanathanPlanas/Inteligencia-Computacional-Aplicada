import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.decomposition import PCA
import os

# ==============================================
# Fase 1 -- Carrega imagens disponíveis
# ==============================================
part1 = 'subject0'
part2 = 'subject'
part3 = ['.centerlight', '.glasses', '.happy', '.leftlight', '.noglasses',
         '.normal', '.rightlight', '.sad', '.sleepy', '.surprised', '.wink']

Nind = 15  # Quantidade de indivíduos (classes)
Nexp = len(part3)  # Quantidade de expressões

X = []  # Matriz que acumula imagens vetorizadas
Y = []  # Matriz que acumula o rótulo (identificador) do indivíduo

for i in range(1, Nind + 1):  # Índice para os indivíduos
    print(f'Individuo: {i}')
    for j in range(Nexp):  # Índice para expressões
        # Monta o nome do arquivo de imagem
        if i < 10:
            nome = f'{part1}{i}{part3[j]}'
        else:
            nome = f'{part2}{i}{part3[j]}'

        if not os.path.exists(nome):
            print(f'Aviso: arquivo {nome} não encontrado, pulando.')
            continue

        # Leitura e redimensionamento da imagem
        img = imread(nome, as_gray=True)
        img_resized = resize(img, (30, 30), anti_aliasing=True)

        # Converte para double precision
        img_double = img_resized.astype(np.float64)

        # Vetorização (empilhamento das colunas)
        a = img_double.flatten(order='F')  # 'F' para empilhar colunas como MATLAB

        # Rótulo = índice do indivíduo
        ROT = i

        X.append(a)
        Y.append(ROT)

# Converte para arrays numpy
X = np.array(X).T  # Cada coluna é uma imagem vetorizada
Y = np.array(Y)

# ==============================================
# Aplicação de PCA (PCACOV)
# ==============================================
# Calcula a covariância das imagens
cov_matrix = np.cov(X.T)
pca = PCA()
pca.fit(cov_matrix)

# Seleciona q primeiros componentes
q = 25
Vq = pca.components_[:q, :]
X_pca = Vq @ X

# Variância explicada acumulada
VEq = np.cumsum(pca.explained_variance_ratio_)
plt.figure()
plt.plot(VEq, 'r-', linewidth=3)
plt.xlabel('Autovalor')
plt.ylabel('Variância explicada acumulada')
plt.show()

# ==============================================
# Formata saída
# ==============================================
Z = np.vstack([X_pca, Y])
Z = Z.T  # Cada linha é um vetor de atributos + rótulo

# Salva arquivo
np.savetxt('recfaces.dat', Z)
