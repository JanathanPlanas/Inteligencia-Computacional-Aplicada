import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.decomposition import PCA
import os

# ==============================================
# Pasta das imagens
# ==============================================
image_folder = r'G:\Meu Drive\UFC\MESTRADO\Int Computacional Aplicada\Kit_projeto_FACES'

part1 = 'subject0'
part2 = 'subject'
part3 = ['.centerlight', '.glasses', '.happy', '.leftlight', '.noglasses',
         '.normal', '.rightlight', '.sad', '.sleepy', '.surprised', '.wink']

Nind = 15
Nexp = len(part3)

X = []
Y = []

for i in range(1, Nind + 1):
    print(f'Indivíduo: {i}')
    for j in range(Nexp):
        prefix = part1 if i < 10 else part2
        filename = f'{prefix}{i}{part3[j]}'  # coloque a extensão correta
        caminho = os.path.join(image_folder, filename)

        if not os.path.exists(caminho):
            print(f'Aviso: arquivo {caminho} não encontrado, pulando.')
            continue

        # Leitura e conversão para grayscale se necessário
        img = imread(caminho)
        if img.ndim == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        img_resized = resize(img, (30, 30), anti_aliasing=True)

        # Vetorização
        a = img_resized.astype(np.float64).flatten(order='F')
        X.append(a)
        Y.append(i)

# Converte para arrays numpy
X = np.array(X)  # (n_amostras, n_features)
Y = np.array(Y)

print(f"Dimensão de X: {X.shape}")
if X.shape[0] == 0:
    raise RuntimeError("Nenhuma imagem foi carregada. Verifique o caminho e a extensão dos arquivos!")

# ==============================================
# PCA para descorrelação
# ==============================================
pca = PCA()  # mantém todos os componentes possíveis
X_pca = pca.fit_transform(X)

# Variância explicada acumulada
VEq = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8, 5))
plt.plot(VEq, 'r-', linewidth=2)
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Explicada Acumulada')
plt.title('Análise de PCA - Variância Explicada')
plt.grid()
plt.show()

# ==============================================
# Formata saída e salva
# ==============================================
Z = np.hstack([X_pca, Y.reshape(-1, 1)])
output_file = os.path.join('data', 'processed', 'recfaces_pca.dat')
os.makedirs(os.path.dirname(output_file), exist_ok=True)
np.savetxt(output_file, Z, fmt='%.6f')
print(f"Arquivo '{output_file}' salvo com sucesso.")
