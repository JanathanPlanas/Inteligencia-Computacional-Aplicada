import numpy as np
from PIL import Image
import os

# ==============================================
# Configurações
# ==============================================
image_folder = r'G:\Meu Drive\UFC\MESTRADO\Int Computacional Aplicada\Kit_projeto_FACES'
part1 = 'subject0'
part2 = 'subject'
part3 = ['.centerlight', '.glasses', '.happy', '.leftlight', '.noglasses',
         '.normal', '.rightlight', '.sad', '.sleepy', '.surprised', '.wink']

Nind = 15
Nexp = len(part3)
img_size = (30, 30)  # ajuste aqui o tamanho das imagens

X = []
Y = []

# ==============================================
# Pré-processamento
# ==============================================
for i in range(1, Nind + 1):
    print(f'Individuo: {i}')
    for j in range(Nexp):
        # Nome do arquivo com caminho completo
        if i < 10:
            nome = os.path.join(image_folder, f'{part1}{i}{part3[j]}')
        else:
            nome = os.path.join(image_folder, f'{part2}{i}{part3[j]}')

        if not os.path.exists(nome):
            print(f'Aviso: arquivo {nome} não encontrado, pulando.')
            continue

        # Leitura e redimensionamento
        img = Image.open(nome).convert('L')  # 'L' -> grayscale
        img_resized = img.resize(img_size)
        img_double = np.array(img_resized, dtype=np.float64)

        # Vetorização (empilhamento das colunas)
        a = img_double.flatten(order='F')

        # Rótulo = índice do indivíduo
        ROT = i

        X.append(a)
        Y.append(ROT)

# Converte para arrays numpy
X = np.array(X).T  # cada coluna = imagem vetorizada
Y = np.array(Y)

# Salva sem PCA
Z = np.vstack([X, Y])
Z = Z.T  # cada linha = vetor de atributos + rótulo
np.savetxt('recfaces_no_pca.dat', Z)

print("Pré-processamento concluído. Arquivo 'recfaces_no_pca.dat' salvo.")
