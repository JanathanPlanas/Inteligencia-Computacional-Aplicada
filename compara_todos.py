import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier

def evaluate_classifier(model, data, n_repetitions, train_percent):
    """
    Função reutilizável para treinar e avaliar um classificador múltiplas vezes.

    Args:
        model: O objeto do classificador do scikit-learn.
        data (np.ndarray): O dataset completo (features + labels).
        n_repetitions (int): Número de vezes para repetir o experimento.
        train_percent (int): Porcentagem de dados para o conjunto de treino.

    Returns:
        np.ndarray: Um array com as taxas de acerto de cada repetição.
    """
    # Separa as features (todas as colunas exceto a última) e os rótulos (última coluna)
    X = data[:, :-1]
    y = data[:, -1]
    
    train_size = train_percent / 100.0
    accuracies = []

    for i in range(n_repetitions):
        # Divide os dados em treino e teste
        # stratify=y é importante para manter a proporção das classes nos conjuntos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, stratify=y
        )
        
        # Normaliza os dados para melhor performance de alguns modelos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Treina o modelo
        model.fit(X_train_scaled, y_train)
        
        # Avalia e armazena a acurácia (taxa de acerto)
        accuracy = model.score(X_test_scaled, y_test)
        accuracies.append(accuracy)
        
    return np.array(accuracies)


def main():
    """
    Função principal que orquestra o processo de avaliação.
    """
    # --- Configurações do Experimento ---
    try:
        # Carrega o dataset do arquivo .dat
        data = np.loadtxt('recfaces.dat')
    except FileNotFoundError:
        print("Erro: Arquivo 'recfaces.dat' não encontrado.")
        print("Por favor, certifique-se de que o arquivo está no mesmo diretório do script.")
        return

    NR = 50  # No. de repetições
    P_TRAIN = 80  # Porcentagem de treinamento

    # --- Definição dos Classificadores ---
    classifiers = {
        "Quadratico": QuadraticDiscriminantAnalysis(),
        "Variante 1": QuadraticDiscriminantAnalysis(reg_param=0.01),
        "Variante 2": LinearDiscriminantAnalysis(),
        "Variante 3": LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5),
        "Variante 4": GaussianNB(),
        "MQ": RidgeClassifier()
    }

    results = {}
    execution_times = {}

    # --- Execução e Avaliação ---
    for name, model in classifiers.items():
        print(f"Avaliando o classificador: {name}...")
        start_time = time.time()
        
        # Roda o experimento para o classificador atual
        accuracies = evaluate_classifier(model, data, NR, P_TRAIN)
        
        execution_times[name] = time.time() - start_time
        
        # Armazena as estatísticas e as taxas de acerto
        results[name] = {
            "stats": {
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies)
            },
            "accuracies": accuracies
        }

    # --- Exibição dos Resultados ---
    print("\n" + "="*40)
    print("Estatísticas dos Classificadores (Acurácia)")
    print("="*40)
    for name, result_data in results.items():
        stats = result_data['stats']
        print(f"{name:<12}: Média = {stats['mean_accuracy']:.4f}, Desvio Padrão = {stats['std_accuracy']:.4f}")

    print("\n" + "="*40)
    print("Tempos de Execução (segundos)")
    print("="*40)
    # Criando um DataFrame para uma exibição mais organizada
    times_df = pd.DataFrame(list(execution_times.items()), columns=['Classificador', 'Tempo (s)'])
    print(times_df.to_string(index=False))

    # --- Geração do Gráfico ---
    plt.figure(figsize=(12, 8))
    
    # Prepara os dados para o boxplot
    accuracies_data = [results[name]["accuracies"] for name in classifiers.keys()]
    labels = list(classifiers.keys())
    
    # Cria o boxplot
    sns.boxplot(data=accuracies_data, palette="viridis")
    
    # Configurações do gráfico
    plt.xticks(ticks=np.arange(len(labels)), labels=labels)
    plt.title('Comparação de Desempenho dos Classificadores no Conjunto Coluna', fontsize=16)
    plt.xlabel('Classificador', fontsize=12)
    plt.ylabel('Taxas de Acerto (Acurácia)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == '__main__':
    main()