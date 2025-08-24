# Inteligência Computacional Aplicada - Trabalho de Regressão

Este repositório contém a implementação e avaliação de diversos modelos de regressão para o problema de precificação de imóveis. [cite_start]O projeto foi desenvolvido para a disciplina de Inteligência Computacional Aplicada e tem como objetivo comparar a performance de diferentes algoritmos com os resultados de benchmark publicados no artigo de Yeh & Hsu (2018).

## 🎯 Objetivo

Implementar e avaliar os modelos de Regressão Linear Múltipla (Mínimos Quadrados), Perceptron, MLP (1 e 2 camadas ocultas), SVR e LSSVR para o "Real estate valuation data set". A análise inclui a geração de histogramas de resíduos, gráficos de dispersão e o cálculo de métricas de performance, comparando-as com as reportadas na literatura.

## 📊 Dataset

[cite_start]O conjunto de dados utilizado é o **Real estate valuation data set**, que contém 414 transações imobiliárias com 6 features preditoras e 1 variável alvo (preço por unidade de área)[cite: 30]. O dataset está incluído neste repositório.

## 🤖 Modelos Implementados

O script `main.py` treina, avalia e compara os seguintes modelos:

* Regressão Linear (Mínimos Quadrados)
* Perceptron (implementado com `SGDRegressor`)
* Rede MLP com 1 camada oculta
* Rede MLP com 2 camadas ocultas
* Support Vector Regression (SVR) com kernel RBF
* Least Squares Support Vector Regression (LSSVR)

## 🚀 Como Executar

Para executar a análise e replicar os resultados, siga os passos abaixo.

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/JanathanPlanas/Inteligencia-Computacional-Aplicada.git](https://github.com/JanathanPlanas/Inteligencia-Computacional-Aplicada.git)
    cd Inteligencia-Computacional-Aplicada
    ```

2.  **Crie um ambiente virtual (Recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute o script principal:**
    ```bash
    python main.py
    ```

Ao ser executado, o script irá processar os dados, treinar cada um dos modelos, exibir os gráficos de análise (histograma de resíduos e dispersão) e, ao final, imprimir uma tabela consolidada com as métricas de performance no console.

## 📈 Resultados

A execução do script gera uma análise comparativa completa. A tabela abaixo resume os resultados obtidos, que serviram de base para a comparação com o artigo de referência. O modelo **SVR (RBF)** demonstrou a melhor performance geral.

| Modelo | RMSE (Teste) | R² (Teste) | MAE (Teste) |
| :--- | :---: | :---: | :---: |
| Linear Regression | 8.17 | 0.58 | 5.92 |
| Perceptron | 8.15 | 0.58 | 5.91 |
| MLP (1 camada) | 8.90 | 0.50 | 6.97 |
| MLP (2 camadas) | 7.10 | 0.68 | 5.01 |
| **SVR (RBF)** | **6.71** | **0.72** | **4.57** |
| LSSVR (Linear) | 8.05 | 0.59 | 5.71 |

## 📄 Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 📚 Artigo de Referência

Yeh, I. C., & Hsu, T. K. (2018). Building real estate valuation models with comparative approach through case-based reasoning. [cite_start]*Applied Soft Computing*, 65, 260-271[cite: 2].