# PROJETO DE MACHINE LEARNING UniSATC: Predição e Classificação da Qualidade de Vinhos

Este repositório contém o trabalho final da disciplina de Machine Learning, focado na aplicação de técnicas de regressão e classificação para analisar e prever a qualidade de vinhos.

##  AUTORES

* Jorge Luiz Madeira Pires
* Cristhian Cardoso Bertan

## 1. Visão Geral do Projeto

O objetivo principal deste projeto é construir um pipeline completo de Machine Learning. Abordaremos duas frentes:

* **Regressão:** Para prever a nota de qualidade do vinho como um valor contínuo (0 a 10).
* **Classificação:** Para categorizar os vinhos em faixas de qualidade (e.g., Baixa, Média, Alta).

## 2. Dataset Utilizado

O dataset utilizado é o **Wine Quality Dataset**. Ele contém diversas variáveis físico-químicas de vinhos, juntamente com uma nota de qualidade sensorial (variando de 0 a 10), que é nossa variável alvo (`quality`).


### Colunas do Dataset:

As colunas incluem características como:
* `fixed acidity` (acidez fixa)
* `volatile acidity` (acidez volátil)
* `citric acid` (ácido cítrico)
* `residual sugar` (açúcar residual)
* `chlorides` (cloretos)
* `free sulfur dioxide` (dióxido de enxofre livre)
* `total sulfur dioxide` (dióxido de enxofre total)
* `density` (densidade)
* `pH`
* `sulphates` (sulfatos)
* `alcohol` (álcool)
* `type` (tipo de vinho: 'red' ou 'white')
* `quality` (nota de qualidade sensorial, 0-10) - **Variável Alvo**

## 3. Estrutura do Projeto

A estrutura do diretório do projeto é a seguinte:

```text
projeto_final_ml/
├── data/
│   └── group_4_winequality.csv  # Pasta dataset
├── main.py                      # O script  com todo a pipeline
├── README.md                    # Este arquivo
└── requirements.txt             # Lista de bibliotecas Python necessárias
```

## 4. Como Executar o Projeto

Para executar este projeto, siga os passos abaixo:

### 4.1. Pré-requisitos

Certifique-se de ter o Python 3.9+ instalado. Recomenda-se o uso de um ambiente virtual para gerenciar as dependências.

### 4.2. Configuração do Ambiente

1.  **Clone o Repositório, comando:**
   ```bash
    git clone https://github.com/JorgePires279/satc_ml_dataset_vinhos.git
   ```
2.  **Instale as dependencia, comando:**
      ```bash
     pip install -r requirements.txt
      ```
### 4.3. Execução do Script Principal

1.  **Navegue até o diretório raiz do projeto** no terminal (onde está o `main.py`).
2.  **Execute o script:**
    ```bash
    python main.py
    ```
3.  **Observe a Saída:** O script imprimirá logs no terminal sobre cada etapa do pipeline. Os gráficos (histogramas, scatter plots, matrizes de correlação e de confusão, importância das features) serão exibidos em janelas separadas, nescessario fechar a janela atual para avançar para a próxima.

## 5. Partes do Projeto

O projeto é dividido em duas partes principais, conforme o objetivo:

### 5.1. Parte 1 — Regressão (Predição de Nota de Qualidade)

**Objetivo:** Prever a nota de qualidade do vinho como um valor contínuo (0 a 10).

**Etapas Abordadas:**

* **Análise Exploratória de Dados (AED):** Geração de histogramas, scatter plots e matriz de correlação para entender a distribuição dos dados e o relacionamento entre as variáveis.
* **Limpeza e Pré-processamento:**
    * Tratamento de valores ausentes (com preenchimento via mediana).
    * Conversão de tipos de dados (forçando colunas como 'alcohol', 'chlorides', 'density' para numérico, tratando erros de formato).
    * Tratamento de outliers extremos em 'chlorides' e 'density' via `np.clip` e correção de escala.
    * Codificação da coluna categórica `type` (`red`/`white`) usando One-Hot Encoding.
    * Padronização das features (StandardScaler).
* **Modelagem:** Aplicação de múltiplos modelos de regressão, incluindo:
    * Regressão Linear
    * Random Forest Regressor
    * Gradient Boosting Regressor
* **Avaliação:** Utilização de métricas de regressão: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error) e R² (Coeficiente de Determinação).
* **Ajuste de Hiperparâmetros:** Aplicação de `GridSearchCV` para otimizar os hiperparâmetros do `RandomForestRegressor`.
* **Discussão:** Interpretação de coeficientes (para modelos lineares) e importância das features (para modelos baseados em árvores), analisando o impacto das variáveis na predição da qualidade.

### 5.2. Parte 2 — Classificação (Faixas de Qualidade)

**Objetivo:** Transformar o problema em classificação categórica, agrupando os vinhos por faixas de qualidade sensorial.

**Etapas Abordadas:**

* **Preparação:**
    * **Discretização da variável `quality`:** Criação de uma nova variável alvo categórica, agrupando as notas em faixas (e.g., 'Baixa', 'Média', 'Alta'). A faixa escolhida foi: `quality <= 4` -> 'Baixa', `4 < quality <= 6` -> 'Média', `quality > 6` -> 'Alta'.
    * Codificação de rótulos (`LabelEncoder`) para o target categórico.
    * Análise da distribuição das classes e discussão sobre desbalanceamento (com opção de `SMOTE` comentada no código).
    * Padronização das features.
* **Modelagem:** Teste de múltiplos classificadores, incluindo:
    * Regressão Logística
    * Árvore de Decisão
    * Random Forest Classifier
* **Avaliação:** Utilização de métricas de classificação relevantes, como Acurácia, Precisão, Recall e F1-Score (ponderados para lidar com possível desbalanceamento). Geração e análise de matrizes de confusão.
* **Ajuste de Hiperparâmetros:** Aplicação de `GridSearchCV` para otimizar os hiperparâmetros do `RandomForestClassifier`.
* **Discussão:** Análise detalhada dos erros de classificação (via matriz de confusão), comparação de desempenho entre os modelos, e identificação de features decisivas para a classificação da qualidade.

## 6. Resultados e Conclusões

**Análise dos Erros: Houve confusão entre as classes? Quais?**

Sim, houve confusão significativa entre as classes, principalmente devido ao desbalanceamento do dataset. A classe "Média" é a categoria majoritária, e os modelos frequentemente classificaram vinhos de qualidade "Baixa" e "Alta" incorretamente como "Média". Isso se deve ao viés do modelo em favor da classe mais abundante, impactando negativamente a precisão e o recall das classes minoritárias.

**Algum modelo se saiu melhor? Alguma feature foi decisiva?**

Sim, o Random Forest Classifier (Ajustado com GridSearchCV) demonstrou ser o melhor modelo de classificação. Ele superou os demais em acurácia (0.8353) e F1-Score ponderado (0.8329), mostrando que o ajuste de hiperparâmetros foi eficaz.

As features mais decisivas para a classificação da qualidade do vinho foram: álcool, acidez volátil, sulfatos, dióxido de enxofre total e ácido cítrico. Dentre elas, o álcool e a acidez volátil destacaram-se como os atributos físico-químicos mais importantes para distinguir as categorias de qualidade.
