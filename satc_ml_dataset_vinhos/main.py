# Abaixo, as bibliotecas utilizadas no projeto para que sejam exibidos os resultados conforme estabelecido na proposta do projeto:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore') # Ignora avisos aplicados automaticamente pelos filtros para uma execução mais limpa

# 1. Define o caminho para o dataset que será utilizado
CAMINHO_DATASET = 'data\\group_4_winequality.csv'

# 2. Carregamento dos dados conforme estabelecido nas configurações iniciais
print("Carregando os dados...")
try:
    # Ao carregar os dados, já é utilizado por padrão o ponto como separador decimal no arquivo .csv, e então, é armazenado na variável df_vinho abaixo:
    df_vinho = pd.read_csv(CAMINHO_DATASET)
    print("Dados carregados com sucesso!")
    print(f"Formato dos dados: {df_vinho.shape}")
    print("\nPrimeiras 5 linhas do dataset (original):")
    print(df_vinho.head())
    # Caso o data-set não seja encontrado no caminho especificado na variável CAMINHO_DATASET, o código irá executar o exception abaixo:
except FileNotFoundError:
    print(f"ERRO: Dataset não encontrado em '{CAMINHO_DATASET}'. Verifique o caminho.")
    print("Certificar-se de que o arquivo 'group_4_winequality.csv' está dentro da pasta 'data' no mesmo diretório do código.")
    exit()

# 3. Pré-processamento inicial (correção dos tipos e formatos das colunas conforme o tipo dos valores
print("\nPré-processamento inicial: Correção de tipos e formato")

# Colunas que deveriam ser numéricas mas podem ter valores string problemáticos
colunas_numericas_potencialmente_problema = ['chlorides', 'density', 'alcohol'] # Colunas adicionadas pois houveram problemas de tipos de valores (string, number, etc.) na chamada da função info()

for coluna in colunas_numericas_potencialmente_problema:
    if coluna in df_vinho.columns:
        # Tenta converter para numérico, caso aconteçam erros, os valores são convertidos para NaN
        # Antes de converter, limpamos os caracteres que poderiam ser utilizados como separadores de milhares ou algo do gênero
        # O erro '100.333.333.333.333' sugere que temos pontos extras na separação dos milhares
        # Portanto, vamos remover todos os pontos, exceto o último
        # Esse processo descrito acima, pode ou não ser necessário, tudo depende do padrão real dos dados
        df_vinho[coluna] = df_vinho[coluna].astype(str).apply(
            lambda x: x.replace('.', '', x.count('.') - 1) if x.count('.') > 1 else x
        )
        df_vinho[coluna] = pd.to_numeric(df_vinho[coluna], errors='coerce') # Esta faz com que os valores que não puderem ser substituídos, sejam substituídos por NaN

print("\nInformações gerais do dataset após correção de tipos:")
df_vinho.info()

print("\nVerificando valores ausentes após a conversão de tipos:")
print(df_vinho.isnull().sum())

# 4. Tratamento de valores nulos (resultantes da conversão 'coerce' ou já existentes)
# Optamos por preencher com a mediana para não introduzir ruído de outliers na média
print("\nTratando valores ausentes com a mediana da coluna...")
for coluna in df_vinho.columns:
    if df_vinho[coluna].isnull().any():
        if df_vinho[coluna].dtype != 'object': # Apenas para colunas numéricas
            mediana_coluna = df_vinho[coluna].median()
            df_vinho[coluna].fillna(mediana_coluna, inplace=True)
            print(f"  Preenchido NaN na coluna '{coluna}' com a mediana ({mediana_coluna:.2f}).")
        else: # Se não, para a coluna 'quality', se os dados forem convertidos para NaN por engano:
            # Vamos remover as linhas, pois não podem haver dados faltantes para o treinamento do modelo
            if coluna == 'quality':
                contagem_nan_quality = df_vinho['quality'].isnull().sum()
                if contagem_nan_quality > 0:
                    df_vinho.dropna(subset=['quality'], inplace=True)
                    print(f"  Removidas {contagem_nan_quality} linhas com NaN na coluna 'quality' (alvo).")
            else: # Para outras colunas 'object', preencher com a moda ou valor 'desconhecido' tipo string
                moda_coluna = df_vinho[coluna].mode()[0]
                df_vinho[coluna].fillna(moda_coluna, inplace=True)
                print(f"  Preenchido NaN na coluna '{coluna}' com a moda ('{moda_coluna}').")

print("\nVerificando valores ausentes após tratamento:")
print(df_vinho.isnull().sum())
print(f"Formato dos dados após tratamento de NaN: {df_vinho.shape}") # Esperamos que o resutlado seja zero ou próximo de zero

# Após o tratamento de NaN e a correção de tipos, vamos verificar novamente os resultados
print("\nEstatísticas descritivas após o processamento inicial:")
print(df_vinho.describe())

# Selecionando os outliers em nas colunas 'density' e 'chlorides'
# Pelas estatísticas descritivas, as colunas 'density' e 'chlorides' ainda apresentaram valores muito além do esperado
# Valores máximos de 3793 para density e 975 para chlorides, com medias muito menores. Portanto, isso sugere algum problema nesses valores

# Conforme a descrição do dataset original com os dados dos vinhos, os valores "normais" são:
# chlorides: aprox 0.012 - 0.61 e density: aproximadamente 0.99 - 1.03
limite_superior_chlorides = df_vinho['chlorides'].quantile(0.99) # 99% para não cortar todos os valores mais altos
limite_inferior_chlorides = df_vinho['chlorides'].quantile(0.01) # 1%
df_vinho['chlorides'] = np.clip(df_vinho['chlorides'], limite_inferior_chlorides, limite_superior_chlorides)

limite_superior_density = df_vinho['density'].quantile(0.99)
limite_inferior_density = df_vinho['density'].quantile(0.01)
# Para density, se houver valores na casa dos 900+ ou 9000+, eles devem ser divididos por 1000
# O erro `100.333.333.333.333` sugere que o valor real pode ser `0.100...` ou `100.333...`
# Vamos tentar uma correção para density, supondo que valores muito altos foram digitados com algum erro
df_vinho['density'] = df_vinho['density'].apply(lambda x: x / 1000 if x > 100 else x) # Esta linha aplica uma limitação para valores extremos. Se a densidade é > 100, divide por 1000
df_vinho['density'] = np.clip(df_vinho['density'], limite_inferior_density, limite_superior_density)

print("\nEstatísticas descritivas após tratamento de outliers extremos em chlorides e density:")
print(df_vinho[['chlorides', 'density']].describe())

# Parte 1 - Início da regressão
# O código acima e o restante do código abaixo é praticamente o mesmo. Porém, agora vamos utilizar o dataset limpo e com os tipos de valores corretos

# Análise exploratória
print("\nAnálise exploratória de dados (dados limpos)")

# Histogramas para cada variável
print("\nGerando histogramas das variáveis (dados limpos)...")
df_vinho.hist(bins=15, figsize=(18, 12))
plt.suptitle('Histogramas das variáveis físico-químicas e qualidade do vinho (dados limpos)', y=1.02)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()

# Ao aplicara a matriz de correlação, dessa vez deve funcionará.
print("\nGerando matriz de correlação (dados limpos)...")
plt.figure(figsize=(12, 10))
# Precisamos fazer com que todas as colunas sejam numéricas antes de calcular a correlação entre elas
# 'type' é uma coluna categórica, então deve ser excluída da correlação, pois só vamos utilizar dados do tipo númerico
colunas_numericas_para_corr = df_vinho.select_dtypes(include=np.number).columns
sns.heatmap(df_vinho[colunas_numericas_para_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de correlação das variáveis do vinho (dados limpos)')
plt.show()


# 6. Limpeza e pré-processamento
print("\nLimpeza e pré-processamento (continuação)")

# A coluna 'type' é categórica, portanto, precisa ser convertida para numérica
# Usaremos o método de conversão one-hot encoder para a coluna 'type', pois não faz sentido dizer que red é maior que white
df_vinho_processado = pd.get_dummies(df_vinho, columns=['type'], drop_first=True, prefix='type')
print("\nDataset após one-hot encoding para a coluna 'type':")
print(df_vinho_processado.head())

# Separação de features (X) e target (y)
# 'quality' é o alvo. 'type_white' é a nova feature.
caracteristicas = df_vinho_processado.drop('quality', axis=1)
qualidade_alvo = df_vinho_processado['quality']

# 7. Padronização (utilizar o StandartScaller da biblioteca sklearn)
print("\nPadronizando as features...")
escalador = StandardScaler()
# O StandarScaler faz com que a média dos dados seja 0 e o desvio padrão seja igual a 1. Isso garante que os dados fiquem em uma escala comum, simplificando o processo de reconhecimento pela LLM model e evita o Bias
caracteristicas_escaladas = escalador.fit_transform(caracteristicas)
caracteristicas_escaladas = pd.DataFrame(caracteristicas_escaladas, columns=caracteristicas.columns)

# Divisão em conjuntos de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(caracteristicas_escaladas, qualidade_alvo, test_size=0.2, random_state=42)
print(f"Dados divididos: Treino ({X_treino.shape[0]} amostras), Teste ({X_teste.shape[0]} amostras)")

# 8. Modelagem (regressão)
print("\nModelagem (regressão)")
modelos_regressao = {
    'Regressão Linear': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
}

resultados_regressao = {}

for nome_modelo, modelo in modelos_regressao.items():
    print(f"\nTreinando {nome_modelo}...")
    modelo.fit(X_treino, y_treino)
    y_predito = modelo.predict(X_teste)

    # 9. Avaliação da regressão
    rmse = np.sqrt(mean_squared_error(y_teste, y_predito))
    mae = mean_absolute_error(y_teste, y_predito)
    r2 = r2_score(y_teste, y_predito)

    resultados_regressao[nome_modelo] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}

    print(f"Resultados para {nome_modelo}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

# 10. Ajuste de hiperparâmetros (GridSearchCV exemplo para Random Forest Regressor)
print("\nAjuste de hiperparâmetros (regressão - GridSearchCV para Random Forest)")
parametros_grid_rf = {
    'n_estimators': [100, 200], # Reduzindo para 2 para agilizar o exemplo
    'max_features': [0.6, 0.8], # Reduzindo para 2 para agilizar o exemplo
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_regressor = RandomForestRegressor(random_state=42)
busca_em_grade_rf = GridSearchCV(estimator=rf_regressor, param_grid=parametros_grid_rf,
                              cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
print("Iniciando busca em grade para Random Forest Regressor...")
busca_em_grade_rf.fit(X_treino, y_treino)

melhor_rf_regressor = busca_em_grade_rf.best_estimator_
y_predito_melhor_rf = melhor_rf_regressor.predict(X_teste)

melhor_rmse_rf = np.sqrt(mean_squared_error(y_teste, y_predito_melhor_rf))
melhor_mae_rf = mean_absolute_error(y_teste, y_predito_melhor_rf)
melhor_r2_rf = r2_score(y_teste, y_predito_melhor_rf)

print("\nMelhores hiperparâmetros para Random Forest Regressor:")
print(busca_em_grade_rf.best_params_)
print(f"Melhores resultados com Random Forest (ajustado):")
print(f"  RMSE: {melhor_rmse_rf:.4f}")
print(f"  MAE: {melhor_mae_rf:.4f}")
print(f"  R²: {melhor_r2_rf:.4f}")

resultados_regressao['Random Forest Regressor (ajustado)'] = {'RMSE': melhor_rmse_rf, 'MAE': melhor_mae_rf, 'R²': melhor_r2_rf}

# Comparativo de todos os resultados obtidos através da regressão
print("\nComparativo final dos modelos de regressão")
for nome, metricas in resultados_regressao.items():
    print(f"\n{nome}:")
    for nome_metrica, valor in metricas.items():
        print(f"  {nome_metrica}: {valor:.4f}")

# 11. Discussão da regressão
print("\nDiscussão da regressão")
# Interpretar coeficientes para modelos lineares
if 'Regressão Linear' in modelos_regressao:
    modelo_linear = modelos_regressao['Regressão Linear']
    if hasattr(modelo_linear, 'coef_'):
        print("\nCoeficientes da Regressão Linear:")
        # Colunas usadas para treino são as de caracteristicas_escaladas
        coeficientes = pd.Series(modelo_linear.coef_, index=caracteristicas_escaladas.columns).sort_values(ascending=False)
        print(coeficientes)
        print("\nVariáveis com maior impacto (Regressão Linear - em valor absoluto):")
        print(coeficientes.abs().sort_values(ascending=False).head(5))

# Importância das Features para modelos baseados em árvores (Random Forest)
if 'Random Forest Regressor' in modelos_regressao and hasattr(melhor_rf_regressor, 'feature_importances_'):
    print("\nImportância das Features (Random Forest Regressor):")
    # Colunas usadas para treino são as de caracteristicas_escaladas
    importancias_features = pd.Series(melhor_rf_regressor.feature_importances_, index=caracteristicas_escaladas.columns).sort_values(ascending=False)
    print(importancias_features)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importancias_features.values, y=importancias_features.index, palette='viridis')
    plt.title('Importância das Features (Random Forest Regressor)')
    plt.xlabel('Importância Relativa')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

print("\nAnálise crítica dos resultados da regressão:")
print("- Compare os RMSE, MAE e R² dos modelos. Qual se saiu melhor e por quê?")
print("- Os modelos baseados em árvores (Random Forest, Gradient Boosting) geralmente têm melhor desempenho do que a Regressão Linear em dados não lineares ou com interações complexas.")
print("- A importância das features revela quais características físico-químicas são mais preditivas para a qualidade do vinho. Quais se destacaram?")
print("- A Regressão Linear nos dá insights sobre a direção do impacto (positivo/negativo) dos recursos na qualidade. Por exemplo, 'alcohol' geralmente tem um coeficiente positivo.")
print("- O ajuste de hiperparâmetros (GridSearchCV) geralmente melhora o desempenho do modelo, mas exige mais tempo de processamento. Houve melhora significativa?")

# Parte 2 - Classificação (faixas de qualidade)
print("\n\nParte 2 — Classificação (faixas de qualidade)")

# Criar nova variável alvo categórica
def classificar_qualidade(qualidade_numerica):
    if qualidade_numerica <= 4:
        return 'Baixa'
    elif qualidade_numerica <= 6:
        return 'Média'
    else:
        return 'Alta'

print("\nDiscretizando a variável 'quality' para criar categorias de qualidade...")
df_vinho['categoria_qualidade'] = df_vinho['quality'].apply(classificar_qualidade)
print("Contagem de vinhos por categoria de qualidade:")
print(df_vinho['categoria_qualidade'].value_counts())

# Verificar distribuição das classes (desbalanceamento)
plt.figure(figsize=(8, 6))
sns.countplot(x='categoria_qualidade', data=df_vinho, palette='viridis',
              order=df_vinho['categoria_qualidade'].value_counts().index)
plt.title('Distribuição das classes de qualidade do vinho')
plt.xlabel('Categoria de qualidade')
plt.ylabel('Número de vinhos')
plt.show()

# Discussão sobre o desbalanceamento:
print("\nAnálise de desbalanceamento de classes")
contagem_classes = df_vinho['categoria_qualidade'].value_counts()
print(contagem_classes)
tamanho_min_classe = contagem_classes.min()
tamanho_max_classe = contagem_classes.max()
if tamanho_max_classe / tamanho_min_classe > 2:
    print(f"ATENÇÃO: As classes estão desbalanceadas. A maior classe ({tamanho_max_classe}) é {tamanho_max_classe/tamanho_min_classe:.2f} vezes maior que a menor classe ({tamanho_min_classe}).")
    print("Considerar técnicas de balanceamento como Oversampling (SMOTE) ou Undersampling durante o pré-processamento de classificação.")
else:
    print("As classes parecem razoavelmente balanceadas ou o desbalanceamento não é severo o suficiente para exigir rebalanceamento imediato.")


# Preparação (classificação)
# Usar o df_vinho_processado que já tem a coluna 'type' codificada
caracteristicas_clf = df_vinho_processado.drop(['quality', 'categoria_qualidade'], axis=1, errors='ignore') # 'categoria_qualidade' ainda não está em df_vinho_processado
alvo_clf = df_vinho['categoria_qualidade']

# Codificação de rótulos (LabelEncoder para o target)
codificador_rotulos = LabelEncoder()
alvo_clf_codificado = codificador_rotulos.fit_transform(alvo_clf)
print(f"\nClasses codificadas: {list(codificador_rotulos.classes_)} -> {list(range(len(codificador_rotulos.classes_)))}")

# Padronização das features (reaplicar, pois as características podem ter mudado após o one-hot encoding)
escalador_clf = StandardScaler()
caracteristicas_clf_escaladas = escalador_clf.fit_transform(caracteristicas_clf)
caracteristicas_clf_escaladas = pd.DataFrame(caracteristicas_clf_escaladas, columns=caracteristicas_clf.columns)

# Divisão em conjuntos de treino e teste (estratificado para não perdermos a proporção das classes)
X_treino_clf, X_teste_clf, y_treino_clf, y_teste_clf = train_test_split(
    caracteristicas_clf_escaladas, alvo_clf_codificado, test_size=0.2, random_state=42, stratify=alvo_clf_codificado
)
print(f"Dados para classificação divididos: Treino ({X_treino_clf.shape[0]} amostras), Teste ({X_teste_clf.shape[0]} amostras)")
print("Distribuição das classes no treino (após split estratificado):")
valores_unicos, contagens = np.unique(y_treino_clf, return_counts=True)
for i, val in enumerate(valores_unicos):
    print(f"  Classe {codificador_rotulos.inverse_transform([val])[0]}: {contagens[i]} ({contagens[i]/len(y_treino_clf)*100:.2f}%)")

# Modelagem (classificação)
print("\nModelagem (classificação) ")
modelos_classificacao = {
    'Regressão logística': LogisticRegression(random_state=42, max_iter=1000),
    'Árvore de decisão': DecisionTreeClassifier(random_state=42),
    'Random Forest Classifier': RandomForestClassifier(random_state=42),
}

resultados_classificacao = {}

for nome_modelo, modelo in modelos_classificacao.items():
    print(f"\nTreinando {nome_modelo}...")
    modelo.fit(X_treino_clf, y_treino_clf)
    y_predito_clf = modelo.predict(X_teste_clf)

    # Avaliação (classificação)
    acuracia = accuracy_score(y_teste_clf, y_predito_clf)
    precisao = precision_score(y_teste_clf, y_predito_clf, average='weighted', zero_division=0)
    recall = recall_score(y_teste_clf, y_predito_clf, average='weighted', zero_division=0)
    f1 = f1_score(y_teste_clf, y_predito_clf, average='weighted', zero_division=0)
    matriz_confusao = confusion_matrix(y_teste_clf, y_predito_clf)
    relatorio_classificacao = classification_report(y_teste_clf, y_predito_clf, target_names=codificador_rotulos.classes_, zero_division=0)

    resultados_classificacao[nome_modelo] = {
        'Acurácia': acuracia,
        'Precisão (Ponderada)': precisao,
        'Recall (Ponderado)': recall,
        'F1-Score (Ponderado)': f1,
        'Matriz de Confusão': matriz_confusao,
        'Relatório de classificação': relatorio_classificacao
    }

    print(f"Resultados para {nome_modelo}:")
    print(f"  Acurácia: {acuracia:.4f}")
    print(f"  Precisão (Ponderada): {precisao:.4f}")
    print(f"  Recall (Ponderado): {recall:.4f}")
    print(f"  F1-Score (Ponderado): {f1:.4f}")
    print("\nRelatório de classificação:\n", relatorio_classificacao)
    print("Matriz de confusão:\n", matriz_confusao)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues',
                xticklabels=codificador_rotulos.classes_, yticklabels=codificador_rotulos.classes_)
    plt.title(f'Matriz de confusão para {nome_modelo}')
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.show()

# Ajuste de hiperparâmetros (GridSearchCV para Random Forest Classifier)
print("\nAjuste de hiperparâmetros (classificação - GridSearchCV para Random Forest)")
parametros_grid_rf_clf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [None, 'balanced']
}

rf_classificador = RandomForestClassifier(random_state=42)
busca_em_grade_rf_clf = GridSearchCV(estimator=rf_classificador, param_grid=parametros_grid_rf_clf,
                                  cv=3, n_jobs=-1, verbose=1, scoring='f1_weighted')
print("Iniciando busca em grade para Random Forest Classifier...")
busca_em_grade_rf_clf.fit(X_treino_clf, y_treino_clf)

melhor_rf_classificador = busca_em_grade_rf_clf.best_estimator_
y_predito_melhor_rf_clf = melhor_rf_classificador.predict(X_teste_clf)

melhor_acuracia_rf_clf = accuracy_score(y_teste_clf, y_predito_melhor_rf_clf)
melhor_f1_rf_clf = f1_score(y_teste_clf, y_predito_melhor_rf_clf, average='weighted', zero_division=0)
melhor_relatorio_clf_rf_clf = classification_report(y_teste_clf, y_predito_melhor_rf_clf, target_names=codificador_rotulos.classes_, zero_division=0)
melhor_matriz_confusao_rf_clf = confusion_matrix(y_teste_clf, y_predito_melhor_rf_clf)

resultados_classificacao['Random Forest Classifier (Ajustado)'] = {
    'Acurácia': melhor_acuracia_rf_clf,
    'F1-Score (Ponderado)': melhor_f1_rf_clf,
    'Relatório de Classificação': melhor_relatorio_clf_rf_clf,
    'Matriz de Confusão': melhor_matriz_confusao_rf_clf
}

print("\nMelhores hiperparâmetros para Random Forest Classifier:")
print(busca_em_grade_rf_clf.best_params_)
print(f"Melhores resultados com Random Forest Classifier (ajustado):")
print(f"  Acurácia: {melhor_acuracia_rf_clf:.4f}")
print(f"  F1-Score (Ponderado): {melhor_f1_rf_clf:.4f}")
print("\nRelatório de Classificação (Ajustado):\n", melhor_relatorio_clf_rf_clf)
print("Matriz de Confusão (Ajustado):\n", melhor_matriz_confusao_rf_clf)

plt.figure(figsize=(8, 6))
sns.heatmap(melhor_matriz_confusao_rf_clf, annot=True, fmt='d', cmap='Blues',
            xticklabels=codificador_rotulos.classes_, yticklabels=codificador_rotulos.classes_)
plt.title('Matriz de Confusão para Random Forest Classifier (Ajustado)')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.show()

# Discussão (classificação)
print("\nDiscussão (classificação)")
print("\nAnálise dos erros (matrizes de confusão):")
print("- Observe as matrizes de confusão. As células fora da diagonal principal indicam erros de classificação.")
print("- Quais classes foram mais frequentemente confundidas entre si? Por exemplo, vinhos 'Média' foram classificados como 'Baixa' ou 'Alta'? Isso pode indicar que as características físico-químicas não são distintivas o suficiente para diferenciar essas classes ou que o modelo precisa de mais dados/ajustes.")

print("\nComparativo de modelos de classificação:")
print("- Qual modelo se saiu melhor em termos de acurácia, F1-score, precisão e recall? Um F1-score alto é desejável, especialmente com classes desbalanceadas.")

if 'Random Forest Classifier' in modelos_classificacao and hasattr(melhor_rf_classificador, 'feature_importances_'):
    print("\nImportância das Features (Random Forest Classifier):")
    importancias_features_clf = pd.Series(melhor_rf_classificador.feature_importances_, index=caracteristicas_clf.columns).sort_values(ascending=False)
    print(importancias_features_clf)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importancias_features_clf.values, y=importancias_features_clf.index, palette='viridis')
    plt.title('Importância das Features (Random Forest Classifier)')
    plt.xlabel('Importância Relativa')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    print("\nAlguma feature foi decisiva para a classificação? Compare com a análise de regressão.")

print("\nConsiderações Finais do Projeto:")
print("- Reflita sobre as escolhas de faixas para a classificação. Como diferentes faixas (ranges) impactam o balanceamento e o desempenho do modelo?")
print("- Quais são as limitações dos modelos que você usou? Eles seriam adequados para uso em produção?")
print("- Sugestões para trabalhos futuros: Mais dados, engenharia de features mais elaborada (criação de novas features a partir das existentes), explorar outros modelos (e.g., redes neurais), validação cruzada mais robusta.")