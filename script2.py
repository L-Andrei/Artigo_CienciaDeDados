import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carregue o seu dataset correto
df = pd.read_csv("data_limpo.csv")

# O iloc[:, :-1] seleciona todas as linhas e todas as colunas, EXCETO a última (features)
X = df.iloc[:, :-1]

# O iloc[:, -1] seleciona todas as linhas, mas APENAS a última coluna (alvo)
y = df.iloc[:, -1]

# Ajustado para usar a string exata que o seu dataset possui para malwares
rotulo_malware = 'Malware' 

# Lista de proporções para a base de teste (30%, 25%, 20%, 15%, 10%)
tamanhos_teste = [0.30, 0.25, 0.20, 0.15, 0.10]

# Lista vazia para ir guardando os resultados de cada treinamento
resultados_finais = []

for test_size in tamanhos_teste:
    # Calculando as porcentagens para exibição
    treino_pct = int((1 - test_size) * 100)
    teste_pct = int(test_size * 100)
    distribuicao = f"{treino_pct}-{teste_pct}"
    
    print(f"Treinando modelo com distribuição {distribuicao}...")
    
    # Dividindo os dados conforme a proporção atual do loop
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Inicializando e treinando o modelo Random Forest
    modelo = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)
    
    # Gerando as previsões
    previsoes = modelo.predict(X_test)
    
    # Calculando as métricas solicitadas
    acuracia = accuracy_score(y_test, previsoes)
    precisao_malware = precision_score(y_test, previsoes, pos_label=rotulo_malware)
    recall_malware = recall_score(y_test, previsoes, pos_label=rotulo_malware)
    f1_malware = f1_score(y_test, previsoes, pos_label=rotulo_malware)
    
    # Salvando os resultados deste ciclo no dicionário
    resultados_finais.append({
        'Distribuicao': distribuicao,
        'Acuracia_Geral': round(acuracia, 4),
        'Precision_Malware': round(precisao_malware, 4),
        'Recall_Malware': round(recall_malware, 4),
        'F1_Score_Malware': round(f1_malware, 4)
    })

# Convertendo a lista de resultados em um DataFrame para melhor visualização
df_resultados = pd.DataFrame(resultados_finais)

print("\n--- Resumo de Todos os Treinamentos ---")
print(df_resultados.to_string(index=False))

# Descomente a linha abaixo se quiser salvar a tabela de resultados em um arquivo CSV na sua máquina
# df_resultados.to_csv("resultados_comparativos_distribuicoes.csv", index=False)