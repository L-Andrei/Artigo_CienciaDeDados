import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("data_limpo.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

rotulo_malware = 'Malware' 

tamanhos_teste = [0.30, 0.25, 0.20, 0.15, 0.10]
resultados_finais = []

for test_size in tamanhos_teste:
    treino_pct = int(round((1 - test_size) * 100))
    teste_pct = int(round(test_size * 100))
    distribuicao = f"{treino_pct}-{teste_pct}"
    
    print(f"Executando validação cruzada para distribuição {distribuicao}...")
    
    # Configuração da Validação Cruzada Estratificada (5 iterações por distribuição)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=42)
    
    # O classificador mantém os mesmos hiperparâmetros já definidos metodologicamente
    modelo = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    
    # Listas temporárias para armazenar os resultados das 5 rodadas
    acuracias = []
    precisoes = []
    recalls = []
    f1s = []
    
    for train_index, test_index in sss.split(X, y):
        # Separação dos dados para a rodada atual
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Treinamento e previsão
        modelo.fit(X_train, y_train)
        previsoes = modelo.predict(X_test)
        
        # Cálculo e armazenamento das métricas desta rodada
        acuracias.append(accuracy_score(y_test, previsoes))
        precisoes.append(precision_score(y_test, previsoes, pos_label=rotulo_malware))
        recalls.append(recall_score(y_test, previsoes, pos_label=rotulo_malware))
        f1s.append(f1_score(y_test, previsoes, pos_label=rotulo_malware))
    
    # Consolidação dos resultados calculando a média das 5 rodadas
    resultados_finais.append({
        'Distribuicao': distribuicao,
        'Acuracia_Geral': round(np.mean(acuracias), 4),
        'Precision_Malware': round(np.mean(precisoes), 4),
        'Recall_Malware': round(np.mean(recalls), 4),
        'F1_Score_Malware': round(np.mean(f1s), 4)
    })

df_resultados = pd.DataFrame(resultados_finais)

print("\n--- Resumo de Todos os Treinamentos (Validação Cruzada) ---")
print(df_resultados.to_string(index=False))

# df_resultados.to_csv("resultados_comparativos_cv.csv", index=False)