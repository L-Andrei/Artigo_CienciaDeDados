import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Carregamento dos dados
df = pd.read_csv("data_limpo.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Testando TODAS as distribuições para encontrar o ponto exato de falha
tamanhos_teste = [0.30, 0.25, 0.20, 0.15, 0.10]
nomes_distribuicao = ['70-30', '75-25', '80-20', '85-15', '90-10']

print("--- Análise de Falhas Práticas por Distribuição ---\n")

for tamanho, nome in zip(tamanhos_teste, nomes_distribuicao):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=tamanho, random_state=42)
    modelo = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    
    taxa_falsos_negativos = [] # Malwares não detectados
    taxa_falsos_positivos = [] # Benignos bloqueados injustamente
    
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        modelo.fit(X_train, y_train)
        previsoes = modelo.predict(X_test)
        
        # Gera a matriz de confusão
        cm = confusion_matrix(y_test, previsoes, labels=['Benign', 'Malware'])
        
        verdadeiros_negativos = cm[0, 0]
        falsos_positivos = cm[0, 1]
        falsos_negativos = cm[1, 0]
        verdadeiros_positivos = cm[1, 1]
        
        # Cálculo das taxas percentuais
        fnr = falsos_negativos / (falsos_negativos + verdadeiros_positivos)
        fpr = falsos_positivos / (falsos_positivos + verdadeiros_negativos)
        
        taxa_falsos_negativos.append(fnr)
        taxa_falsos_positivos.append(fpr)
        
    media_fnr = np.mean(taxa_falsos_negativos) * 100
    media_fpr = np.mean(taxa_falsos_positivos) * 100
    
    print(f"Distribuição {nome}:")
    print(f"  Falsos Negativos (Malwares que passaram): {media_fnr:.2f}%")
    print(f"  Falsos Positivos (Benignos bloqueados)  : {media_fpr:.2f}%\n")