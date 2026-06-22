import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

# Carregamento da base de dados
df = pd.read_csv("data_limpo.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Cálculo da frequência média de cada característica por classe
medias_malware = X[y == 'Malware'].mean()
medias_benigno = X[y == 'Benign'].mean() # Ajuste 'Benign' se o seu rótulo for diferente (ex: 0)

tamanhos_teste = [0.30, 0.25, 0.20, 0.15, 0.10]
nomes_distribuicao = ['70-30', '75-25', '80-20', '85-15', '90-10']

print("--- Extração de Importância das Características (Top 5 por Classe) ---")

for tamanho, nome in zip(tamanhos_teste, nomes_distribuicao):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=tamanho, random_state=42)
    modelo = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    
    pesos_rodadas = []
    
    # Extrai os pesos em cada uma das 5 dobras da validação cruzada
    for train_index, test_index in sss.split(X, y):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        modelo.fit(X_train, y_train)
        pesos_rodadas.append(modelo.feature_importances_)
    
    # Calcula a média dos pesos para a distribuição atual
    pesos_medios = np.mean(pesos_rodadas, axis=0)
    
    # Organiza os resultados em um DataFrame decrescente
    df_pesos = pd.DataFrame({'Caracteristica': X.columns, 'Peso': pesos_medios})
    df_pesos = df_pesos.sort_values(by='Peso', ascending=False)
    
    top_malware = []
    top_benigno = []
    
    # Separa as 5 principais para cada classe cruzando com a frequência
    for index, row in df_pesos.iterrows():
        feature = row['Caracteristica']
        peso = row['Peso']
        
        if medias_malware[feature] > medias_benigno[feature]:
            if len(top_malware) < 5:
                top_malware.append((feature, peso))
        else:
            if len(top_benigno) < 5:
                top_benigno.append((feature, peso))
        
        if len(top_malware) == 5 and len(top_benigno) == 5:
            break
            
    print(f"\nDistribuição {nome}:")
    print("  Indicadores de Malware:")
    for feat, peso in top_malware:
        print(f"    {feat}: {peso:.4f}")
    print("  Indicadores de Aplicativos Benignos:")
    for feat, peso in top_benigno:
        print(f"    {feat}: {peso:.4f}")
