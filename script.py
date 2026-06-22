import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Carregue o seu dataset (usando o arquivo limpo como base)
df = pd.read_csv("data_limpo.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Divisão fixa para que a variação seja apenas pelo número de árvores
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Lista de valores de n_estimators para testar
# Você pode ajustar esses valores conforme necessário
n_estimators_list = [ 100, 200, 300, 400, 500]
f1_results = []

print("Iniciando testes de n_estimators...")

for n in n_estimators_list:
    print(f"Treinando com {n} árvores...")
    
    # Inicializa e treina o modelo
    modelo = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)
    
    # Previsão e cálculo do F1-Score
    previsoes = modelo.predict(X_test)
    score = f1_score(y_test, previsoes, pos_label='Malware')
    
    f1_results.append(score)

# Plotagem do gráfico
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, f1_results, marker='o', linestyle='-', color='b')
plt.title('Influência do n_estimators no F1-Score')
plt.xlabel('Número de Estimadores (Árvores)')
plt.ylabel('F1-Score (Malware)')
plt.grid(True)

# Adiciona os valores no gráfico para facilitar a leitura
for i, txt in enumerate(f1_results):
    plt.annotate(f"{txt:.4f}", (n_estimators_list[i], f1_results[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()
