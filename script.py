import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregue o seu dataset correto substituindo o nome do arquivo abaixo
df = pd.read_csv("data.csv")

# O iloc[:, :-1] seleciona todas as linhas e todas as colunas, EXCETO a última (features)
X = df.iloc[:, :-1]

# O iloc[:, -1] seleciona todas as linhas, mas APENAS a última coluna (alvo)
y = df.iloc[:, -1]

# Dividindo os dados com 80% para treino e 20% para teste (test_size=0.20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Inicializando o modelo Random Forest com processamento paralelo (n_jobs=-1)
modelo = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)

# Treinando o modelo apenas com a parcela de 80% dos dados
modelo.fit(X_train, y_train)

# Fazendo com que o modelo tente adivinhar a natureza dos 20% de dados separados
previsoes = modelo.predict(X_test)

# Conferindo os resultados comparando as previsões com o gabarito (y_test)
acuracia = accuracy_score(y_test, previsoes)
print(f"Acurácia do modelo: {acuracia * 100:.2f}%\n")

print("Relatório de Classificação:")
print(classification_report(y_test, previsoes))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, previsoes))