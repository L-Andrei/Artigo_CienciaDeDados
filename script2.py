import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

df = pd.read_csv("data_limpo.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

n_estimators_list = list(range(100, 550, 50))
test_sizes = {0.30: "70-30", 0.20: "80-20"}

f1_scores_divisao = {"70-30": [], "80-20": []}

print("A iniciar o treino para geração do gráfico...")

for test_size, proporcao in test_sizes.items():
    print(f"A processar a divisão {proporcao}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    for n in n_estimators_list:
        modelo = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
        modelo.fit(X_train, y_train)
        
        previsoes = modelo.predict(X_test)
        f1 = f1_score(y_test, previsoes, pos_label='Malware')
        f1_scores_divisao[proporcao].append(f1)

print("Treino concluído. A gerar a figura...")

plt.figure(figsize=(10, 6))

plt.plot(n_estimators_list, f1_scores_divisao["70-30"], marker='o', linestyle='-', color='b', label='Divisão 70-30')
plt.plot(n_estimators_list, f1_scores_divisao["80-20"], marker='s', linestyle='--', color='r', label='Divisão 80-20')

plt.title('Comparativo de Evolução do F1-Score: 70-30 vs 80-20')
plt.xlabel('Quantidade de Sub-árvores (n_estimators)')
plt.ylabel('F1-Score')
plt.xticks(n_estimators_list)
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)

plt.savefig("comparativo_f1_score.png", dpi=300, bbox_inches='tight')

plt.show()

print("Figura guardada com sucesso como 'comparativo_f1_score.png'.")