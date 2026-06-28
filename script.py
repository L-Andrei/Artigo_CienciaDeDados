import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, accuracy_score, precision_score, 
                             recall_score, confusion_matrix)

df = pd.read_csv("data_limpo.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

test_sizes = [0.30, 0.20]
n_estimators_list = list(range(100, 550, 50))

resultados = []

print("A iniciar os testes do Random Forest...")

for test_size in test_sizes:
    proporcao = "70-30" if test_size == 0.30 else "80-20"
    print(f"\n--- A avaliar a divisão {proporcao} ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    for n in n_estimators_list:
        modelo = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
        modelo.fit(X_train, y_train)
        
        previsoes = modelo.predict(X_test)
        
        f1 = round(f1_score(y_test, previsoes, pos_label='Malware'), 4)
        acc = round(accuracy_score(y_test, previsoes), 4)
        prec = round(precision_score(y_test, previsoes, pos_label='Malware'), 4)
        rec = round(recall_score(y_test, previsoes, pos_label='Malware'), 4)
        
        cm = confusion_matrix(y_test, previsoes)
        tn, fp, fn, tp = cm.ravel()
        
        resultados.append({
            'Divisao': proporcao,
            'N_Arvores': n,
            'Acuracia': acc,
            'Precisao': prec,
            'Recall': rec,
            'F1_Score': f1,
            'Falsos_Positivos': fp,
            'Falsos_Negativos': fn
        })
        
        print(f"Treino com {n} árvores concluído.")

df_resultados = pd.DataFrame(resultados)
print("\nResultados Finais:")
print(df_resultados.to_string(index=False))

df_resultados.to_csv("resultados_experimento.csv", index=False)
print("\nResultados guardados no ficheiro 'resultados_experimento.csv'.")