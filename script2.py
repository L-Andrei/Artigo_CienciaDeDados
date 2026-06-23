import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

df = pd.read_csv("data_limpo.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

rotulo_malware = 'Malware'

# Em vez de "chutar" a string da classe benigna (ex: 'Benigno'), detecta
# automaticamente qual é o outro rótulo presente nos dados. Isso evita
# erros de divisão por zero quando o rótulo real está escrito de forma
# diferente do esperado (ex: 'benigno', 'Benign', com espaço extra, etc).
classes_presentes = sorted(y.unique())
print(f"Rótulos encontrados na coluna de classe: {classes_presentes}")

outras_classes = [c for c in classes_presentes if c != rotulo_malware]
if len(outras_classes) != 1:
    raise ValueError(
        f"Esperava encontrar exatamente 2 classes (Malware + 1 outra), "
        f"mas encontrei: {classes_presentes}. Verifique se 'rotulo_malware' "
        f"está escrito exatamente como aparece no dataset."
    )
rotulo_benigno = outras_classes[0]
print(f"Classe benigna detectada automaticamente como: '{rotulo_benigno}'")

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

    # Listas temporárias para armazenar os resultados das 5 rodadas (TESTE)
    acuracias = []
    precisoes = []
    recalls = []
    f1s = []
    fn_rates = []   # falsos negativos / (falsos negativos + verdadeiros positivos)
    fp_rates = []   # falsos positivos / (falsos positivos + verdadeiros negativos)

    # Listas para acompanhar o desempenho no próprio TREINO (overfitting/underfitting)
    f1s_treino = []
    acuracias_treino = []

    for train_index, test_index in sss.split(X, y):
        # Separação dos dados para a rodada atual
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Treinamento
        modelo.fit(X_train, y_train)

        # --- Avaliação no conjunto de TESTE ---
        previsoes = modelo.predict(X_test)

        acuracias.append(accuracy_score(y_test, previsoes))
        precisoes.append(precision_score(y_test, previsoes, pos_label=rotulo_malware))
        recalls.append(recall_score(y_test, previsoes, pos_label=rotulo_malware))
        f1s.append(f1_score(y_test, previsoes, pos_label=rotulo_malware))

        # Matriz de confusão explícita para extrair FN e FP corretamente
        # Ordem fixa de labels: [Benigno, Malware]
        # tn, fp
        # fn, tp
        tn, fp, fn, tp = confusion_matrix(
            y_test, previsoes, labels=[rotulo_benigno, rotulo_malware]
        ).ravel()

        fn_rates.append(fn / (fn + tp))  # entre os malwares reais, quantos passaram batido
        fp_rates.append(fp / (fp + tn))  # entre os benignos reais, quantos foram bloqueados à toa

        # --- Avaliação no conjunto de TREINO (para detectar overfitting) ---
        previsoes_treino = modelo.predict(X_train)
        acuracias_treino.append(accuracy_score(y_train, previsoes_treino))
        f1s_treino.append(f1_score(y_train, previsoes_treino, pos_label=rotulo_malware))

    # Consolidação: média E desvio padrão das 5 rodadas
    resultados_finais.append({
        'Distribuicao': distribuicao,

        'Acuracia_Teste_Media': round(np.mean(acuracias), 4),
        'Acuracia_Teste_Std': round(np.std(acuracias), 4),

        'Precision_Media': round(np.mean(precisoes), 4),
        'Precision_Std': round(np.std(precisoes), 4),

        'Recall_Media': round(np.mean(recalls), 4),
        'Recall_Std': round(np.std(recalls), 4),

        'F1_Teste_Media': round(np.mean(f1s), 4),
        'F1_Teste_Std': round(np.std(f1s), 4),

        'FN_Rate_Media(%)': round(np.mean(fn_rates) * 100, 2),
        'FN_Rate_Std(%)': round(np.std(fn_rates) * 100, 2),

        'FP_Rate_Media(%)': round(np.mean(fp_rates) * 100, 2),
        'FP_Rate_Std(%)': round(np.std(fp_rates) * 100, 2),

        # Métricas de treino, para comparar com o teste e embasar a discussão
        # sobre overfitting (90-10) e underfitting (70-30)
        'Acuracia_Treino_Media': round(np.mean(acuracias_treino), 4),
        'F1_Treino_Media': round(np.mean(f1s_treino), 4),

        # Gap treino-teste: quanto maior, mais sinal de overfitting
        'Gap_F1_Treino_Teste': round(np.mean(f1s_treino) - np.mean(f1s), 4),
    })

df_resultados = pd.DataFrame(resultados_finais)

print("\n--- Resumo de Todos os Treinamentos (Validação Cruzada, média ± desvio padrão) ---")
print(df_resultados.to_string(index=False))

df_resultados.to_csv("resultados_comparativos_cv.csv", index=False)