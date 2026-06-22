import pandas as pd
from sklearn.model_selection import train_test_split

# Carregamento do dataset utilizado nos experimentos
df = pd.read_csv("data_limpo.csv")

# Separação entre features (X) e a coluna alvo (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Definição das proporções de teste e seus respectivos rótulos
tamanhos_teste = [0.30, 0.25, 0.20, 0.15, 0.10]
nomes_distribuicao = ['70-30', '75-25', '80-20', '85-15', '90-10']

print("--- Distribuição de Classes por Particionamento ---\n")

for tamanho, nome in zip(tamanhos_teste, nomes_distribuicao):
    # O particionamento deve ser idêntico ao usado no modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tamanho, random_state=42)
    
    # Contagem das classes em cada subconjunto
    contagem_treino = y_train.value_counts()
    contagem_teste = y_test.value_counts()
    
    # Extração robusta para lidar com rótulos em texto ou numéricos (1 e 0)
    treino_malware = contagem_treino.get('Malware', contagem_treino.get(1, 0))
    treino_benigno = contagem_treino.get('Benign', contagem_treino.get(0, 0))
    
    teste_malware = contagem_teste.get('Malware', contagem_teste.get(1, 0))
    teste_benigno = contagem_teste.get('Benign', contagem_teste.get(0, 0))
    
    print(f"Distribuição {nome}:")
    print(f"  Treinamento ({len(y_train)} amostras totais):")
    print(f"    Malwares: {treino_malware} | Benignos: {treino_benigno}")
    print(f"  Teste ({len(y_test)} amostras totais):")
    print(f"    Malwares: {teste_malware} | Benignos: {teste_benigno}\n")
