import pandas as pd

def analisar_dados_faltantes(caminho_arquivo):
    try:
        # Carrega o dataset
        # Nota: Se o seu csv usar um separador diferente de vírgula, adicione sep=';' por exemplo.
        df = pd.read_csv(caminho_arquivo)
        
        print(f"--- Análise de Dados Faltantes: {caminho_arquivo} ---")
        print(f"Total de linhas: {df.shape[0]}")
        print(f"Total de colunas: {df.shape[1]}\n")

        # Conta a quantidade de valores nulos (NaN) por coluna
        nulos_por_coluna = df.isnull().sum()
        
        # Filtra apenas as colunas que realmente têm dados faltando
        colunas_com_nulos = nulos_por_coluna[nulos_por_coluna > 0]
        
        if colunas_com_nulos.empty:
            print("✅ Excelente! Não há nenhum dado faltando no seu arquivo .csv.")
        else:
            print("⚠️ Foram encontrados dados faltando nas seguintes colunas:\n")
            
            # Cria um DataFrame para exibir a quantidade e a porcentagem de forma organizada
            analise = pd.DataFrame({
                'Valores Faltantes': colunas_com_nulos,
                'Porcentagem (%)': (colunas_com_nulos / len(df)) * 100
            })
            
            # Formata a porcentagem para duas casas decimais
            analise['Porcentagem (%)'] = analise['Porcentagem (%)'].round(2)
            
            # Exibe o resultado ordenado da coluna com mais nulos para a com menos
            print(analise.sort_values(by='Valores Faltantes', ascending=False))
            
            print(f"\nTotal geral de células vazias no dataset: {df.isnull().sum().sum()}")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado. Verifique o caminho.")
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo: {e}")

# --- Execução do Script ---
# Substitua 'seu_arquivo.csv' pelo nome ou caminho do seu arquivo
nome_do_arquivo = 'data.csv' 
analisar_dados_faltantes(nome_do_arquivo)
