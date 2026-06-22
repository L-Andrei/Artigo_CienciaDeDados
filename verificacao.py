import pandas as pd

def limpar_dataset(caminho_arquivo, caminho_saida):
    try:
        # Carrega o dataset original
        df = pd.read_csv(caminho_arquivo)
        
        print(f"--- Processando o Dataset: {caminho_arquivo} ---")
        print(f"Tamanho original: {df.shape[0]} linhas e {df.shape[1]} colunas\n")

        # Verifica e separa as colunas preenchidas somente com 0 ou 1
        colunas_so_zeros = [col for col in df.columns if (df[col] == 0).all()]
        colunas_so_uns = [col for col in df.columns if (df[col] == 1).all()]
        colunas_para_remover = colunas_so_zeros + colunas_so_uns

        if colunas_para_remover:
            print(f"⚠️ Removendo {len(colunas_para_remover)} coluna(s) com valor constante (só 0 ou só 1).")
        else:
            print("✅ Nenhuma coluna com valor constante encontrada.")

        # Verifica e separa as linhas preenchidas somente com 0 ou 1 (avaliando até a penúltima coluna)
        df_features = df.iloc[:, :-1]
        linhas_so_zeros = df_features.index[(df_features == 0).all(axis=1)].tolist()
        linhas_so_uns = df_features.index[(df_features == 1).all(axis=1)].tolist()
        linhas_para_remover = linhas_so_zeros + linhas_so_uns

        if linhas_para_remover:
            print(f"⚠️ Removendo {len(linhas_para_remover)} linha(s) com valor constante (só 0 ou só 1 nas features).")
        else:
            print("✅ Nenhuma linha com valor constante nas features encontrada.")

        # Remove as colunas e linhas identificadas do dataframe
        df_limpo = df.drop(columns=colunas_para_remover)
        df_limpo = df_limpo.drop(index=linhas_para_remover)
        
        # Reseta o índice para manter a numeração organizada após as exclusões
        df_limpo = df_limpo.reset_index(drop=True)

        # Salva o novo dataframe limpo em um arquivo CSV
        df_limpo.to_csv(caminho_saida, index=False)
        
        print(f"\n--- Limpeza Concluída ---")
        print(f"Tamanho do novo dataset: {df_limpo.shape[0]} linhas e {df_limpo.shape[1]} colunas")
        print(f"Dataset salvo com sucesso em: '{caminho_saida}'\n")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado. Verifique o caminho.")
    except Exception as e:
        print(f"Ocorreu um erro ao processar o arquivo: {e}")

# --- Execução do Script ---
# Variáveis com os nomes do arquivo original e do novo arquivo gerado
nome_do_arquivo_entrada = 'data.csv' 
nome_do_arquivo_saida = 'data_limpo.csv'

limpar_dataset(nome_do_arquivo_entrada, nome_do_arquivo_saida)