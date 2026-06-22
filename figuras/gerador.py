import matplotlib.pyplot as plt

# Dados do seu experimento atualizados com a Validação Cruzada
distribuicoes = ['70-30', '75-25', '80-20', '85-15', '90-10']
f1_scores = [0.9613, 0.9601, 0.9627, 0.9609, 0.9611]

# Configuração da figura (tamanho padrão acadêmico)
plt.figure(figsize=(8, 6))

# Cores: Destacando o 80-20 (terceiro item) com uma cor diferente
cores = ['#AEC7E8', '#AEC7E8', '#1F77B4', '#AEC7E8', '#AEC7E8'] 

# Criando as barras
barras = plt.bar(distribuicoes, f1_scores, color=cores, width=0.6, edgecolor='black')

# Títulos e Rótulos
plt.title('Desempenho do F1-Score por Distribuição de Particionamento', fontsize=14, pad=15)
plt.xlabel('Distribuição (Treinamento - Teste)', fontsize=12)
plt.ylabel('F1-Score (Classe Malware)', fontsize=12)

# Ajuste fino do Eixo Y adaptado para os novos limites dos dados
plt.ylim(0.9590, 0.9635)

# Adicionando os rótulos de dados no topo de cada barra
for barra in barras:
    altura = barra.get_height()
    # Adiciona o texto alinhado ao centro, logo acima da barra
    plt.text(barra.get_x() + barra.get_width() / 2, altura + 0.0001, 
             f'{altura:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Adicionando linhas de grade horizontais sutis para facilitar a leitura
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Ajusta as margens para não cortar nenhum texto
plt.tight_layout()

# Salva a imagem em alta resolução (300 dpi) na mesma pasta do script
plt.savefig('grafico_f1_score_cv.png', dpi=300, bbox_inches='tight')

# Exibe o gráfico na tela
plt.show()