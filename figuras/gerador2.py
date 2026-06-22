import matplotlib.pyplot as plt
import numpy as np

# Nomes simplificados para caberem melhor no gráfico (removendo prefixos redundantes se necessário, mas mantendo a essência)
features_malware = [
    'READ_PHONE_STATE', 
    'RECEIVE_BOOT_COMPLETED', 
    'INSTALL_SHORTCUT', 
    'ACCESS_FINE_LOCATION', 
    'ACCESS_COARSE_LOCATION'
]

features_benign = [
    'c2dm.permission.RECEIVE', 
    'WAKE_LOCK', 
    'URL;->openConnection', 
    'BILLING', 
    'getLastKnownLocation'
]

# Pesos extraídos do seu log para Malware
peso_malware = {
    '70-30': [0.2101, 0.0838, 0.0481, 0.0430, 0.0429],
    '75-25': [0.2088, 0.0785, 0.0477, 0.0441, 0.0438],
    '80-20': [0.2033, 0.0834, 0.0486, 0.0411, 0.0428],
    '85-15': [0.2122, 0.0823, 0.0488, 0.0412, 0.0421],
    '90-10': [0.2063, 0.0830, 0.0456, 0.0430, 0.0429]
}

# Pesos extraídos do seu log para Benignos
peso_benign = {
    '70-30': [0.0628, 0.0254, 0.0199, 0.0186, 0.0188],
    '75-25': [0.0629, 0.0256, 0.0214, 0.0192, 0.0183],
    '80-20': [0.0612, 0.0268, 0.0215, 0.0198, 0.0190],
    '85-15': [0.0635, 0.0257, 0.0203, 0.0202, 0.0200],
    '90-10': [0.0636, 0.0263, 0.0205, 0.0191, 0.0196]
}

distribuicoes = list(peso_malware.keys())
cores = ['#D6EAF8', '#85C1E9', '#3498DB', '#2874A6', '#1B4F72'] # Degradê de azul

# Configuração da figura com 2 subplots (um em cima do outro)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Configuração matemática para o agrupamento das barras
y_malware = np.arange(len(features_malware))
y_benign = np.arange(len(features_benign))
altura_barra = 0.15

# Plotando o gráfico de Malware (Ax1)
for i, dist in enumerate(distribuicoes):
    posicoes_y = y_malware + (i * altura_barra) - (altura_barra * 2)
    ax1.barh(posicoes_y, peso_malware[dist], altura_barra, label=dist, color=cores[i], edgecolor='black', linewidth=0.5)

ax1.set_yticks(y_malware)
ax1.set_yticklabels(features_malware, fontsize=11)
ax1.set_title('Top 5 Permissões - Indicadores de Malware', fontsize=14, pad=10, fontweight='bold')
ax1.set_xlabel('Peso da Característica (Feature Importance)', fontsize=12)
ax1.invert_yaxis() # Inverte para a mais importante ficar no topo
ax1.grid(axis='x', linestyle='--', alpha=0.7)
ax1.legend(title="Distribuição", loc='lower right')

# Plotando o gráfico de Benignos (Ax2)
for i, dist in enumerate(distribuicoes):
    posicoes_y = y_benign + (i * altura_barra) - (altura_barra * 2)
    ax2.barh(posicoes_y, peso_benign[dist], altura_barra, label=dist, color=cores[i], edgecolor='black', linewidth=0.5)

ax2.set_yticks(y_benign)
ax2.set_yticklabels(features_benign, fontsize=11)
ax2.set_title('Top 5 Permissões - Indicadores de Aplicativos Benignos', fontsize=14, pad=10, fontweight='bold')
ax2.set_xlabel('Peso da Característica (Feature Importance)', fontsize=12)
ax2.invert_yaxis()
ax2.grid(axis='x', linestyle='--', alpha=0.7)
ax2.legend(title="Distribuição", loc='lower right')

# Ajusta o espaçamento para evitar sobreposição de textos
plt.tight_layout(pad=3.0)

# Salva a imagem
plt.savefig('grafico_importancia_caracteristicas.png', dpi=300, bbox_inches='tight')

# Exibe na tela
plt.show()
