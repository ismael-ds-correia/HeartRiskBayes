import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm

input_path = 'data/processed/heart_disease.csv'
output_dir = 'reports/feature_plots'
os.makedirs(output_dir, exist_ok=True)

with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

header = lines[0].strip().split(',')
col_idx = {col: idx for idx, col in enumerate(header)}

status_idx = col_idx['Heart Disease Status']

# Defina quais features são categóricas (adicione mais se necessário)
categorical_features = [
    'Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 'Diabetes',
    'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol',
    'Alcohol Consumption'
]

for feature in header:
    if feature == 'Heart Disease Status':
        continue

    idx = col_idx[feature]
    group_0 = []
    group_1 = []

    for line in lines[1:]:
        values = line.strip().split(',')
        val = values[idx].strip().lower()
        status = values[status_idx].strip().lower()
        if val in ['', 'nan']:
            continue
        if status in ['0', 'no']:
            group_0.append(val)
        elif status in ['1', 'yes']:
            group_1.append(val)

    plt.figure(figsize=(8, 6))
    if feature in categorical_features:
        # Proporção para cada categoria
        categories = sorted(set(group_0 + group_1))
        props_0 = [group_0.count(cat) / len(group_0) if group_0 else 0 for cat in categories]
        props_1 = [group_1.count(cat) / len(group_1) if group_1 else 0 for cat in categories]
        x = np.arange(len(categories))
        width = 0.35
        plt.bar(x - width/2, props_0, width, label='Sem risco cardíaco', color='blue', alpha=0.7)
        plt.bar(x + width/2, props_1, width, label='Com risco cardíaco', color='red', alpha=0.7)
        plt.xticks(x, categories)
        plt.ylabel('Proporção')
        plt.title(f'Proporção de {feature} por Grupo de Risco Cardíaco')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
    else:
        # Distribuição normal para numéricas
        try:
            group_0 = [float(v) for v in group_0]
            group_1 = [float(v) for v in group_1]
        except ValueError:
            continue  # pula se não conseguir converter para float

        if not group_0 or not group_1:
            continue

        all_vals = np.array(group_0 + group_1)
        x_grid = np.linspace(all_vals.min(), all_vals.max(), 200)
        mean_0, std_0 = np.mean(group_0), np.std(group_0)
        mean_1, std_1 = np.mean(group_1), np.std(group_1)
        density_0 = norm.pdf(x_grid, mean_0, std_0)
        density_1 = norm.pdf(x_grid, mean_1, std_1)
        plt.plot(x_grid, density_0, label='P(x|0) - Sem risco cardíaco', color='blue')
        plt.plot(x_grid, density_1, label='P(x|1) - Com risco cardíaco', color='red')
        plt.fill_between(x_grid, density_0, alpha=0.2, color='blue')
        plt.fill_between(x_grid, density_1, alpha=0.2, color='red')
        plt.xlabel(feature)
        plt.ylabel('Densidade Normal')
        plt.title(f'Verossimilhança de {feature} para Cada Grupo (Normal)')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{feature}_vs_status.png'))
    plt.close()