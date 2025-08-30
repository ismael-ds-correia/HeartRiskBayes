import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from scipy.stats import norm

input_path = 'data/raw/heart_2020_cleaned.csv'
output_dir = 'reports/feature_plots'
os.makedirs(output_dir, exist_ok=True)

# lê com csv.reader para respeitar aspas e vírgulas dentro de campos
with open(input_path, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    rows = list(reader)

if not rows:
    raise SystemExit(f"Arquivo vazio: {input_path}")

header = rows[0]
col_idx = {col: idx for idx, col in enumerate(header)}

# localizar coluna de status de forma robusta
possible_status_names = ['Heart Disease Status', 'HeartDisease', 'Heart_Disease', 'HeartDiseaseStatus', 'Heart Disease', 'HeartDisease ']
status_col = next((name for name in possible_status_names if name in col_idx), None)
if status_col is None:
    print("Cabeçalho do CSV encontrado:", header)
    raise KeyError("Coluna de status cardíaco não encontrada no CSV. Verifique o cabeçalho.")
status_idx = col_idx[status_col]

categorical_features = [
    'Smoking',
    'AlcoholDrinking',
    'Stroke',
    'DiffWalking',
    'Sex',
    'AgeCategory',
    'Race',
    'Diabetic',
    'PhysicalActivity',
    'GenHealth',
    'Asthma',
    'KidneyDisease',
    'SkinCancer'
]

processed = 0
skipped = 0

for feature in header:
    if feature == status_col:
        continue

    idx = col_idx[feature]
    group_0 = []
    group_1 = []

    for values in rows[1:]:
        # protege contra linhas malformadas
        if idx >= len(values) or status_idx >= len(values):
            continue
        val = values[idx].strip().lower()
        status = values[status_idx].strip().lower()
        if val in ['', 'nan']:
            continue
        if status in ['0', 'no']:
            group_0.append(val)
        elif status in ['1', 'yes']:
            group_1.append(val)

    # detecta rapidamente se não há dados suficientes
    if not group_0 and not group_1:
        print(f"Pulando {feature}: sem valores válidos em nenhum grupo.")
        skipped += 1
        continue

    plt.figure(figsize=(8, 6))
    try:
        if feature in categorical_features:
            categories = sorted(set(group_0 + group_1))
            props_0 = [group_0.count(cat) / len(group_0) if group_0 else 0 for cat in categories]
            props_1 = [group_1.count(cat) / len(group_1) if group_1 else 0 for cat in categories]
            x = np.arange(len(categories))
            width = 0.35
            plt.bar(x - width/2, props_0, width, label='Sem risco cardíaco', color='blue', alpha=0.7)
            plt.bar(x + width/2, props_1, width, label='Com risco cardíaco', color='red', alpha=0.7)
            plt.xticks(x, categories, rotation=45, ha='right')
            plt.ylabel('Proporção')
            plt.title(f'Proporção de {feature} por Grupo de Risco Cardíaco')
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.5)
        else:
            # tenta converter para float; se falhar, pula (provavelmente feature categórica não listada)
            try:
                g0 = [float(v) for v in group_0]
                g1 = [float(v) for v in group_1]
            except ValueError:
                print(f"Pulando {feature}: valores não numéricos e feature não marcada como categórica.")
                skipped += 1
                plt.close()
                continue

            if not g0 or not g1:
                print(f"Pulando {feature}: grupo vazio após conversão numérica.")
                skipped += 1
                plt.close()
                continue

            all_vals = np.array(g0 + g1)
            x_grid = np.linspace(all_vals.min(), all_vals.max(), 200)
            mean_0, std_0 = np.mean(g0), np.std(g0)
            mean_1, std_1 = np.mean(g1), np.std(g1)
            density_0 = norm.pdf(x_grid, mean_0, std_0 if std_0 > 0 else 1e-6)
            density_1 = norm.pdf(x_grid, mean_1, std_1 if std_1 > 0 else 1e-6)
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
        processed += 1
    except Exception as e:
        print(f"Erro ao processar {feature}: {e}")
        skipped += 1
        plt.close()

print(f"Gerados: {processed}, Pulados: {skipped}")