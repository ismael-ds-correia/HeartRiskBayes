import csv
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from models.byes_classifier import *

# === Carrega CSV ===
def load_csv(filepath):
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Converte valores para int quando possível
    for row in rows:
        for key, val in row.items():
            if val.isdigit():
                row[key] = int(val)
    return rows

# === Divide treino/teste ===
def train_test_split(rows, test_size=0.2, seed=42):
    random.seed(seed)
    rows_copy = rows[:]
    random.shuffle(rows_copy)
    split_idx = int(len(rows_copy) * (1 - test_size))
    return rows_copy[:split_idx], rows_copy[split_idx:]

if __name__ == "__main__":
    data = load_csv("data/processed/heart_disease.csv")
    train_rows, test_rows = train_test_split(data, test_size=0.2)

    model = NaiveBayesClassifier()
    model.fit(train_rows, target_col="HeartDisease")

    # === Avaliação ===
    y_true = [row["HeartDisease"] for row in test_rows]
    y_pred = [model.predict(row) for row in test_rows]

    # 1. Probabilidades a priori
    print("=== Probabilidades a priori ===")
    for cls, prob in model.class_probs.items():
        print(f"P(HeartDisease={cls}) = {prob:.3f}")

    # 2. Alguns exemplos de posteriori
    print("\n=== Exemplos de probabilidades a posteriori ===")
    for i, row in enumerate(test_rows[:5]):
        probs = model.predict_proba(row)
        print(f"Paciente {i} | Verdadeiro={row['HeartDisease']} | Probs={probs}")

    # 3. Matriz de confusão
    conf_matrix = Counter((yt, yp) for yt, yp in zip(y_true, y_pred))
    print("\n=== Matriz de confusão ===")
    print("Verdadeiro=0, Predito=0:", conf_matrix[(0, 0)])
    print("Verdadeiro=0, Predito=1:", conf_matrix[(0, 1)])
    print("Verdadeiro=1, Predito=0:", conf_matrix[(1, 0)])
    print("Verdadeiro=1, Predito=1:", conf_matrix[(1, 1)])

    # 4. Precisão, Recall e F1-score
    tp = conf_matrix[(1, 1)]
    fp = conf_matrix[(0, 1)]
    fn = conf_matrix[(1, 0)]
    tn = conf_matrix[(0, 0)]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(test_rows)

    print("\n=== Métricas ===")
    print(f"Acurácia: {accuracy:.3f}")
    print(f"Precisão: {precision:.3f}")
    print(f"Recall:   {recall:.3f}")
    print(f"F1-score: {f1:.3f}")

    # === Matriz de confusão como gráfico ===
    conf_matrix = np.array([
        [conf_matrix[(0, 0)], conf_matrix[(0, 1)]],
        [conf_matrix[(1, 0)], conf_matrix[(1, 1)]]
    ])

    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix, cmap="Blues", norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(cax)

    # Anotações nos quadrados
    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, str(val), va="center", ha="center", color="black", fontsize=12, fontweight="bold")

    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    plt.title("Matriz de Confusão")
    plt.show()