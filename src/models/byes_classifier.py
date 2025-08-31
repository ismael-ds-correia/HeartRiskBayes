import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}

    def fit(self, rows, target_col):
        total = len(rows)
        class_counts = defaultdict(int)

        # Conta as classes
        for row in rows:
            class_counts[row[target_col]] += 1

        # Probabilidades a priori
        self.class_probs = {cls: count / total for cls, count in class_counts.items()}

        # Inicializa feature_probs
        self.feature_probs = {cls: defaultdict(lambda: defaultdict(int)) for cls in class_counts}

        # Conta atributos condicionados pela classe
        for row in rows:
            cls = row[target_col]
            for feature, value in row.items():
                if feature == target_col:
                    continue
                self.feature_probs[cls][feature][value] += 1

        # Normaliza para probabilidades
        for cls, features in self.feature_probs.items():
            for feature, values in features.items():
                total_values = sum(values.values())
                for val, count in values.items():
                    self.feature_probs[cls][feature][val] = count / total_values

    def predict_proba(self, row):
        probs = {}
        for cls, prior in self.class_probs.items():
            prob = math.log(prior)  # log evita underflow
            for feature, value in row.items():
                if feature not in self.feature_probs[cls]:
                    continue
                prob += math.log(self.feature_probs[cls][feature].get(value, 1e-6))
            probs[cls] = prob

        # Converte de log de volta para probabilidades
        max_log = max(probs.values())
        exp_probs = {cls: math.exp(p - max_log) for cls, p in probs.items()}
        total = sum(exp_probs.values())
        return {cls: p / total for cls, p in exp_probs.items()}

    def predict(self, row):
        probs = self.predict_proba(row)
        return max(probs, key=probs.get)
