from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
from scipy.spatial.distance import cosine


class EvaluationMetrics:
    def __init__(self) -> None:
        pass

    # Silhouette Skoru
    def silhoutte_metric(self, X, labels):
        return silhouette_score(X, labels)

    # Davies-Bouldin İndeksi
    def davies_bouldin_metric(self, X, labels):
        return davies_bouldin_score(X, labels)

    # Calinski-Harabasz Skoru
    def calinski_harabasz_metric(self, X, labels):
        return calinski_harabasz_score(X, labels)

    def random_vectors_cosine_similarity(self, vectors: list, k: int):
        n = len(vectors)
        similarity_sum = 0

        number_of_vectors = int(k * (n / 100))
        if number_of_vectors <= 1:
            return 1.0

        if number_of_vectors >= n:
            number_of_vectors = n

        # k adet rastgele indeks seç
        random_indices = np.random.choice(n, number_of_vectors, replace=False)

        # Seçilen rastgele vektörlerin birbirlerine olan benzerliklerini hesapla
        for i in random_indices:
            for j in random_indices:
                if i != j:  # Aynı vektörlerin benzerliği 1 olacağı için hesaplamaya gerek yok
                    similarity_sum += 1 - cosine(vectors[i], vectors[j])

        # Ortalama benzerliği hesapla
        return similarity_sum / (number_of_vectors * (number_of_vectors - 1))
