import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class Clustering_Methods:
    def __init__(self, cluster_count):
        self.cluster_count = cluster_count

    def random_forest_clustering(self, data: pd.DataFrame):
        # Extract embeddings, labels, and text
        embeddings = np.array(data["embeddings"].tolist())
        labels = data["label"].values  # Convert to array for consistent indexing
        texts = data["text"].values  # Convert to array for consistent indexing

        # Split data
        X_train, X_test, train_labels, test_labels, train_texts, test_texts = train_test_split(
            embeddings, labels, texts, test_size=0.2, random_state=42
        )

        # Convert to DataFrame to align indices properly
        test_texts = pd.DataFrame(test_texts, columns=["test_texts"]).reset_index(drop=True)
        test_labels = pd.DataFrame(test_labels, columns=["test_labels"]).reset_index(drop=True)

        # Train the Random Forest model
        random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        random_forest.fit(X_train, train_labels)
        predicted_labels = random_forest.predict(X_test)

        # Convert predicted labels to DataFrame
        predicted_labels = pd.DataFrame(predicted_labels, columns=["predicted_labels"])

        # Combine into result DataFrame
        result_df = pd.concat([test_texts, test_labels, predicted_labels], axis=1)
        result_df["test_embeddings"] = list(X_test)

        # Display result
        print(result_df.head(3))

        # Calculate accuracy and classification report
        accuracy = accuracy_score(test_labels["test_labels"], predicted_labels["predicted_labels"])
        return (
            result_df,
            classification_report(test_labels["test_labels"], predicted_labels["predicted_labels"]),
            accuracy,
        )

    def svm_clustering(self, data: pd.DataFrame):
        # Extract embeddings, labels, and text
        embeddings = np.array(data["embeddings"].tolist())
        labels = data["label"].values  # Convert to array for consistent indexing
        texts = data["text"].values  # Convert to array for consistent indexing

        # Split data
        X_train, X_test, train_labels, test_labels, train_texts, test_texts = train_test_split(
            embeddings, labels, texts, test_size=0.2, random_state=42
        )

        # Convert to DataFrame to align indices properly
        test_texts = pd.DataFrame(test_texts, columns=["test_texts"]).reset_index(drop=True)
        test_labels = pd.DataFrame(test_labels, columns=["test_labels"]).reset_index(drop=True)

        svm = SVC(kernel="linear")  # Lineer kernel ile SVM modelini oluştur
        svm.fit(X_train, train_labels)
        predicted_labels = svm.predict(X_test)

        # Convert predicted labels to DataFrame
        predicted_labels = pd.DataFrame(predicted_labels, columns=["predicted_labels"])

        # Combine into result DataFrame
        result_df = pd.concat([test_texts, test_labels, predicted_labels], axis=1)
        result_df["test_embeddings"] = list(X_test)

        # Display result
        print(result_df.head(3))

        accuracy = accuracy_score(test_labels["test_labels"], predicted_labels["predicted_labels"])
        return (
            result_df,
            classification_report(test_labels["test_labels"], predicted_labels["predicted_labels"]),
            accuracy,
        )

    def naive_bayes_clustering(self, data: pd.DataFrame):
        # Extract embeddings, labels, and text
        embeddings = np.array(data["embeddings"].tolist())
        labels = data["label"].values
        texts = data["text"].values

        # Split data
        X_train, X_test, train_labels, test_labels, train_texts, test_texts = train_test_split(
            embeddings, labels, texts, test_size=0.2, random_state=42
        )

        # Scale data to non-negative range
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to DataFrame to align indices properly
        test_texts = pd.DataFrame(test_texts, columns=["test_texts"]).reset_index(drop=True)
        test_labels = pd.DataFrame(test_labels, columns=["test_labels"]).reset_index(drop=True)

        # Train MultinomialNB
        naive_bayes = MultinomialNB()
        naive_bayes.fit(X_train, train_labels)
        predicted_labels = naive_bayes.predict(X_test)

        # Convert predicted labels to DataFrame
        predicted_labels = pd.DataFrame(predicted_labels, columns=["predicted_labels"])

        # Combine into result DataFrame
        result_df = pd.concat([test_texts, test_labels, predicted_labels], axis=1)
        result_df["test_embeddings"] = list(X_test)

        # Display result
        print(result_df.head(3))

        accuracy = accuracy_score(test_labels["test_labels"], predicted_labels["predicted_labels"])
        return (
            result_df,
            classification_report(test_labels["test_labels"], predicted_labels["predicted_labels"]),
            accuracy,
        )

    def custom_model(self, data: pd.DataFrame):
        pass
