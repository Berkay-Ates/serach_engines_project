import dataclasses
import pandas as pd
from typing import Protocol
from typing_extensions import override


class HyperParameters(Protocol):
    """
    This is a protocol class that defines the structure of a message.
    """

    def get_parameters(self): ...


class SVMHyperParameters(HyperParameters):
    @override
    def __init__(self, c=1.0, kernel="rbf"):
        # if c is str convert to float
        if isinstance(c, str):
            c = float(c)

        self._c = c
        self._kernel = kernel

    @override
    def get_parameters(self):
        # return as a dict
        return {"c": self._c, "kernel": self._kernel}


class RandomForestHyperParameters(HyperParameters):
    @override
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        # convert those values to int if they are str
        if isinstance(n_estimators, str):
            n_estimators = int(n_estimators)
        if isinstance(max_depth, str):
            max_depth = int(max_depth)
        if isinstance(min_samples_split, str):
            min_samples_split = int(min_samples_split)
        if isinstance(min_samples_leaf, str):
            min_samples_leaf = int(min_samples_leaf)

        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    @override
    def get_parameters(self):
        # return as a dict
        return {
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf,
        }


class NaiveBayesHyperParameters(HyperParameters):
    @override
    def __init__(self, alpha=1.0):
        # if alpha is str convert to float
        if isinstance(alpha, str):
            alpha = float(alpha)
        self._alpha = alpha

    @override
    def get_parameters(self):
        # return as a dict
        return {"alpha": self._alpha}


class EmbedParameters:
    def __init__(
        self,
        clustering_method="No Clustering",
        dimension_reducer="No Reducer",
        data_path="No Path",
        data_column_name="No Column",
        data: pd.DataFrame = "no Data Frame",
        embedding_method="No Embed",
        cluster_count: int = 2,
        hyper_parameters: HyperParameters = None,
    ) -> None:
        self._embedding_method = embedding_method
        self._clustering_method = clustering_method
        self._cluster_count = cluster_count
        self._dimension_reducer = dimension_reducer
        self._data_path = data_path
        self._data_column_name = data_column_name
        self._data = data
        self._hyper_parameters = hyper_parameters

    def __str__(self):
        return (
            f"Clustering Method: {self._clustering_method}\n"
            f"Dimension Reducer: {self._dimension_reducer}\n"
            f"Data Path: {self._data_path}\n"
            f"Data Column Name: {self._data_column_name}\n"
            f"Embedding Method: {self._embedding_method}\n"
            f"Cluster Count: {self._cluster_count}\n"
        )

    def __copy__(self):
        # MyClass sınıfının bir kopyasını oluştururken içindeki DataFrame'in kopyasını da oluşturuyoruz
        return EmbedParameters(
            clustering_method=self._clustering_method,
            dimension_reducer=self._dimension_reducer,
            data_path=self._data_path,
            data_column_name=self._data_column_name,
            data=self._data.copy() if self._data is not None else None,
            embedding_method=self._embedding_method,
            cluster_count=self._cluster_count,
        )

    def is_inited(self):
        if (
            self._embedding_method != "No Embed"
            and self._clustering_method != "No Clustering"
            and self._dimension_reducer != "No Reducer"
            and self._data_path != "No Path"
            and self._data_column_name != "No Column"
        ):
            return True

        return False

    @property
    def data(self):
        return self._data

    @property
    def embedding_method(self):
        return self._embedding_method

    @property
    def clustering_method(self):
        return self._clustering_method

    @property
    def cluster_count(self):
        return self._cluster_count

    @property
    def dimension_reducer(self):
        return self._dimension_reducer

    @property
    def data_path(self):
        return self._data_path

    @property
    def data_column_name(self):
        return self._data_column_name

    @property
    def hyper_parameters(self):
        return self._hyper_parameters

    @data.setter
    def data(self, hyper_parameters: HyperParameters):
        self._hyper_parameters = hyper_parameters

    @data.setter
    def data(self, data: pd.DataFrame):
        self._data = data

    @embedding_method.setter
    def embedding_method(self, embedding):
        self._embedding_method = embedding

    @clustering_method.setter
    def clustering_method(self, clustering_method):
        self._clustering_method = clustering_method

    @cluster_count.setter
    def cluster_count(self, cluster_count):
        self._cluster_count = cluster_count

    @dimension_reducer.setter
    def dimension_reducer(self, dimension_reducer):
        self._dimension_reducer = dimension_reducer

    @data_path.setter
    def data_path(self, data_path):
        self._data_path = data_path

    @data_column_name.setter
    def data_column_name(self, data_column_name):
        self._data_column_name = data_column_name
