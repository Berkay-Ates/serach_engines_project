from modules.embeddings.embedding_methods import EmbeddingMethods
from modules.clustering.clustering_methods import Clustering_Methods
from modules.dimension_reducer.dimension_reducer_methods import DimensionReducerMethods


class Constants:

    _embedding_lists = [
        method
        for method in dir(EmbeddingMethods)
        if callable(getattr(EmbeddingMethods, method)) and not method.startswith("__")
    ]
    _embedding_lists.insert(0, "")

    _dimension_reducer_lists = [
        method
        for method in dir(DimensionReducerMethods)
        if callable(getattr(DimensionReducerMethods, method)) and not method.startswith("__")
    ]
    _dimension_reducer_lists.insert(0, "")

    _clustering_lists = [
        method
        for method in dir(Clustering_Methods)
        if callable(getattr(Clustering_Methods, method)) and not method.startswith("__")
    ]
    _clustering_lists.insert(0, "")

    def __init__(self) -> None:
        pass

    @property
    def embedding_list(self):
        return self._embedding_lists

    @property
    def dimension_reducer_list(self):
        return self._dimension_reducer_lists

    @property
    def clustering_list(self):
        return self._clustering_lists
