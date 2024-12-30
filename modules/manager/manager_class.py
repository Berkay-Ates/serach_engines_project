import numpy as np
from modules.model.embed_param_model import EmbedParameters, HyperParameters
from modules.constants.constants import Constants
from modules.embeddings.embedding_methods import EmbeddingMethods
from modules.clustering.clustering_methods import Clustering_Methods
from modules.dimension_reducer.dimension_reducer_methods import DimensionReducerMethods
import pandas as pd


class Manager:
    def __init__(self, parameters: EmbedParameters) -> None:
        self.constants = Constants()
        self.parameters = parameters
        self.hyper_parameters: HyperParameters = parameters.hyper_parameters

    def embed_words(self):
        embedding = self.parameters.embedding_method
        print(embedding)
        return EmbeddingMethods().__run_function__(
            embedding, list(self.parameters.data[self.parameters.data_column_name])
        )

    def cluster_words(self):
        clustering = self.parameters.clustering_method
        print(clustering, "--------------------------------")
        if clustering == "random_forest_clustering":
            return Clustering_Methods(self.parameters.cluster_count).random_forest_clustering(
                self.parameters.data, self.hyper_parameters
            )

        elif clustering == "svm_clustering":
            return Clustering_Methods(self.parameters.cluster_count).svm_clustering(
                self.parameters.data, self.hyper_parameters
            )

        elif clustering == "naive_bayes_clustering":
            return Clustering_Methods(self.parameters.cluster_count).naive_bayes_clustering(
                self.parameters.data, self.hyper_parameters
            )

    def dimension_reducer(self, data: pd.DataFrame):
        dimension_reducer = self.parameters.dimension_reducer
        print(dimension_reducer, "-----------------------------------")
        if dimension_reducer == "pca_dimension_reduction":
            return DimensionReducerMethods().pca_dimension_reduction(list(data))

        elif dimension_reducer == "mds_dimension_reduction":
            return DimensionReducerMethods().mds_dimension_reduction(list(data))

        elif dimension_reducer == "tsne_dimension_reduction":
            return DimensionReducerMethods().tsne_dimension_reduction(np.array(data))
        else:
            print("did not match any dimension reduction methods so pca going to be used ")
            return DimensionReducerMethods().pca_dimension_reduction(list(data))
