from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import numpy as np


class DimensionReducerMethods:
    def __init__(self) -> None:
        pass

    def pca_dimension_reduction(self, data: list) -> np.ndarray:
        pca = PCA(n_components=2)
        return pca.fit_transform(data)

    # def mds_dimension_reduction(self, data: list) -> np.ndarray:
    #     mds = MDS(n_components=2)
    #     return mds.fit_transform(data)

    def tsne_dimension_reduction(self, data: np.array) -> np.ndarray:
        tsne_reduction = TSNE(n_components=2)
        return tsne_reduction.fit_transform(data)
