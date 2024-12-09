from PyQt5.QtWidgets import *
from matplotlib import pyplot as plt
import pandas as pd
from view_py_files.result_page import Ui_MainWindow as ResultPage
from modules.model.embed_param_model import EmbedParameters
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from utilities.show_messages import show_error_message, notification_message
from modules.evaluation_metrics.evaluation_metrics import EvaluationMetrics

from modules.manager.manager_class import Manager


class ResultPageController(QMainWindow):

    def __init__(self, parameters: EmbedParameters) -> None:
        super().__init__()

        self.result_page = ResultPage()
        self.result_page.setupUi(self)
        self.parameters = parameters
        self.manager = Manager(self.parameters)
        self.embed_results = []
        self.labels = []
        self.percentage = 0

        self.bar_index = 0
        self.bar_graphs = []
        self.clustered_graph = None

        self.result_df: pd.DataFrame = None
        self.reduced_dim = None

        self.result_page.label_cluster_count.setText("<b>Cluster Count:</b> " + str(parameters.cluster_count))
        self.result_page.label_cluster_method.setText(parameters.clustering_method)
        self.result_page.label_dimension_reducer.setText("<b>Dim. Reduc:</b> " + parameters.dimension_reducer)
        self.result_page.label_dataset_name.setText("<b>Dataset Name:</b> " + parameters.data_path.split("/")[-1])

        self.result_page.label_embedding.setText(self.parameters.embedding_method)
        self.result_page.pushButton_cosine_similarity.clicked.connect(self.cosine_sim_calculate)
        self.result_page.pushButton_increase_percentage.clicked.connect(self.increase_percentage)
        self.result_page.pushButton_decrease_percentage.clicked.connect(self.decrease_percentage)

        self.result_page.pushButton_estimated_labels.clicked.connect(self.show_test_predicted_labels_graf)
        self.result_page.pushButton_labels.clicked.connect(self.shot_test_labels_graf)

        self.set_values()

    def increase_percentage(self):
        if self.percentage == 100:
            show_error_message("Percentage cannot be more than 100%")
            return

        self.percentage = self.percentage + 10
        self.result_page.label_cluster_vector_percentage.setText(str(self.percentage) + "%")

    def decrease_percentage(self):
        if self.percentage == 0:
            show_error_message("Percentage cannot be less than 0%")
            return

        self.percentage = self.percentage - 10
        self.result_page.label_cluster_vector_percentage.setText(str(self.percentage) + "%")

    def cosine_sim_calculate(self):
        labels = self.result_df["predicted_labels"]
        if self.embed_results is None or labels is None:
            show_error_message("There must be error with the vectors which are expected to use for cosine similarity")

        if self.percentage == 0:
            notification_message("Please increase percentage")
            return

        similarities = []
        labels_set = set(labels)
        for label in labels_set:
            current_label_indexes = [self.embed_results[i] for i in range(len(labels)) if labels[i] == label]
            similarity = EvaluationMetrics().random_vectors_cosine_similarity(current_label_indexes, self.percentage)
            similarities.append([label, similarity])
        self.set_table_widget(similarities, self.result_page.tableWidget_result)

    def set_values(self):
        self.result_page.tableWidget_result.setColumnCount(2)
        self.result_page.tableWidget_result.setHorizontalHeaderLabels(["Cluster", "Cos Sim"])
        self.result_page.tableWidget_result.setColumnWidth(0, 60)
        self.result_page.tableWidget_result.setColumnWidth(1, self.result_page.tableWidget_result.width() - 10)

        self.embed_results = self.manager.embed_words()
        self.manager.parameters.data["embeddings"] = self.embed_results
        self.result_df, self.labels, self.accuracy = self.manager.cluster_words()

        self.result_page.label_silhoutte_score.setText(self.labels)
        self.result_page.label_accuracy.setText(f"Accuracy: {self.accuracy:.5f}")

        self.reduced_dim = self.manager.dimension_reducer(list(self.result_df["test_embeddings"]))

        # DataFrame'leri birleştirme
        self.show_plot(self.reduced_dim, [0 if i == "ham" else 1 for i in self.result_df["test_labels"]])
        self.set_table(self.result_df)

    def shot_test_labels_graf(self):
        self.show_plot(self.reduced_dim, [0 if i == "ham" else 1 for i in self.result_df["test_labels"]])

    def show_test_predicted_labels_graf(self):
        self.show_plot(self.reduced_dim, [0 if i == "ham" else 1 for i in self.result_df["predicted_labels"]])

    def show_plot(self, reduced_dim, labels):
        fig, ax = plt.subplots(figsize=(9, 6))
        # Grafiği çizin (örneğin, scatter plot olarak)
        scatter = ax.scatter(
            reduced_dim[:, 0],
            reduced_dim[:, 1],
            c=labels,
        )

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title("Clustered Words")

        handles, labels = scatter.legend_elements()
        legend1 = ax.legend(handles, labels, loc="lower right", title="Classes")
        ax.add_artist(legend1)
        self.clustered_graph = fig
        canvas = FigureCanvas(fig)
        self.result_page.gridLayout_cluster_graph.addChildWidget(canvas)

    def set_table(self, dt: pd.DataFrame):
        self.clear_columns()
        self.result_page.tableWidget_clustered_data.setColumnCount(len(dt.columns))
        self.result_page.tableWidget_clustered_data.setHorizontalHeaderLabels(dt.columns)
        if dt.memory_usage(deep=True).sum() / (1024 * 1024) > 20:
            show_error_message("The data frame is larger than 20Mb so first 300 lines just showed in table")
            self.result_page.tableWidget_clustered_data.setRowCount(300)
            for row_idx, row_data in dt.head(300).iterrows():
                for col_idx, cell_data in enumerate(row_data):
                    item = QTableWidgetItem(str(cell_data))
                    self.result_page.tableWidget_clustered_data(row_idx, col_idx, item)
        else:
            self.result_page.tableWidget_clustered_data.setRowCount(len(dt.index))
            for row_idx, row_data in dt.iterrows():
                for col_idx, cell_data in enumerate(row_data):
                    item = QTableWidgetItem(str(cell_data))
                    self.result_page.tableWidget_clustered_data.setItem(int(row_idx), int(col_idx), item)

        for idx in range(len(dt.columns)):
            self.result_page.tableWidget_clustered_data.setColumnWidth(idx, 300)

    def clear_columns(self):
        self.result_page.tableWidget_clustered_data.clearContents()
        self.result_page.tableWidget_clustered_data.setRowCount(0)
        self.result_page.tableWidget_clustered_data.setColumnCount(0)

    def set_table_widget(self, list_data, table: QTableWidget):
        table.clearContents()  # Clear existing content
        table.setRowCount(0)  # Reset row count
        for item in list_data:
            row_index = table.rowCount()
            table.insertRow(row_index)
            for column, data in enumerate(item):
                item = QTableWidgetItem(str(data))
                table.setItem(row_index, column, item)
