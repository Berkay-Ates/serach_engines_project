from PyQt5.QtWidgets import *
from matplotlib import pyplot as plt
import pandas as pd
from modules.manager.manager_class import Manager
from view_py_files.multiple_cmp_result_page import Ui_MainWindow as ComparisionResultPage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from modules.model.embed_param_model import EmbedParameters
from modules.evaluation_metrics.evaluation_metrics import EvaluationMetrics

from utilities.show_messages import show_error_message, notification_message


class MultipleComparisionResultController(QMainWindow):

    def __init__(self, param1: EmbedParameters, param2: EmbedParameters) -> None:
        super().__init__()
        self.main_comp_page_view = ComparisionResultPage()
        self.main_comp_page_view.setupUi(self)
        self.param1 = param1
        self.param2 = param2

        print("data1:col", self.param1.data_column_name)

        self.manager1 = Manager(self.param1)
        self.manager2 = Manager(self.param2)

        self.embed_results1 = None
        self.result_df1 = None
        self.labels1 = None
        self.accuracy1 = None
        self.reduced_dim1 = None

        self.embed_results2 = None
        self.result_df2 = None
        self.labels2 = None
        self.accuracy2 = None
        self.reduced_dim2 = None

        self.set_params(self.param1, self.param2)
        self.main_comp_page_view.pushButton_get_result1.clicked.connect(
            lambda: self.set_parameters1(
                self.set_values(
                    self.manager1,
                    self.param1,
                ),
                self.main_comp_page_view.gridLayout_1,
            )
        )

        self.main_comp_page_view.pushButton_get_result2.clicked.connect(
            lambda: self.set_parameters2(
                self.set_values(
                    self.manager2,
                    self.param2,
                ),
                self.main_comp_page_view.gridLayout_2,
            )
        )

        self.main_comp_page_view.pushButton_label_res1.clicked.connect(self.set_labels_plot1)
        self.main_comp_page_view.pushButton_label_res2.clicked.connect(self.set_labels_plot2)
        self.main_comp_page_view.pushButton_estimated_res1.clicked.connect(self.set_estimated_labels_plot1)
        self.main_comp_page_view.pushButton_estimated_res2.clicked.connect(self.set_estimated_labels_plot2)

    def set_labels_plot1(self):
        self.show_plot(
            self.reduced_dim1,
            [0 if i == "ham" else 1 for i in self.result_df1["test_labels"]],
            self.main_comp_page_view.gridLayout_1,
        )

    def set_labels_plot2(self):
        self.show_plot(
            self.reduced_dim2,
            [0 if i == "ham" else 1 for i in self.result_df2["test_labels"]],
            self.main_comp_page_view.gridLayout_2,
        )

    def set_estimated_labels_plot1(self):
        self.show_plot(
            self.reduced_dim1,
            [0 if i == "ham" else 1 for i in self.result_df1["predicted_labels"]],
            self.main_comp_page_view.gridLayout_1,
        )

    def set_estimated_labels_plot2(self):
        self.show_plot(
            self.reduced_dim2,
            [0 if i == "ham" else 1 for i in self.result_df2["predicted_labels"]],
            self.main_comp_page_view.gridLayout_2,
        )

    def set_parameters1(self, params, widget):
        self.embed_results1, self.result_df1, self.labels1, self.accuracy1, self.reduced_dim1 = params
        self.show_plot(self.reduced_dim1, [0 if i == "ham" else 1 for i in self.result_df1["test_labels"]], widget)
        self.main_comp_page_view.label_res_report1.setText(self.labels1)
        self.main_comp_page_view.label_accuracy_1.setText("<b>Accuracy:</b> " + f"{self.accuracy1:.5f}")
        self.set_table(self.result_df1, self.main_comp_page_view.tableWidget_1)

    def set_parameters2(self, params, widget):
        self.embed_results2, self.result_df2, self.labels2, self.accuracy2, self.reduced_dim2 = params
        self.show_plot(self.reduced_dim2, [0 if i == "ham" else 1 for i in self.result_df2["test_labels"]], widget)
        self.main_comp_page_view.label_res_report2.setText(self.labels2)
        self.main_comp_page_view.label_accuracy_2.setText("<b>Accuracy:</b> " + f"{self.accuracy2:.5f}")
        self.set_table(self.result_df2, self.main_comp_page_view.tableWidget_2)

    def set_values(
        self,
        manager: Manager,
        params: EmbedParameters,
    ):
        embed_results = manager.embed_words()
        manager.parameters.data["embeddings"] = embed_results
        result_df, labels, accuracy = manager.cluster_words()
        reduced_dim = manager.dimension_reducer(list(result_df["test_embeddings"]))

        # self.show_plot(reduced_dim, [0 if i == "ham" else 1 for i in self.result_df["test_labels"]], widget)

        return embed_results, result_df, labels, accuracy, reduced_dim

    def show_plot(self, reduced_dim, labels, widget):
        # Burada Matplotlib grafiğini oluşturun
        fig, ax = plt.subplots(figsize=(6, 6))
        # Grafiği çizin (örneğin, scatter plot olarak)
        scatter = ax.scatter(reduced_dim[:, 0], reduced_dim[:, 1], c=labels)

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title("Clustered Words")

        handles, labels = scatter.legend_elements()
        legend1 = ax.legend(handles, labels, loc="lower right", title="Classes")
        ax.add_artist(legend1)
        self.clustered_graph = fig
        canvas = FigureCanvas(fig)
        widget.addChildWidget(canvas)

    def set_params(self, param1: EmbedParameters, param2: EmbedParameters):
        self.main_comp_page_view.tableWidget_1.setColumnCount(4)
        self.main_comp_page_view.tableWidget_1.setHorizontalHeaderLabels(
            ["test_texts", "test_labels", "test_labels", "predicted_labels"]
        )

        self.main_comp_page_view.tableWidget_2.setColumnCount(4)
        self.main_comp_page_view.tableWidget_2.setHorizontalHeaderLabels(
            ["test_texts", "test_labels", "test_labels", "predicted_labels"]
        )

        self.main_comp_page_view.label_cluster_count_1.setText("<b>Cluster Count:</b> " + str(param1.cluster_count))
        self.main_comp_page_view.label_cluster_count_tab2_1.setText(
            "<b>Cluster Count:</b> " + str(param1.cluster_count)
        )

        self.main_comp_page_view.label_cluster_method_1.setText(
            "<b>Clustering Method:</b> " + str(param1.clustering_method)
        )
        self.main_comp_page_view.label_cluster_method_tab2_1.setText(
            "<b>Clustering Method:</b> " + str(param1.clustering_method)
        )

        self.main_comp_page_view.label_dataset_name_1.setText(
            "<b>Dataset Name:</b> " + str(param1.data_path.split("/")[-1])
        )
        self.main_comp_page_view.label_dataset_name_tab2_1.setText(
            "<b>Dataset Name:</b> " + str(param1.data_path.split("/")[-1])
        )
        self.main_comp_page_view.label_dimension_reducer_1.setText(
            "<b>Dim. Reduc:</b> " + str(param1.dimension_reducer)
        )
        self.main_comp_page_view.label_dimension_reducer_tab2_1.setText(
            "<b>Dim. Reduc:</b> " + str(param1.dimension_reducer)
        )

        self.main_comp_page_view.label_cluster_count_2.setText("<b>Cluster Count:</b> " + str(param2.cluster_count))
        self.main_comp_page_view.label_cluster_count_tab2_2.setText(
            "<b>Cluster Count:</b> " + str(param2.cluster_count)
        )
        self.main_comp_page_view.label_cluster_method_2.setText(
            "<b>Clustering Method:</b> " + str(param2.clustering_method)
        )
        self.main_comp_page_view.label_cluster_method_tab2_2.setText(
            "<b>Clustering Method:</b> " + str(param2.clustering_method)
        )
        self.main_comp_page_view.label_dataset_name_2.setText(
            "<b>Dataset Name:</b> " + str(param2.data_path.split("/")[-1])
        )
        self.main_comp_page_view.label_dataset_name_tab2_2.setText(
            "<b>Dataset Name:</b> " + str(param2.data_path.split("/")[-1])
        )
        self.main_comp_page_view.label_dimension_reducer_2.setText(
            "<b>Dim. Reduc:</b> " + str(param2.dimension_reducer)
        )
        self.main_comp_page_view.label_dimension_reducer_tab2_2.setText(
            "<b>Dim. Reduc:</b> " + str(param2.dimension_reducer)
        )
        self.main_comp_page_view.label_embedding_1.setText("<b> " + str(param1.embedding_method) + " </b>")
        self.main_comp_page_view.label_embedding_tab2_1.setText("<b> " + str(param1.embedding_method) + " </b>")
        self.main_comp_page_view.label_embedding_2.setText("<b> " + str(param2.embedding_method) + " </b>")
        self.main_comp_page_view.label_embedding_tab2_2.setText("<b> " + str(param2.embedding_method) + " </b>")

    def set_table_widget(self, list_data, table: QTableWidget):
        table.clearContents()  # Clear existing content
        table.setRowCount(0)  # Reset row count
        for item in list_data:
            row_index = table.rowCount()
            table.insertRow(row_index)
            for column, data in enumerate(item):
                item = QTableWidgetItem(str(data))
                table.setItem(row_index, column, item)

    def set_table(self, dt: pd.DataFrame, widget):
        self.clear_columns(widget)
        widget.setColumnCount(len(dt.columns))
        widget.setHorizontalHeaderLabels(dt.columns)
        if dt.memory_usage(deep=True).sum() / (1024 * 1024) > 20:
            show_error_message("The data frame is larger than 20Mb so first 300 lines just showed in table")
            widget.setRowCount(300)
            for row_idx, row_data in dt.head(300).iterrows():
                for col_idx, cell_data in enumerate(row_data):
                    item = QTableWidgetItem(str(cell_data))
                    widget(row_idx, col_idx, item)
        else:
            widget.setRowCount(len(dt.index))
            for row_idx, row_data in dt.iterrows():
                for col_idx, cell_data in enumerate(row_data):
                    item = QTableWidgetItem(str(cell_data))
                    widget.setItem(int(row_idx), int(col_idx), item)

        for idx in range(len(dt.columns)):
            widget.setColumnWidth(idx, 300)

    def clear_columns(self, widget):
        widget.clearContents()
        widget.setRowCount(0)
        widget.setColumnCount(0)
