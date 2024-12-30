from PyQt5.QtWidgets import *
import matplotlib
from view_py_files.main_page import Ui_MainWindow as MainPage
from .data_page_controller import DataPageController
from .result_page_controller import ResultPageController
from .multiple_cmp_select_page_cntrl import MultipleComparisionSelectController


import os
import pandas as pd
from utilities.show_messages import show_error_message, notification_message
from modules.model.embed_param_model import (
    EmbedParameters,
    SVMHyperParameters,
    NaiveBayesHyperParameters,
    RandomForestHyperParameters,
)
from modules.constants.constants import Constants


class MainPageController(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.main_page_view = MainPage()
        self.data_page_controller = None
        self.multiple_comp_select_page_contr = None
        self.main_page_view.setupUi(self)
        self.add_combobox_items()
        self.hide_hyper_parameters()

        self.result_pages = []

        self.data = None

        self.embedding = ""
        self.clustering = ""
        self.dimension_reducer = ""
        self.cluster_count = 2
        self.data_path = None
        self.data_column = None

        self.main_page_view.actionData_2.triggered.connect(self.menu_clicked)
        self.main_page_view.actionMultipleComparision_2.triggered.connect(self.multiple_select_clicked)
        self.main_page_view.listWidget_data_columns.itemClicked.connect(self.select_column)
        self.main_page_view.comboBox_dataset_select.currentTextChanged.connect(self.set_data)
        self.main_page_view.pushButton_cluster_increase.clicked.connect(self.cluster_increase)
        self.main_page_view.pushButton_cluster_decrease.clicked.connect(self.cluster_decrease)

        self.main_page_view.pushButton_show_results.clicked.connect(self.show_result)
        self.main_page_view.comboBox_embedding.currentIndexChanged.connect(self.set_embedding)
        self.main_page_view.comboBox_clustering.currentIndexChanged.connect(self.set_clustering)
        self.main_page_view.comboBox_dimension_reducer.currentIndexChanged.connect(self.set_dimension_reducer)

        self.main_page_view.pushButton_delete_data.clicked.connect(self.delete_dataset)
        self.main_page_view.pushButton_refresh_list.clicked.connect(self.add_data_files)

    def hide_hyper_parameters(self):
        self.main_page_view.groupBox_naive_bayes.setVisible(False)
        self.main_page_view.groupBox_random_forest.setVisible(False)
        self.main_page_view.groupBox_svm.setVisible(False)

    def show_hyper_parameters(self, clustering_method):
        self.hide_hyper_parameters()
        if clustering_method == "naive_bayes_clustering":
            self.main_page_view.groupBox_naive_bayes.setVisible(True)
        elif clustering_method == "random_forest_clustering":
            self.main_page_view.groupBox_random_forest.setVisible(True)
        elif clustering_method == "svm_clustering":
            self.main_page_view.groupBox_svm.setVisible(True)

    def get_hyper_parameters(self):
        if self.clustering == "naive_bayes_clustering":
            return NaiveBayesHyperParameters(alpha=self.main_page_view.lineEdit_naive_bayes_alpha.text())
        elif self.clustering == "random_forest_clustering":
            return RandomForestHyperParameters(
                n_estimators=self.main_page_view.lineEdit_rf_n_estim.text(),
                max_depth=self.main_page_view.lineEdit_rf_max_dept.text(),
                min_samples_split=self.main_page_view.lineEdit_rf_min_samp_split.text(),
                min_samples_leaf=self.main_page_view.lineEdit_rf_min_samp_leaf.text(),
            )
        elif self.clustering == "svm_clustering":
            return SVMHyperParameters(
                c=self.main_page_view.lineEdit_svm_c.text(),
                kernel=self.main_page_view.comboBox_svm_kernel.currentText(),
            )

        return None

    def delete_dataset(self):
        if self.data_path is not None:
            result = notification_message(f"The file {self.data_path} would be deleted")
            if result:
                os.remove(self.data_path)
                self.clear_columns()
                self.set_column_list([])

    def select_column(self):
        if self.main_page_view.listWidget_data_columns.count() > 0:
            self.data_column = self.main_page_view.listWidget_data_columns.selectedItems()[0].text()

    def set_dimension_reducer(self):
        self.dimension_reducer = self.main_page_view.comboBox_dimension_reducer.currentText()

    def set_clustering(self):
        self.clustering = self.main_page_view.comboBox_clustering.currentText()
        self.hide_hyper_parameters()
        self.show_hyper_parameters(self.clustering)

    def set_embedding(self):
        self.embedding = self.main_page_view.comboBox_embedding.currentText()

    def show_result(self):
        if (
            self.embedding != ""
            and self.clustering != ""
            and self.dimension_reducer != ""
            and self.cluster_count
            and self.data_path
            and self.data_column
            and self.data is not None
        ):
            matplotlib.pyplot.close()
            parameters = EmbedParameters(
                embedding_method=self.embedding,
                clustering_method=self.clustering,
                cluster_count=self.cluster_count,
                dimension_reducer=self.dimension_reducer,
                data_path=self.data_path,
                data_column_name=self.data_column,
                data=self.data,
                hyper_parameters=self.get_hyper_parameters(),
            )
            page = ResultPageController(parameters)
            self.result_pages.append(page)
            page.show()
        else:
            show_error_message(
                "You have to select all parameters \n Embedding, Clustering, Dimension Reducer, Cluster Count, Data"
            )

        self.cleanup_result_pages()  # Gösterilmeyen sayfaları temizle

    def cleanup_result_pages(self):
        visible_pages = [page for page in self.result_pages if page.isVisible()]
        self.result_pages = visible_pages

    def cluster_increase(self):
        current = self.main_page_view.lineEdit_cluster_count.text()
        current = int(current) + 1
        self.cluster_count = current
        self.main_page_view.lineEdit_cluster_count.setText(str(current))

    def cluster_decrease(self):
        current = self.main_page_view.lineEdit_cluster_count.text()
        if int(current) > 2:
            current = int(current) - 1
            self.cluster_count = current
            self.main_page_view.lineEdit_cluster_count.setText(str(current))
        else:
            show_error_message("cluster count should be 2 or more")

    def set_data(self):
        df = self.main_page_view.comboBox_dataset_select.currentText()

        if df == "":
            return
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        parent_directory = os.path.dirname(current_directory)
        data_directory = os.path.join(parent_directory, "Data")
        self.data_path = f"{data_directory}/{df}"
        self.data_column = None
        try:
            dt = pd.read_csv(self.data_path)
            self.data = dt
        except Exception as e:
            show_error_message(e)

        self.set_table(dt=self.data)
        self.set_column_list(self.data.columns)

    def add_combobox_items(self):
        constants = Constants()
        self.main_page_view.comboBox_clustering.clear()
        self.main_page_view.comboBox_embedding.clear()
        self.main_page_view.comboBox_dimension_reducer.clear()

        for item in constants._clustering_lists:
            self.main_page_view.comboBox_clustering.addItem(item)
        self.main_page_view.comboBox_clustering.setPlaceholderText("Select a Clustering")

        for item in constants.embedding_list:
            self.main_page_view.comboBox_embedding.addItem(item)
        self.main_page_view.comboBox_embedding.setPlaceholderText("Select a Embedding")

        for item in constants.dimension_reducer_list:
            self.main_page_view.comboBox_dimension_reducer.addItem(item)
        self.main_page_view.comboBox_dimension_reducer.setPlaceholderText("Select a Dimension Reducer")

        self.add_data_files()

    def add_data_files(self):
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        parent_directory = os.path.dirname(current_directory)
        data_directory = os.path.join(parent_directory, "Data")
        self.main_page_view.comboBox_dataset_select.clear()
        self.main_page_view.comboBox_dataset_select.addItem("")

        files = os.listdir(data_directory)
        for file in files:
            self.main_page_view.comboBox_dataset_select.addItem(file)

    def menu_clicked(self):
        if self.data_page_controller is None:
            self.data_page_controller = DataPageController()
            self.data_page_controller.show()
        if not self.data_page_controller.isVisible():
            self.data_page_controller = DataPageController()
            self.data_page_controller.show()

    def multiple_select_clicked(self):
        if self.multiple_comp_select_page_contr is None:
            self.multiple_comp_select_page_contr = MultipleComparisionSelectController()
            self.multiple_comp_select_page_contr.show()
        if not self.multiple_comp_select_page_contr.isVisible():
            self.multiple_comp_select_page_contr = MultipleComparisionSelectController()
            self.multiple_comp_select_page_contr.show()

    def set_table(self, dt: pd.DataFrame):
        self.clear_columns()
        self.main_page_view.tableWidget_data_preview.setColumnCount(len(dt.columns))
        self.main_page_view.tableWidget_data_preview.setHorizontalHeaderLabels(dt.columns)
        if dt.memory_usage(deep=True).sum() / (1024 * 1024) > 20:
            show_error_message("The data frame is larger than 20Mb so first 300 lines just showed in table")
            self.main_page_view.tableWidget_data_preview.setRowCount(300)
            for row_idx, row_data in dt.head(300).iterrows():
                for col_idx, cell_data in enumerate(row_data):
                    item = QTableWidgetItem(str(cell_data))
                    self.main_page_view.tableWidget_data_preview.setItem(row_idx, col_idx, item)
        else:
            self.main_page_view.tableWidget_data_preview.setRowCount(len(dt.index))
            for row_idx, row_data in dt.iterrows():
                for col_idx, cell_data in enumerate(row_data):
                    item = QTableWidgetItem(str(cell_data))
                    self.main_page_view.tableWidget_data_preview.setItem(row_idx, col_idx, item)

        for idx in range(len(dt.columns)):
            self.main_page_view.tableWidget_data_preview.setColumnWidth(
                idx,
                int(self.main_page_view.tableWidget_data_preview.width() / len(dt.columns) - 25),
            )

    def clear_columns(self):
        self.main_page_view.tableWidget_data_preview.clearContents()
        self.main_page_view.tableWidget_data_preview.setRowCount(0)
        self.main_page_view.tableWidget_data_preview.setColumnCount(0)

    def set_column_list(self, columns: list):
        self.main_page_view.listWidget_data_columns.clear()

        for item in columns:
            value = QListWidgetItem(item)
            self.main_page_view.listWidget_data_columns.addItem(value)
