import os
from PyQt5.QtWidgets import *
import pandas as pd
from controllers.data_popup_controller import DataPopupController
from controllers.multiple_cmp_result_page_cntrl import MultipleComparisionResultController
from modules.constants.constants import Constants
from view_py_files.multiple_cmp_select_page import Ui_MainWindow as ComparisionSelectPage
from modules.model.embed_param_model import EmbedParameters

from utilities.show_messages import show_error_message, notification_message


class MultipleComparisionSelectController(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.main_comp_page_view = ComparisionSelectPage()
        self.main_comp_page_view.setupUi(self)

        self.add_combobox_items()

        self.parameters1 = EmbedParameters()
        self.parameters2 = EmbedParameters()

        self.result_pages = []
        self.dataset_pages = []

        self.main_comp_page_view.pushButton_cluster_increase1.clicked.connect(
            lambda: self.cluster_increase(self.main_comp_page_view.lineEdit_cluster_count1, self.parameters1)
        )
        self.main_comp_page_view.pushButton_cluster_decrease1.clicked.connect(
            lambda: self.cluster_decrease(self.main_comp_page_view.lineEdit_cluster_count1, self.parameters1)
        )

        self.main_comp_page_view.pushButton_cluster_increase2.clicked.connect(
            lambda: self.cluster_increase(self.main_comp_page_view.lineEdit_cluster_count2, self.parameters2)
        )
        self.main_comp_page_view.pushButton_cluster_decrease2.clicked.connect(
            lambda: self.cluster_decrease(self.main_comp_page_view.lineEdit_cluster_count2, self.parameters2)
        )

        self.main_comp_page_view.pushButton_refresh_dataset1.clicked.connect(self.add_data_files)

        self.main_comp_page_view.comboBox_dataset_select1.currentIndexChanged.connect(
            lambda: self.set_data(
                self.parameters1,
                self.main_comp_page_view.comboBox_dataset_select1,
                self.main_comp_page_view.listWidget_data_columns1,
            )
        )

        self.main_comp_page_view.comboBox_dataset_select2.currentIndexChanged.connect(
            lambda: self.set_data(
                self.parameters2,
                self.main_comp_page_view.comboBox_dataset_select2,
                self.main_comp_page_view.listWidget_data_columns2,
            )
        )

        self.main_comp_page_view.comboBox_embedding1.currentIndexChanged.connect(
            lambda: self.set_embedding(self.parameters1, self.main_comp_page_view.comboBox_embedding1)
        )
        self.main_comp_page_view.comboBox_embedding2.currentIndexChanged.connect(
            lambda: self.set_embedding(self.parameters2, self.main_comp_page_view.comboBox_embedding2)
        )

        self.main_comp_page_view.comboBox_clustering1.currentIndexChanged.connect(
            lambda: self.set_clustering(self.parameters1, self.main_comp_page_view.comboBox_clustering1)
        )

        self.main_comp_page_view.comboBox_clustering2.currentIndexChanged.connect(
            lambda: self.set_clustering(self.parameters2, self.main_comp_page_view.comboBox_clustering2)
        )

        self.main_comp_page_view.comboBox_dimension_reducer1.currentIndexChanged.connect(
            lambda: self.set_dimension_reducer(self.parameters1, self.main_comp_page_view.comboBox_dimension_reducer1)
        )

        self.main_comp_page_view.comboBox_dimension_reducer2.currentIndexChanged.connect(
            lambda: self.set_dimension_reducer(self.parameters2, self.main_comp_page_view.comboBox_dimension_reducer2)
        )

        self.main_comp_page_view.listWidget_data_columns1.itemClicked.connect(
            lambda: self.select_column(self.parameters1, self.main_comp_page_view.listWidget_data_columns1)
        )

        self.main_comp_page_view.listWidget_data_columns2.itemClicked.connect(
            lambda: self.select_column(self.parameters2, self.main_comp_page_view.listWidget_data_columns2)
        )

        self.main_comp_page_view.pushButton_show_results.clicked.connect(self.show_result_page)
        self.main_comp_page_view.pushButton_pop_up_dataset1.clicked.connect(
            lambda: self.show_dataset_popup(self.parameters1)
        )

        self.main_comp_page_view.pushButton_pop_up_dataset2.clicked.connect(
            lambda: self.show_dataset_popup(self.parameters2)
        )

    def show_dataset_popup(self, params: EmbedParameters):
        if params.data_path == "No Path":
            show_error_message("You should select a dataset")
            return
        page = DataPopupController(params.data_path)
        self.dataset_pages.append(page)
        page.show()
        self.cleanup_pages()

    def show_result_page(self):
        if not self.parameters1.is_inited() or not self.parameters2.is_inited():
            show_error_message("You have to select all parameters for embedding and clustering!")
            return

        print(self.parameters1, self.parameters2)
        print(self.parameters1.data.columns)
        print(self.parameters2.data.columns)

        # * dataframe diger sayfada islenecegi icin bozulabilir dolayisiyla kopyasini yollayalim
        page = MultipleComparisionResultController(self.parameters1.__copy__(), self.parameters2.__copy__())
        self.result_pages.append(page)
        page.show()

        # Gösterilmeyen sayfaları temizle
        self.cleanup_pages()

    def cleanup_pages(self):
        self.result_pages = [page for page in self.result_pages if page.isVisible()]
        self.dataset_pages = [page for page in self.dataset_pages if page.isVisible()]

    def select_column(self, parameters: EmbedParameters, widget: QListWidget):
        if len(parameters.data.columns) > 0:
            parameters.data_column_name = widget.selectedItems()[0].text()

    def set_dimension_reducer(self, params: EmbedParameters, comboBox: QComboBox):
        params.dimension_reducer = comboBox.currentText()

    def set_clustering(self, params: EmbedParameters, comboBox: QComboBox):
        params.clustering_method = comboBox.currentText()

    def set_embedding(self, params: EmbedParameters, comboBox: QComboBox):
        params.embedding_method = comboBox.currentText()

    def set_data(self, embedding: EmbedParameters, widget, widgetList):
        df = widget.currentText()

        if df == "":
            return
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        parent_directory = os.path.dirname(current_directory)
        data_directory = os.path.join(parent_directory, "Data")
        embedding.data_path = f"{data_directory}/{df}"
        self.data_column = None
        try:
            dt = pd.read_csv(embedding.data_path)
            embedding.data = dt
        except Exception as e:
            show_error_message(e)

        self.set_column_list(embedding.data.columns, widgetList)

    def set_column_list(self, columns: list, widget):
        widget.clear()

        for item in columns:
            value = QListWidgetItem(item)
            widget.addItem(value)

    def cluster_increase(self, widget, params: EmbedParameters):
        current = widget.text()
        current = int(current) + 1
        params.cluster_count = current
        widget.setText(str(current))

    def cluster_decrease(self, widget, params: EmbedParameters):
        current = widget.text()
        if int(current) > 2:
            current = int(current) - 1
            params.cluster_count = current
            widget.setText(str(current))
        else:
            show_error_message("cluster count should be 2 or more")

    def add_combobox_items(self):
        constants = Constants()
        self.main_comp_page_view.comboBox_clustering1.clear()
        self.main_comp_page_view.comboBox_embedding1.clear()
        self.main_comp_page_view.comboBox_dimension_reducer1.clear()
        self.main_comp_page_view.comboBox_clustering2.clear()
        self.main_comp_page_view.comboBox_embedding2.clear()
        self.main_comp_page_view.comboBox_dimension_reducer2.clear()

        for item in constants._clustering_lists:
            self.main_comp_page_view.comboBox_clustering1.addItem(item)
            self.main_comp_page_view.comboBox_clustering2.addItem(item)

        self.main_comp_page_view.comboBox_clustering1.setPlaceholderText("Select a Clustering")
        self.main_comp_page_view.comboBox_clustering2.setPlaceholderText("Select a Clustering")

        for item in constants.embedding_list:
            self.main_comp_page_view.comboBox_embedding1.addItem(item)
            self.main_comp_page_view.comboBox_embedding2.addItem(item)
        self.main_comp_page_view.comboBox_embedding1.setPlaceholderText("Select a Embedding")
        self.main_comp_page_view.comboBox_embedding2.setPlaceholderText("Select a Embedding")

        for item in constants.dimension_reducer_list:
            self.main_comp_page_view.comboBox_dimension_reducer1.addItem(item)
            self.main_comp_page_view.comboBox_dimension_reducer2.addItem(item)

        self.main_comp_page_view.comboBox_dimension_reducer1.setPlaceholderText("Select a Dimension Reducer")
        self.main_comp_page_view.comboBox_dimension_reducer2.setPlaceholderText("Select a Dimension Reducer")

        self.add_data_files()

    def add_data_files(self):
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        parent_directory = os.path.dirname(current_directory)
        data_directory = os.path.join(parent_directory, "Data")
        self.main_comp_page_view.comboBox_dataset_select1.clear()
        self.main_comp_page_view.comboBox_dataset_select2.clear()
        self.main_comp_page_view.comboBox_dataset_select1.addItem("")
        self.main_comp_page_view.comboBox_dataset_select2.addItem("")

        files = os.listdir(data_directory)
        for file in files:
            self.main_comp_page_view.comboBox_dataset_select1.addItem(file)
            self.main_comp_page_view.comboBox_dataset_select2.addItem(file)
