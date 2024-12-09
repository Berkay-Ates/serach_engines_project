# multi dimensional scaling
import os
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog
from view_py_files.data_page import Ui_MainWindow as DataPageView

import pandas as pd
from pathlib import Path
import re

from utilities.show_messages import (
    show_error_message,
    show_info_message,
    notification_message,
)
from utilities.nltk_tools import remove_stopwords, snowball_stemming, wordnet_lemmatizer


class DataPageController(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.data_page_view = DataPageView()
        self.data_page_view.setupUi(self)
        self.data_page_view.pushButton_select_data_from_pc.clicked.connect(self.select_data_from_pc)

        self.add_data_files_to_list()

        self.data = None
        self.data_path = None
        self.column_operations = {}
        self.selected_data_column = None

        self.checboxes = self.data_page_view.groupBox_data_processing.findChildren(QCheckBox)

        self.data_page_view.checkBox_lowercase.clicked.connect(self.lowercase_data)
        self.data_page_view.checkBox_clear_punc_char.clicked.connect(self.clear_punct_spec_char)
        self.data_page_view.checkBox_remove_numbers.clicked.connect(self.remove_numbers)
        self.data_page_view.checkBox_lemmatize.clicked.connect(self.lemmatize_data)
        self.data_page_view.checkBox_stemming.clicked.connect(self.stem_data)

        self.data_page_view.checkBox_remove_stop_words.clicked.connect(self.remove_stop_words)

        self.data_page_view.pushButton_reload_data.clicked.connect(self.reload_data)
        self.data_page_view.listWidget_selected_data_columns.itemClicked.connect(self.data_column_drop)

        self.data_page_view.pushButton_apply_regex.clicked.connect(self.apply_regex)
        self.data_page_view.pushButton_save_dataset.clicked.connect(self.save_dataset)
        self.data_page_view.pushButton_drop_values.clicked.connect(self.data_drop_values)
        self.data_page_view.tableWidget_data.itemSelectionChanged.connect(self.select_column)
        self.data_page_view.comboBox_select_saved_data.currentIndexChanged.connect(self.select_data_from_saved_files)
        self.data_page_view.checkBox_remove_duplicates.clicked.connect(self.remove_duplicates)
        self.data_page_view.checkBox_remove_none.clicked.connect(self.remove_none_rows)

    def remove_duplicates(self, value):
        try:
            if value is True:
                self.data = self.data.drop_duplicates()
                self.data_page_view.checkBox_remove_duplicates.setDisabled(True)
                self.set_table(self.data)
            else:
                show_error_message("Select a data Columns")
                self.data_page_view.checkBox_remove_duplicates.setChecked(not value)
        except Exception as e:
            show_error_message(e)

    def add_data_files_to_list(self):
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        parent_directory = os.path.dirname(current_directory)
        data_directory = os.path.join(parent_directory, "Data")
        self.data_page_view.comboBox_select_saved_data.clear()
        self.data_page_view.comboBox_select_saved_data.addItem("Select From Saved Datasets")

        files = os.listdir(data_directory)
        for file in files:
            self.data_page_view.comboBox_select_saved_data.addItem(file)

    def select_column(self):
        try:
            idx = self.data_page_view.tableWidget_data.selectedItems()[0].column()
            self.selected_data_column = self.data_page_view.tableWidget_data.horizontalHeaderItem(idx).text()
            self.data_page_view.label_selected_data_column.setText(f"Selected Column: {self.selected_data_column}")
            for checkbox in self.checboxes:
                if checkbox.objectName() in self.column_operations[self.selected_data_column]:
                    checkbox.setDisabled(True)
                    checkbox.setChecked(True)
                else:
                    checkbox.setChecked(False)
                    checkbox.setDisabled(False)

        except Exception as e:
            print(e)
            show_error_message(e)

    def save_dataset(self):
        data_name = self.data_page_view.lineEdit_name_dataset.text()

        if len(data_name.split(" ")) != 1 or data_name == "" or self.data is None:
            show_error_message("there should not be space in the data name or you should select data ")
            return
        self.data.to_csv(f"Data/{data_name}.csv", index=False)
        show_info_message(f"{data_name} saved to Data/{data_name}")

    def apply_regex(self):
        regex_text = self.data_page_view.lineEdit_regex.text()
        if regex_text and self.selected_data_column is not None:
            try:
                compiled_regex = re.compile(regex_text)
                self.data[self.selected_data_column] = self.data[self.selected_data_column].str.replace(
                    compiled_regex, "", regex=True
                )
                self.set_table(self.data)
            except Exception as e:
                show_error_message(e)

    def data_column_drop(self, selected_column):
        if selected_column.text() == self.selected_data_column:
            show_error_message("You cannot drop selected data colum, please try to remove unselected data columns")
            return
        drop_result = notification_message(f"Do you want to drop {selected_column.text()} ?")
        if drop_result == True:
            self.data = self.data.drop(columns=[selected_column.text()])
            self.set_table(self.data)
            self.set_column_list(self.data.columns)

    def reload_data(self):
        if self.data_path is not None:
            self.selected_data_column = None
            dt = pd.read_csv(self.data_path)
            self.data = dt
            self.set_table(dt)
            self.set_column_list(dt.columns)
            self.init_col_operations()

            for checkbox in self.checboxes:
                checkbox.setChecked(False)
                checkbox.setDisabled(False)

    def data_drop_values(self):
        value = self.data_page_view.lineEdit_data_percentage.text()
        if value != "" and value.isdigit():
            selected_rows = self.data.sample(frac=1 - int(value) / 100)
            self.data = self.data.drop(selected_rows.index)
            self.set_table(self.data)

    def remove_stop_words(self, value):
        try:
            if value is True and self.selected_data_column is not None:
                self.data[self.selected_data_column] = self.data[self.selected_data_column].apply(remove_stopwords)
                self.data_page_view.checkBox_remove_stop_words.setDisabled(True)
                self.column_operations[self.selected_data_column].append(
                    self.data_page_view.checkBox_remove_stop_words.objectName()
                )
                self.set_table(self.data)
            else:
                show_info_message("Select a data Columns")
                self.data_page_view.checkBox_remove_stop_words.setChecked(not value)
        except Exception as e:
            print(e)
            show_error_message(e)

    def stem_data(self, value):
        try:
            if value is True and self.selected_data_column is not None:
                self.data[self.selected_data_column] = self.data[self.selected_data_column].apply(snowball_stemming)
                self.data_page_view.checkBox_stemming.setDisabled(True)
                self.column_operations[self.selected_data_column].append(
                    self.data_page_view.checkBox_stemming.objectName()
                )
                self.set_table(self.data)
            else:
                show_info_message("Select a data Column")
                self.data_page_view.checkBox_stemming.setChecked(not value)
        except Exception as e:
            print(e)
            show_error_message(e)

    def lemmatize_data(self, value):
        try:
            if value is True and self.selected_data_column is not None:
                self.data[self.selected_data_column] = self.data[self.selected_data_column].apply(wordnet_lemmatizer)
                self.data_page_view.checkBox_lemmatize.setDisabled(True)
                self.column_operations[self.selected_data_column].append(
                    self.data_page_view.checkBox_lemmatize.objectName()
                )
                self.set_table(self.data)
            else:
                show_info_message("Select a data Column")
                self.data_page_view.checkBox_lemmatize.setChecked(not value)
        except Exception as e:
            print(e)
            show_error_message(e)

    def remove_none_rows(self, value):
        try:
            if value is True:
                self.data.dropna(inplace=True, how="all")
                self.data_page_view.checkBox_remove_none.setDisabled(True)
                self.set_table(self.data)
            else:
                show_error_message("Select a data Columns")
                self.data_page_view.checkBox_remove_none.setChecked(not value)
        except Exception as e:
            show_error_message(e)

    def remove_numbers(self, value):
        try:
            if value is True and self.selected_data_column is not None:
                self.data[self.selected_data_column] = self.data[self.selected_data_column].str.replace(
                    r"\b\d+\b", "", regex=True
                )
                self.data_page_view.checkBox_remove_numbers.setDisabled(True)
                self.column_operations[self.selected_data_column].append(
                    self.data_page_view.checkBox_remove_numbers.objectName()
                )
                self.set_table(self.data)

            else:
                show_info_message("Select a data Column")
                self.data_page_view.checkBox_remove_numbers.setChecked(not value)
        except Exception as e:
            print(e)
            show_error_message(e)

    def clear_punct_spec_char(self, value):
        try:
            if value is True and self.selected_data_column is not None:
                self.data[self.selected_data_column] = self.data[self.selected_data_column].apply(
                    lambda x: re.sub(r"[^\w\s]", "", x)
                )
                self.data_page_view.checkBox_clear_punc_char.setDisabled(True)
                self.column_operations[self.selected_data_column].append(
                    self.data_page_view.checkBox_clear_punc_char.objectName()
                )
                self.set_table(self.data)
            else:
                show_info_message("Select a data Column")
                self.data_page_view.checkBox_clear_punc_char.setChecked(not value)
        except Exception as e:
            print(e)
            show_error_message(e)

    def lowercase_data(self, value):
        try:
            if value is True and self.selected_data_column is not None:
                self.data[self.selected_data_column] = self.data[self.selected_data_column].str.lower()
                self.data_page_view.checkBox_lowercase.setDisabled(True)
                self.column_operations[self.selected_data_column].append(
                    self.data_page_view.checkBox_lowercase.objectName()
                )
                self.set_table(self.data)
            else:
                show_info_message("Select a data Column")
                self.data_page_view.checkBox_lowercase.setChecked(not value)
        except Exception as e:
            print(e)
            show_error_message(e)

    def select_data_from_pc(self):
        fname, _ = QFileDialog.getOpenFileName(caption="Select CSV Data File", filter="CSV Files (*.csv)")

        if fname:
            path = Path(str(fname))
            fname = "Dataset Location: " + str(fname)
            self.data_page_view.label_selected_dataset_location.setText(fname)
            dt = pd.read_csv(path)
            self.data = dt
            self.data_path = path
            self.set_table(dt)
            self.set_column_list(dt.columns)
            self.init_col_operations()
            for checkbox in self.checboxes:
                checkbox.setChecked(False)
                checkbox.setDisabled(False)
            self.selected_data_column = None

    def select_data_from_saved_files(self):
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        parent_directory = os.path.dirname(current_directory)
        data_directory = os.path.join(parent_directory, "Data")
        df = self.data_page_view.comboBox_select_saved_data.currentText()
        self.data_path = f"{data_directory}/{df}"

        try:
            dt = pd.read_csv(self.data_path)
            self.data = dt

        except Exception as e:
            show_error_message(e)

        self.set_table(dt=self.data)
        self.set_column_list(self.data.columns)
        self.selected_data_column = None
        self.init_col_operations()
        for checkbox in self.checboxes:
            checkbox.setChecked(False)
            checkbox.setDisabled(False)

    def set_table(self, dt: pd.DataFrame):
        self.data_page_view.tableWidget_data.setColumnCount(len(dt.columns))
        self.data_page_view.tableWidget_data.setHorizontalHeaderLabels(dt.columns)
        if dt.memory_usage(deep=True).sum() / (1024 * 1024) > 20:
            show_info_message("The data frame is larger than 20Mb so first 300 lines just showed in table")
            self.data_page_view.tableWidget_data.setRowCount(300)
            for row_idx, row_data in dt.head(300).iterrows():
                for col_idx, cell_data in enumerate(row_data):
                    item = QTableWidgetItem(str(cell_data))
                    self.data_page_view.tableWidget_data.setItem(row_idx, col_idx, item)
        else:
            self.data_page_view.tableWidget_data.setRowCount(len(dt.index))
            for row_idx, row_data in dt.iterrows():
                for col_idx, cell_data in enumerate(row_data):
                    item = QTableWidgetItem(str(cell_data))
                    self.data_page_view.tableWidget_data.setItem(row_idx, col_idx, item)

        for idx in range(len(dt.columns)):
            self.data_page_view.tableWidget_data.setColumnWidth(
                idx, int(self.data_page_view.tableWidget_data.width() / len(dt.columns) - 25)
            )

    def init_col_operations(self):
        self.column_operations = {}
        for col in self.data.columns:
            self.column_operations[col] = []

    def clear_columns(self):
        self.data_page_view.tableWidget_data.clearContents()
        self.data_page_view.tableWidget_data.setRowCount(0)
        self.data_page_view.tableWidget_data.setColumnCount(0)

    def set_column_list(self, columns: list):
        self.data_page_view.listWidget_selected_data_columns.clear()

        for item in columns:
            value = QListWidgetItem(item)
            self.data_page_view.listWidget_selected_data_columns.addItem(value)
