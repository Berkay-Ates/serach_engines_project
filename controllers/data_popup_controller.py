from PyQt5.QtWidgets import *
import pandas as pd
from view_py_files.data_popup import Ui_MainWindow as MainWindow
from utilities.show_messages import show_error_message, notification_message


class DataPopupController(QMainWindow):

    def __init__(self, data_path) -> None:
        super().__init__()

        self.main_window = MainWindow()
        self.main_window.setupUi(self)
        self.data_path = data_path
        self.set_table(pd.read_csv(self.data_path))

    def set_table(self, dt: pd.DataFrame):
        self.clear_columns()
        width = self.main_window.menubar.width()
        self.main_window.tableWidget_data.setColumnCount(len(dt.columns))
        self.main_window.tableWidget_data.setHorizontalHeaderLabels(dt.columns)
        if dt.memory_usage(deep=True).sum() / (1024 * 1024) > 20:
            show_error_message("The data frame is larger than 20Mb so first 300 lines just showed in table")
            self.main_window.tableWidget_data.setRowCount(300)
            for row_idx, row_data in dt.head(300).iterrows():
                for col_idx, cell_data in enumerate(row_data):
                    item = QTableWidgetItem(str(cell_data))
                    self.main_window.tableWidget_data.setItem(row_idx, col_idx, item)
        else:
            self.main_window.tableWidget_data.setRowCount(len(dt.index))
            for row_idx, row_data in dt.iterrows():
                for col_idx, cell_data in enumerate(row_data):
                    item = QTableWidgetItem(str(cell_data))
                    self.main_window.tableWidget_data.setItem(row_idx, col_idx, item)

        for idx in range(len(dt.columns)):
            self.main_window.tableWidget_data.setColumnWidth(
                idx, int(width / len(dt.columns)) - int(150 / len(dt.columns))
            )

    def clear_columns(self):
        self.main_window.tableWidget_data.clearContents()
        self.main_window.tableWidget_data.setRowCount(0)
        self.main_window.tableWidget_data.setColumnCount(0)
