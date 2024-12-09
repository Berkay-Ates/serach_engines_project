from PyQt5.QtWidgets import *


def show_error_message(message):
    error_message_box = QMessageBox()
    error_message_box.setIcon(QMessageBox.Critical)
    error_message_box.setText(str(message))
    error_message_box.setWindowTitle("Error")
    error_message_box.setStandardButtons(QMessageBox.Ok)
    error_message_box.exec_()


def show_info_message(message):
    info_message_box = QMessageBox()
    info_message_box.setIcon(QMessageBox.Information)
    info_message_box.setText(str(message))
    info_message_box.setWindowTitle("Info")
    info_message_box.setStandardButtons(QMessageBox.Ok)
    info_message_box.exec_()


def notification_message(message):
    message_object = QMessageBox()
    message_object.setText(message)
    message_object.setWindowTitle("Dataset Deletiond!")
    message_object.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    message_object.setEscapeButton(QMessageBox.No)
    result = message_object.exec_()

    if result == QMessageBox.Yes:
        return True

    return False
