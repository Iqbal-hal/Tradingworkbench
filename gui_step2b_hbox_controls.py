from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox
)
from PyQt5.QtGui import QFont

class HBoxControlsDemo(QMainWindow):
    def __init__(self):
        super().__init__()

        # Central widget and horizontal layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Label
        label = QLabel("Name:")
        label.setFont(QFont("Arial", 10))
        layout.addWidget(label)

        # Text input
        name_input = QLineEdit()
        name_input.setPlaceholderText("Enter your name")
        layout.addWidget(name_input)

        # Checkbox
        agree_checkbox = QCheckBox("I agree")
        layout.addWidget(agree_checkbox)

        # Button
        submit_btn = QPushButton("Submit")
        layout.addWidget(submit_btn)

        self.setWindowTitle("HBox Layout with Controls")
        self.setGeometry(100, 100, 600, 100)

app = QApplication([])
window = HBoxControlsDemo()
window.show()
app.exec_()