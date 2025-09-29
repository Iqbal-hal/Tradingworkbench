from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QPushButton, QLineEdit
)
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Change TextBox Color Example")
        self.resize(400, 200)

        # Central widget + layout
        central = QWidget()
        layout = QVBoxLayout(central)

        # Text box
        self.textbox = QLineEdit("Type something here...")
        self.textbox.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.textbox)

        # Button
        self.button = QPushButton("Change Color")
        self.button.clicked.connect(self.change_color)  # connect signal to slot
        layout.addWidget(self.button)

        self.setCentralWidget(central)

    def change_color(self):
        """Slot to change the text box background color."""
        # Toggle between two colors for demo
        current_style = self.textbox.styleSheet()
        if "lightgreen" in current_style:
            self.textbox.setStyleSheet("background-color: lightblue; color: black;")
        else:
            self.textbox.setStyleSheet("background-color: lightgreen; color: black;")


app = QApplication([])
win = MainWindow()
win.show()
app.exec_()