from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QLabel
from PyQt5.QtGui import QPalette, QColor

class ColorDemo(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create central widget (main content area)
        central_widget = QWidget()
        central_widget.setAutoFillBackground(True)
        palette = central_widget.palette()
        palette.setColor(QPalette.Window, QColor("lightblue"))  # Light blue background
        central_widget.setPalette(palette)

        # Set it as the central widget of the window
        self.setCentralWidget(central_widget)

        # Add layout and label to visualize
        layout = QHBoxLayout(central_widget)
        label = QLabel("This is the central widget area")
        layout.addWidget(label)

        self.setWindowTitle("Central Widget Demo")
        self.setGeometry(100, 100, 500, 300)

app = QApplication([])
window = ColorDemo()
window.show()
app.exec_()