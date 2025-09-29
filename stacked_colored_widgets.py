from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel
)
from PyQt5.QtCore import Qt


def make_colored_box(name, color, min_width=100, min_height=50):
    """Create a QLabel with background color set inline."""
    label = QLabel(name)
    label.setAlignment(Qt.AlignCenter)
    label.setMinimumSize(min_width, min_height)
    label.setStyleSheet(f"background-color: {color}; color: white;")
    return label


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stacked Widgets Example")
        self.resize(600, 400)

        # Main horizontal layout
        layout_1 = QHBoxLayout()

        # --- Column 1 ---
        col1_widget = QWidget()
        col1_layout = QVBoxLayout(col1_widget)
        col1_layout.addWidget(make_colored_box("Pink", "pink"))
        layout_1.addWidget(col1_widget)

        # --- Column 2 ---
        col2_widget = QWidget()
        col2_layout = QVBoxLayout(col2_widget)
        col2_layout.addWidget(make_colored_box("Red", "red"))
        col2_layout.addWidget(make_colored_box("Green", "green"))
        col2_layout.addWidget(make_colored_box("Orange", "orange"))
        layout_1.addWidget(col2_widget)

        # --- Column 3 ---
        col3_widget = QWidget()
        col3_layout = QVBoxLayout(col3_widget)
        col3_layout.addWidget(make_colored_box("Blue", "blue"))
        col3_layout.addWidget(make_colored_box("Purple", "purple"))
        layout_1.addWidget(col3_widget)

        # Equal widths
        layout_1.setStretch(0, 1)
        layout_1.setStretch(1, 1)
        layout_1.setStretch(2, 1)

        # Central widget
        main_widget = QWidget()
        main_widget.setLayout(layout_1)
        self.setCentralWidget(main_widget)


App = QApplication([])
Win = MainWindow()
Win.show()
App.exec_()