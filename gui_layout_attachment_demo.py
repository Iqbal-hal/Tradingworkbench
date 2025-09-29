from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel
)
from PyQt5.QtGui import QPalette, QColor, QFont


class LayoutAttachmentDemo(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Central widget with horizontal layout (Style 1: attach at creation) ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # This attaches the layout directly to central_widget
        main_layout = QHBoxLayout(central_widget)

        # --- Left panel: using QHBoxLayout(parent) style ---
        left_panel = QWidget()
        left_panel.setAutoFillBackground(True)
        palette = left_panel.palette()
        palette.setColor(QPalette.Window, QColor("#a8dadc"))  # light teal
        left_panel.setPalette(palette)

        left_label = QLabel("Left Panel\n(HBoxLayout(parent))")
        left_label.setFont(QFont("Arial", 10, QFont.Bold))
        main_layout.addWidget(left_panel)

        # Add label inside left panel
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(left_label)

        # --- Right panel: using QVBoxLayout() + setLayout() style ---
        right_panel = QWidget()
        right_panel.setAutoFillBackground(True)
        palette = right_panel.palette()
        palette.setColor(QPalette.Window, QColor("#f4a261"))  # orange
        right_panel.setPalette(palette)

        # Create layout first, then attach it
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        right_label = QLabel("Right Panel\n(QVBoxLayout() + setLayout())")
        right_label.setFont(QFont("Arial", 10, QFont.Bold))
        right_layout.addWidget(right_label)

        # Add right panel to main layout
        main_layout.addWidget(right_panel)

        # Window setup
        self.setWindowTitle("Layout Attachment Demo")
        self.setGeometry(100, 100, 700, 300)


app = QApplication([])
window = LayoutAttachmentDemo()
window.show()
app.exec_()