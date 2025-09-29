from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPalette, QColor, QFont

class SidebarDemo(QMainWindow):
    def __init__(self):
        super().__init__()

        # Central widget and horizontal layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Sidebar widget with vertical layout
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout()
        sidebar.setLayout(sidebar_layout)

        # Set background color for sidebar
        sidebar.setAutoFillBackground(True)
        palette = sidebar.palette()
        palette.setColor(QPalette.Window, QColor("lightgray"))
        sidebar.setPalette(palette)

        # Add title and buttons to sidebar
        title = QLabel("Sidebar")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        sidebar_layout.addWidget(title)

        for label in ["Home", "Settings", "About"]:
            btn = QPushButton(label)
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()  # Push buttons to top

        # Add sidebar to main layout
        main_layout.addWidget(sidebar)

        # Add placeholder content area
        content = QLabel("Main Content Area")
        content.setStyleSheet("background-color: lightblue; padding: 20px;")
        main_layout.addWidget(content)

        self.setWindowTitle("Step 2: Sidebar Layout")
        self.setGeometry(100, 100, 600, 300)

app = QApplication([])
window = SidebarDemo()
window.show()
app.exec_()