from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QTextEdit
)
from PyQt5.QtGui import QFont, QPalette, QColor

class FullLayoutDemo(QMainWindow):
    def __init__(self):
        super().__init__()

        # Central widget and horizontal layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Sidebar setup
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout()
        sidebar.setLayout(sidebar_layout)

        # Sidebar background color
        sidebar.setAutoFillBackground(True)
        palette = sidebar.palette()
        palette.setColor(QPalette.Window, QColor("#dfe6e9"))  # light gray
        sidebar.setPalette(palette)

        # Sidebar content
        title = QLabel("Sidebar")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        sidebar_layout.addWidget(title)

        for label in ["Dashboard", "Settings", "Help"]:
            btn = QPushButton(label)
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar)

        # Main content area
        content_area = QWidget()
        content_layout = QVBoxLayout()
        content_area.setLayout(content_layout)

        # Content background color
        content_area.setAutoFillBackground(True)
        palette = content_area.palette()
        palette.setColor(QPalette.Window, QColor("#d0e6f7"))  # light blue
        content_area.setPalette(palette)

        # Add controls to content area
        heading = QLabel("User Information")
        heading.setFont(QFont("Arial", 12, QFont.Bold))
        content_layout.addWidget(heading)

        name_input = QLineEdit()
        name_input.setPlaceholderText("Enter your name")
        content_layout.addWidget(name_input)

        email_input = QLineEdit()
        email_input.setPlaceholderText("Enter your email")
        content_layout.addWidget(email_input)

        agree_checkbox = QCheckBox("Subscribe to newsletter")
        content_layout.addWidget(agree_checkbox)

        country_select = QComboBox()
        country_select.addItems(["India", "USA", "Germany", "Japan"])
        content_layout.addWidget(country_select)

        comments = QTextEdit()
        comments.setPlaceholderText("Additional comments...")
        content_layout.addWidget(comments)

        submit_btn = QPushButton("Submit")
        content_layout.addWidget(submit_btn)

        main_layout.addWidget(content_area)

        self.setWindowTitle("Step 3: Sidebar + Main Controls")
        self.setGeometry(100, 100, 800, 500)

app = QApplication([])
window = FullLayoutDemo()
window.show()
app.exec_()