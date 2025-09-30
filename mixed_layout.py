import sys
from PyQt6.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QWidget, 
    QHBoxLayout, 
    QVBoxLayout, 
    QPushButton,
    QLabel
)

class ExampleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Central Widget and Layout Demo")
        self.setGeometry(200, 200, 600, 250)  # x, y, width, height
        self.init_ui()

    def init_ui(self):
        """Sets up the UI elements."""

        # 1. Create central widget (the main container).
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 2. Create the MAIN layout: horizontal (left vs right areas).
        main_layout = QHBoxLayout(central_widget)

        # --- LEFT SIDE: Vertical layout with buttons ---
        left_layout = QVBoxLayout()
        btn_a = QPushButton("A (top)")
        btn_b = QPushButton("B (middle)")
        btn_c = QPushButton("C (bottom)")

        left_layout.addWidget(btn_a)
        left_layout.addWidget(btn_b)
        left_layout.addWidget(btn_c)

        # Put left layout into a container widget
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # --- RIGHT SIDE: Horizontal layout with label + buttons ---
        right_layout = QHBoxLayout()
        button1 = QPushButton("Button 1")
        button2 = QPushButton("Button 2")
        label1 = QLabel("Label 1")
        button3 = QPushButton("Button 3")

        right_layout.addWidget(button1)
        right_layout.addWidget(button2)
        right_layout.addWidget(label1)
        right_layout.addWidget(button3)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # --- Add both sides to the main horizontal layout ---
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)


# --- Boilerplate code to run the application ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ExampleWindow()
    window.show()
    sys.exit(app.exec())
