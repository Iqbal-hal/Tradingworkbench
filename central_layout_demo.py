from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QLabel

class SimpleWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Step 1: Create a central widget (the main container)
        central_widget = QWidget()

        # Step 2: Set it as the central area of the QMainWindow
        self.setCentralWidget(central_widget)
        

        # Step 3: Create a horizontal layout and attach it to the central widget
        main_layout = QHBoxLayout(central_widget)

        # Add something to visualize the layout
        label = QLabel("Hello from the left side!")
        main_layout.addWidget(label)

        self.setWindowTitle("Minimal Layout Demo")
        self.setGeometry(100, 100, 400, 200)

app = QApplication([])
window = SimpleWindow()
window.show()
app.exec_()