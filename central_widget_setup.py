import sys
from PyQt6.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QWidget, 
    QHBoxLayout, 
    QPushButton,
    QLabel
)

class ExampleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Central Widget and Layout Demo")
        self.setGeometry(200, 200, 500, 150) # x, y, width, height
        self.init_ui()

    def init_ui(self):
        """Sets up the UI elements."""

        # 1. Create a blank container to hold all our other widgets.
        # Think of this as the main canvas for our window's content area.
        central_widget = QWidget()
        
        # 2. Set this blank container as the central content area of the main window.
        # Now, it fills the space inside the window's frame.
        self.setCentralWidget(central_widget)

        # 3. Create a layout manager that arranges things horizontally (left-to-right)
        # and tell it to manage the space inside our 'central_widget'.
        main_layout = QHBoxLayout(central_widget)

        # --- Now, we add widgets to the LAYOUT ---
        # The layout will automatically place them inside the central_widget for us.
        
        # Create three buttons
        button1 = QPushButton("Button 1")
        button2 = QPushButton("Button 2")
        button3 = QPushButton("Button 3")

        label1 = QLabel("Label 1")
        
        # Add the buttons to the horizontal layout
        main_layout.addWidget(button1)
        main_layout.addWidget(button2)
         # Add the label to the layout
        main_layout.addWidget(label1)
        main_layout.addWidget(button3)

       

        # You don't need to specify positions; the QHBoxLayout handles it.

# --- Boilerplate code to run the application ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ExampleWindow()
    window.show()
    sys.exit(app.exec())