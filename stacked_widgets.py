from PyQt5.QtWidgets import (QMainWindow,QApplication, QWidget, QHBoxLayout, QVBoxLayout)
from color_wd import color
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stacked Widgets Example")
        self.resize(400, 300)
        layout_1 = QHBoxLayout()
        layout_2 = QVBoxLayout()
        layout_3=QVBoxLayout()

        #vertical 1
        layout_2.addWidget(color("red"))
        layout_2.addWidget(color("green"))
        layout_2.addWidget(color("orange"))

        #vertical 2
        layout_3.addWidget(color("blue"))
        layout_3.addWidget(color("purple"))

        # Horizontal all
        layout_1.addWidget(color("Pink"))
        layout_1.addLayout(layout_2)
        layout_1.addLayout(layout_3)

        # Show all

        main_widget=QWidget()
        main_widget.setLayout(layout_1)
        self.setCentralWidget(main_widget)

App=QApplication([])
Win=MainWindow()
Win.show()
App.exec_()# stacked_widgets.py





       

