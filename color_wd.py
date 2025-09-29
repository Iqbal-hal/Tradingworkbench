# color_wd.py
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

def color(name, text=None, min_width=100, min_height=50):
    """
    Create a QLabel with a given background color that can expand
    to fill its layout cell.
    """
    widget = QLabel(text if text else name.capitalize())
    widget.setAlignment(Qt.AlignCenter)

    # Instead of fixed size, allow expansion
    widget.setMinimumSize(min_width, min_height)
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # Apply background color
    widget.setAutoFillBackground(True)
    palette = widget.palette()
    palette.setColor(QPalette.Window, QColor(name))
    palette.setColor(QPalette.WindowText, Qt.white)
    widget.setPalette(palette)

    return widget