"""
main.py
Main entry point for Trading Workbench PyQt application
"""

import sys
from PyQt5.QtWidgets import QApplication
from trading_workbench import TradingWorkbench

def main():
    # Create the application
    app = QApplication(sys.argv)
    
    """
Explains how a PyQt5 window object inherits the 'show()' method.

In a PyQt5 application, a window object, such as an instance of a
class that inherits from QMainWindow or QWidget, gains access to
the 'show()' method through inheritance from its base class in the
PyQt5 framework. Calling the 'show()' method makes the window visible
to the user within the application's event loop.
"""
    window = TradingWorkbench()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()