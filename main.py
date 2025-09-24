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
    
    # Create and show the main window
    window = TradingWorkbench()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()