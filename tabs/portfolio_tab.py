"""
tabs/portfolio_tab.py
Portfolio analysis tab (placeholder)
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

class PortfolioAnalysisTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Portfolio Analysis - Coming Soon"))
        self.setLayout(layout)
    
    def on_data_loaded(self, data):
        pass