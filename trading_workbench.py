"""
trading_workbench.py
Main application window for Trading Workbench
"""

import os
import pandas as pd
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                             QPushButton, QLabel, QTabWidget, QMessageBox,
                             QStatusBar, QFileDialog, QListWidget, QTableWidget,
                             QTableWidgetItem, QGroupBox, QComboBox, QSpinBox,
                             QDoubleSpinBox, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon

# Local project imports
from data_manager import DataManager
from tabs.data_manager_tab import DataManagerTab
from tabs.fundamentals_tab import FundamentalsTab
from tabs.technical_tab import TechnicalAnalysisTab
from tabs.portfolio_tab import PortfolioAnalysisTab

class TradingWorkbench(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_manager = DataManager()
        self.current_data = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the main user interface"""
        self.setWindowTitle("Trading Workbench")
        self.setGeometry(100, 100, 1400, 900)  # x, y, width, height
        
        # Set application icon (optional)
        # self.setWindowIcon(QIcon('icon.png'))
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Setup navigation sidebar
        self.setup_sidebar(main_layout)
        
        # Setup main content area
        self.setup_content_area(main_layout)
        
        # Setup status bar
        self.setup_status_bar()
        
    def setup_sidebar(self, main_layout):
        """Create the navigation sidebar"""
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout()
        
        # Application title
        title = QLabel("Trading Workbench")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title)
        
        sidebar_layout.addSpacing(20)
        
        # Navigation buttons
        nav_buttons = [
            ("📊 Data Manager", self.show_data_manager),
            ("🔍 Fundamentals", self.show_fundamentals),
            ("📈 Technical Analysis", self.show_technical),
            ("📊 Portfolio Analysis", self.show_portfolio),
            ("📋 Reports", self.show_reports),
            ("⚙️ Settings", self.show_settings)
        ]
        
        for text, handler in nav_buttons:
            btn = QPushButton(text)
            btn.setFixedHeight(45)
            btn.setFont(QFont("Arial", 10))
            btn.clicked.connect(handler)
            sidebar_layout.addWidget(btn)
        
        sidebar_layout.addStretch()  # Push everything to the top
        
        # Current file info
        self.file_info_label = QLabel("No data loaded")
        self.file_info_label.setWordWrap(True)
        self.file_info_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 3px;")
        sidebar_layout.addWidget(self.file_info_label)
        
        sidebar.setLayout(sidebar_layout)
        main_layout.addWidget(sidebar)
    
    def setup_content_area(self, main_layout):
        """Create the main content area with tabs"""
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.data_tab = DataManagerTab(self)
        self.fundamentals_tab = FundamentalsTab(self)
        self.technical_tab = TechnicalAnalysisTab(self)
        self.portfolio_tab = PortfolioAnalysisTab(self)
        
        # Add tabs to widget
        self.tab_widget.addTab(self.data_tab, "📊 Data Manager")
        self.tab_widget.addTab(self.fundamentals_tab, "🔍 Fundamentals")
        self.tab_widget.addTab(self.technical_tab, "📈 Technical Analysis")
        self.tab_widget.addTab(self.portfolio_tab, "📊 Portfolio Analysis")
        
        main_layout.addWidget(self.tab_widget)
    
    def setup_status_bar(self):
        """Setup the status bar at the bottom"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        status_bar.showMessage("Ready")
    
    # Navigation methods
    def show_data_manager(self):
        self.tab_widget.setCurrentIndex(0)
    
    def show_fundamentals(self):
        if self.current_data is not None:
            self.tab_widget.setCurrentIndex(1)
        else:
            QMessageBox.warning(self, "No Data", "Please load data first!")
    
    def show_technical(self):
        if self.current_data is not None:
            self.tab_widget.setCurrentIndex(2)
        else:
            QMessageBox.warning(self, "No Data", "Please load data first!")
    
    def show_portfolio(self):
        if self.current_data is not None:
            self.tab_widget.setCurrentIndex(3)
        else:
            QMessageBox.warning(self, "No Data", "Please load data first!")
    
    def show_reports(self):
        QMessageBox.information(self, "Reports", "Reports feature coming soon!")
    
    def show_settings(self):
        QMessageBox.information(self, "Settings", "Settings feature coming soon!")
    
    def load_data(self, file_path):
        """Load data and update all tabs"""
        try:
            self.current_data = self.data_manager.load_csv(file_path)
            self.file_info_label.setText(f"Loaded: {os.path.basename(file_path)}\n"
                                       f"Rows: {len(self.current_data):,}\n"
                                       f"Stocks: {self.current_data['Stock'].nunique()}")
            
            # Update all tabs with new data
            self.data_tab.on_data_loaded(self.current_data)
            self.fundamentals_tab.on_data_loaded(self.current_data)
            self.technical_tab.on_data_loaded(self.current_data)
            self.portfolio_tab.on_data_loaded(self.current_data)
            
            self.statusBar().showMessage(f"Successfully loaded {len(self.current_data):,} rows")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load data:\n{str(e)}")
            return False