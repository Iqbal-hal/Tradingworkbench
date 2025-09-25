"""
tabs/data_manager_tab.py
Data management tab functionality
"""

import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTableWidget, QTableWidgetItem, QGroupBox,
                             QFileDialog, QMessageBox, QListWidget)
from PyQt5.QtCore import Qt

class DataManagerTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        
        # Load button
        load_btn = QPushButton("Load CSV File")
        load_btn.clicked.connect(self.load_csv_file)
        load_btn.setFixedHeight(40)
        file_layout.addWidget(load_btn)
        
        # Recent files list
        file_layout.addWidget(QLabel("Recent Files:"))
        self.recent_files_list = QListWidget()
        file_layout.addWidget(self.recent_files_list)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Data preview group
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        
        # Data statistics
        self.stats_label = QLabel("No data loaded")
        preview_layout.addWidget(self.stats_label)
        
        # Data table
        self.preview_table = QTableWidget()
        preview_layout.addWidget(self.preview_table)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        self.setLayout(layout)
    
    def load_csv_file(self):
        """Open file dialog to load CSV"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)")
        
        if file_path:
            success = self.main_window.load_data(file_path)
            if success:
                self.add_to_recent_files(file_path)
    
    def add_to_recent_files(self, file_path):
        """Add file to recent files list"""
        # Simple implementation - just show the filename
        file_name = os.path.basename(file_path)
        self.recent_files_list.addItem(file_name)
    
    def on_data_loaded(self, data):
        """Update UI when new data is loaded"""
        # Update statistics
        stats_text = (f"Rows: {len(data):,} | "
                     f"Stocks: {data['Stock'].nunique()} | "
                     f"Date Range: {data['Date'].min().strftime('%Y-%m-%d')} to "
                     f"{data['Date'].max().strftime('%Y-%m-%d')}")
        self.stats_label.setText(stats_text)
        
        # Update preview table (show first 20 rows)
        self.update_preview_table(data.head(20))
    
    def update_preview_table(self, data):
        """Update the data preview table"""
        self.preview_table.setRowCount(len(data))
        self.preview_table.setColumnCount(len(data.columns))
        self.preview_table.setHorizontalHeaderLabels(data.columns)
        
        for row_idx, (_, row_data) in enumerate(data.iterrows()):
            for col_idx, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                self.preview_table.setItem(row_idx, col_idx, item)
        
        self.preview_table.resizeColumnsToContents()