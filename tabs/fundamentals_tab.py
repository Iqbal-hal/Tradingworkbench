"""
tabs/fundamentals_tab.py
Fundamental analysis tab
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTableWidget, QTableWidgetItem, QGroupBox,
                             QDoubleSpinBox, QComboBox, QCheckBox)
from PyQt5.QtCore import Qt

class FundamentalsTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.data = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Filters group
        filters_group = QGroupBox("Fundamental Filters")
        filters_layout = QVBoxLayout()
        
        # P/E ratio filter
        pe_layout = QHBoxLayout()
        pe_layout.addWidget(QLabel("P/E Ratio:"))
        self.pe_min = QDoubleSpinBox()
        self.pe_min.setRange(0, 1000)
        self.pe_min.setValue(0)
        self.pe_max = QDoubleSpinBox()
        self.pe_max.setRange(0, 1000)
        self.pe_max.setValue(50)
        pe_layout.addWidget(self.pe_min)
        pe_layout.addWidget(QLabel("to"))
        pe_layout.addWidget(self.pe_max)
        filters_layout.addLayout(pe_layout)
        
        # Dividend yield filter
        dy_layout = QHBoxLayout()
        dy_layout.addWidget(QLabel("Dividend Yield (%):"))
        self.dy_min = QDoubleSpinBox()
        self.dy_min.setRange(0, 100)
        self.dy_min.setValue(0)
        self.dy_max = QDoubleSpinBox()
        self.dy_max.setRange(0, 100)
        self.dy_max.setValue(10)
        dy_layout.addWidget(self.dy_min)
        dy_layout.addWidget(QLabel("to"))
        dy_layout.addWidget(self.dy_max)
        filters_layout.addLayout(dy_layout)
        
        # Apply filters button
        apply_btn = QPushButton("Apply Filters")
        apply_btn.clicked.connect(self.apply_filters)
        filters_layout.addWidget(apply_btn)
        
        filters_group.setLayout(filters_layout)
        layout.addWidget(filters_group)
        
        # Results table
        self.results_table = QTableWidget()
        layout.addWidget(QLabel("Filtered Results:"))
        layout.addWidget(self.results_table)
        
        self.setLayout(layout)
    
    def on_data_loaded(self, data):
        """Called when new data is loaded"""
        self.data = data
        self.apply_filters()  # Auto-apply filters with default values
    
    def apply_filters(self):
        """Apply fundamental filters to data"""
        if self.data is None:
            return
        
        filtered_data = self.data.copy()
        
        # Apply P/E filter
        filtered_data = filtered_data[
            (filtered_data['P/E'] >= self.pe_min.value()) & 
            (filtered_data['P/E'] <= self.pe_max.value())
        ]
        
        # Apply dividend yield filter (convert percentage to decimal)
        filtered_data = filtered_data[
            (filtered_data['DY'] >= self.dy_min.value() / 100) & 
            (filtered_data['DY'] <= self.dy_max.value() / 100)
        ]
        
        self.display_results(filtered_data)
    
    def display_results(self, data):
        """Display filtered results in table"""
        if len(data) == 0:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            return
        
        # Get latest record for each stock
        latest_data = data.sort_values('Date').groupby('Stock').last().reset_index()
        
        self.results_table.setRowCount(len(latest_data))
        columns_to_show = ['Stock', 'P/E', 'EPS', 'DY', 'P/B', 'D/E', 'PEG']
        self.results_table.setColumnCount(len(columns_to_show))
        self.results_table.setHorizontalHeaderLabels(columns_to_show)
        
        for row_idx, (_, row_data) in enumerate(latest_data.iterrows()):
            for col_idx, col_name in enumerate(columns_to_show):
                value = row_data[col_name]
                item = QTableWidgetItem(f"{value:.2f}" if isinstance(value, (int, float)) else str(value))
                self.results_table.setItem(row_idx, col_idx, item)
        
        self.results_table.resizeColumnsToContents()