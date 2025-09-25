"""
data_manager.py
Handles data loading, validation, and management
"""

import pandas as pd
import os

class DataManager:
    def __init__(self):
        self.predefined_columns = [
            "Date", "Stock", "Open", "High", "Low", "Close", "Volume", 
            "P/E", "EPS", "Earning Growth", "MCap", "P/B", "DY", "D/E", "PEG"
        ]
    
    def load_csv(self, file_path):
        """Load and validate CSV file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the data
        df = pd.read_csv(file_path)
        
        # Basic validation
        missing_cols = set(self.predefined_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {', '.join(missing_cols)}")
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date and stock
        df = df.sort_values(['Date', 'Stock']).reset_index(drop=True)
        
        return df
    
    def validate_data(self, df):
        """Perform data quality checks"""
        issues = []
        
        # Check for missing values in critical columns
        critical_cols = ['Date', 'Stock', 'Open', 'High', 'Low', 'Close']
        for col in critical_cols:
            missing = df[col].isna().sum()
            if missing > 0:
                issues.append(f"{col}: {missing} missing values")
        
        # Check for negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            negative = (df[col] < 0).sum()
            if negative > 0:
                issues.append(f"{col}: {negative} negative values")
        
        return issues