# TradingWorkbench - CSV Selection Module

## Overview

The `step1_select_input.py` module is the first step in the TradingWorkbench application pipeline, providing a comprehensive interface for selecting, uploading, and validating CSV files containing OHLC (Open, High, Low, Close) financial data with volume information.

## Features

### File Management
- Lists CSV files from multiple directories
- Handles file uploads with automatic naming and duplicate prevention
- Provides comprehensive file information (size, modification dates, row count)

### Data Validation
**Header Validation:**
- Checks for required columns (date, open, high, low, close, volume)
- Detects column name variations using flexible alias matching
- Provides detailed error messages

**Data Quality Validation:**
- Checks for missing values
- Validates numeric data types
- Detects negative values where inappropriate
- Identifies duplicate records
- Validates date parsing and sorting
- Handles large files with sampling

### User Experience
- Collapsible sections for better organization
- Visual feedback with icons and colors
- Example data format display
- File history tracking
- Responsive layout design

### Error Handling
- Comprehensive error handling with informative messages
- Graceful degradation for partial failures
- Large file handling with sampling to avoid memory issues

## Installation & Setup

This module is part of the TradingWorkbench Streamlit application. Ensure you have the required dependencies:

```bash
pip install streamlit pandas
```

## Usage

### Basic Usage
1. Launch the TradingWorkbench application
2. Navigate to "1 — Select OHLC CSV" in the sidebar
3. Select a file from the dropdown or upload a new CSV file
4. Review the validation results
5. Proceed with valid files or force proceed with issues

### File Format Requirements
Your CSV should include these columns (case insensitive):
- **Date**: Date/time of each observation
- **Open**: Opening price
- **High**: Highest price during the period
- **Low**: Lowest price during the period
- **Close**: Closing price
- **Volume**: Trading volume

Optional columns:
- **Ticker/Symbol**: Security identifier (if multiple securities in file)

### Example Data Format
| date       | open  | high  | low   | close | volume  | ticker |
|------------|-------|-------|-------|-------|---------|--------|
| 2023-01-01 | 100.0 | 102.0 | 99.5  | 101.5 | 1000000 | AAPL   |
| 2023-01-02 | 101.5 | 103.0 | 100.5 | 102.0 | 1200000 | AAPL   |
| 2023-01-03 | 102.3 | 104.5 | 101.0 | 103.5 | 950000  | AAPL   |

## Module Structure

### Configuration Constants
```python
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT, "input_data")
UPLOADS_DIR = os.path.join(INPUT_DIR, "uploads")
LARGE_FILE_MB_WARN = 50
REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
TICKER_CANDIDATES = {"ticker", "symbol", "scrip", "security", "name"}
```

### Key Functions

#### File Management
- `ensure_dirs()`: Creates necessary directories if they don't exist
- `list_csv_files()`: Finds and returns all CSV files in input directories
- `save_upload()`: Handles file uploads with duplicate prevention
- `get_file_info()`: Retrieves comprehensive file metadata

#### Validation
- `detect_mapping()`: Creates mapping between standardized and actual column names
- `header_check()`: Validates CSV headers and checks for required columns
- `enhanced_deep_validation()`: Performs comprehensive data quality checks

#### Utilities
- `human_size()`: Converts byte sizes to human-readable format
- `sanitize_filename()`: Removes special characters from filenames

### UI Components
The module creates a comprehensive user interface with:
- File selection dropdown
- File upload capability
- File information display
- Data preview
- Validation results with expandable sections
- Action buttons with conditional logic

## Integration

This module integrates with the broader TradingWorkbench application through:

1. **Session State**: Stores selected file information for subsequent steps
2. **File System**: Reads from and writes to designated directories
3. **Validation Results**: Provides comprehensive validation data for downstream processing

## Error Handling

The module implements comprehensive error handling with:
- Try-catch blocks around file operations
- Detailed error messages for different failure modes
- Graceful fallbacks for partial failures
- Large file sampling to avoid memory issues

## Extension Points

The module can be extended by:
1. Adding new column aliases to the detection system
2. Implementing additional validation checks
3. Supporting additional file formats
4. Adding more file metadata display
5. Enhancing the user interface with additional visualizations

## Technical Notes

- Uses pandas for data manipulation and validation
- Implements efficient file operations with proper resource management
- Includes type hints for better code documentation and IDE support
- Follows modular design principles with single-responsibility functions

## License

This module is part of the TradingWorkbench application. Please refer to the main project for licensing information.

## Support

For issues related to this module, please check:
1. File format compliance with the requirements
2. File permissions in the input_data directory
3. Available memory for large file handling

---

*This document was generated based on the implementation of `step1_select_input.py` in the TradingWorkbench application.*