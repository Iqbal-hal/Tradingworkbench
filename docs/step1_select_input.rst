Step 1: CSV Input Selection
===========================

Overview
--------

This module handles CSV file selection and validation for OHLC trading data in the TradingWorkbench application. It provides functionality for:

- Listing available CSV files
- Validating CSV structure and content
- Detecting column mappings for OHLC data
- Sanitizing filenames and handling uploads

Main Functions
--------------

.. autofunction:: step1_select_input.render

   The main rendering function that creates the Streamlit interface for CSV selection.

Validation Functions
--------------------

.. autofunction:: step1_select_input.header_check

   Validates CSV headers against expected OHLC patterns.

.. autofunction:: step1_select_input.enhanced_deep_validation

   Performs comprehensive validation of CSV content and structure.

.. autofunction:: step1_select_input.detect_mapping

   Detects and maps CSV columns to standard OHLCV (Open, High, Low, Close, Volume) format.

Utility Functions
-----------------

.. autofunction:: step1_select_input.ensure_dirs

   Ensures required directories exist, creating them if necessary.

.. autofunction:: step1_select_input.list_csv_files

   Lists all CSV files in the designated directory.

.. autofunction:: step1_select_input.human_size

   Converts file size in bytes to a human-readable format.

.. autofunction:: step1_select_input.sanitize_filename

   Sanitizes filenames to remove potentially problematic characters.

.. autofunction:: step1_select_input.save_upload

   Handles file uploads and saves them to the appropriate location.

.. autofunction:: step1_select_input.get_file_info

   Retrieves metadata and information about CSV files.

Module Reference
----------------

For a complete technical reference of all module components:

.. automodule:: step1_select_input
   :members:
   :show-inheritance:
   :undoc-members:
   :no-index:  # This prevents duplicate index entries