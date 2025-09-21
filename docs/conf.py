# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# -- Path Setup --------------------------------------------------------------

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath('..'))        # Project root
sys.path.insert(0, os.path.abspath('../pages'))  # Pages directory
sys.path.insert(0, os.path.abspath('../src'))    # If you have src code
sys.path.insert(0, os.path.abspath('../support_files'))  # Support files


# -- Project information -----------------------------------------------------

project = 'Trading Workbench'
copyright = f'{datetime.now().year}, Iqbal'
author = 'Iqbal'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

# Add essential extensions
extensions = [
    'sphinx.ext.autodoc',        # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',       # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',       # Add links to highlighted source code
    'sphinx.ext.githubpages',    # Enable GitHub Pages integration
    'sphinx.ext.autosummary',    # Generate automatic summaries
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_mock_imports = ["pandas", "numpy", "matplotlib", "ccxt", "ta", "yfinance", "streamlit","glob"]  # Add any heavy dependencies here

# Templates path
templates_path = ['_templates']

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'env', 'venv', 'workbenchenv']

# -- Options for HTML output -------------------------------------------------

# Use a more modern theme (requires installing sphinx_rtd_theme)
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Static files path
html_static_path = ['_static']

# Custom CSS
html_css_files = [
    'custom.css',
]

# Logo
html_logo = '../assets/logo.png' if os.path.exists('../assets/logo.png') else None

# Favicon
html_favicon = '../assets/favicon.ico' if os.path.exists('../assets/favicon.ico') else None

# -- Extension configuration -------------------------------------------------

# Auto-summary generation
autosummary_generate = True

# -- Custom setup ------------------------------------------------------------

def skip_member(app, what, name, obj, skip, options):
    """Skip documenting certain members."""
    # Skip private methods (starting with _) unless specifically included
    if name.startswith('_') and name not in ['__init__']:
        return True
    return skip

def setup(app):
    app.connect('autodoc-skip-member', skip_member)
    
    # Add custom CSS
    app.add_css_file('custom.css')