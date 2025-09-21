# build_docs.py
import os
import subprocess

def update_documentation():
    # Run sphinx-apidoc to generate stub files for new modules
    subprocess.run([
        'sphinx-apidoc', 
        '-o', 'docs/generated', 
        '../pages',
        '--separate'  # Create separate pages for each module
    ])
    
    # Build the documentation
    subprocess.run(['sphinx-build', '-b', 'html', 'docs', 'docs/_build'])

if __name__ == '__main__':
    update_documentation()