import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"The directory is: {ROOT}")
INPUT_DIR = os.path.join(ROOT, "input_data")
print(f"\nThe directory is: {INPUT_DIR}")
print(os.path.abspath(__file__))