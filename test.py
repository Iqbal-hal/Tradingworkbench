import os
import glob

#ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#print(f"The directory is: {ROOT}")
#INPUT_DIR = os.path.join(ROOT, "input_data")
#print(f"\nThe directory is: {INPUT_DIR}")
#print(os.path.abspath(__file__))

folder_path = os.path.dirname(os.path.abspath(__file__))
print(f"The directory is: {folder_path}")
INPUT_DATA = os.path.join(folder_path, "input_data")
print(f"\n'input_data' relative path: {INPUT_DATA}")
pattern = os.path.join(INPUT_DATA, "*.csv")
print(f"\nThe pattern is: {pattern}")
files = glob.glob(pattern)
print(f"\nThe files are: {files}")
for file in files:
    print(f"\nThe file is: {file}")

