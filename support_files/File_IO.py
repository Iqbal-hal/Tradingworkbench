import pandas as pd
from datetime import datetime
import os
from pathlib import Path

current_dir = os.getcwd()
current_datetime = datetime.now()
time_now = current_datetime.strftime('%d%m%Y%H%M%S')


def filename_formatter(filename, _order):    
    _filename=os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    
    order = True
    if _order == 'A':
        order = True
        _filename = _filename + '_ASC_'
    elif _order == 'D':
        order = False
        _filename = _filename + '_DSC_'
    filename = _filename + time_now + ext
    return filename, order


def convert_order(_order):
    order = True
    if _order == 'A':
        order = True
    elif _order == 'D':
        order = False
    return order


# Copy downloaded data in the csv format to working directory
def save_df_to_csv(_df, filename, _order,target_dir):
    #file name with extension .csv
    #order 'A' for Ascending 'D' for descending
    filename, order = filename_formatter(filename, _order)

    # Robust sort: if 'Date' is a column sort by ['Stock','Date'] otherwise sort by index then by Stock (stable)
    try:
        if 'Date' in _df.columns:
            _df = _df.sort_values(by=['Stock', 'Date'], ascending=[True, order], kind='mergesort')
        else:
            _df = _df.sort_index(ascending=order)
            if 'Stock' in _df.columns:
                _df = _df.sort_values(by=['Stock'], kind='mergesort')
    except Exception:
        pass

    try:
        # Create target_dir if it doesn't exist and write using full path (avoid chdir)
        cd = os.getcwd()
        full_target_dir = os.path.join(cd, target_dir)
        os.makedirs(full_target_dir, exist_ok=True)
        out_path = os.path.join(full_target_dir, filename)
        _df.to_csv(out_path, float_format='%.5f')
        print(f"File saved successfully as {out_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Copy downloaded data in the .pkl format to working directory
def save_df_to_pkl(_df, filename, _order):
    #file name with extension .pkl
    #order 'A' for Ascending 'D' for descending
    filename, order = filename_formatter(filename, _order)
    _df = _df.sort_values(by=['Stock', 'Date'], ascending=[True, order])

    try:
        cd = os.getcwd()
        sd = os.path.join(cd, 'sub_dir')
        os.chdir(sd)
        _df.to_pickle(filename)
        print(f"File saved successfully as {filename}")
        os.chdir(cd)
    except Exception as e:
        print(f"An error occurred: {e}")


def read_csv_to_df(filename, _order,source_dir,dayfirst=True):
    # make 'files' as the current working directory
    # read  *.csv/sub_dir
    cd = os.getcwd()
    sd = os.path.join(cd, source_dir)
    os.chdir(sd)

    order = convert_order(_order)
    _df = pd.read_csv(filename)
    # Safe column normalization: trim whitespace and remove BOM/zero-width spaces
    try:
        def _clean_col(c):
            s = str(c)
            # remove BOM, zero-width spaces, and trim
            return (
                s.replace('\ufeff', '').replace('\u200b', '').strip()
            )
        _df.columns = [_clean_col(c) for c in _df.columns]
    except Exception:
        pass
    # Normalize Date to datetime (day-first by default) and ensure it's a column, not index
    if 'Date' in _df.columns:
        _df['Date'] = pd.to_datetime(_df['Date'], dayfirst=dayfirst, errors='coerce')

    # If Date somehow got set as index upstream, reset it back to a normal column
    try:
        if _df.index.name == 'Date':
            _df.reset_index(inplace=True)
    except Exception:
        pass

    # Final guarantee: if Date exists, ensure dtype is datetime
    if 'Date' in _df.columns:
        _df['Date'] = pd.to_datetime(_df['Date'], errors='coerce')

    # Stable sort: preserve natural row order; sort by Stock if present
    try:
        if 'Stock' in _df.columns and 'Date' in _df.columns:
            _df = _df.sort_values(by=['Stock', 'Date'], ascending=[True, order], kind='mergesort')
        elif 'Stock' in _df.columns:
            _df = _df.sort_values(by=['Stock'], kind='mergesort')
    except Exception:
        pass

    os.chdir(cd)

    return _df


def read_pkl_to_df(filename, _order):
    order = convert_order(_order)
    _df = pd.read_pickle(filename)
    if 'Date' in _df.columns:
        _df['Date'] = pd.to_datetime(_df['Date'])
        _df.set_index('Date', inplace=True)  # set Date column as index
    _df.sort_index(axis=0, inplace=True)  # index based sorting compulsory for slicing
    if 'Stock' in _df.columns:
        _df = _df.sort_values(by=['Stock'], kind='mergesort')
    return _df


def convert_csv_to_pkl(filename_csv, filename_pkl, _order):
    df = read_csv_to_df(filename_csv, _order)
    save_df_to_pkl(df, filename_pkl, _order)


def convert_pkl_to_csv(filename_csv, filename_pkl, _order):
    df = read_pkl_to_df(filename_pkl, _order)
    save_df_to_csv(df, filename_csv, _order)


def change_cwd(sd):
    """
    Change working directory to the given subdirectory name/path.
    Behavior:
      - If `sd` is an absolute path and exists, chdir to it.
      - Else try to chdir relative to the current working directory.
      - Else fall back to chdir relative to this package directory (recommended for project-local folders).
    """
    # 1) absolute path
    try:
        if os.path.isabs(sd) and os.path.isdir(sd):
            os.chdir(sd)
            return
    except Exception:
        pass

    # 2) try relative to current working directory
    try:
        os.chdir(sd)
        return
    except Exception:
        pass

    # 3) fallback: resolve relative to this module's package directory
    pkg_dir = Path(__file__).resolve().parent
    target = pkg_dir / sd
    if target.exists():
        os.chdir(str(target))
        return

    # final attempt: try to create the directory under package and chdir into it
    try:
        target.mkdir(parents=True, exist_ok=True)
        os.chdir(str(target))
        return
    except Exception as e:
        # Let the original exception surface if we cannot change directory
        raise


def get_cwd():
    os.chdir(current_dir)
    # print("Changed to Current directory:", os.getcwd())


# This code is exclusively made for reading from .csv file and format ,filter,sort etc
def df_slicer(_df, _order,start_date,end_date):
    order = convert_order(_order)    
    _df = _df.loc[:,:]
    # index based sorting compulsory for slicing
    _df.sort_index(axis=0, inplace=True)
    _df = _df.loc[start_date:end_date]
    # Stable sort by Stock while preserving date order in the index
    if 'Stock' in _df.columns:
        _df = _df.sort_values(by=['Stock'], kind='mergesort')
    return _df
