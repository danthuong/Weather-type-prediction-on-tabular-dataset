import os
import shutil
import kagglehub
import pandas as pd

def download_dataset(dataset_name, target_dir):
    """
    Download an entire Kaggle dataset and store it into a specific folder.
    Uses kagglehub (no API token required).

    Parameters:
        dataset_name (str): e.g. 'zynicide/wine-reviews'
        target_dir (str): e.g. './data/wine_reviews'

    Returns:
        str: Path to dataset in target_dir
    """
    source_dir = kagglehub.dataset_download(dataset_name)

    os.makedirs(target_dir, exist_ok=True)

    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    print(f"Dataset saved to: {os.path.abspath(target_dir)}")
    return os.path.abspath(target_dir)

def get_missings(df, columns):
    """
    Hàm trả về dataframe chứa số lượng giá trị null của các cột được chỉ định trong danh sách columns
    Args:
        df (pd.DataFrame): DataFrame cần kiểm tra giá trị null
        columns (list): Danh sách tên các cột cần kiểm tra giá trị null
    Returns:
        pd.DataFrame: DataFrame chứa số lượng giá trị null của các cột được chỉ định
    """
    dict_nulls = {}
    for col in columns:
        dict_nulls[col] = df[col].isnull().sum()

    df_nulls = pd.DataFrame(
        data=list(dict_nulls.values()),
        index=list(dict_nulls.keys()),
        columns=["MissingNumber"],
    )
    return df_nulls


def get_missings_percentage(df, columns):
    """
    Hàm trả về dataframe chứa phần trăm giá trị null của các cột được chỉ định trong danh sách columns
    Args:
        df (pd.DataFrame): DataFrame cần kiểm tra giá trị null
        columns (list): Danh sách tên các cột cần kiểm tra giá trị null
    Returns:
        pd.DataFrame: DataFrame chứa phần trăm giá trị null của các cột được chỉ định
    """
    dict_nulls = {}
    for col in df.columns:
        percentage_null_values = str(round(df[col].isnull().sum() / len(df), 2)) + "%"
        dict_nulls[col] = percentage_null_values

    df_nulls = pd.DataFrame(
        data=list(dict_nulls.values()),
        index=list(dict_nulls.keys()),
        columns=["%Missing"],
    )
    return df_nulls