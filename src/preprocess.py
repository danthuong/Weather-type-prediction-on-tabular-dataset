import os
import shutil
import kagglehub

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