# --- Standard library ---
from pathlib import Path   # robust path handling (Windows/Linux/Mac)
import shutil              # move/copy files if we want later
import os                  # read env vars (for Kaggle credentials)

# --- Third-party ---
# pip install kagglehub
import kagglehub           # lightweight helper that fetches Kaggle datasets locally

def main():
    """
    Downloads the Indian currency notes dataset from Kaggle and tells you where it lives.
    We *won't* rearrange/split anything yet—just download and confirm the path.
    """
    # 1) Which dataset? (owner/dataset-slug on Kaggle)
    dataset_id = "gauravsahani/indian-currency-notes-classifier"

    # 2) Download (kagglehub caches; subsequent runs reuse local copy)
    path = kagglehub.dataset_download(dataset_id)

    # 3) Print the path so you can inspect it
    print("Path to dataset files:", path)

    # 4) (Optional, next steps) You could mirror it into our repo's data/raw:
    # raw_dir = Path("data/raw")
    # raw_dir.mkdir(parents=True, exist_ok=True)
    # shutil.copytree(path, raw_dir / "indian_currency_notes", dirs_exist_ok=True)
    # print("Copied into:", raw_dir / "indian_currency_notes")

if __name__ == "__main__":
    main()
