#!/usr/bin/env python3
"""
Test loading data from the DeepDialogue Parquet file.
"""

import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

def main():
    print("Testing Parquet file approach for DeepDialogue dataset...")

    # The dataset URL
    parquet_url = "https://huggingface.co/datasets/SALT-Research/DeepDialogue-orpheus/resolve/main/data/train-00000-of-00001.parquet"

    # Create temp directory
    data_dir = Path(__file__).parent.parent / "temp_data"
    data_dir.mkdir(exist_ok=True)
    parquet_path = data_dir / "train.parquet"

    # Download if needed (or use cached version)
    if not parquet_path.exists():
        print(f"Downloading Parquet file from {parquet_url}...")
        response = requests.get(parquet_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        print(f"Total size: {total_size / 1024 / 1024:.1f} MB")

        with open(parquet_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    else:
        print(f"Using cached Parquet file at {parquet_path}")

    # Load the first few rows to check structure
    print("\nLoading sample data from Parquet file...")
    df = pd.read_parquet(parquet_path, columns=None)  # Load all columns

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Number of unique conversations: {df['conversation_id'].nunique()}")

    # Show first few rows
    print("\nFirst 3 rows:")
    print(df.head(3))

    # Check unique speakers
    if 'speaker' in df.columns:
        print(f"\nUnique speakers: {df['speaker'].unique()}")

    # Show a sample of text fields
    print("\nSample text fields from first row:")
    first_row = df.iloc[0]
    for field in ['original_text', 'substituted_text', 'cleaned_text']:
        if field in df.columns:
            text = str(first_row[field])[:200]
            print(f"\n{field}: {text}...")

if __name__ == "__main__":
    main()