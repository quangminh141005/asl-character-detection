import pandas as pd
from src.config import CSV_FILE

def load_data(csv_file: str = CSV_FILE) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    feature_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    print(f"Loaded {len(df)} rows, {df['label'].nunique()} labels, "
          f"{df['video_id'].nunique()} unique videos.")
    
    return df

def get