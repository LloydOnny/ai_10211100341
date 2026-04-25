"""
Lloyd Onny — 10211100341

Clean Ghana election CSV for downstream chunking / indexing.
"""

import os

import pandas as pd


def clean_election_data(input_path, output_path):
    df = pd.read_csv(input_path)
    # Basic cleaning: drop empty columns, fill NAs, strip whitespace
    df = df.dropna(axis=1, how='all')
    df = df.fillna('')
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(
            lambda x: x.replace("\xa0", " ").strip() if isinstance(x, str) else x
        )
    df.to_csv(output_path, index=False)
    print(f"Cleaned election data saved to {output_path}")

if __name__ == "__main__":
    input_csv = os.path.join('data', 'Ghana_Election_Result.csv')
    output_csv = os.path.join('data', 'Ghana_Election_Result_clean.csv')
    clean_election_data(input_csv, output_csv)
