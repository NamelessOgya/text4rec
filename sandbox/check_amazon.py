"""
    poetry run python -m sandbox.check_amazon
"""

import pandas as pd
import json
import pickle as pk

def check_raw_data():
    # 読み込み
    df = pd.read_json("./Data/amazon/amazon.json", lines=True)

    print(df.head(30))  # dict や list になる

    df = df[['reviewerID', 'asin', 'overall','unixReviewTime']]
    print(df.head(30))

def check_preprocessed_data():
    with open("./Data/preprocessed/ml-1m_min_rating4-min_uc5-min_sc0-splitleave_one_out/dataset.pkl", "rb") as f:
        obj = pk.load(f)
    
    print(type(obj))


if __name__ == "__main__":
    check_preprocessed_data()
    