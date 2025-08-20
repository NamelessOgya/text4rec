"""
    poetry run python -m sandbox.check_movielens
"""

import pandas as pd
import pickle as pk

if __name__ == "__main__":
    with open("./Data/preprocessed/ml-1m_min_rating4-min_uc5-min_sc0-splitleave_one_out/dataset.pkl", "rb") as f:
        obj = pk.load(f)
    
    print(type(obj))
    