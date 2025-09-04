"""
    poetry run python -m sandbox.analyze_dataset
    {'train': train,
    'val': val,
    'test': test,
    'umap': umap,
    'smap': smap,
    'smap_r': smap_r,
    'item_text': item_text} 
"""


import pickle
from datasets.amazon import AmazonDataset

path = "./Data/preprocessed/amazon_min_rating4-min_uc5-min_sc0-splitleave_one_out/dataset.pkl"

def main():
    with open(path, "rb") as f:
        dataset = pickle.load(f)

    print(type(dataset["item_text"]))
    print(len(dataset["item_text"]))
    

if __name__ == '__main__':
    main()
