"""
    poetry run python -m sandbox.test_ml1m_dataset
"""

from options import args
from datasets import dataset_factory

if __name__ == '__main__':

    args.dataset_code = 'ml-1m'  # Set the dataset code to 'amazon'
    dataset = dataset_factory(args)

    print(f"dataset code: {dataset.code()}")
    print(f"dataset url: {dataset.url()}")
    print(f"dataset file name: {dataset.all_raw_file_names()}")

    print("downloading...")
    dataset.maybe_download_raw_dataset()
    dataset.preprocess()

    # print("load df...")
    # ratings_df = dataset.load_ratings_df()
    # print(ratings_df.head())