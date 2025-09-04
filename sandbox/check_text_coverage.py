import pandas as pd
import os
import sys

# 親ディレクトリをシステムパスに追加して、datasetsモジュールをインポートできるようにする
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.amazon import AmazonDataset
from types import SimpleNamespace

print("Analyzing item text coverage...")

# AmazonDatasetのインスタンスを作成するために、ダミーのargsオブジェクトを作成
# このスクリプトではmin_ratingなどは影響しない
dummy_args = SimpleNamespace(
    min_rating=0, min_uc=0, min_sc=0, split='leave_one_out'
)
dataset_instance = AmazonDataset(dummy_args)

# --- 1. メタデータファイルの分析 ---
try:
    print("\n--- 1. Analyzing metadata file (metadata.json) ---")
    folder_path = dataset_instance._get_rawdata_folder_path()
    file_path = folder_path.joinpath('metadata.json')
    meta_df = pd.read_json(file_path, lines=True)
    
    total_meta_items = meta_df['asin'].nunique()
    print(f"Total unique items in metadata: {total_meta_items}")

    # テキストを持つアイテムを処理
    text_df = dataset_instance.load_item_text_df()
    items_with_text = len(text_df)
    print(f"Items with valid text (after processing): {items_with_text}")

    items_without_text = total_meta_items - items_with_text
    percentage_without_text = (items_without_text / total_meta_items) * 100 if total_meta_items > 0 else 0
    print(f"Items excluded due to missing/empty text: {items_without_text} ({percentage_without_text:.2f}%)")

except FileNotFoundError:
    print("Error: metadata.json not found. Please run the download/preprocessing first.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during metadata analysis: {e}")
    sys.exit(1)


# --- 2. 評価データとの関連を分析 ---
try:
    print("\n--- 2. Analyzing interaction data (amazon.json) ---")
    ratings_df = dataset_instance.load_ratings_df()
    total_rating_items = ratings_df['sid'].nunique()
    print(f"Total unique items in ratings data: {total_rating_items}")

    # 評価データに含まれるアイテムのうち、テキストを持つもの
    items_in_ratings_with_text = ratings_df['sid'].isin(text_df['sid']).sum()
    unique_items_in_ratings_with_text = ratings_df[ratings_df['sid'].isin(text_df['sid'])]['sid'].nunique()
    
    print(f"Unique items in ratings data that also have text: {unique_items_in_ratings_with_text}")

    # 評価データから除外されるアイテム数
    excluded_from_ratings = total_rating_items - unique_items_in_ratings_with_text
    percentage_excluded_from_ratings = (excluded_from_ratings / total_rating_items) * 100 if total_rating_items > 0 else 0
    print(f"Items in ratings data excluded due to no text: {excluded_from_ratings} ({percentage_excluded_from_ratings:.2f}%)")

    print("\n--- Summary ---")
    print(f"Out of {total_rating_items} items that users interacted with, {unique_items_in_ratings_with_text} items will be used for training/evaluation because they have valid text data.")

except FileNotFoundError:
    print("Error: amazon.json not found. Please run the download/preprocessing first.")
except Exception as e:
    print(f"An error occurred during rating analysis: {e}")
