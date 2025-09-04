import pickle
import pandas as pd
import os

# プロジェクトのルートディレクトリからの相対パス
relative_path = 'Data/preprocessed/amazon_min_rating4-min_uc5-min_sc0-splitleave_one_out/dataset.pkl'

# 絶対パスに変換
# スクリプトがプロジェクトルートから実行されることを想定
dataset_path = os.path.abspath(relative_path)

try:
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    if 'item_text' in data:
        item_text_dict = data['item_text']
        
        print(f"Total number of items with text: {len(item_text_dict)}")
        
        # すべてのテキストをSeriesに変換して調査
        text_series = pd.Series(list(item_text_dict.values()))
        
        # ユニークなテキストの数を確認
        num_unique_texts = text_series.nunique()
        print(f"Number of unique texts: {num_unique_texts}")

        if num_unique_texts == 1:
            print("Warning: All item texts are identical.")
        elif len(item_text_dict) > 0 and num_unique_texts < len(item_text_dict) / 2:
            print("Warning: There are many duplicate texts.")

        # 最初の5件のテキストを表示して目視確認
        print("\n--- First 5 item texts ---")
        for i, text in enumerate(text_series.head(5)):
            print(f"Item {i+1}: {str(text)[:300]}...")
        
        if len(text_series) > 5:
            print("\n--- Random 5 item texts ---")
            random_samples = text_series.sample(5)
            for i, text in enumerate(random_samples):
                print(f"Item {i+1}: {str(text)[:300]}...")

    else:
        print("Error: 'item_text' not found in the dataset.")

except FileNotFoundError:
    print(f"Error: Dataset file not found at {dataset_path}")
    print("Please make sure the preprocessing has been run.")
except Exception as e:
    print(f"An error occurred: {e}")
