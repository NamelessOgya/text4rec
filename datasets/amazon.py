from .base import AbstractDataset
import pickle
import shutil

import pandas as pd

from datetime import date
from pathlib import Path
import tempfile
from .utils import *
import os

class AmazonDataset(AbstractDataset):
    def __init__(self, args):
        super().__init__(args)
        self.amazon_url = args.amazon_url
        self.amazon_metadata_url = args.amazon_metadata_url

    @classmethod
    def code(cls):
        return 'amazon'

    def url(self):
        return self.amazon_url

    def metadata_url(self):
        return self.amazon_metadata_url

    @classmethod
    def is_zipfile(cls):
        return False
    
    @classmethod
    def is_gzfile(cls):
        return True

    @classmethod
    def zip_file_content_is_folder(cls):
        return False

    @classmethod
    def all_raw_file_names(cls):
        # 例: Digital_Music.json と metadata.json
        return ['amazon.json', 'metadata.json']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('amazon.json')
        df = pd.read_json(file_path, lines=True)
        # 必要なカラムだけ抽出
        df = df[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
        df = df.rename(columns={
            'reviewerID': 'uid',
            'asin': 'sid',
            'overall': 'rating',
            'unixReviewTime': 'timestamp'
        })
        # timestampをint型に変換
        df['timestamp'] = df['timestamp'].astype(int)
        return df
    
    # ダウンロード部分をオーバーライドする。
    def _download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # レビューファイルをダウンロード
        ratings_dst = folder_path / "amazon.json"
        print("Downloading ratings file...")
        with tempfile.TemporaryDirectory() as td:
            tmproot = Path(td)
            tmpzip = tmproot / "file.gz"
            tmpfile = tmproot / "amazon.json"
            # ダウンロード & 解凍
            download(self.url(), tmpzip)
            ungzip(tmpzip, tmpfile)   # 出力はファイルを期待
            # 既存があれば上書き
            ratings_dst.unlink(missing_ok=True)
            shutil.move(str(tmpfile), str(ratings_dst))
        print("Downloaded raw ratings dataset to:", ratings_dst)

        # メタデータファイルをダウンロード
        metadata_dst = folder_path / "metadata.json"
        print("Downloading metadata file...")
        with tempfile.TemporaryDirectory() as td:
            tmproot = Path(td)
            tmpzip = tmproot / "file.gz"
            tmpfile = tmproot / "metadata.json"
            # ダウンロード & 解凍
            download(self.metadata_url(), tmpzip)
            ungzip(tmpzip, tmpfile)   # 出力はファイルを期待
            # 既存があれば上書き
            metadata_dst.unlink(missing_ok=True)
            shutil.move(str(tmpfile), str(metadata_dst))
        print("Downloaded raw metadata dataset to:", metadata_dst)

    def make_rowwise_text_col(self, series: pd.Series):
        text = ""
        text += "title\n"
        text += series["title"] if pd.notnull(series["title"]) else "NULL"

        text += "\nbrand\n"
        text += series["brand"] if pd.notnull(series["brand"]) else "NULL"

        text += "\ndescription\n"
        text += str(series["description"])
        
        if isinstance(series["description"], list):
            # descriptionはtextで与えられるので、改行を挟んで文字列に起こす。
            for i in series["description"]:
                text += str(i)
                text += "\n"
        elif pd.notnull(series["description"]):
            text += "NULL"
        else:
            text += str(series["description"])


        return text

    def load_item_text_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('metadata.json')
        df = pd.read_json(file_path, lines=True)
        
        # 必要なカラムだけ抽出
        df = df[['asin', 'title', 'brand', 'description']]
        df = df.rename(columns={'asin': 'sid'})
        
        # descriptionがリストの場合、最初の要素を取得。それ以外はそのまま。
        print("combine text data...")
        df['text'] = df.apply(self.make_rowwise_text_col,axis = 1)
        # テキストが空、またはNaNの行を削除
        df = df.dropna(subset=['text'])
        df = df[df['text'] != '']
        # df = df(("asin", "text"))

        return df

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
            
        self.maybe_download_raw_dataset()
        
        # 評価データとテキストデータを読み込む
        ratings_df = self.load_ratings_df()
        text_df = self.load_item_text_df()
        
        # テキスト情報を持つアイテムの評価のみに絞り込む
        print(f"### debug: ratings len before join {len(ratings_df)}")
        ratings_df = ratings_df[ratings_df['sid'].isin(text_df['sid'])]
        print(f"### debug: ratings len after join {len(ratings_df)}")

        # base.pyと同様の前処理を実行
        df = self.make_implicit(ratings_df)
        df = self.filter_triplets(df)
        df, umap, smap, smap_r = self.densify_index(df)
        print(f"### debug: ratings len after filtering {len(df)}")
        
        # テキストデータのsidも新しいindexに変換
        text_df['sid'] = text_df['sid'].map(smap)
        # フィルタリングで除外されたアイテムをテキストデータからも削除
        print(f"### debug: text len before filtering {len(text_df)}")
        text_df = text_df.dropna(subset=['sid'])
        print(f"### debug: text len after filtering {len(text_df)}")

        text_df['sid'] = text_df['sid'].astype(int)
        
        # 新しいsidをキーにしたテキストの辞書を作成
        item_text = pd.Series(text_df.text.values, index=text_df.sid).to_dict()

        train, val, test = self.split_df(df, len(umap))
        
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap,
                   'smap_r': smap_r,
                   'item_text': item_text} # item_text をデータセットに追加
                   
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)
