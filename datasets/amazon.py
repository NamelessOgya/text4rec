from .base import AbstractDataset

import pandas as pd

from datetime import date
from pathlib import Path
import tempfile
from .utils import *
import os

class AmazonDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'amazon'

    @classmethod
    def url(cls):
        # 例: Digital Music
        return 'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION_5.json.gz'

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
        # 例: Digital_Music.json
        return ['amazon.json']

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
        dst = folder_path / "amazon.json"

        with tempfile.TemporaryDirectory() as td:
            tmproot = Path(td)
            tmpzip = tmproot / "file.gz"
            tmpfile = tmproot / "amazon.json"

            # ダウンロード & 解凍
            download(self.url(), tmpzip)
            ungzip(tmpzip, tmpfile)   # 出力はファイルを期待

            # 既存があれば上書き
            dst.unlink(missing_ok=True)
            shutil.move(str(tmpfile), str(dst))

        print("Downloaded raw dataset to:", dst)