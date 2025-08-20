from .base import AbstractDataset

import pandas as pd

from datetime import date
from pathlib import Path
import tempfile
from .utils import *
import os
import pickle


class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'movies.dat',
                'ratings.dat',
                'users.dat']

    def load_ratings_df(self):
        """
            ratings.datの読み込み
            uid, sid, rating, timestampのカラムを持つDataFrameを返す   
            uid: ユーザID
            sid: アイテムID
            rating: レーティング
            timestamp: タイムスタンプ
            return: DataFrame with columns ['uid', 'sid', 'rating', 'timestamp']
        """
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    
    def make_implicit(self, df):
        """
            ratingがmin_rating以上のものを抽出する。
            df: DataFrame with columns ['uid', 'sid', 'rating', 'timestamp']    
            return: DataFrame with implicit ratings
        """
        print('Turning into implicit ratings')
        df = df[df['rating'] >= self.min_rating] #特定のratingより高いものを抽出。
        # return df[['uid', 'sid', 'timestamp']]
        return df
    
    def filter_triplets(self, df):
        """
            登場回数が少ないuser, itemを除外する。
            min_sc: itemの最小登場回数
            min_uc: userの最小登場回数
            df: DataFrame with columns ['uid', 'sid', 'rating', 'timestamp']
            return: Filtered DataFrame
        """
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size() #sid出現回数を数える。
            good_items = item_sizes.index[item_sizes >= self.min_sc] #一定以上登場しているものに限定
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size() #uid出現回数を数える。
            good_users = user_sizes.index[user_sizes >= self.min_uc] #一定以上登場しているものに限定
            df = df[df['uid'].isin(good_users)]

        return df
    
    def densify_index(self, df):
        """
            userのmappingとitemのmappingを作成し、indexを0から始まる連番に変換する。
            df, umap, smap
        """
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']))}
        smap = {s: i for i, s in enumerate(set(df['sid']))}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap
    
    def _download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if self.is_zipfile():
            tmproot = Path(tempfile.mkdtemp())
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            download(self.url(), tmpzip)
            unzip(tmpzip, tmpfolder)
            if self.zip_file_content_is_folder():
                tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, folder_path)
            shutil.rmtree(tmproot)
            print()
        else:
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(self.url(), tmpfile)
            folder_path.mkdir(parents=True)
            shutil.move(tmpfile, folder_path.joinpath('ratings.csv'))
            shutil.rmtree(tmproot)
            print()



