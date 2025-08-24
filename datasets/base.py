from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @classmethod
    def is_zipfile(cls):
        return True

    @classmethod
    def is_gzfile(cls):
        return False

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @abstractmethod
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
        pass

    def load_dataset(self):
        """
        Load the preprocessed dataset.
        """
        self.preprocess()
        if self.args.generate_item_embeddings:
            self.generate_and_save_item_embeddings()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def preprocess(self):
        pass
        
    def generate_and_save_item_embeddings(self):
        print('Generating and saving item embeddings...')
        dataset_path = self._get_preprocessed_dataset_path()
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        if 'item_text' not in dataset:
            print('item_text not found in dataset. Skipping embedding generation.')
            return

        item_text_dict = dataset['item_text']
        smap = dataset['smap']
        num_items = max(smap.values()) + 1

        texts_to_embed = []
        placeholder_text = ""
        for i in range(num_items):
            texts_to_embed.append(item_text_dict.get(i, placeholder_text))

        device = torch.device(self.args.device)
        tokenizer = AutoTokenizer.from_pretrained(self.args.embedding_model_name)
        model = AutoModel.from_pretrained(self.args.embedding_model_name).to(device)

        embeddings = get_e5_embedding(texts_to_embed, model, tokenizer, device, self.args.embedding_batch_size)

        embedding_path = self._get_preprocessed_folder_path().joinpath('item_embeddings.npy')
        np.save(embedding_path, embeddings)
        print(f'Embeddings saved to {embedding_path}')


    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and \
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            
            print('Raw data already exists. Skip downloading')  
            return
        print("Raw file doesn't exist. Downloading...")
        self._download_raw_dataset()
    
    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        df = self.make_implicit(df) #特定のratingより高いものを抽出。
        df = self.filter_triplets(df) #user, item出現回数が高いものに限定
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

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

    def split_df(self, df, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
            train, val, test = {}, {}, {}
            for user in range(user_count):
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        elif self.args.split == 'holdout':
            print('Splitting')
            np.random.seed(self.args.dataset_split_seed)
            eval_set_size = self.args.eval_set_size

            # Generate user indices
            permuted_index = np.random.permutation(user_count)
            train_user_index = permuted_index[                :-2*eval_set_size]
            val_user_index   = permuted_index[-2*eval_set_size:  -eval_set_size]
            test_user_index  = permuted_index[  -eval_set_size:                ]

            # Split DataFrames
            train_df = df.loc[df['uid'].isin(train_user_index)]
            val_df   = df.loc[df['uid'].isin(val_user_index)]
            test_df  = df.loc[df['uid'].isin(test_user_index)]

            # DataFrame to dict => {uid : list of sid's}
            train = dict(train_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            val   = dict(val_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            test  = dict(test_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            return train, val, test
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')

    def _download_raw_dataset(self):
        """
        maybe_download_raw_dataset() でダウンロードが必要になった際の挙動は
        各Datasetクラスで実装する。
        """
        pass
