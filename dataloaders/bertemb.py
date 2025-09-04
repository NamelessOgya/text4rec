from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory

import torch
import torch.utils.data as data_utils
import numpy as np


class BertEmbeddingDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.item_embeddings = torch.from_numpy(np.load(args.item_embedding_path)).float()
        self.embedding_dim = self.item_embeddings.shape[1]
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        code = args.train_negative_sampler_code
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          self.save_folder)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_sampler = test_negative_sampler

    @classmethod
    def code(cls):
        return 'bert_embedding'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertEmbeddingTrainDataset(self.train, self.max_len, self.mask_prob, self.item_count, self.rng, self.item_embeddings, self.embedding_dim)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        dataset = BertEmbeddingEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_sampler, self.item_embeddings, self.embedding_dim)
        return dataset


class BertEmbeddingTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, num_items, rng, item_embeddings, embedding_dim):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob # Note: this will be used to decide IF we mask, not WHERE
        self.num_items = num_items
        self.rng = rng
        self.item_embeddings = item_embeddings
        self.embedding_dim = embedding_dim
        self.padding_embedding = torch.zeros(1, self.embedding_dim)
        self.mask_embedding = torch.zeros(1, self.embedding_dim) # A specific embedding for the MASK token
        self.embeddings_with_padding = torch.cat((self.padding_embedding, self.item_embeddings), 0)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]

        # Ensure sequence is long enough to mask
        if len(seq) < 2:
            # If sequence is too short, we can't create a meaningful triplet.
            # A robust way is to skip, but for simplicity, we'll duplicate items.
            while len(seq) < 2:
                seq.extend(seq)

        # 1. Choose a position to mask (anywhere except the last item)
        mask_idx = self.rng.randint(0, len(seq) - 1)
        
        # 2. Get the positive item ID and its embedding
        positive_id = seq[mask_idx]
        positive_embedding = self.embeddings_with_padding[positive_id]

        # 3. Sample a negative item ID and get its embedding
        negative_id = self.rng.randint(1, self.num_items)
        while negative_id == positive_id:
            negative_id = self.rng.randint(1, self.num_items)
        negative_embedding = self.embeddings_with_padding[negative_id]

        # 4. Create the masked input sequence of embeddings
        input_embeddings = []
        for i, item_id in enumerate(seq):
            embedding = self.mask_embedding if i == mask_idx else self.embeddings_with_padding[item_id].unsqueeze(0)
            input_embeddings.append(embedding)

        # 5. Pad the sequence of embeddings
        input_embeddings = input_embeddings[-self.max_len:]
        padding_len = self.max_len - len(input_embeddings)
        input_embeddings = [self.padding_embedding] * padding_len + input_embeddings
        
        input_tensor = torch.cat(input_embeddings, 0)
        
        # The mask position needs to be adjusted for padding
        final_mask_idx = mask_idx + padding_len

        return input_tensor, positive_embedding, negative_embedding, torch.LongTensor([final_mask_idx])

    def _getseq(self, user):
        return self.u2seq[user]



class BertEmbeddingEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_sampler, item_embeddings, embedding_dim):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_sampler = negative_sampler
        self.item_embeddings = item_embeddings
        self.embedding_dim = embedding_dim
        self.padding_embedding = torch.zeros(1, self.embedding_dim)
        self.mask_embedding = torch.zeros(1, self.embedding_dim)
        self.embeddings_with_padding = torch.cat((self.padding_embedding, self.item_embeddings), 0)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq_ids = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_sampler.sample_for_user(user)

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq_ids = seq_ids + [self.mask_token]
        
        seq_embeddings = []
        for item_id in seq_ids:
            if item_id == self.mask_token:
                seq_embeddings.append(self.mask_embedding)
            else:
                seq_embeddings.append(self.embeddings_with_padding[item_id].unsqueeze(0))

        seq_embeddings = seq_embeddings[-self.max_len:]
        padding_len = self.max_len - len(seq_embeddings)
        seq_embeddings = [self.padding_embedding] * padding_len + seq_embeddings
        
        seq_tensor = torch.cat(seq_embeddings, 0)

        return seq_tensor, torch.LongTensor(candidates), torch.LongTensor(labels)

