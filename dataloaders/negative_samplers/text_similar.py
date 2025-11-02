from .base import AbstractNegativeSampler
from tqdm import trange
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TextSimilarNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'text_similar'

    def __init__(self, train, val, test, user_count, item_count, sample_size, seed, save_folder, item_embedding_path):
        super().__init__(train, val, test, user_count, item_count, sample_size, seed, save_folder)
        self.item_embeddings = np.load(item_embedding_path)

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        
        # Pre-calculate similarity matrix if memory allows
        # This is a trade-off between speed and memory
        # For now, we calculate similarity on the fly
        
        negative_samples = {}
        print('Sampling textually similar negative items')
        
        for user in trange(self.user_count):
            if isinstance(self.train[user][1], tuple):
                seen = set(x[0] for x in self.train[user])
                seen.update(x[0] for x in self.val[user])
                seen.update(x[0] for x in self.test[user])
            else:
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])

            # Get the last item the user interacted with as the positive item
            if isinstance(self.train[user][-1], tuple):
                last_item_id = self.train[user][-1][0]
            else:
                last_item_id = self.train[user][-1]

            positive_item_embedding = self.item_embeddings[last_item_id].reshape(1, -1)
            
            # Calculate cosine similarity between the positive item and all other items
            similarities = cosine_similarity(positive_item_embedding, self.item_embeddings).flatten()
            
            # Get indices of items sorted by similarity (descending)
            sorted_indices = np.argsort(similarities)[::-1]
            
            samples = []
            for item_idx in sorted_indices:
                if len(samples) >= self.sample_size:
                    break
                if item_idx not in seen and item_idx != last_item_id:
                    samples.append(item_idx + 1) # item_id is 1-based

            negative_samples[user] = samples

        return negative_samples
