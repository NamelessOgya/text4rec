from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset  # Keep the original dataset instance
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.save_folder = self.dataset._get_preprocessed_folder_path()
        
        loaded_data = self.dataset.load_dataset()
        self.train = loaded_data['train']
        self.val = loaded_data['val']
        self.test = loaded_data['test']
        self.umap = loaded_data['umap']
        self.smap = loaded_data['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        # Attach the inverse map to the original dataset instance
        self.dataset.idx2item = {v: k for k, v in self.smap.items()}

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass
