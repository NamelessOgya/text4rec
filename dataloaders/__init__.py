from .base import AbstractDataloader
from .bert import BertDataloader
from .bertemb import BertEmbeddingDataloader
from .ae import AEDataloader
from .sasrec import SASEmbDataloader
from datasets import dataset_factory

DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    AEDataloader.code(): AEDataloader,
    BertEmbeddingDataloader.code(): BertEmbeddingDataloader,
    SASEmbDataloader.code(): SASEmbDataloader,
}

def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
