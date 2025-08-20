from .ml_1m import ML1MDataset
from .ml_20m import ML20MDataset
from .amazon import AmazonDataset


DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML20MDataset.code(): ML20MDataset,
    AmazonDataset.code(): AmazonDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
