from .bert import BERTModel
from .bertemb import BERTEmbeddingModel
from .dae import DAEModel
from .vae import VAEModel
from .sasrec import SASRecModel

MODELS = {
    BERTModel.code(): BERTModel,
    DAEModel.code(): DAEModel,
    VAEModel.code(): VAEModel,
    BERTEmbeddingModel.code(): BERTEmbeddingModel,
    SASRecModel.code(): SASRecModel,
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
