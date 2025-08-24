from .bert import BERTModel
from .bertemb import BERTEmbeddingModel
from .dae import DAEModel
from .vae import VAEModel

MODELS = {
    BERTModel.code(): BERTModel,
    DAEModel.code(): DAEModel,
    VAEModel.code(): VAEModel,
    BERTEmbeddingModel.code(): BERTEmbeddingModel,
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
