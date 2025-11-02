from .popular import PopularNegativeSampler
from .random import RandomNegativeSampler
from .text_similar import TextSimilarNegativeSampler

NEGATIVE_SAMPLERS = {
    PopularNegativeSampler.code(): PopularNegativeSampler,
    RandomNegativeSampler.code(): RandomNegativeSampler,
    TextSimilarNegativeSampler.code(): TextSimilarNegativeSampler,
}

def negative_sampler_factory(code, train, val, test, user_count, item_count, sample_size, seed, save_folder, item_embedding_path=None):
    negative_sampler = NEGATIVE_SAMPLERS[code]
    if code == 'text_similar':
        return negative_sampler(train, val, test, user_count, item_count, sample_size, seed, save_folder, item_embedding_path)
    return negative_sampler(train, val, test, user_count, item_count, sample_size, seed, save_folder)
