import gensim
import numpy as np
import pandas as pd


def getVectors(data):
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    tokens = list(data['token'])

    def vectorize(token):
        try:
            res = model[str(token)]
        except KeyError:
            res = np.zeros((300,))
        return res

    vectorized_tokens = [vectorize(token) for token in tokens]
    return vectorized_tokens