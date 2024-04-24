import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import gensim
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from math import sin, cos

def sample(df, seed=42):
    train, test = train_test_split(df, test_size=0.2, random_state=seed)
    dfs = [0] * 12
    oobs = [0] * 12
    for i, label in enumerate(set(list(train['label']))):
        dfs[i] = resample(df[df['label'] == label], n_samples=len(df) // 12, random_state=seed)
        oobs[i] = df.loc[(~df.index.isin(dfs[i].index)) & (df['label'] == label)]
    return pd.concat(dfs), pd.concat(oobs), test

def getVectors(data):
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    tokens = list(data['token'])
    positions = list(data['token_index'])

    def vectorize(token, index=None):
        try:
            position = np.array([sin(index / (10000 ** (2 * i / 300))) if i % 2 == 0 else cos(index / (10000 ** (2 * i / 300))) for i in range(300)])
            word = model[str(token)]
            res = np.add(word, position)
        except KeyError:
            res = np.array([sin(index / (10000 ** (2 * i / 300))) if i % 2 == 0 else cos(index / (10000 ** (2 * i / 300))) for i in range(300)])
        return res

    vectorized_tokens = []
    for i in range(len(tokens)):
        vectorized_tokens.append(vectorize(tokens[i], positions[i]))
    return vectorized_tokens


if __name__ == "__main__":
    df = pd.read_csv('data.csv', index_col=0)
    train, validation, test = sample(df)
    vectorized_vectors_train, vectorized_vectors_validation, vectorized_vectors_test = np.array(getVectors(train)), np.array(getVectors(validation)), np.array(getVectors(test))

    encoder = LabelEncoder()
    onehot = OneHotEncoder()

    trailing_space_train = np.array([int(item) for item in list(train['trailing_space'])])
    train_X = np.column_stack((vectorized_vectors_train, trailing_space_train))
    train_y = onehot.fit_transform(np.array(encoder.fit_transform(train['label'])).reshape(-1, 1)).toarray()

    trailing_space_validation = np.array([int(item) for item in list(validation['trailing_space'])])
    validation_X = np.column_stack((vectorized_vectors_validation, trailing_space_validation))
    validation_y = onehot.transform(np.array(encoder.transform(validation['label'])).reshape(-1, 1)).toarray()

    trailing_space_test = np.array([int(item) for item in list(test['trailing_space'])])
    test_X = np.column_stack((vectorized_vectors_test, trailing_space_test))
    test_y = onehot.transform(np.array(encoder.transform(test['label'])).reshape(-1, 1)).toarray()
    train.to_csv('train_full.csv')
    test.to_csv('test_full.csv')
    validation.to_csv('validation_full.csv')

    np.save('train_X.npy', train_X)
    np.save('train_y.npy', train_y)
    np.save('validation_X.npy', validation_X)
    np.save('validation_y.npy', validation_y)
    np.save('test_X.npy', test_X)
    np.save('test_y.npy', test_y)