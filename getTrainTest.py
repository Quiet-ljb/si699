from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd

def sample(df, seed=42):
    train, test = train_test_split(df, test_size=0.2, random_state=seed)
    dfs = [0] * 12
    oobs = [0] * 12
    for i, label in enumerate(set(list(train['label']))):
        dfs[i] = resample(df[df['label'] == label], n_samples=len(df) // 12, random_state=seed)
        oobs[i] = df.loc[(~df.index.isin(dfs[i].index)) & (df['label'] == label)]
    return pd.concat(dfs), pd.concat(oobs), test