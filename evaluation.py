import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


train = pd.read_csv('/home/wushiyu/SI699/train_full.csv', lineterminator='\n')
test = pd.read_csv('/home/wushiyu/SI699/test_full.csv', lineterminator='\n')

encoder = LabelEncoder()
onehot = OneHotEncoder()

onehot.fit(np.array(encoder.fit_transform(train['label'])).reshape(-1, 1))

train_pred = np.load('/home/wushiyu/SI699/resnet_train_full.npy')
test_pred = np.load('/home/wushiyu/SI699/resnet_test_full.npy')

train_pred = encoder.inverse_transform(onehot.inverse_transform(train_pred).reshape(1, -1)[0])
test_pred = encoder.inverse_transform(onehot.inverse_transform(test_pred).reshape(1, -1)[0])

print('\ntrain set')
train_labels = list(train['label'].unique())
train_mat = confusion_matrix(train['label'], train_pred, labels=train_labels)
print('labels: ')
print(train_labels)
print('confusion matrix: ')
print(train_mat)
for i, row in enumerate(train_mat):
    print('----')
    print(f'label: {train_labels[i]}')
    try:
        print(f'Recall score: {row[i] / sum(row)}')
    except:
        print('Recall score: NaN')
    fp = 0
    for j in range(len(train_mat)):
        if j != i:
            fp += train_mat[j][i]
    try:
        print(f'F1 score: {row[i] / (row[i] + 0.5 * (fp + sum(row) - row[i]))}')
    except:
        print(f'F1 score: NaN')

print('\ntest set')
test_mat = confusion_matrix(test['label'], test_pred, labels=list(train['label'].unique()))
print('labels: ')
print(list(train['label'].unique()))
print('confusion matrix: ')
print(test_mat)
for i, row in enumerate(test_mat):
    print('----')
    print(f'label: {train_labels[i]}')
    try:
        print(f'Recall score: {row[i] / sum(row)}')
    except:
        print('Recall score: NaN')
    fp = 0
    for j in range(len(test_mat)):
        if j != i:
            fp += test_mat[j][i]
    try:
        print(f'F1 score: {row[i] / (row[i] + 0.5 * (fp + sum(row) - row[i]))}')
    except:
        print(f'F1 score: NaN')
