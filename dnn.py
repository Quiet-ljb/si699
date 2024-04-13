import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

if __name__ == "__main__":
    train_X = np.load('/home/wushiyu/SI699/train_X.npy')
    train_y = np.load('/home/wushiyu/SI699/train_y.npy')
    validation_X = np.load('/home/wushiyu/SI699/validation_X.npy')
    validation_y = np.load('/home/wushiyu/SI699/validation_y.npy')
    test_X = np.load('/home/wushiyu/SI699/test_X.npy')
    test_y = np.load('/home/wushiyu/SI699/test_y.npy')

    model = models.Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(2048, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(12, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['categorical_crossentropy'])

    model.fit(train_X, train_y, epochs=3, batch_size=256, validation_data=(validation_X, validation_y))

    
    pred_train = model.predict(train_X)
    pred = model.predict(test_X)
    np.save('baseline_train_full.npy', pred_train)
    np.save('baseline_test_full.npy', pred)