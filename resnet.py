import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def res_block(x, unit1, unit2, activation1, activation2):
    fx = layers.Dense(unit1, activation=activation1)(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Dense(unit2, activation=activation2)(x)
    out = layers.Add()([x, fx])
    out = layers.ReLU()(out)
    out = layers.BatchNormalization()(out)
    return out


if __name__ == "__main__":
    train_X = np.load('train_X.npy')
    train_y = np.load('train_y.npy')
    validation_X = np.load('validation_X.npy')
    validation_y = np.load('validation_y.npy')
    test_X = np.load('test_X.npy')
    test_y = np.load('test_y.npy')

    inputs = layers.Input(shape=((301,)))
    x = layers.Dense(2048, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = res_block(x, 2048, 2048, 'relu', 'relu')
    x = res_block(x, 2048, 2048, 'relu', 'relu')
    x = res_block(x, 2048, 2048, 'relu', 'relu')
    x = res_block(x, 2048, 2048, 'relu', 'relu')
    x = res_block(x, 2048, 2048, 'relu', 'relu')
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = res_block(x, 1024, 1024, 'relu', 'relu')
    x = res_block(x, 1024, 1024, 'relu', 'relu')
    x = res_block(x, 1024, 1024, 'relu', 'relu')
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = res_block(x, 512, 512, 'relu', 'relu')
    x = res_block(x, 512, 512, 'relu', 'relu')
    x = res_block(x, 512, 512, 'relu', 'relu')
    outputs = layers.Dense(12, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['categorical_crossentropy'])

    model.fit(train_X, train_y, epochs=3, batch_size=256, validation_data=(validation_X, validation_y))

    pred_train = model.predict(train_X)
    pred = model.predict(test_X)
    np.save('resnet_train_full.npy', pred_train)
    np.save('resnet_test_full.npy', pred)
