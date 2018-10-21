import pandas as pd
import keras
import tensorflow as tf
tf.python_io.control_flow_ops = tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adadelta

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

'''
google ML을 이용하여 학습시킬 준비.
'''

if __name__ == '__main__':
    img_rows, img_cols = 48, 48
    total_rows, train_rows, test_rows = 32300, 28710, 32300 # 실제 학습 시킬때
    # total_rows, train_rows, test_rows = 150, 100, 150 # 간단하게 테스트 할때

    full_data = pd.read_csv('./fer2013.csv')
    print(full_data.values.shape) # (35887,3)

    data = full_data.values
    pixels = data[:total_rows, 1]


    X = np.zeros((pixels.shape[0], img_rows * img_cols))

    for ix in range(X.shape[0]):
        p = pixels[ix].split(' ')
        for iy in range(X.shape[1]):
            X[ix, iy] = int(p[iy])

    # np.save('facial_data_X', X)
    # np.save('facial_labels', data[:, 0])
    # x = np.load('./facial_data_X.npy')
    # y = np.load('./facial_labels.npy')
    x = X
    y = data[:total_rows, 0]

    # 평균 값을 뺀다음에 표준편차로 나누자!
    # x -= np.mean(x, axis=0)
    # x /= np.std(x, axis=0)

    # 학습 데이터
    X_train = x[0:train_rows,:]
    Y_train = y[0:train_rows]
    print(X_train.shape , Y_train.shape)
    # 테스트 데이터
    X_crossval = x[train_rows:total_rows,:]
    Y_crossval = y[train_rows:total_rows]
    print (X_crossval.shape , Y_crossval.shape)

    # 사진 모양으로 바꾸고
    X_train = X_train.reshape((X_train.shape[0], 1, img_rows, img_cols))
    X_crossval = X_crossval.reshape((X_crossval.shape[0], 1, img_rows, img_cols))

    print(X_train.shape)

    model = Sequential()
    model.add(Convolution2D(64, (5, 5), border_mode='valid',
                            input_shape=(1, img_rows, img_cols), data_format='channels_first'))
    model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))

    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(Convolution2D(128, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(Convolution2D(128, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))

    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
    model.add(Dropout(0.2))

    model.add(Dense(7, activation='softmax'))

    ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=ada,
                  metrics=['accuracy'])
    model.summary()

    y_ = np_utils.to_categorical(y)
    Y_train = y_[:train_rows]
    Y_crossval = y_[train_rows:test_rows]
    print(X_crossval.shape, model.input_shape, Y_crossval.shape)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)

    filepath='output/Model.{epoch:02d}-{val_acc:.4f}.hdf5'
    checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

    # 모델 학습
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=128),
                        steps_per_epoch=len(X_train) / 128,

                        validation_data=(X_crossval, Y_crossval),
                        # samples_per_epoch=X_train.shape[0],
                        callbacks=[checkpointer])

    saver = tf.train.Saver()
    saver.save(K.get_session(), './keras_model.ckpt')


    # scores = model.evaluate(X_crossval, Y_crossval, batch_size=128)
    # print(scores) # 손실, 정확

    '''
    Keras에서 만든 모델을 저장할 때는 다음과 같은 룰을 따릅니다.
    
    모델은 JSON 파일 또는 YAML 파일로 저장한다. 
    Weight는 H5 파일로 저장한다.
    '''
    '''
    # 모델 저장
    model_json = model.to_json()
    with open("model.json", "w") as json_file :
        json_file.write(model_json)

    model.save("weight.h5")    # 가중치 저장
    '''
    '''
    # 모델 로드하기
    from keras.models import model_from_json
    json_file = open("model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # 가중치 로드하기
    model.load_weights("Model.59-0.6053.hdf5")
    '''

    '''
    # 모델 컴파일 후 평가
    loaded_model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    
    # model evaluation
    score = loaded_model.evaluate(X,Y,verbose=0)
    
    print("%s : %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    '''
    # for ix in range(10):
    #     plt.figure(ix)
    #     plt.imshow(X_train[ix].reshape(48, 48), cmap='gray')
    #     print(np.argmax(pred[ix]), np.argmax(y_[ix]))
    # plt.show()