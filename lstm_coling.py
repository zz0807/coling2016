#coding: utf-8
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np
from keras.optimizers import rmsprop

np.random.seed(1337)    #保持一致性

def lstm_coling(index,x_train,y_train,x_test,y_test,
                        layer_num,cell_num,nb_epoch,
                        dropout,lr,batch_size,decay=1e-6):

    maxlen = 20 # cut texts after this number of words (among top max_features most common words)
    # 限定最大词数

    # batch_size = 25
    len_wv = 50

    weights_filepath = u"saved_model/"+str(index)+'-maxf_develop_weights.epoch_{epoch:02d}-val_acc_{val_acc:.2f}.hdf5'

    # Memory 足够时用
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='float64')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float64')

    x_train = x_train.reshape((len(x_train), maxlen, len_wv))
    x_test = x_test.reshape((len(x_test), maxlen, len_wv))

    # step 1: train / test 均2分类
    y_test = np_utils.to_categorical(y_test, 42)  # 必须使用固定格式表示标签
    y_train = np_utils.to_categorical(y_train, 42)  # 必须使用固定格式表示标签 一共 42分类

    # =================================以上数据读取============================

    print('Build model...')

    model = Sequential()
    # Stacked LSTM
    #GRU
    if layer_num < 2:
        model.add(LSTM(cell_num, input_shape=(maxlen, len_wv)))

    else:
        model.add(LSTM(cell_num, return_sequences=True, input_shape=(maxlen, len_wv)))

        for i in range(0, layer_num - 2):

            model.add(LSTM(cell_num, return_sequences=True))
            # model.add(GRU(cell_num, return_sequences=True))
        model.add(LSTM(cell_num))
        # model.add(GRU(cell_num))

    model.add(Dropout(dropout))

    model.add(Dense(42))
    model.add(Activation('softmax'))

    _rmsprop = rmsprop(lr=lr)

    model.compile(loss='categorical_crossentropy', optimizer=_rmsprop,
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test),
              nb_epoch=nb_epoch, shuffle=True,
              callbacks=[ModelCheckpoint(weights_filepath, monitor='val_acc',
                                         verbose=1, save_best_only=True, mode='max'),
                         EarlyStopping(monitor='val_acc', verbose=1, patience=30, mode='max')])

