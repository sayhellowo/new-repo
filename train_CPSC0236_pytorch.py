import tensorflow as tf
# from keras.layers import Dense, Dropout, Activation, Flatten,  Input, Reshape, CuDNNGRU
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional, LeakyReLU
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape, GRU
from keras.layers import Convolution1D, MaxPool1D, GlobalAveragePooling1D, concatenate, AveragePooling1D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.models import Model
from keras import initializers, regularizers, constraints
import numpy as np
from keras.layers.normalization import BatchNormalization
import scipy.io as sio
import os
from AttentionWithContext import *
import tensorflow.keras as keras
import csv
# from keras.utils import multi_gpu_model
import math
from Base_generator import *



def model_define():
    main_input = Input(shape=(72000, 12), dtype='float32', name='main_input')
    x = Convolution1D(12, 3, padding='same')(main_input)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 48, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    cnnout = Dropout(0.2)(x)
    x = Bidirectional(GRU(12, input_shape=(2250, 12), return_sequences=True, return_state=False, reset_after=True))(
        cnnout)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = AttentionWithContext()(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    main_output = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=main_input, outputs=main_output)

    return model


if __name__ == "__main__":
    # 确保安装了tensorflow的GPU版本
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # 使用第一张GPU卡
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 参数设置
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    ecg_size = (72000, 12)
    batch_size = 64
    num_classes = 9
    epochs = 10000
    leadsLabel = np.asarray(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'all'])
    num_folds = 10
    ecg_path = '/mnt/ecg/sample/CPSC_TrainingSet'
    save_path = '/tmp'

    # 读取每个样本的标签
    with open(ecg_path + '/REFERENCE.csv', 'r') as csvfile:
        X = np.array(list(csv.reader(csvfile)))[1:, 0:2]

    np.random.seed(13)
    train = np.arange(0, X.shape[0], 1)
    np.random.shuffle(train)
    for lead in range(13):
        # 定义好的模型
        model = model_define()

        # 使用GPU训练
        # model = multi_gpu_model(model, 1)  # GPU个数为2

        # 查看模型的参数
        model.summary()

        # 设置损失函数loss、优化器optimizer、准确性评价函数metrics
        # model.compile(optimizer=keras.optimizers.Adam(1e-3),  # Low learning rate
        #     loss='binary_crossentropy',
        #     metrics=['categorical_accuracy'])
        model.compile(optimizer=keras.optimizers.Adam(1e-3),  # Low learning rate
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # 利用生成器，分批次向模型送入数据训练
        model.fit_generator(Base_generator(train, batch_size, X, ecg_path, lead, ecg_size), epochs=epochs,
                            verbose=1, workers=1, steps_per_epoch=len(train) // batch_size)

        # 保存模型
        model.save(save_path+"/model_"+leadsLabel[lead]+".h5")

