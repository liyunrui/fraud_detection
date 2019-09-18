"""
Reference:https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import regularizers
import gc

def autoencoder(input_dim, encoding_dim):
    """
    architecture of autoencoder, we consider this as a dimension reduction method. 
    encoding_dim: int
    input_dim: int.
    """
    from keras.layers import Input, Dense
    from keras.models import Model
    
    input_layer = Input(shape=(input_dim, ))

    encoder = Dense(encoding_dim, activation="tanh",
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

    decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    
    return autoencoder

def build_model(autoencoder,X_train,X_test,nb_epoch = 100,batch_size = 32):
    """
    X_train: data only including normal transation data

    X_test: data both including fradulant and normal data 
    """
    autoencoder.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath="../models/model.h5",
                                   monitor='val_loss',
                                   verbose=0,
                                   save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', 
                                  min_delta=0, 
                                  patience=5, 
                                  verbose=0, 
                                  mode='auto', 
                                  baseline=None, 
                                  restore_best_weights=False)

#     tensorboard = TensorBoard(log_dir='/media/old-tf-hackers-7/logs',
#                               histogram_freq=0,
#                               write_graph=True,
#                               write_images=True)
    history = autoencoder.fit(X_train, X_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        verbose=1,
                        callbacks=[checkpointer, earlystopper]).history
    return history


train = pd.read_csv("/data/yunrui_li/fraud/dataset/train.csv")
test = pd.read_csv("/data/yunrui_li/fraud/dataset/test.csv")


CATEGORY = ['ecfg', 'flbmk', 'flg_3dsmk', 'insfg', 
            'ovrlt', 'scity', 'csmcu', 'cano', 
            'mchno', 'hcefg', 'bacno', 'contp', 'etymd', 'acqic']


X_train = train[train.fraud_ind == 0]
X_train = X_train.drop(['fraud_ind'], axis=1)
X_train = X_train.drop(['txkey'], axis=1)

X_test = test
X_test = X_test.drop(['txkey'], axis=1)


CATEGORY = ['ecfg', 'flbmk', 'flg_3dsmk', 'insfg', 
            'ovrlt', 'scity', 'csmcu', 'cano', 
            'mchno', 'hcefg', 'bacno', 'contp', 'etymd', 'acqic']

df = pd.concat([train, test], axis = 0)
for cat in CATEGORY:
    le = preprocessing.LabelEncoder()
    le.fit(df[cat].tolist())
    df[cat] = le.transform(df[cat].tolist())
    X_train[cat] = le.transform(X_train[cat].tolist()) 
    X_test[cat] = le.transform(X_test[cat].tolist()) 

print ("=" * 100)
print ("finished label encoding")

del train,test
gc.collect()

scaler = MinMaxScaler()
data = df[X_train.columns.tolist()]
scaler.fit(data)

X_train = X_train.values
X_test = X_test.values

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

del df, data
gc.collect()

input_dim = X_train.shape[1]
encoding_dim = 14
print ('number of raw features', input_dim)
print ('number of normal data', X_train.shape[0])


autoencoder = autoencoder(input_dim, encoding_dim)

print (autoencoder.summary())

history = build_model(autoencoder,X_train,X_test,nb_epoch = 100,batch_size = 32)
