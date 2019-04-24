import numpy as np
import os
import pickle
import datetime
import sys

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate, Conv2D, MaxPooling2D, LSTM, GRU, concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
from keras import regularizers
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras import backend as K

from keras_self_attention import SeqSelfAttention

sys.path.append('/home/jantunes/Desktop/joana/bs/code/')
from utils import pick_best_decision_threshold, print_performace_metrics, \
    convert_word_level_predictions_to_interview_level_predictions

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def convert_to_sequences(X, y, N, y_video_ids):
    y_video_ids = [i for x in y_video_ids for i in x]

    X_seq = []
    y_seq = []
    y_video_ids_seq = []

    for i in np.arange(0, X.shape[0] - N, N):
        if len(X.shape) == 4:
            X_seq.append(X[i:i + N, :, :, :])
        elif len(X.shape) == 2:
            X_seq.append(X[i:i + N, :])
        else:
            print("wrong input size!")
        y_seq.append(y[i:i + N])
        y_video_ids_seq.append(y_video_ids[i:i + N])

    X_seq = np.asarray(X_seq)
    y_seq = np.asarray(y_seq)
    y_video_ids_seq = np.asarray(y_video_ids_seq)

    return X_seq, y_seq, y_video_ids_seq


basepath = "/home/jantunes/Desktop/joana/bs/"
model_inputs_dir = basepath + "model_inputs/"
out_path = basepath + "models/"

if not os.path.exists(out_path):
    os.makedirs(out_path)

bool_save_model = False

bool_normalize_Xs = False

N = 30  # segment len is N words

# repeat same experiment for LSTM with and without glove + emotion information

for [bool_include_emotion_information, bool_include_semantic_information] in [[False, False],
                                                                              [False, True],
                                                                              [True, False],
                                                                              [True, True]
                                                                             ]:

    if not bool_include_emotion_information and not bool_include_semantic_information:
        f = ""
        print("Training and testing LSTM with ONLY spectral data ...")
    elif not bool_include_emotion_information and bool_include_semantic_information:
        f = "_g_"
        print("Training and testing LSTM with spectral data and semantic data...")
    elif bool_include_emotion_information and not bool_include_semantic_information:
        f = "_e_"
        print("Training and testing LSTM with spectral data and emotion descriptors data...")
    else:
        f = "_e_g_"
        print("Training and testing LSTM with spectral, semantic, and emotion descriptors data...")

    # Load LSTM inputs

    X_train = pickle.load(open(model_inputs_dir + "X_train_" + f + "embedded.pkl", "rb"))

    y_train = pickle.load(open(model_inputs_dir + "y_train.pkl", "rb"))
    y_train = np.asarray([item for sublist in y_train for item in sublist])

    X_devel = pickle.load(open(model_inputs_dir + "X_devel_" + f + "embedded.pkl", "rb"))

    y_devel = pickle.load(open(model_inputs_dir + "y_devel.pkl", "rb"))
    y_devel = np.asarray([item for sublist in y_devel for item in sublist])

    video_ids_train = pickle.load(open(model_inputs_dir + "y_video_ids_train.pkl", "rb"))
    video_ids_devel = pickle.load(open(model_inputs_dir + "y_video_ids_devel.pkl", "rb"))

    # convert input to sequences of 50 words

    X_train, y_train_seq, y_video_ids_train_seq = convert_to_sequences(X_train, y_train, N, video_ids_train)
    X_devel, y_devel_seq, y_video_ids_devel_seq = convert_to_sequences(X_devel, y_devel, N, video_ids_devel)

    # Define LSTM architecture

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=X_train.shape[1:]))
    model.add(LSTM(16, return_sequences=True))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    try:
        model = multi_gpu_model(model, len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
        print("Will train model using multiple GPUs... ")
    except:
        pass
        print("Will train model using a single GPU... ")

    # Set Training Parameters

    learning_rate = 0.000001
    epochs = 30
    batch_size = 32

    opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=0, epsilon=1e-4, mode='min')

    # print model summary

    # model.summary()

    # train model

    print("Initializing LSTM training...")
    history = model.fit(x=X_train,
                        y=np.expand_dims(y_train_seq, axis=2),
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_devel, np.expand_dims(y_devel_seq, axis=2)),
                        callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                        shuffle=True,
                        verbose=0
                        )

    # save model

    now = datetime.datetime.now()
    name = 'lstm' + f + "_" + now.strftime("%Y%m%d_%H%M")
    if bool_save_model:
        model.save(out_path + name + '.h5')

    # evaluate LSTM performance

    print("Evaluating LSTM ...")
    y_pred_train = model.predict(X_train)
    y_pred_devel = model.predict(X_devel)

    # convert continuous predictions to binary

    y_pred_train, y_pred_devel = pick_best_decision_threshold(y_train_seq.flatten(), y_pred_train.flatten(),
                                                              y_pred_devel.flatten())

    # print word level performance

    print("Computing word level performance ...")
    print_performace_metrics(y_train_seq.flatten(), y_pred_train, comment="performace for train: ")
    print_performace_metrics(y_devel_seq.flatten(), y_pred_devel, comment="performace for devel: ")

    # convert word level predictions to interview level predictions

    y_train_interview = convert_word_level_predictions_to_interview_level_predictions(y_train_seq.flatten(),
                                                                                      y_video_ids_train_seq)
    y_devel_interview = convert_word_level_predictions_to_interview_level_predictions(y_devel_seq.flatten(),
                                                                                      y_video_ids_devel_seq)
    y_pred_train_interview = convert_word_level_predictions_to_interview_level_predictions(y_pred_train,
                                                                                           y_video_ids_train_seq)
    y_pred_devel_interview = convert_word_level_predictions_to_interview_level_predictions(y_pred_devel,
                                                                                           y_video_ids_devel_seq)

    # print interview level performance

    print("Computing interview level performance ...")
    print_performace_metrics(y_train_interview, y_pred_train_interview, comment="performace for train: ")
    print_performace_metrics(y_devel_interview, y_pred_devel_interview, comment="performace for devel: ")

