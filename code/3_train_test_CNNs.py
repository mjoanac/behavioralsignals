import numpy as np
import os
import pickle
import datetime
import sys

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate, Conv2D, MaxPooling2D, LSTM, GRU, concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras import regularizers
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras import backend as K

sys.path.append('/home/jantunes/Desktop/joana/bs/code/')
from utils import pick_best_decision_threshold, print_performace_metrics, convert_word_level_predictions_to_interview_level_predictions


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


basepath = "/home/jantunes/Desktop/joana/bs/"
model_inputs_dir = basepath + "model_inputs/"
out_path = basepath + "models/"

if not os.path.exists(out_path):
    os.makedirs(out_path)

bool_save_model = False
bool_save_embeddings = True

bool_normalize_Xs = False

# repeat same experiment for CNNs with and without glove + emotion information 

for [bool_include_emotion_information, bool_include_semantic_information] in [[False, False],
                                                                              [False, True],
                                                                              [True, False],
                                                                              [True, True]
                                                                              ]:


    if not bool_include_emotion_information and not bool_include_semantic_information:
        f = ""
        print("Training and testing CNN with ONLY spectral data ...")
    elif not bool_include_emotion_information and bool_include_semantic_information:
        f = "_g_"
        print("Training and testing CNN with spectral data and semantic data...")
    elif bool_include_emotion_information and not bool_include_semantic_information:
        f = "_e_"
        print("Training and testing CNN with spectral data and emotion descriptors data...")
    else:
        f = "_e_g_"
        print("Training and testing CNN with spectral, semantic, and emotion descriptors data...")

    # Load model inputs

    X_train = pickle.load(open(model_inputs_dir + "X_train_s.pkl", "rb"))
    X_train = np.expand_dims(np.vstack(X_train), axis=3)

    y_train = pickle.load(open(model_inputs_dir + "y_train.pkl", "rb"))
    y_train = np.hstack((y_train))

    X_devel = pickle.load(open(model_inputs_dir + "X_devel_s.pkl", "rb"))
    X_devel = np.expand_dims(np.vstack(X_devel), axis=3)

    y_devel = pickle.load(open(model_inputs_dir + "y_devel.pkl", "rb"))
    y_devel = np.hstack((y_devel))

    video_ids_train = pickle.load(open(model_inputs_dir + "y_video_ids_train.pkl", "rb"))
    video_ids_devel = pickle.load(open(model_inputs_dir + "y_video_ids_devel.pkl", "rb"))

    print("\n")
    print("Shape of train data and Labels: ")
    print(X_train.shape)
    print(y_train.shape)

    print("\n")
    print("Shape of devel data and Labels: ")
    print(X_devel.shape)
    print(y_devel.shape)

    print("\n")
    print("Average train label: " + str(np.mean(y_train)))
    print("Average devel label: " + str(np.mean(y_devel)))

    if bool_include_emotion_information:

        print("Loading emotion data...")

        X_train_e = pickle.load(open(model_inputs_dir + "X_train_e.pkl", "rb"))
        X_train_e = np.vstack(X_train_e)

        X_devel_e = pickle.load(open(model_inputs_dir + "X_devel_e.pkl", "rb"))
        X_devel_e = np.vstack(X_devel_e)

        print(X_train_e.shape)
        print(X_devel_e.shape)

    if bool_include_semantic_information:

        print("Loading semantic data...")

        X_train_g = pickle.load(open(model_inputs_dir + "X_train_g.pkl", "rb"))
        X_train_g = np.vstack(X_train_g)

        X_devel_g = pickle.load(open(model_inputs_dir + "X_devel_g.pkl", "rb"))
        X_devel_g = np.vstack(X_devel_g)

        print(X_train_g.shape)
        print(X_devel_g.shape)

    # Define CNN model architecture 

    input_spectrograms = Input(X_train.shape[1:])

    model_cnn = Conv2D(32, (3, 3))(input_spectrograms)
    model_cnn = Activation("relu")(model_cnn)
    model_cnn = BatchNormalization()(model_cnn)
    model_cnn = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(model_cnn)

    model_cnn = Conv2D(64, (5, 5))(model_cnn)
    model_cnn = Activation("relu")(model_cnn)
    model_cnn = BatchNormalization()(model_cnn)
    model_cnn = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(model_cnn)

    model_cnn = Conv2D(64, (5, 5))(model_cnn)
    model_cnn = Activation("relu")(model_cnn)
    model_cnn = BatchNormalization()(model_cnn)
    model_cnn = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(model_cnn)

    model_flat = Flatten()(model_cnn)

    if bool_include_emotion_information:
        input_emotion = Input(X_train_e.shape[1:])
        model_flat = concatenate([model_flat, input_emotion], axis=1)

    if bool_include_semantic_information:
        input_gloves = Input(X_train_g.shape[1:])
        model_flat = concatenate([model_flat, input_gloves], axis=1)

    model_top = Dense(256, kernel_regularizer=regularizers.l2(0.1))(model_flat)
    model_top = Activation('relu')(model_top)
    model_top = Dropout(0.2)(model_top)

    model_top2 = Dense(64, kernel_regularizer=regularizers.l2(0.1))(model_top)
    model_top2 = Activation('relu')(model_top2)
    model_top2 = Dropout(0.5)(model_top2)

    model_out = Dense(1, kernel_regularizer=regularizers.l2(0.05))(model_top2)
    model_out = Activation('sigmoid')(model_out)

    if not bool_include_emotion_information and not bool_include_semantic_information:
        model = Model(inputs=[input_spectrograms], outputs=model_out)
    elif not bool_include_emotion_information and bool_include_semantic_information:
        model = Model(inputs=[input_spectrograms, input_gloves], outputs=model_out)
    elif bool_include_emotion_information and not bool_include_semantic_information:
        model = Model(inputs=[input_spectrograms, input_emotion], outputs=model_out)
    else:
        model = Model(inputs=[input_spectrograms, input_emotion, input_gloves], outputs=model_out)

    #try:
    #    model = multi_gpu_model(model,len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
    #    print("Will train model using multiple GPUs... ")
    #except:
    #    pass
    #    print("Will train model using a single GPU... ")

    # Set Training Parameters

    learning_rate = 0.00005
    epochs = 30
    batch_size = 32

    # Define model optimizers and calls

    opt = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    # print model summary 
 
    model.summary()


    # define model inputs
 
    if not bool_include_emotion_information and not bool_include_semantic_information:
        X_train_all = [X_train]
        X_devel_all = [X_devel]
    elif not bool_include_emotion_information and bool_include_semantic_information:
        X_train_all = [X_train, X_train_g]
        X_devel_all = [X_devel, X_devel_g]
    elif bool_include_emotion_information and not bool_include_semantic_information:
        X_train_all = [X_train, X_train_e]
        X_devel_all = [X_devel, X_devel_e]
    else:
        X_train_all = [X_train, X_train_e, X_train_g]
        X_devel_all = [X_devel, X_devel_e, X_devel_g]

    # train model

    print("Initializing CNN training...")
    history = model.fit(x=X_train_all,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_devel_all, y_devel),
                        callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                        shuffle=True,
                        verbose=2
                        )

    # save model

    now = datetime.datetime.now()
    name = 'cnn' + f + "_" + now.strftime("%Y%m%d_%H%M")
    if bool_save_model:
        model.save(out_path + name + '.h5')

    # evaluate CNN performance

    print("Evaluating CNN ...")
    y_pred_train = model.predict(X_train_all)
    y_pred_devel = model.predict(X_devel_all)

    # convert continuous predictions to binary

    y_pred_train, y_pred_devel = pick_best_decision_threshold(y_train, y_pred_train, y_pred_devel)

    # print word level performance

    print("Computing word level performance ...")
    print_performace_metrics(y_train, y_pred_train, comment="performace for train: ")
    print_performace_metrics(y_devel, y_pred_devel, comment="performace for devel: ")

    # convert word level predictions to interview level predictions

    y_train_interview = convert_word_level_predictions_to_interview_level_predictions(y_train, video_ids_train)
    y_devel_interview = convert_word_level_predictions_to_interview_level_predictions(y_devel, video_ids_devel)
    y_pred_train_interview = convert_word_level_predictions_to_interview_level_predictions(y_pred_train, video_ids_train)
    y_pred_devel_interview = convert_word_level_predictions_to_interview_level_predictions(y_pred_devel, video_ids_devel)

    # print interview level performance

    print("Computing interview level performance ...")
    print_performace_metrics(y_train_interview, y_pred_train_interview, comment="performace for train: ")
    print_performace_metrics(y_devel_interview, y_pred_devel_interview, comment="performace for devel: ")

    # remove top of network and compute embedding of data

    if bool_save_embeddings:

        print("Embedding spectral data...")
        model_bottom = Model(inputs=model.input, outputs=model.layers[13].output)

        X_train_embed = model_bottom.predict(X_train_all)
        X_devel_embed = model_bottom.predict(X_devel_all)

        pickle.dump(X_train_embed, open(model_inputs_dir + "X_train_"+ f +"embedded.pkl", "wb"))
        pickle.dump(X_devel_embed, open(model_inputs_dir + "X_devel_" + f + "embedded.pkl", "wb"))
