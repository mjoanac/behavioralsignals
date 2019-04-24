import os
import random
import pickle
import glob
import numpy as np

def unpikle(path):

    data = []
    try:
        path = glob.glob(path)[0]
        data = pickle.load(open(path,"rb"))
    except IndexError:
        pass

    return data


def load_spectrograms(file_id, path_to_features):

    data = unpikle(path_to_features + file_id + "_spectrograms.pkl")
    data = np.asarray(data)

    return data


def load_emotions(file_id, path_to_features):

    data = unpikle(path_to_features + file_id + "_emotions.pkl")
    data = np.asarray(data)

    return data


def load_gloves(file_id, path_to_features):

    data = unpikle(path_to_features + file_id + "_gloves.pkl")
    data = np.asarray(data)

    return data


def load_data(ids_labels,
              path_to_features,
              bool_load_spectrograms,
              bool_load_emotion,
              bool_load_words_and_gloves):

    X_s = []
    X_e = []
    X_g = []
    y = []
    y_ids = []

    for line in ids_labels:

        file_id = line[0]
        label = line[-1]
        n = 0

        print("Generating X_" + file_id + "...")

        if bool_load_spectrograms:
            X_s.append(load_spectrograms(file_id, path_to_features))
            n = X_s[-1].shape[0]
        if bool_load_emotion:
            X_e.append(load_emotions(file_id, path_to_features))
            n = X_e[-1].shape[0]
        if bool_load_words_and_gloves:
            X_g.append(load_gloves(file_id, path_to_features))
            n = X_g[-1].shape[0]

        y.append([int(label)] * n)
        y_ids.append([file_id] * n)

    return X_s, X_e, X_g, y, y_ids


basepath = "/home/jantunes/Desktop/joana/bs/"
path_to_labels = basepath + "data/labels.txt"
path_to_features = basepath + "features/"
path_to_audio_data = basepath + "data/wavs/"
out_dir = basepath + "model_inputs/"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

bool_load_spectrograms = True
bool_load_emotion = True
bool_load_words_and_gloves = True

bool_shuffle_train_devel_slpit = False

with open(path_to_labels) as f:
    ids_labels = f.readlines()
ids_labels = [line.rstrip().split("\t") for line in ids_labels]

train_devel_split = 0.7
n = int(len(ids_labels) * train_devel_split)

if bool_shuffle_train_devel_slpit:

    ids_labels_shuffle = ids_labels
    random.shuffle(ids_labels_shuffle)
    ids_labels_train = ids_labels_shuffle[:n]
    ids_labels_devel = ids_labels_shuffle[n:]

    print(ids_labels_train)
    print(ids_labels_devel)

else:

    ids_labels_train = ids_labels[:n]
    ids_labels_devel = ids_labels[n:]

X_train_s, X_train_e, X_train_g, y_train, y_video_ids_train = load_data(ids_labels_train,
                                                                        path_to_features,
                                                                        bool_load_spectrograms,
                                                                        bool_load_emotion,
                                                                        bool_load_words_and_gloves)

X_devel_s, X_devel_e, X_devel_g, y_devel, y_video_ids_devel = load_data(ids_labels_devel,
                                                                        path_to_features,
                                                                        bool_load_spectrograms,
                                                                        bool_load_emotion,
                                                                        bool_load_words_and_gloves)

print("Dumping ...")

pickle.dump(X_train_s, open(out_dir + "X_train_s.pkl", "wb"))
pickle.dump(X_train_e, open(out_dir + "X_train_e.pkl", "wb"))
pickle.dump(X_train_g, open(out_dir + "X_train_g.pkl", "wb"))
pickle.dump(y_train, open(out_dir + "y_train.pkl", "wb"))
pickle.dump(y_video_ids_train, open(out_dir + "y_video_ids_train.pkl", "wb"))

pickle.dump(X_devel_s, open(out_dir + "X_devel_s.pkl", "wb"))
pickle.dump(X_devel_e, open(out_dir + "X_devel_e.pkl", "wb"))
pickle.dump(X_devel_g, open(out_dir + "X_devel_g.pkl", "wb"))
pickle.dump(y_devel, open(out_dir + "y_devel.pkl", "wb"))
pickle.dump(y_video_ids_devel, open(out_dir + "y_video_ids_devel.pkl", "wb"))

