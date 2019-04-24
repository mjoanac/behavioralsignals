import json
import pandas as pd
import csv
import numpy as np
import re
import os
import pickle
import glob
from sklearn import metrics


def decontract(phrase):

    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    return phrase.split(" ")[0]


def load_glove_model(path):

    # load pre trained GloVe word embedding model

    model = pd.read_csv(path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    # to get word vector:
    # words.loc[w].as_matrix()

    return model


def load_emotion_json(path, dic_type="one_hot"):

    # load emotion data from json

    try:
        path_to_emotion = glob.glob(os.path.dirname(path) + "/*" + os.path.basename(path)[:-5] + "*" + dic_type + ".pkl")[0]
        emotion_data = pickle.load(open(path_to_emotion, "rb"))

    except IndexError:

        print("loading Emotion data from path: " + path)

        if dic_type == "one_hot":

            dic = {1.0: [1, 0, 0, 0, 0, 0],
                   2.0: [0, 1, 0, 0, 0, 0],
                   3.0: [0, 0, 1, 0, 0, 0],
                   4.0: [0, 0, 0, 1, 0, 0],
                   5.0: [0, 0, 0, 0, 1, 0],
                   6.0: [0, 0, 0, 0, 0, 1],
                   7.0: [0, 0, 0, 0, 0, 0]
                   }

        elif dic_type == "categorical":

            dic = {1.0: "Happy",
                   2.0: "Neutral",
                   3.0: "Angry",
                   4.0: "Sad",
                   5.0: "Frustrated",
                   6.0: "Ambiguous",
                   7.0: "None"
                   }
        else:
            print("wrong dictionary type!")
            return []

        with open(path) as json_file:
            data = json.load(json_file)

        emotion_data = []

        for frame in data["frames"]:
            if frame["speakers"] is not None:
                for speaker in frame["speakers"][:1]:
                    if speaker["emotion"]:
                        emotion_data.append(dic[speaker["emotion"]["framelevel"]])
                    else:
                        if dic_type == "one_hot":
                            emotion_data.append([0, 0, 0, 0, 0, 0])
                        else:
                            emotion_data.append("None")
            else:
                if dic_type == "one_hot":
                    emotion_data.append([0, 0, 0, 0, 0, 0])
                else:
                    emotion_data.append("None")

        pickle.dump(emotion_data, open(path[:-5] + dic_type + ".pkl", "wb"))

    return emotion_data


def load_asr_json(path, glove_model):

    # load asr data from json

    try:
        path_to_asr = glob.glob(os.path.dirname(path) + "/*" + os.path.basename(path)[:-5] + "*.pkl")[0]
        [asr_data, asr_data_glove] = pickle.load(open(path_to_asr, "rb"))

    except IndexError:
        print("loading ASR data from path: " + path)

        with open(path) as json_file:
            data = json.load(json_file)

        asr_data = []
        asr_data_glove = []

        for word_data in data["words"]:

            word = word_data["w"]

            try:
                w = glove_model.loc[word].values
                asr_data.append([word_data["st"], word_data["et"], word_data["w"]])
                asr_data_glove.append(w)

            except KeyError:
                try:
                    w = glove_model.loc[decontract(word)].values
                    asr_data.append([word_data["st"], word_data["et"], word_data["w"]])
                    asr_data_glove.append(w)

                except KeyError:
                    # print("word not found: " + word)
                    pass

        pickle.dump([asr_data, asr_data_glove], open(path[:-5]+".pkl","wb"))

    return asr_data, asr_data_glove


def compute_word_level_emotions(asr_data, emotion_data, dic_type="one_hot"):

    # compute word level emotions by retrieving the emotions for a given word
    # through their timestamps, then choose most common

    word_level_emotions = []

    for t_start, t_end, word in asr_data:

        data_from_emotion = emotion_data[int(t_start * 10):int(t_end * 10)]

        if data_from_emotion:
            if data_from_emotion.count(data_from_emotion[0]) == len(data_from_emotion):
                most_common_emotion = data_from_emotion[0]
            else:
                if dic_type == "one_hot":
                    idx_max = np.argmax(np.sum(np.asarray(data_from_emotion), axis=0))
                    most_common_emotion = [0] * len(data_from_emotion[0])
                    most_common_emotion[idx_max] = 1
                else:
                    most_common_emotion = max(set(data_from_emotion), key=data_from_emotion.count)
        else:
            if dic_type == "one_hot":
                most_common_emotion = [0, 0, 0, 0, 0, 0]
            else:
                most_common_emotion = "None"

        word_level_emotions.append(most_common_emotion)

    return word_level_emotions


def normalize_X(X, X_mean, X_std):

    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            if not all(X[i,:,j])==0:
                X[i,:,j] = (X[i,:,j] - X_mean) / X_std

    return X


def pick_best_decision_threshold(y_train, y_pred_train, y_pred_devel):

    best_theta = 0.5
    best_performance = metrics.accuracy_score(y_train, np.asarray([1 if y >= best_theta else 0 for y in y_pred_train]))

    for theta in np.linspace(0, 1, 11):
        y_pred_train_bin = np.asarray([1 if y >= theta else 0 for y in y_pred_train])
        performance = metrics.accuracy_score(y_train, y_pred_train_bin)
        if performance > best_performance:
            best_performance = performance
            best_theta = theta

    print(best_theta)
    print(best_performance)

    y_pred_train_bin = np.asarray([1 if y >= best_theta else 0 for y in y_pred_train])
    y_pred_devel_bin = np.asarray([1 if y >= best_theta else 0 for y in y_pred_devel])

    return y_pred_train_bin, y_pred_devel_bin


def print_performace_metrics(y,y_pred, phonemes="", comment="performance: "):

    print(comment)
    print(" ".join(phonemes) + '\t' +
          str(np.mean(y)) + '\t' +
          str(metrics.accuracy_score(y, y_pred)) + '\t' +
          str(metrics.f1_score(y, y_pred, average="weighted")) + '\t' +
          str(metrics.precision_score(y, y_pred, average="weighted")) + '\t' +
          str(metrics.recall_score(y, y_pred, average="weighted")) + '\t' +
          str(metrics.recall_score(y, y_pred, average="macro"))
          )
    print(metrics.classification_report(y, y_pred))
    print(metrics.confusion_matrix(y, y_pred))

    return


def convert_word_level_predictions_to_interview_level_predictions(y, ids):

    ids = [i for x in ids for i in x]

    y_interview = []
    
    s = sorted(set(ids))
    
    for id in s:    
        idx = [i for i, x in enumerate(ids) if x == id]
        y_subset = [y[i] for i in idx]
        y_subset = np.average(y_subset)
        y_subset = np.round(y_subset)
        y_interview.append(int(y_subset))

    y_interview = np.asarray(y_interview)

    return y_interview
