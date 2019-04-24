import glob
import librosa
import numpy as np
import os
import pickle
import sys

sys.path.append('/home/jantunes/Desktop/joana/bs/code/')
from utils import load_glove_model, load_emotion_json, load_asr_json, compute_word_level_emotions

def compute_word_level_spectrograms(path_to_wav, asr_data):

    # load spectrograms and split into words segments using ars results
    try:
        path_to_s = glob.glob(os.path.dirname(path_to_wav) + "/*" + os.path.basename(path_to_wav)[:-4] + "*.pkl")[0]
        S_words = pickle.load(open(path_to_s, "rb"))

    except IndexError:

        print("Extracting word level mel spectrograms from path: " + path_to_wav)

        y, sr = librosa.load(path_to_wav, sr=16000)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=160)

        S_words = []

        for a, b, _ in asr_data:

            start = int(np.min([a*100, b*100])) # 1 frame = 10 ms
            end = int(np.max([a*100,b*100]))
            N = 70 # 700 ms

            if end - start >= N:
                mid = (end - start) / 2 + start
                s = S[:,int(mid-N/2):int(mid-N/2)+N]
            elif end - start < 10:
                s = np.zeros((S.shape[0], N))

            elif end < S.shape[1]:
                s = np.hstack([S[:, start:end], np.zeros((S.shape[0], int(N - (end - start))))])
            else:
                s = np.zeros((S.shape[0], N))

            S_words.append(s)

        S_words = np.asarray(S_words)

        pickle.dump(S_words, open(path_to_wav[:-4]+".pkl","wb"))

    return S_words


basepath = "/home/jantunes/Desktop/joana/bs/"
path_to_labels = basepath + "data/labels.txt"
path_to_audio_data = basepath + "data/wavs/"
out_path = basepath + "features/"
path_to_GloVe_model = basepath + "misc/glove.twitter.27B.25d.txt"

if not os.path.exists(out_path):
    os.makedirs(out_path)

with open(path_to_labels) as f:
    ids_labels = f.readlines()
ids_labels = [line.rstrip().split("\t") for line in ids_labels]

glove_model = load_glove_model(path_to_GloVe_model)

# fore each file of the dataset extract asr features, emotion features and spectral features + save features to
# feature dir

for line in ids_labels:

    file_id = line[0]
    label = line[-1]

    path_to_emotion = glob.glob(basepath + "data/emotions_frame/*" + file_id + "*.json")[0]
    path_to_asr = glob.glob(basepath + "data/asr/*" + file_id + "*.json")[0]
    path_to_wav = glob.glob(basepath + "data/wavs/*" + file_id + "*.wav")[0]

    emotion_data = load_emotion_json(path_to_emotion, dic_type="one_hot")
    asr_data, asr_data_glove = load_asr_json(path_to_asr, glove_model)

    word_level_emotions = compute_word_level_emotions(asr_data, emotion_data, dic_type="one_hot")
    S_words = compute_word_level_spectrograms(path_to_wav, asr_data)

    pickle.dump(asr_data_glove, open(out_path + file_id + "_gloves.pkl", "wb"))
    pickle.dump(word_level_emotions, open(out_path + file_id + "_emotions.pkl", "wb"))
    pickle.dump(S_words, open(out_path + file_id + "_spectrograms.pkl", "wb"))


