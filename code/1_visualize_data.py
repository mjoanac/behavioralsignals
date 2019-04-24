import glob
import sys
import numpy as np
import collections
import random
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
from sklearn.manifold import TSNE

sys.path.append('/home/jantunes/Desktop/joana/bs/code/')
from utils import decontract, load_glove_model, load_emotion_json, load_asr_json, compute_word_level_emotions


def plot_results(X_train, X_devel, y_train, y_devel, model_type, feature_type, title):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=pd.DataFrame(y_train)[0], s=50)
    ax.set_title(title)
    plt.colorbar(scatter)
    fig.savefig(basepath + model_type + "_prediction_X_train_" + feature_type + ".png")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X_devel[:, 0], X_devel[:, 1], c=pd.DataFrame(y_devel)[0], s=50)
    ax.set_title(title)
    plt.colorbar(scatter)
    fig.savefig(basepath + model_type + "_prediction_X_devel_" + feature_type + ".png")

    return


def cluster(X_train, X_devel, y_train, y_devel, n=2):

    print("Custering data...")

    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X_train)

    y_pred_train = kmeans.predict(X_train)
    y_pred_devel = kmeans.predict(X_devel)

    print("\n")
    print("Classification report for model performance on TRAIN data:")
    print("\n")
    print(metrics.classification_report(y_train, y_pred_train))
    print("\n")
    print("Classification report for model performance on DEVEL data:")
    print("\n")
    print(metrics.classification_report(y_devel, y_pred_devel))

    return kmeans, y_pred_train, y_pred_devel


def classify(X_train, X_devel, y_train, y_devel, c=1):

    print("Training classifier on data...")

    clf = svm.SVC(C=c, kernel='linear')
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_devel = clf.predict(X_devel)

    print("\n")
    print("Classification report for model performance on TRAIN data:")
    print("\n")
    print(metrics.classification_report(y_train, y_pred_train))
    print("\n")
    print("Classification report for model performance on DEVEL data:")
    print("\n")
    print(metrics.classification_report(y_devel, y_pred_devel))

    return clf, y_pred_train, y_pred_devel

# %%

basepath = "/home/jantunes/Desktop/joana/bs/"
path_to_labels = basepath + "data/labels.txt"
path_to_GloVe_model = basepath + "misc/glove.twitter.27B.25d.txt"

glove_model = load_glove_model(path_to_GloVe_model)

with open(path_to_labels) as f:
    ids_labels = f.readlines()
ids_labels = [line.rstrip().split("\t") for line in ids_labels]

# Load categorical emotion information for positive and negative videos

emotion_data_categorical_positive = []
emotion_data_categorical_negative = []

for line in ids_labels:

    file_id = line[0]
    label = line[-1]

    path_to_emotion = glob.glob(basepath + "data/emotions_frame/*" + file_id + "*.json")[0]

    if label == "1":
        emotion_data_categorical_positive += load_emotion_json(path_to_emotion, dic_type="categorical")

    else:
        emotion_data_categorical_negative += load_emotion_json(path_to_emotion, dic_type="categorical")

# Generate categorical histogram plots of emotion data for positive and negative partitions of data

fig1 = plt.figure()
df1 = pd.DataFrame({"non-depressed": emotion_data_categorical_negative})
df1["non-depressed"].value_counts().plot("bar").set_title("Non-depressed")
fig1.savefig(basepath + "emotion_distribution_non_depressed.png")

fig2 = plt.figure()
df2 = pd.DataFrame({"depressed": emotion_data_categorical_positive})
df2["depressed"].value_counts().plot("bar").set_title("Depressed")
fig2.savefig(basepath + "emotion_distribution_depressed.png")

# Load one-hot emotion information for train and devel partitions of data (70/30 split), average over each video

split = 0.7
N = int(len(ids_labels) * split)

ids_labels_train = ids_labels[:N]
ids_labels_devel = ids_labels[N:]

emotion_data_one_hot_train = []
emotion_data_one_hot_devel = []

y_train = []
y_devel = []

for line in ids_labels_train:
    file_id = line[0]
    path_to_emotion = glob.glob(basepath + "data/emotions_frame/*" + file_id + "*.json")[0]
    e = load_emotion_json(path_to_emotion, dic_type="one_hot")
    emotion_data_one_hot_train.append(np.sum(e,axis=0)/np.sum(e))
    y_train += [int(line[-1])]

for line in ids_labels_devel:
    file_id = line[0]
    path_to_emotion = glob.glob(basepath + "data/emotions_frame/*" + file_id + "*.json")[0]
    e = load_emotion_json(path_to_emotion, dic_type="one_hot")
    emotion_data_one_hot_devel.append(np.sum(e,axis=0)/np.sum(e))
    y_devel += [int(line[-1])]

# Reduce data dimentionality to 2 with PCA

pca_emotion = PCA(n_components=2, svd_solver='full')
pca_emotion.fit(emotion_data_one_hot_train)

X_train_emotion = pca_emotion.transform(emotion_data_one_hot_train)
X_devel_emotion = pca_emotion.transform(emotion_data_one_hot_devel)


# Plot ground truth for emotion features

plot_results(X_train_emotion, X_devel_emotion, y_train, y_devel, "ground_truth", "emotion", "Ground truth for emotion features projected in 2D")

# Train Kmeans on train partition and plot

print("\n")
print("Clustering emotion data in original dimentions: ")
_, _, _ = cluster(emotion_data_one_hot_train, emotion_data_one_hot_devel, y_train, y_devel)

print("\n")
print("Clustering emotion data projected in 2 dimentions using PCA: ")
_, y_pred_train_kmeans, y_pred_devel_kmeans = cluster(X_train_emotion, X_devel_emotion, y_train, y_devel)

plot_results(X_train_emotion, X_devel_emotion, y_pred_train_kmeans, y_pred_devel_kmeans, "kmeans", "emotion", "Predicted K-Means Clusters")

# Train SVM and plot

print("\n")
print("Training SVM using emotion data in original dimentions: ")
_, _, _ = classify(emotion_data_one_hot_train, emotion_data_one_hot_devel, y_train, y_devel)
print("\n")
print("Training SVM using emotion data projected in 2 dimentions using PCA: ")
_, y_pred_train_svm, y_pred_devel_svm = classify(X_train_emotion, X_devel_emotion, y_train, y_devel, c=0.01)

plot_results(X_train_emotion, X_devel_emotion, y_pred_train_svm, y_pred_devel_svm, "svm", "emotion", "Labels estimated from SVM")

# Load word level information: words, semantic vectors, emotions at word level

words = []
gloves = []
emotions = []
emotions_one_hot =[]
labels = []

for line in ids_labels:

    file_id = line[0]
    label = line[-1]

    path_to_asr = glob.glob(basepath + "data/asr/*"+file_id+"*.json")[0]
    path_to_emotion = glob.glob(basepath + "data/emotions_frame/*" + file_id + "*.json")[0]

    asr_data, asr_data_glove = load_asr_json(path_to_asr, glove_model)

    emotion_data = load_emotion_json(path_to_emotion, dic_type="categorical")
    emotion_data_one_hot = load_emotion_json(path_to_emotion, dic_type="one_hot")

    word_level_emotions = compute_word_level_emotions(asr_data, emotion_data, dic_type="categorical")
    word_level_emotions_one_hot = compute_word_level_emotions(asr_data, emotion_data_one_hot, dic_type="one_hot")

    words.append(asr_data)
    gloves.append(asr_data_glove)
    emotions.append(word_level_emotions)
    emotions_one_hot.append(word_level_emotions_one_hot)
    labels.append(label)

# Compute average glove for each video and reduce dim to 2d w/ pca

gloves_ave = [np.mean(sublist, axis=0) for sublist in gloves]

gloves_train = np.asarray(gloves_ave[:N])
gloves_devel = np.asarray(gloves_ave[N:])

X_train_glove = gloves_ave[:N]
X_devel_glove = gloves_ave[N:]

pca_glove = PCA(n_components=2, svd_solver='full')
pca_glove.fit(X_train_glove)

X_train_glove = pca_glove.transform(X_train_glove)
X_devel_glove = pca_glove.transform(X_devel_glove)

# Plot ground truth for glove features

plot_results(X_train_glove, X_devel_glove, y_train, y_devel, "ground_truth", "glove", "Ground truth for GloVe features projected in 2D")

# Train Kmeans on train partition and plot

print("\n")
print("Clustering GloVe data in original dimentions: ")
_, _, _ = cluster(gloves_train, gloves_devel, y_train, y_devel)

print("\n")
print("Clustering GloVe data projected in 2 dimentions using PCA: ")
_, y_pred_train_kmeans, y_pred_devel_kmeans = cluster(X_train_glove, X_devel_glove, y_train, y_devel)

plot_results(X_train_glove, X_devel_glove, y_pred_train_kmeans, y_pred_devel_kmeans, "kmeans", "glove", "Predicted K-Means Clusters")

# Train SVM and plot

print("\n")
print("Training SVM using GloVe data in original dimentions: ")
_, _, _ = classify(gloves_train, gloves_devel, y_train, y_devel)
print("\n")
print("Training SVM using GloVe data projected in 2 dimentions using PCA: ")
_, y_pred_train_svm, y_pred_devel_svm = classify(X_train_glove, X_devel_glove, y_train, y_devel)

plot_results(X_train_glove, X_devel_glove, y_pred_train_svm, y_pred_devel_svm, "svm", "glove", "Labels estimated from SVM")

# remove stopwords

words_flat = [item for sublist in words for item in sublist]
gloves_flat = [item for sublist in gloves for item in sublist]
emotions_flat = [item for sublist in emotions for item in sublist]
emotions_one_hot_flat = [item for sublist in emotions_one_hot for item in sublist]

words_no_stopwords = []
gloves_no_stopwords = []
emotions_no_stopwords = []
emotions_one_hot_no_stopwords = []
labels_no_stopwords = []

stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
             'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
             'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
             'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
             'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
             'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
             'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
             'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
             'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
             'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'um', 'like', 'yeah', 'uh', 'oh', 'ah']

for w, g, e, e_oh in zip(words_flat, gloves_flat, emotions_flat, emotions_one_hot_flat):
    if w[2] not in stopwords and not all(g) == 0 and decontract(w[2]) not in stopwords:
            words_no_stopwords.append(w[2])
            gloves_no_stopwords.append(g)
            emotions_no_stopwords.append(e)
            emotions_one_hot_no_stopwords.append(e_oh)

# reduce dimensionality of gloves using t-SNE

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
gloves_2d = tsne_model.fit_transform(np.asarray(gloves_no_stopwords))

# select top N most common words per sentiment and plot

print("Selecting and plotting top N most common words per sentiment... ")

top_n = {}

plt.figure(figsize=(16, 16))

emotions = ["Happy", "Neutral", "Angry", "Sad", "Frustrated",
            "Ambiguous", "None"]

colors = cm.rainbow(np.linspace(0, 1, len(emotions)))

for emotion, color in zip(emotions, colors):

    idx = [i for i, x in enumerate(emotions_no_stopwords) if x == emotion]

    if idx:

        tmp_words = [decontract(words_no_stopwords[i]) for i in idx]
        tmp_gloves = [gloves_no_stopwords[i] for i in idx]
        top_n[emotion] = collections.Counter(tmp_words).most_common(10)

        print("\n")
        print('Top 10 words uttered with emotion "' + emotion + '":')
        print(top_n[emotion][:10])

        top_n_gloves = []
        top_n_words = []
        for w in top_n[emotion]:
            top_n_gloves.append(tmp_gloves[tmp_words.index(w[0])])
            top_n_words.append(w)

        top_n_gloves = np.asarray(top_n_gloves)

        r_x = (random.random()-0.5)*0.1
        r_y = (random.random()-0.5)*0.1

        plt.scatter(top_n_gloves[:, 0] + r_x , top_n_gloves[:, 1] + r_y, s=[w[1] for w in top_n_words]*5000, c=color)
        for i in range(top_n_gloves.shape[0]):
            plt.annotate(top_n_words[i][0],
                         xy=(top_n_gloves[i, 0] + r_x, top_n_gloves[i, 1] + r_y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

plt.savefig(basepath + "scatter_top_n_words_per_emotion.png")

# plot correlation between variables in original dimensions

print("Computing and plotting correlations between variables... ")

data = np.hstack([emotions_one_hot_no_stopwords, gloves_no_stopwords])
d = {'e_d0': data[:,0], 'e_d1': data[:,1], 'e_d2': data[:,2], 'e_d3': data[:,3], 'e_d4': data[:,4],
     'e_d5': data[:,5],
     'g_d0': data[:,6], 'g_d1': data[:,7], 'g_d2': data[:,8], 'g_d3': data[:,9], 'g_d4': data[:,10],
     'g_d5': data[:,11], 'g_d6': data[:,12], 'g_d7': data[:,13], 'g_d8': data[:,14], 'g_d9': data[:,15],
     'g_d10': data[:,16], 'g_d11': data[:,17], 'g_d12': data[:,18], 'g_d13': data[:,19], 'g_d14': data[:,20],
     'g_d15': data[:,21], 'g_d16': data[:,22], 'g_d17': data[:,23], 'g_d18': data[:,24], 'g_d19': data[:,25],
     'g_d20': data[:,26], 'g_d21': data[:,27], 'g_d22': data[:,28], 'g_d23': data[:,29], 'g_d24': data[:,30]}
df = pd.DataFrame(data=d)
corr = df.corr()
fig, ax = plt.subplots()
ax.matshow(corr)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.savefig(basepath + "variable_correlations.png")

# plot correlation between variables in 2-D

data = np.hstack([pca_emotion.transform(emotions_one_hot_no_stopwords), pca_glove.transform(gloves_no_stopwords)])
d = {'emotion_dim1': data[:,0], 'emotion_dim2': data[:,1], 'glove_dim1': data[:,2], 'glove_dim2': data[:,3]}
df = pd.DataFrame(data=d)
corr = df.corr()
fig, ax = plt.subplots()
ax.matshow(corr)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.savefig(basepath + "variable_correlations_2d.png")

