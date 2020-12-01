from string import punctuation

import numpy as np
import pandas as pd
import sklearn
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
from joblib.numpy_pickle_utils import xrange
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, Dropout, GlobalAveragePooling1D, Dense

from batch_gen import batch_gen
from preprocess_twitter import tokenize as tokenizer_g


def tokenize(tweet):
    return tokenizer_g(tweet)


def split_and_remove_punctuation_and_stopwords(tweet):
    text = ''.join([c for c in tweet if c not in punctuation])
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return words


# convert list of words to list of indices in the vocabulary
def gen_sequence(tweet):
    i = 0
    for word in tweet:
        tweet[i] = word2vec_model.wv.vocab[word].index
        i += 1
    return tweet


# words count from the vocab in form of tuples (i.e (3, 244) means word with index 3 appears 244 times)
def get_embedding_weights():
    return [(voc.index, voc.count) for k, voc in word2vec_model.wv.vocab.items()]


def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)


def fast_text_model(sequence_length):
    ft_model = Sequential()
    ft_model.add(Embedding(len(word2vec_model.wv.vocab)+1, 50, input_length=sequence_length))
    ft_model.add(Dropout(0.5))
    ft_model.add(GlobalAveragePooling1D())
    ft_model.add(Dense(2, activation='softmax'))
    ft_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return ft_model


def train_fasttext(X, y, model, epochs=10, batch_size=128):
    cv_object = KFold(n_splits=10, shuffle=True, random_state=42)
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    sentence_len = X.shape[1]
    lookup_table = np.zeros_like(model.layers[0].get_weights()[0])
    for train_index, test_index in cv_object.split(X):
        shuffle_weights(model)
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        for epoch in xrange(epochs):
            for X_batch in batch_gen(X_temp, batch_size):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]
        class_weights = {}
        class_weights[0] = np.where(y_temp == 0)[0].shape[0]/float(len(y_temp))
        class_weights[1] = np.where(y_temp == 1)[0].shape[0]/float(len(y_temp))
        try:
            y_temp = np_utils.to_categorical(y_temp, num_classes=2)
        except Exception as e:
            print(e)
        y_pred = model.predict_on_batch(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        p += precision_score(y_test, y_pred, average='weighted')
        p1 += precision_score(y_test, y_pred, average='micro')
        r += recall_score(y_test, y_pred, average='weighted')
        r1 += recall_score(y_test, y_pred, average='micro')
        f1 += f1_score(y_test, y_pred, average='weighted')
        f11 += f1_score(y_test, y_pred, average='micro')

    print("macro results are")
    print("average precision is %f" % (p/10))
    print("average recall is %f" % (r/10))
    print("average f1 is %f" % (f1/10))
    print("----------------------------------")
    print("micro results are")
    print("average precision is %f" % (p1/10))
    print("average recall is %f" % (r1/10))
    print("average f1 is %f" % (f11/10))
    return lookup_table/float(10)


############################
# MAIN APP
############################

# Preparing the text data
df = pd.read_csv("../dataset/tweet_user_data_final.csv")

df = df[["tweet_text", "is_hate"]]

df["tweet_text"] = df["tweet_text"].apply(tokenize).apply(split_and_remove_punctuation_and_stopwords)

# generates vocabulary
word2vec_model = Word2Vec(sentences=df["tweet_text"], size=100, window=5, min_count=1, workers=4)

# generates sequence converting strings into integers from the vocab
df["tweet_text"] = df["tweet_text"].apply(gen_sequence)

X = np.array(df["tweet_text"])

y = np.array(df["is_hate"])

# max list length
MAX_SEQUENCE_LENGTH = max(map(lambda x: len(x), X))

# using pad_sequences, all lists have the same length, padding 0 where needed
data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

data, y = sklearn.utils.shuffle(data, y)

model = fast_text_model(data.shape[1])

_ = train_fasttext(data, y, model, 50)


