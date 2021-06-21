from string import punctuation

import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import STOPWORDS
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

import classification_config as clf_config
from text_classification.preprocess_twitter import tokenize as tokenizer_g

test_size = clf_config.TEST_SIZE


def tokenize(tweet):
    return tokenizer_g(tweet)


def split_and_remove_punctuation_and_stopwords(tweet):
    text = ''.join([c for c in tweet if c not in punctuation])
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return words


# Preparing the text data
df = pd.read_csv("../dataset/tweet_user_data_final.csv")

df["tweet_text"] = df["tweet_text"].apply(tokenize).apply(split_and_remove_punctuation_and_stopwords)

# uncomment the line below for equally balanced dataset
# df.drop(df.index[:1100][df[:1100]["is_hate"] == 0], inplace=True)

text = df["tweet_text"]

# let's use Keras.Tokenizer to convert our sentences to vectors
token = Tokenizer()

token.fit_on_texts(text)

vocab_size = len(token.word_index) + 1

# converts text to sequence of vocab index (i.e [3, 5, 11, 1, 9, 7])
encoded_text = token.texts_to_sequences(text)

# get max length of sentences in corpus to create padded sentences of this length
max_length = len(max(encoded_text, key=len))

df["tweet_text"] = encoded_text

attributes = ["tweet_text"] + clf_config.chosen_attributes_list

X = np.array(df[attributes])

y = np.array(df["is_hate"])

# return how many user attributes we use to train our dataset (-1 because we remove the tweet_text)
_, user_attributes_count = X.shape
user_attributes_count -= 1

# MODEL BUILDING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=clf_config.TEST_SIZE, stratify=y)

X_train_text = pad_sequences(X_train[:, 0], maxlen=max_length, padding="post")
X_test_text = pad_sequences(X_test[:, 0], maxlen=max_length, padding="post")

X_train_user = np.delete(X_train, 0, 1).astype("float32")
X_test_user = np.delete(X_test, 0, 1).astype("float32")
