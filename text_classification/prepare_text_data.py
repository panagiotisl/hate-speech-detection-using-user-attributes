from string import punctuation

import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import STOPWORDS
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

import classification_config as clf_config
from preprocess_twitter import tokenize as tokenizer_g


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

text = df["tweet_text"]

# let's use Keras.Tokenizer to convert our sentences to vectors
token = Tokenizer()

token.fit_on_texts(text)

vocab_size = len(token.word_index) + 1

# converts text to sequence of vocab index (i.e [3, 5, 11, 1, 9, 7])
encoded_text = token.texts_to_sequences(text)

# get max length of sentences in corpus to create padded sentences of this length
max_length = len(max(encoded_text, key=len))

X = pad_sequences(encoded_text, maxlen=max_length, padding="post")

y = np.array(df["is_hate"])

# MODEL BUILDING
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=clf_config.TEST_SIZE, stratify=y)

