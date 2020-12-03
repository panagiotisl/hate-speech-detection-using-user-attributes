from string import punctuation

import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import STOPWORDS
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from preprocess_twitter import tokenize as tokenizer_g


def tokenize(tweet):
    return tokenizer_g(tweet)


def split_and_remove_punctuation_and_stopwords(tweet):
    text = ''.join([c for c in tweet if c not in punctuation])
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return words


############################
# MAIN APP
############################

# Preparing the text data
df = pd.read_csv("../dataset/tweet_user_data_final.csv")

df = df[["tweet_text", "is_hate"]]

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

y = df["is_hate"]


######### GLOVE VECTORS
GLOVE_FILE = "C:\\Users\\giorg\\Documents\\Thesis\\GloveModelFile\\glove.twitter.27B.50d.txt"

glove_vectors = dict()

file = open(GLOVE_FILE, encoding="utf-8")

for line in file:
    values = line.split()
    word = values[0]
    vectors = np.asarray(values[1:])
    glove_vectors[word] = vectors
file.close()

word_vector_matrix = np.zeros((vocab_size, 50))

for word, index in token.word_index.items():
    vector = glove_vectors.get(word)
    if vector is not None:
        word_vector_matrix[index] = vector


########### MODEL BUILDING
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

vector_size = 50

model = Sequential()

# use this for custom embeddings
# model.add(Embedding(vocab_size, vector_size, input_length=max_length, trainable=False))
# or use this to enable GloVe pretrained embeddings
model.add(Embedding(vocab_size, vector_size, input_length=max_length, weights=[word_vector_matrix], trainable=False))
model.add(Flatten())
model. add(Dense(1, activation="sigmoid"))

model.compile(optimizer=Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)

loss, accuracy = model.evaluate(X, y, verbose=0)
print('Accuracy: %f' % (accuracy*100))

