from tensorflow.keras.layers import Embedding, Dense, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import classification_config as clf_config
import prepare_text_data as data
from glove_producer import produce_glove_vector_matrix

# the following two lines are used to create the plot of the neural network model
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

# GLOVE VECTORS
word_vector_matrix = produce_glove_vector_matrix(clf_config.EMBEDDING_DIM, data.vocab_size, data.token)

# LSTM model
model = Sequential()
# use this for custom embeddings
# model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
# or use this to enable GloVe pretrained embeddings
model.add(Embedding(data.vocab_size, clf_config.EMBEDDING_DIM, input_length=data.max_length, weights=[word_vector_matrix]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

model.fit(data.X_train, data.y_train, epochs=clf_config.EPOCHS, verbose=1, validation_split=clf_config.VALIDATION_SIZE)

loss, accuracy = model.evaluate(data.X_test, data.y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))
