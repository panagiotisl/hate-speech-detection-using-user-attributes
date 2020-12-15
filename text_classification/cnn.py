from tensorflow.keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import classification_config as clf_config
import prepare_text_data as data
from glove_producer import produce_glove_vector_matrix

# GLOVE VECTORS
word_vector_matrix = produce_glove_vector_matrix(clf_config.EMBEDDING_DIM, data.vocab_size, data.token)

# CNN model
model = Sequential()
# use this for custom embeddings
# model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
# or use this to enable GloVe pretrained embeddings
model.add(Embedding(data.vocab_size, clf_config.EMBEDDING_DIM, input_length=data.max_length, weights=[word_vector_matrix]))
model.add(Conv1D(filters=64, kernel_size=5, activation="relu"))
model.add(MaxPooling1D(pool_size=5))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

model.fit(data.X_train_partial, data.y_train_partial, epochs=clf_config.EPOCHS, verbose=1, validation_data=(data.X_val, data.y_val))

loss, accuracy = model.evaluate(data.X_test, data.y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))
