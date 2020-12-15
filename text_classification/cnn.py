from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import classification_config as clf_config
import prepare_text_data as data
from glove_producer import produce_glove_vector_matrix

# import data from classification_config.py
embedding_dim = clf_config.EMBEDDING_DIM
validation_size = clf_config.VALIDATION_SIZE
epochs = clf_config.EPOCHS
test_size = clf_config.TEST_SIZE

# import values from prepare_text_data.py
vocab_size = data.vocab_size
token = data.token
max_length = data.max_length
X = data.X
y = data.y


# GLOVE VECTORS
word_vector_matrix = produce_glove_vector_matrix(embedding_dim, vocab_size, token)

# MODEL BUILDING
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=test_size, stratify=y)

# we will split the training data in valuation dataset
val_size = int(len(X_train)*validation_size)
X_val = X_train[:val_size]
X_train_partial = X_train[val_size:]
y_val = y_train[:val_size]
y_train_partial = y_train[val_size:]

# CNN model
model = Sequential()
# use this for custom embeddings
# model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
# or use this to enable GloVe pretrained embeddings
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length, weights=[word_vector_matrix]))
model.add(Conv1D(filters=32, kernel_size=5, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

class_weight = {
    0: 1,
    1: 5
}
model.fit(X_train_partial, y_train_partial, epochs=epochs, verbose=1, validation_data=(X_val, y_val), class_weight=class_weight)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))
