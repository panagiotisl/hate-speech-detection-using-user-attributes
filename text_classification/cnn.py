from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from tensorflow.keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import classification_config as clf_config
import prepare_text_data as data
from glove_producer import produce_glove_vector_matrix
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd

# the following two lines are used to create the plot of the neural network model
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

# GLOVE VECTORS
word_vector_matrix = produce_glove_vector_matrix(clf_config.EMBEDDING_DIM, data.vocab_size, data.token)

# CNN model
model = Sequential()
# use this for custom embeddings
model.add(Embedding(data.vocab_size, clf_config.EMBEDDING_DIM, input_length=data.max_length))
# or use this to enable GloVe pretrained embeddings
# model.add(Embedding(data.vocab_size, clf_config.EMBEDDING_DIM, input_length=data.max_length, weights=[word_vector_matrix]))
model.add(Conv1D(filters=64, kernel_size=10, activation="relu"))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# plot the model
# plot_model(model, to_file='../nn_plots/cnn/model_plot.png', show_shapes=True, show_layer_names=True)

history = model.fit(data.X_train, data.y_train, epochs=clf_config.EPOCHS, verbose=0,
                    validation_split=clf_config.VALIDATION_SIZE)

pred = (model.predict(data.X_test) > 0.5).astype("int32")

print('Accuracy:', accuracy_score(data.y_test, pred))
print('F1 score:', f1_score(data.y_test, pred, average="weighted"))
print('Recall:', recall_score(data.y_test, pred, average="weighted"))
print('Precision:', precision_score(data.y_test, pred, average="weighted"))

# Confusion matrix
conf_m = metrics.confusion_matrix(data.y_test, pred)
print(conf_m)
df_cm = pd.DataFrame(conf_m, range(2), range(2))
sn.set(font_scale=1)
sn.heatmap(df_cm, annot=True, fmt='d')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("confusion_cnn.png")


# epochs = range(1, clf_config.EPOCHS+1)
#
# # ACCURACY
# plt.plot(epochs, history.history['accuracy'])
# plt.plot(epochs, history.history['val_accuracy'])
# plt.title('Training and Validation Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epochs')
# plt.legend(['train', 'val'], loc='upper left')
# plt.savefig('cnn_accuracy.png')

# LOSS
# plt.plot(epochs, history.history['loss'], 'g')
# plt.plot(epochs, history.history['val_loss'], 'b')
# plt.title('Training and Validation Loss')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(['train', 'val'], loc='upper left')
# plt.savefig('cnn_loss.png')

# loss, accuracy = model.evaluate(data.X_test, data.y_test, verbose=0)
# print('Accuracy: %f' % (accuracy*100))
# print('Loss: %f' % (loss*100))
