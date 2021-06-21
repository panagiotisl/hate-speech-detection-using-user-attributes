from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import classification_config as clf_config
import prepare_data as data
from glove_producer import produce_glove_vector_matrix

from tensorflow.keras.layers import Embedding, Dense, Input, LSTM, Concatenate, Bidirectional
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt

import pandas as pd
import seaborn as sn

# the following two lines are used to create the plot of the neural network model
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

# GLOVE VECTORS
word_vector_matrix = produce_glove_vector_matrix(clf_config.EMBEDDING_DIM, data.vocab_size, data.token)

# MODEL 1
input_1 = Input(shape=(data.max_length,))
embedding_layer = Embedding(data.vocab_size, clf_config.EMBEDDING_DIM, weights=[word_vector_matrix])(input_1)
LSTM_layer = Bidirectional(LSTM(100, dropout=0.5))(embedding_layer)
dense_text_layer = Dense(40, activation="sigmoid")(LSTM_layer)

# MODEL 2
input_2 = Input(shape=(data.user_attributes_count,))
dense_layer_1 = Dense(200, activation='sigmoid')(input_2)
dense_layer_2 = Dense(200, activation='sigmoid')(dense_layer_1)
dense_layer_3 = Dense(200, activation='sigmoid')(dense_layer_2)
dense_user_layer = Dense(200, activation='sigmoid')(dense_layer_3)

# concatenate the two outputs
concat_layer = Concatenate()([dense_text_layer, dense_user_layer])

# construct output layer
concatenate_dense_layer_1 = Dense(200, activation="sigmoid")(concat_layer)
concatenate_dense_layer_2 = Dense(200, activation="sigmoid")(concatenate_dense_layer_1)
output = Dense(1, activation="sigmoid")(concatenate_dense_layer_2)

# build final model
model = Model(inputs=[input_1, input_2], outputs=output)

# prepare the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])

# plot the model
# plot_model(model, to_file='../nn_plots/lstm_plus_user_attr/model_plot.png', show_shapes=True, show_layer_names=True)

# train the model
history = model.fit(x=[data.X_train_text, data.X_train_user], verbose=0, y=data.y_train, epochs=clf_config.EPOCHS,
                    validation_split=clf_config.VALIDATION_SIZE, class_weight={0: 1., 1: 1.})

pred = model.predict(x=[data.X_test_text, data.X_test_user])

seq_predictions = list(map(lambda x: 1 if (x > 0.5) else 0, pred))

print('Accuracy:', accuracy_score(data.y_test, seq_predictions))
print('F1 score:', f1_score(data.y_test, seq_predictions, average="weighted"))
print('Recall:', recall_score(data.y_test, seq_predictions, average="weighted"))
print('Precision:', precision_score(data.y_test, seq_predictions, average="weighted"))

conf_m = metrics.confusion_matrix(data.y_test, seq_predictions)

print(conf_m)

df_cm = pd.DataFrame(conf_m, range(2), range(2))
sn.set(font_scale=1)
sn.heatmap(df_cm, annot=True, fmt='d')
plt.xlabel("Actual")
plt.ylabel("Predicted")

attrs = ""
counter = 1
for i in clf_config.chosen_attributes_list:
    attrs += i
    if counter < len(clf_config.chosen_attributes_list):
        attrs += "_"
        counter += 1

plt.savefig("confusion_lstm_"+attrs+".png")

# epochs = range(1, clf_config.EPOCHS+1)
#
# # ACCURACY
# # plt.plot(epochs, history.history['acc'])
# # plt.plot(epochs, history.history['val_acc'])
# # plt.title('Training and Validation Accuracy')
# # plt.ylabel('accuracy')
# # plt.xlabel('epochs')
# # plt.legend(['train', 'val'], loc='lower right')
# # plt.savefig('cnn_plus_user_accuracy.png')
#
#
# # evaluate the model on the test dataset
loss, accuracy = model.evaluate(x=[data.X_test_text, data.X_test_user], y=data.y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))
print('Loss: %f' % (loss*100))
