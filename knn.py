import pandas as pd
import numpy as np
from sklearn import model_selection, metrics
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("dataset/tweet_user_data.csv")

df = df[["user_total_tweets", "tweet_label"]]

prediction_column = "tweet_label"

X = np.array(df.drop([prediction_column], 1))
y = np.array(df[prediction_column])

total_accuracy = 0
total_f1 = 0
total_precision = 0
total_recall = 0
reps = 20

for _ in range(reps):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    knn_clf = KNeighborsClassifier(n_neighbors=15)
    knn_clf.fit(X_train, y_train)
    y_pred = knn_clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    precision = metrics.precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = metrics.recall_score(y_test, y_pred, average="weighted", zero_division=1)
    total_accuracy += acc
    total_f1 += f1
    total_precision += precision
    total_recall += recall


average_accuracy = total_accuracy / reps
average_f1 = total_f1 / reps
average_precision = total_precision /reps
average_recall = total_recall / reps

print(f"average accuracy after {reps} cycles = {average_accuracy}")
print(f"average F1 score after {reps} cycles = {average_f1}")
print(f"average precision score after {reps} cycles = {average_precision}")
print(f"average recall score after {reps} cycles = {average_recall}")

