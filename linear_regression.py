import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection

df = pd.read_csv("dataset/tweet_user_data.csv")

df = df[["user_following", "tweet_label"]]

prediction_column = "tweet_label"

X = np.array(df.drop([prediction_column], 1))
y = np.array(df[prediction_column])

total_accuracy = 0
reps = 20

for _ in range(reps):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    linear_clf = linear_model.LinearRegression()
    linear_clf.fit(X_train, y_train)
    acc = linear_clf.score(X_test, y_test)
    total_accuracy += acc

average_accuracy = total_accuracy / reps

print(f"average accuracy after {reps} cycles = {average_accuracy}")
