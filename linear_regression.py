from sklearn import linear_model, model_selection
import data_config as data

total_accuracy = 0
reps = data.cycles

for _ in range(reps):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data.X, data.y, test_size=data.test_size)
    linear_clf = linear_model.LinearRegression()
    linear_clf.fit(X_train, y_train)
    acc = linear_clf.score(X_test, y_test)
    total_accuracy += acc

average_accuracy = total_accuracy / reps

print(f"average accuracy after {reps} cycles = {average_accuracy}")
