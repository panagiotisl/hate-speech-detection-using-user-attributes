from sklearn import model_selection, svm
from sklearn import metrics
import data_config as data

total_accuracy = 0
total_f1 = 0
total_precision = 0
total_recall = 0
reps = data.cycles

for _ in range(reps):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data.X, data.y, test_size=data.test_size)
    svm_clf = svm.SVC()
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    precision = metrics.precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = metrics.recall_score(y_test, y_pred, average="weighted", zero_division=1)
    total_accuracy += acc
    total_f1 += f1
    total_precision += precision
    total_recall += recall

print(f"average accuracy after {reps} cycles = {total_accuracy / reps}")
print(f"average F1 score after {reps} cycles = {total_f1 / reps}")
print(f"average precision score after {reps} cycles = {total_precision / reps}")
print(f"average recall score after {reps} cycles = {total_recall / reps}")
