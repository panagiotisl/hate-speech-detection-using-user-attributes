from sklearn import model_selection, metrics
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import data_config as data

X_train, X_test, y_train, y_test = model_selection.train_test_split(data.X, data.y, test_size=data.test_size)
knn_clf = KNeighborsClassifier(n_neighbors=data.knn_neighbors)
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict_proba(X_test)

# roc curve for models
fpr, tpr, thresh = roc_curve(y_test, y_pred[:, 1], pos_label=1)

# roc curve for tpr = fpr
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr, tpr, linestyle='--', color='green', label='KNN')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('roc_plots/ROC', dpi=300)
plt.show()
