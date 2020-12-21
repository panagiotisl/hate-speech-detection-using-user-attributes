from sklearn import model_selection
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from user_attributes_classification import classification_config as clf, data_config as data

X_train, X_test, y_train, y_test = model_selection.train_test_split(data.X, data.y, test_size=clf.test_size)
classifier = clf.classifier
classifier.fit(X_train, y_train)
y_pred = classifier.predict_proba(X_test)

# roc curve for nn_plots
fpr, tpr, thresh = roc_curve(y_test, y_pred[:, 1], pos_label=1)

# roc curve for tpr = fpr
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

plt.style.use('seaborn')

classifier = str(clf.classifier).split("(")
# plot roc curves
plt.plot(fpr, tpr, linestyle='--', color='green', label=classifier[0])
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
save_file_name = ''
for f in data.attribute_columns:
    save_file_name += "_" + f
plt.savefig("../roc_plots/" + data.prediction_column + "/" + classifier[0] + save_file_name, dpi=300)
plt.show()
