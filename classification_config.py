from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# #################################################
#   CLASSIFICATION PROPERTIES
#   1. uncomment the classifier you want to run
#   2. choose classification properties
# #################################################

# test size (default 80/20)
test_size = 0.2

# how many train cycles (default 20)
cycles = 20

# classifier
#classifier = KNeighborsClassifier(n_neighbors=15)
classifier = SVC()
