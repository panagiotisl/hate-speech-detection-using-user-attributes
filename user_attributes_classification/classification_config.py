from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb


# #################################################
#   CLASSIFICATION PROPERTIES
#   1. uncomment the classifier you want to run
#   2. choose classification properties
# #################################################

# test size (default 80/20)
test_size = 0.2

# how many train cycles (default 20)
cycles = 1

# classifier
classifier = KNeighborsClassifier(n_neighbors=15)
# classifier = SVC(probability=True)
# classifier = RandomForestClassifier(max_depth=40, random_state=0)
# classifier = GaussianNB()
# classifier = GradientBoostingClassifier(random_state=0)
# classifier = xgb.XGBClassifier(booster='gbtree', n_estimators=1000, learning_rate=0.8, max_depth=40, random_state=23)
