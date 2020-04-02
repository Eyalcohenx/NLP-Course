# TRAIN-SOLVER
import re
import sys
from pickle import dump, load
from collections import defaultdict
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
import warnings

warnings.filterwarnings("ignore", message="Precision is ill-defined")
warnings.filterwarnings("ignore", message="Recall is ill-defined")

possible_tags = load(open("feature_mapping", "rb"))

features_file_location = sys.argv[1]
model_file_location = sys.argv[2]


def process_line(line):
    dict = defaultdict()
    temp = [re.search(r'(.*)=(.*$)', pair).groups() for pair in line.split()[1:]]
    for left, right in temp:
        dict[left] = right
    return line.split()[0], dict


def read_file(file):
    with open(file, 'r') as file:
        return [process_line(line) for line in file.readlines()]


y_train, X_train = zip(*read_file(features_file_location))  # features_file

vec = DictVectorizer()
X_train = vec.fit_transform(list(X_train))

le = preprocessing.LabelEncoder()
le.fit(y_train)

y_train = le.transform(y_train)

# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

clf = LogisticRegression(random_state=0, solver='saga', max_iter=500, multi_class='multinomial', tol=1e-2).fit(X_train,
                                                                                                               y_train)

# find the accuracy of the model
# acc_test = clf.score(X_test, y_test)
# acc_train = clf.score(X_train, y_train)
# print("Train Accuracy:", acc_train)
# print("Test Accuracy:", acc_test)
#
## compute predictions on test features
# pred = clf.predict(X_test)
#
# precision = precision_score(y_test, pred, average="weighted")
# recall = recall_score(y_test, pred, average="weighted")
#
# print("Precision:", precision)
# print("Recall:", recall)

dump(clf, open(model_file_location, "wb"))  # model_file
dump([possible_tags, vec, le], open("feature_mapping", "wb"))
