#TRAIN-SOLVER
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

if __name__ == '__main__':
    features_file_location = sys.argv[1]
    model_file_location = sys.argv[2]
    try:
        feature_mapping_file = sys.argv[3]
    except IndexError:
        feature_mapping_file = "feature_mapping"

    with open(feature_mapping_file, 'rb') as file:
        possible_tags = load(file)


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

    clf = LogisticRegression(random_state=0, solver='saga', max_iter=500, multi_class='multinomial', tol=1e-2).fit(X_train,
                                                                                                                y_train)


    with open(model_file_location, "wb") as file:
        dump(clf, file)
        
    with open(feature_mapping_file, "wb") as file:
        dump([possible_tags, vec, le], file)
