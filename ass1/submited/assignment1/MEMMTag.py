#MEMMTag
import re
import sys
from joblib import dump, load
from collections import defaultdict, namedtuple
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import warnings
from time import time
from ExtractFeatures import signatures, word_to_features
from contextlib import contextmanager
times_dict = defaultdict(int)
@contextmanager
def timethis(label):
    t0 = time()
    yield
    times_dict[label] += time() -  t0
warnings.filterwarnings("ignore", message="Precision is ill-defined")
warnings.filterwarnings("ignore", message="Recall is ill-defined")

input_file_name =sys.argv[1]
model_name =  sys.argv[2]
feature_map_file = sys.argv[3]
out_file_name = sys.argv[4]


def check_availability(element, collection):
    return element in collection


class HasNextIterator:
    def __init__(self, it):
        self._it = iter(it)
        self._next = None

    def __iter__(self):
        return self

    def has_next(self):
        if self._next:
            return True
        try:
            self._next = next(self._it)
            return True
        except StopIteration:
            return False

    def next(self):
        if self._next:
            ret = self._next
            self._next = None
            return ret
        elif self.has_next():
            return self.next()
        else:
            raise StopIteration()


def process_line(line):
    return [re.search(r'(.*)/.*$', pair).groups()[0] for pair in line.split()]

def read_input(file):
    try:
        with open(file, 'r') as f:
            sents = [process_line(line) for line in f.readlines()]
    except AttributeError:
        with open(file, 'r') as f:
            sents = [line.split() for line in f.readlines()]
    return sents

test_sents = read_input(input_file_name)

clf = load(model_name)

feature_mapping = load(feature_map_file)
possible_tags, vec, le = feature_mapping
possible_tags["START"] = {'<START>'}

for k, p in possible_tags.items():
    possible_tags[k] = set(p)

def check_if_rare_with_vectorizer(word):
    if check_availability("w_i=" + word, vec.get_feature_names()):
        return False
    return True

START = '<START>'
END = '<END>'

y_predicted = []

n_tags = len(le.classes_) + 1

def score_sentence(clf, pos_tags_pp_s, pos_tags_p_s, word_features):
    features_list = []
    N = len(word_features)
    features_sets_limits = [0]
    for i, features in enumerate(word_features):
        for _, tag_p in pos_tags_p_s[i]:
            for _, tag_pp in pos_tags_pp_s[i]:
                _features = features.copy()
                _features["tag-1"] = tag_p
                _features["tag-2-1"] = tag_pp + "_" + tag_p
                features_list.append(_features)
        features_sets_limits.append(len(features_list))

    probs = clf.predict_log_proba(vec.transform(features_list))
    scores = np.zeros((N, n_tags, n_tags, n_tags)) - np.inf
    for i in range(N):
        ids_p, ids_pp = [o[0] for o in pos_tags_p_s[i]], [o[0] for o in pos_tags_pp_s[i]]
        ids_p, ids_pp = np.meshgrid(ids_p, ids_pp, indexing='ij')
        ids_p, ids_pp = ids_p.reshape(-1), ids_pp.reshape(-1)
        scores[i, ids_pp, ids_p, :n_tags-1] = probs[features_sets_limits[i]:features_sets_limits[i+1]]
    return scores

all_tags = set()
for k, s in possible_tags.items():
    all_tags = all_tags.union(s)

tag2idx = {tag: i for i, tag in enumerate(le.classes_)}
tag2idx[START] = len(le.classes_)
tag2idx[END] = len(le.classes_) + 1
default_tags = [(i, c) for i, c in enumerate(le.classes_)]

ids_and_tags_for_word = {w: [(tag2idx[t], t) for t in tags] for w, tags in possible_tags.items()}

def get_ids_and_tags(word):
    for sign in signatures:
        if sign.pattern.match(word):
            return ids_and_tags_for_word['^' + sign.name]
    return default_tags

out_file = open(out_file_name, 'w')

def viterbi_predict(test_sents):
    res = []
    for sent in test_sents:
        N, k = len(sent), len(le.classes_)
        scores = np.zeros((N, k + 1, k + 1), dtype=np.float)
        pointers = np.zeros((N, k + 1, k + 1), dtype=np.int)

        it = HasNextIterator(sent)

        if it.has_next():
            it.next()

        if it.has_next():
            word_plus_1 = it.next()
        else:
            word_plus_1 = END

        if it.has_next():
            word_plus_2 = it.next()
        else:
            word_plus_2 = END

        word_minus_2, tag_minus_2_1, tag_minus_1, word_minus_1 = "START", START + "_" + START, START, "START"

        features_list = []            
        pos_tags_p_s = []
        pos_tags_pp_s = []

        for i, tok in enumerate(sent):
            
            features_list.append(word_to_features(tok, check_if_rare_with_vectorizer, word_minus_2, word_minus_1, word_plus_1, word_plus_2))
            pos_tags_p_s.append(ids_and_tags_for_word.setdefault(word_minus_1, default_tags))
            pos_tags_pp_s.append(ids_and_tags_for_word.setdefault(word_minus_2, default_tags))

            word_plus_1 = word_plus_2

            if it.has_next():
                word_plus_2 = it.next()
            else:
                word_plus_2 = END

            word_minus_2, word_minus_1 = word_minus_1, tok

        with timethis('calaulate probs'):
            tag_log_probs = score_sentence(clf, pos_tags_pp_s, pos_tags_p_s, features_list)
    

        for i, tok in enumerate(sent):            
            with timethis('find max'):
                new_scores = scores[i - 1, :, :, np.newaxis] + tag_log_probs[i]
                pointers[i] = np.argmax(new_scores, axis=0)
                I,J = np.ogrid[:n_tags,:n_tags]
                scores[i] = new_scores[pointers[i],I,J] #like scores[i] = np.max(new_scores, axis=0) 

        ys = np.zeros(N, dtype=np.int)
        if N==1:
            ys[0] = np.unravel_index(np.argmax(scores[-1]), (k+1, k+1))[-1]
        elif N==2:
            ys[-2:] = np.unravel_index(np.argmax(scores[-1, :-1, :]), (k, k+1))[-N:]
        else:
            ys[-2:] = np.unravel_index(np.argmax(scores[-1, :-1, :-1]), (k, k))[-N:]
            for i in range(N - 3, -1, -1):
                ys[i] = pointers[i + 2, ys[i + 1], ys[i + 2]]
        res += list(ys)
        predictions_for_line = le.inverse_transform(ys)
        for tok, tag in zip(sent, predictions_for_line):
            print(tok + "/" + tag, end=" ", file=out_file)
        print("", file=out_file)

    return res

y_predicted = viterbi_predict(test_sents)
out_file.close()