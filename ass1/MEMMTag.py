import re
import sys
from joblib import dump, load
from collections import defaultdict, namedtuple
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore", message="Precision is ill-defined")
warnings.filterwarnings("ignore", message="Recall is ill-defined")

input_file_name = 'data/ass1-tagger-dev-input'  # TODO: change to sys.argv[1]
model_name = 'model_file'  # TODO: change to sys.argv[2]
feature_map_file = 'feature_mapping'  # TODO: change to sys.argv[3]
out_file_name = 'predictions'  # TODO: change to sys.argv[4]


def check_availability(element, collection: iter):
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
    return line.split()


def read_file(file):
    with open(file, 'r') as file:
        return [process_line(line) for line in file.readlines()]


# signature = namedtuple('Signature', ['pattern', 'name'])
# signatures = [signature(re.compile(r'^\d+'), 'NUM'), \
#               signature(re.compile(r'^.*\-+.*'), 'HYPHEN'), \
#               signature(re.compile(r'^.*[A-Z].*'), 'CAPITAL'), \
#               ]

signatures = [signature(re.compile(r'^[0-9]{2}'), '2DIGITS'),\
              signature(re.compile(r'^[0-9]{4}'), '4DIGITS'),\
              signature(re.compile(r'^[0-9]+/.{0-9}+'), 'DECIMAL_NUM'),\
              signature(re.compile(r'^[0-9]+,{0-9}+'), 'DECIMAL_NUM'),\
              signature(re.compile(r'^\d+'), 'NUM'),\
              signature(re.compile(r'^\W+$'), 'NON_ALPHANUMERIC'),\
              signature(re.compile(r'^[A-Z][a-z]'), 'Aa'),\
              signature(re.compile(r'^[A-Z]+(\-[A-Z]+)*$'), 'AAA'),\
              signature(re.compile(r'^(\w\.)+\w\.?$'), 'A.B.C'),\
              signature(re.compile(r'(\w\.)+\w'), 'A.B.C-'),\
              signature(re.compile(r'^[A-Z]'), 'A-'),
              signature(re.compile(r'^.*\-+.*'), 'HYPHEN'), \
              signature(re.compile(r'^.*[A-Z].*'), 'CAPITAL')
              ] 

suffixes = ['ing', 'ed', 'less', 'ness', 'es', 'eer', 'er', 'sion', 'ation', 'ity', 'or', 'sion', 'ship', 'el', 'ly',\
            'ward', 'wise', 'ise', 'ize', 'en', 'able', 'ible', 'al', 'ant', 'ary', 'ic', 'ous', 'ive', 'y', 's', \
            'ism', 'ian', 'ment', 'an', 'ist', 'est', 'st']
suffixes.sort(key=len, reverse=True)
prefixes = ['de', 'dis', 're', 'im', 'un', 'non', 'pre', 'extra', 'over', 'anti', 'auto', 'down', 'hyper', \
            'il', 'in', 'ir', 'inter', 'mega', 'mid', 'mis', 'non', 'out', 'post', 'pro', 'pre', 'semi', 'sub',\
            'super', 'tele', 'trans', 'ultra', 'under', 'up']
prefixes.sort(key=len, reverse=True)
signatures += [suffix_signature(sffx) for sffx in suffixes]
signatures += [prefix_signature(prfx) for prfx in prefixes]
#signatures += [signature(re.compile(r''), '*UNK*')]
sign_dict = {sign.name: sign for sign in signatures}

test_sents = read_file(input_file_name)

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


def score_(clf, tag_pp, tag_p, features):
    features["tag-1"] = tag_p
    features["tag-2-1"] = tag_pp + "_" + tag_p
    return clf.predict_log_proba(vec.transform(features))[0]

n_tags = len(le.classes_) + 1

def score_func1(clf, pos_tags_pp, pos_tags_p, features):
    features_list = []
    for _, tag_p in pos_tags_p:
        for _, tag_pp in pos_tags_pp:
            _features = features.copy()
            _features["tag-1"] = tag_p
            _features["tag-2-1"] = tag_pp + "_" + tag_p
            features_list.append(_features)
    return clf.predict_log_proba(vec.transform(features_list)).reshape(len(pos_tags_p), len(pos_tags_pp), -1)

def score_func(clf, pos_tags_pp, pos_tags_p, features):
    features_list = []
    for _, tag_p in pos_tags_p:
        for _, tag_pp in pos_tags_pp:
            _features = features.copy()
            _features["tag-1"] = tag_p
            _features["tag-2-1"] = tag_pp + "_" + tag_p
            features_list.append(_features)
    ids_p, ids_pp = [o[0] for o in pos_tags_p], [o[0] for o in pos_tags_pp]
    ids_p, ids_pp = np.meshgrid(ids_p, ids_pp, indexing='ij')
    ids_p, ids_pp = ids_p.reshape(-1), ids_pp.reshape(-1)
    scores = np.zeros((n_tags, n_tags, len(le.classes_))) - np.inf
    scores[ids_p, ids_pp] = clf.predict_log_proba(vec.transform(features_list))
    return scores
    return clf.predict_log_proba(vec.transform(features_list)).reshape(len(pos_tags_p), len(pos_tags_pp), -1)

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
%pdb on
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

        for i, tok in enumerate(sent):
            scores[i] = -np.inf
            features = defaultdict()

            if check_if_rare_with_vectorizer(tok):
                for sign in signatures:
                    if sign.pattern.match(tok):
                        features[sign.name] = "True"
                try:
                    prefix_1 = tok[0:1]
                    features["prefix_1"] = prefix_1
                except EnvironmentError:
                    pass
                try:
                    prefix_2 = tok[0:2]
                    features["prefix_2"] = prefix_2
                except EnvironmentError:
                    pass
                try:
                    prefix_3 = tok[0:3]
                    features["prefix_3"] = prefix_3
                except EnvironmentError:
                    pass
                try:
                    prefix_4 = tok[0:4]
                    features["prefix_4"] = prefix_4
                except EnvironmentError:
                    pass
                try:
                    suffix_1 = tok[-1:]
                    features["suffix_1"] = suffix_1
                except EnvironmentError:
                    pass
                try:
                    suffix_2 = tok[-2:]
                    features["suffix_2"] = suffix_2
                except EnvironmentError:
                    pass
                try:
                    suffix_3 = tok[-3:]
                    features["suffix_3"] = suffix_3
                except EnvironmentError:
                    pass
                try:
                    suffix_4 = tok[-4:]
                    features["suffix_4"] = suffix_4
                except EnvironmentError:
                    pass
            else:
                features["w_i"] = tok

            features["word-1"] = word_minus_1
            features["word-2"] = word_minus_2
            features["word+1"] = word_plus_1
            features["word+2"] = word_plus_2
            import pdb
            #pdb.set_trace()

            pos_tags_p = ids_and_tags_for_word.setdefault(word_minus_1, get_ids_and_tags(word_minus_1))
            pos_tags_pp = ids_and_tags_for_word.setdefault(word_minus_2, get_ids_and_tags(word_minus_2))
            tag_probs = score_func(clf, pos_tags_pp, pos_tags_p, features)
           # tag_probs1 = score_func1(clf, pos_tags_pp, pos_tags_p, features)
            for itag, tag in ids_and_tags_for_word.setdefault(tok, get_ids_and_tags(tok)):
                for i_p_iter, (itag_p, tag_p) in enumerate(pos_tags_p):
                    for i_pp_iter, (itag_pp, tag_pp) in enumerate(pos_tags_pp):  
                        #assert tag_probs[itag_p, itag_pp, itag] == tag_probs1[i_p_iter, i_pp_iter, itag]
                        score = scores[i - 1, itag_pp, itag_p] + tag_probs[itag_p, itag_pp, itag]
                        if score > scores[i, itag_p, itag]:
                            scores[i, itag_p, itag] = score
                            pointers[i, itag_p, itag] = itag_pp

            # tag_probs = scoring(clf, pos_tags_pp, pos_tags_p, features)
            # for itag, tag in ids_and_tags_for_word.setdefault(tok, get_ids_and_tags(tok)):
            #     for i_p_iter, (itag_p, tag_p) in enumerate(pos_tags_p):
            #         for i_pp_iter, (itag_pp, tag_pp) in enumerate(pos_tags_pp):  
            #             score = scores[i - 1, itag_pp, itag_p] + tag_probs[i_p_iter, i_pp_iter, itag]
            #             if score > scores[i, itag_p, itag]:
            #                 scores[i, itag_p, itag] = score
            #                 pointers[i, itag_p, itag] = itag_pp

            if it.has_next():
                word_plus_1 = it.next()
            else:
                word_plus_1 = END

            if it.has_next():
                word_plus_2 = it.next()
            else:
                word_plus_2 = END

            word_minus_2, word_minus_1 = word_minus_1, tok
        import pdb 
        ys = np.zeros(N, dtype=np.int)
        ys[-2:] = np.unravel_index(np.argmax(scores[-1, :-1, :-1]), (k, k))[-N:]
        for i in range(N - 3, -1, -1):
            ys[i] = pointers[i + 2, ys[i + 1], ys[i + 2]]
        res += list(ys)
    return res

N = 3000
from time import time
t0 = time()
y_predicted = viterbi_predict(test_sents[:N])
print(time() - t0)

# TEST:

def process_line_y(line):
    temp = [re.search(r'(.*)/(.*$)', pair).groups() for pair in line.split()]
    return [t[1] for t in temp]


def read_file_y(file):
    with open(file, 'r') as file:
        return [process_line_y(line) for line in file.readlines()]

#y_predicted = le.transform(y_predicted)
y_test = read_file_y('data/ass1-tagger-dev')
y_test = [val for sublist in y_test[:N] for val in sublist]
y_test = le.transform(y_test)

precision = precision_score(y_test, y_predicted, average="weighted")
recall = recall_score(y_test, y_predicted, average="weighted")
acc = np.mean(y_test == y_predicted)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", acc)