import re
import sys
from pickle import dump
from collections import defaultdict, Counter, namedtuple
from sklearn.feature_extraction import DictVectorizer

signature = namedtuple('Signature', ['pattern', 'name'])

features_file_location = 'features_file'
corpus_file_location = 'data/ass1-tagger-train-5000'


def suffix_signature(suffix):
    return signature(re.compile(r'[\-a-zA-Z]{2,}' + f'{suffix}$'), f'-{suffix}')


def prefix_signature(prefix):
    return signature(re.compile(f'^{prefix}' + r'[\-a-z]{2,}'), f'{prefix}-')


signatures = [signature(re.compile(r'^[0-9]{2}'), '2DIGITS'), \
              signature(re.compile(r'^[0-9]{4}'), '4DIGITS'), \
              signature(re.compile(r'^[0-9]+/.{0-9}+'), 'DECIMAL_NUM'), \
              signature(re.compile(r'^[0-9]+,{0-9}+'), 'DECIMAL_NUM'), \
              signature(re.compile(r'^\d+'), 'NUM'), \
              signature(re.compile(r'^\W+$'), 'NON_ALPHANUMERIC'), \
              signature(re.compile(r'^[A-Z][a-z]'), 'Aa'), \
              signature(re.compile(r'^[A-Z]+(\-[A-Z]+)*$'), 'AAA'), \
              signature(re.compile(r'^(\w\.)+\w\.?$'), 'A.B.C'), \
              signature(re.compile(r'(\w\.)+\w'), 'A.B.C-'), \
              signature(re.compile(r'^[A-Z]'), 'A-'),
              signature(re.compile(r'^.*\-+.*'), 'HYPHEN'), \
              signature(re.compile(r'^.*[A-Z].*'), 'CAPITAL')
              ]

suffixes = ['ing', 'ed', 'less', 'ness', 'es', 'eer', 'er', 'sion', 'ation', 'ity', 'or', 'sion', 'ship', 'el', 'ly', \
            'ward', 'wise', 'ise', 'ize', 'en', 'able', 'ible', 'al', 'ant', 'ary', 'ic', 'ous', 'ive', 'y', 's', \
            'ism', 'ian', 'ment', 'an', 'ist', 'est', 'st']
suffixes.sort(key=len, reverse=True)
prefixes = ['de', 'dis', 're', 'im', 'un', 'non', 'pre', 'extra', 'over', 'anti', 'auto', 'down', 'hyper', \
            'il', 'in', 'ir', 'inter', 'mega', 'mid', 'mis', 'non', 'out', 'post', 'pro', 'pre', 'semi', 'sub', \
            'super', 'tele', 'trans', 'ultra', 'under', 'up']
prefixes.sort(key=len, reverse=True)
signatures += [suffix_signature(sffx) for sffx in suffixes]
signatures += [prefix_signature(prfx) for prfx in prefixes]
sign_dict = {sign.name: sign for sign in signatures}

START = '<START>'
END = '<END>'


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


def suffix_signature(suffix):
    return signature(re.compile(r'[\-a-zA-Z]{2,}' + f'{suffix}$'), f'-{suffix}')


def prefix_signature(prefix):
    return signature(re.compile(f'^{prefix}' + r'[\-a-z]{2,}'), f'{prefix}-')


def process_line(line):
    return [re.search(r'(.*)/(.*$)', pair).groups() for pair in line.split()]


def read_file(file):
    with open(file, 'r') as file:
        return [process_line(line) for line in file.readlines()]


def check_if_rare(tok):
    if tok_counter[tok] >= min_count:
        return False
    else:
        return True


def word_to_features(tok, check_if_rare, word_minus_2, tag_minus_2_1, tag_minus_1, word_minus_1,
                     word_plus_1, word_plus_2):
    features = defaultdict()
    if check_if_rare(tok):
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

    features["tag-1"] = tag_minus_1
    features["tag-2-1"] = tag_minus_2_1
    features["word-1"] = word_minus_1
    features["word-2"] = word_minus_2
    features["word+1"] = word_plus_1
    features["word+2"] = word_plus_2
    return features


possible_tags = defaultdict(set)
possible_tags["START"] = {START}
possible_tags["END"] = {END}

sign_words = defaultdict(list)
sign_words_tag_count = defaultdict(Counter)
sign_rare_words = defaultdict(list)
sign_rare_words_tag_count = defaultdict(Counter)
min_count = 5

train_sents = read_file(corpus_file_location)  # curpus_file
train_data = [pair for sent in train_sents for pair in sent]
tok_counter = Counter(tok for tok, tag in train_data)

for tok, tag in train_data:
    possible_tags[tok].add(tag)
    if tok_counter[tok] < min_count:
        sign_rare_words_tag_count['ALL'].update([tag])
    for sign in signatures:
        if sign.pattern.match(tok):
            possible_tags['^' + sign.name].add(tag)
            break
            sign_words[sign.name].append(tok)
            sign_words_tag_count[sign.name].update([tag])
            if tok_counter[tok] < min_count:
                possible_tags['^' + sign.name].add(tag)
            break

min_count = 5

data_with_signs = [(check_if_rare(tok), tag) for tok, tag in train_data]

features_file = open(features_file_location, "w+")


def create_features(train_sents):
    for sent in train_sents:

        it = HasNextIterator(sent)

        if it.has_next():
            it.next()

        if it.has_next():
            word_plus_1 = it.next()[1]
        else:
            word_plus_1 = END

        if it.has_next():
            word_plus_2 = it.next()[1]
        else:
            word_plus_2 = END

        word_minus_2, tag_minus_2_1, tag_minus_1, word_minus_1 = "START", START + "_" + START, START, "START"

        for tok, tag in sent:

            features = word_to_features(tok, check_if_rare, word_minus_2, tag_minus_2_1, tag_minus_1, word_minus_1, word_plus_1, word_plus_2)

            if it.has_next():
                word_plus_1 = it.next()[1]
            else:
                word_plus_1 = END

            if it.has_next():
                word_plus_2 = it.next()[1]
            else:
                word_plus_2 = END

            word_minus_2, tag_minus_2_1, tag_minus_1, word_minus_1 = word_minus_1, tag_minus_1 + "_" + tag, tag, tok

            vec = DictVectorizer()
            vec.fit_transform(features)
            print(tag, file=features_file, end=" ")
            print(*vec.get_feature_names(), file=features_file)


create_features(train_sents)
dump(possible_tags, open("feature_mapping", "wb"))
features_file.close()
