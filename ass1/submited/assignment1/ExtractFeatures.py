import re
import sys
from pickle import dump
from collections import defaultdict, Counter, namedtuple
from sklearn.feature_extraction import DictVectorizer

signature = namedtuple('Signature', ['pattern', 'name'])
def suffix_signature(suffix):
    return signature(re.compile(r'[\-a-zA-Z]{2,}' + suffix + '$'), '-' + suffix)

def prefix_signature(prefix):
    return signature(re.compile('^' + prefix + r'[\-a-z]{2,}'), prefix + '-')


signatures = [signature(re.compile(r'^[0-9]{2}$'), '2DIGITS'), \
              signature(re.compile(r'^[0-9]{4}$'), '4DIGITS'), \
              signature(re.compile(r'^[0-9]+/.{0-9}+$'), 'DECIMAL_NUM'), \
              signature(re.compile(r'^[0-9]+,{0-9}+$'), 'DECIMAL_NUM'), \
              signature(re.compile(r'^\d+'), 'NUM'), \
              signature(re.compile(r'^\W+$'), 'NON_ALPHANUMERIC'), \
              signature(re.compile(r'^[A-Z][a-z]'), 'Aa'), \
              signature(re.compile(r'^[A-Z]+(\-[A-Z]+)*$'), 'AAA'), \
              signature(re.compile(r'^([A-Z][a-z]\.)+[A-Z][a-z]\.?$'), 'A.B.C'), \
              signature(re.compile(r'([A-Z][a-z]\.)+[A-Z][a-z]'), 'A.B.C-'), \
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

affix_signatures = signatures
affix_signatures += [suffix_signature(sffx) for sffx in suffixes]
affix_signatures += [prefix_signature(prfx) for prfx in prefixes]

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

def process_line(line):
    return [re.search(r'(.*)/(.*$)', pair).groups() for pair in line.split()]
_i_ = 0
max_n = 100

def read_file(file):
    with open(file, 'r') as file:
        return [process_line(line) for line in file.readlines()]

def word_to_features(tok, check_if_rare, word_minus_2, word_minus_1, word_plus_1, word_plus_2, \
                        tag_minus_2_1=None, tag_minus_1=None):
        features = defaultdict()
        if check_if_rare(tok):
            for sign in signatures:
                if sign.pattern.match(tok):
                    features[sign.name] = "True"
            for i in range(1,min(5, len(tok)+1)):
                features["prefix_" + str(i)] = tok[:i]
                features["suffix_" + str(i)] = tok[-i:]
        else:
            features["w_i"] = tok
        for wname, word in zip(['word-2', 'word-1', 'word+1', 'word+2'], \
                                [word_minus_2, word_minus_1, word_plus_1, word_plus_2]):
            features[wname] = word
            if not word in [START, 'START', END, 'END']:
                for sign in affix_signatures:
                    if sign.pattern.match(word):
                        features[wname+'_'+sign.name] = "True"
                        
        if tag_minus_1:
            features["tag-1"] = tag_minus_1
        if tag_minus_2_1:
            features["tag-2-1"] = tag_minus_2_1
        return features


if __name__ == '__main__':
    corpus_file_location = sys.argv[1]
    features_file_location = sys.argv[2]
    try:
        feature_mapping_file = sys.argv[3]
    except IndexError:
        feature_mapping_file = "feature_mapping"

    def check_if_rare(tok):
        if tok_counter[tok] >= min_count:
            return False
        else:
            return True

    possible_tags = defaultdict(set)
    possible_tags["START"] = {START}
    possible_tags["END"] = {END}

    min_count = 5

    train_sents = read_file(corpus_file_location)  # curpus_file
    train_data = [pair for sent in train_sents for pair in sent]
    tok_counter = Counter(tok for tok, tag in train_data)

    for tok, tag in train_data:
        possible_tags[tok].add(tag)
        if tok_counter[tok] < min_count:
            for sign in affix_signatures:
                if sign.pattern.match(tok):
                    possible_tags['^' + sign.name].add(tag)
                    break

    features_file = open(features_file_location, "w+")

    def create_features(train_sents):
        for i, sent in enumerate(train_sents):
            it = HasNextIterator(sent)

            if it.has_next():
                it.next()

            if it.has_next():
                word_plus_1 = it.next()[0]
            else:
                word_plus_1 = END

            if it.has_next():
                word_plus_2 = it.next()[0]
            else:
                word_plus_2 = END

            word_minus_2, tag_minus_2_1, tag_minus_1, word_minus_1 = "START", START + "_" + START, START, "START"

            for tok, tag in sent:

                features = word_to_features(tok, check_if_rare, word_minus_2, word_minus_1, word_plus_1, word_plus_2, tag_minus_2_1, tag_minus_1)

                word_plus_1 = word_plus_2
                if it.has_next():
                    word_plus_2 = it.next()[0]
                else:
                    word_plus_2 = END

                word_minus_2, tag_minus_2_1, tag_minus_1, word_minus_1 = word_minus_1, tag_minus_1 + "_" + tag, tag, tok

                vec = DictVectorizer()
                vec.fit_transform(features)
                print(tag, file=features_file, end=" ")
                print(*vec.get_feature_names(), file=features_file,\
                     end='' if i+1==len(train_sents) and word_plus_1==END else '\n')


    create_features(train_sents)
    features_file.close()

    with open(feature_mapping_file, "wb") as file:
        dump(possible_tags, file=file)

