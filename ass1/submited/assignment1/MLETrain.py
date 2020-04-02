import re
import sys
import numpy as np
from time import time
from collections import defaultdict, Counter, namedtuple

signature = namedtuple('Signature', ['pattern', 'name'])
def suffix_signature(suffix):
    return signature(re.compile(r'[\-a-zA-Z]{2,}' + suffix + '$'), '-' + suffix)

def prefix_signature(prefix):
    return signature(re.compile('^' + prefix + r'[\-a-z]{2,}'), prefix + '-')

signatures = [signature(re.compile(r'^[0-9]{2}$'), '2DIGITS'),\
              signature(re.compile(r'^[0-9]{3}$'), '3DIGITS'),\
              signature(re.compile(r'^[0-9]{4}$'), '4DIGITS'),\
              signature(re.compile(r'^[0-9]+/.{0-9}+'), 'DECIMAL_NUM1'),\
              signature(re.compile(r'^[0-9]+,{0-9}+'), 'DECIMAL_NUM2'),\
              signature(re.compile(r'^\d+'), 'NUM'),\
              signature(re.compile(r'^\W+$'), 'NON_ALPHANUMERIC'),\
              signature(re.compile(r'^[A-Z][a-z]'), 'Aa'),\
              signature(re.compile(r'^[A-Z]+(\-[A-Z]+)*$'), 'AAA'),\
              signature(re.compile(r'^([A-Z][a-z]\.)+[A-Z][a-z]\.?$'), 'A.B.C'),\
              signature(re.compile(r'([A-Z][a-z]\.)+[A-Z][a-z]'), 'A.B.C-'),\
              signature(re.compile(r'^[A-Z]'), 'A-')
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
 
def process_line(line):
    return [re.search(r'(.*)/(.*$)', pair).groups() for pair in line.split()]

def read_file(file):
    with open(file, 'r') as file:
        return [process_line(line) for line in file.readlines()]

START = '<START>'
def estimateMLE(train_sents, min_count=5):
    tok_counter = Counter(tok for sent in train_sents for tok, tag in sent)
    e_counter = defaultdict(int)
    q_counter = defaultdict(int)
    for sent in train_sents:
        tag_minus_2, tag_minus_1 = START, START
        for i, (tok, tag) in enumerate(sent):
            #tok = replace_if_rare(tok)
            if i == 0:
                if sign_dict['Aa'].pattern.match(tok):
                    tok = tok.lower()
            e_counter[(tok, tag)] += 1
            if tok_counter[tok] <= min_count:
                for sign in signatures:
                    if sign.pattern.match(tok):
                        e_counter[('^'+sign.name, tag)] += 1
                else:
                    e_counter[('*UNK*', tag)] += 1
            q_counter[(tag,)] += 1
            q_counter[(tag_minus_1, tag)] += 1
            q_counter[(tag_minus_2, tag_minus_1, tag)] += 1
            tag_minus_2, tag_minus_1 = tag_minus_1, tag
    return e_counter, q_counter

def saveMLE(counts, file):
    with open(file, 'w') as save_file:
        for key, val in counts.items():
            print(*key, file=save_file, end='\t')
            print(val, file=save_file)

if __name__=='__main__':
    try:
        train_file, q_file, e_file = sys.argv[1:]
    except ValueError:
        train_file = 'data/ass1-tagger-train'
        q_file, e_file = 'q.mle', 'e.mle'
    train_sents = read_file(train_file)
    e_counts, q_counts = estimateMLE(train_sents)
    saveMLE(e_counts, e_file)
    saveMLE(q_counts, q_file)

