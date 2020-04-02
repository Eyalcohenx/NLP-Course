import numpy as np
import sys
import re
from itertools import product
from collections import namedtuple, defaultdict, Counter
from MLETrain import signatures, sign_dict

def readMLE(file):
    count = defaultdict(int)
    with open(file, 'r') as f:
        for line in f.readlines():
            key, val = line.split('\t')
            key = key.split(' ')
            count[tuple(key)] = int(val)
    return count

def process_line(line):
    return [re.search(r'(.*)/.*$', pair).groups()[0] for pair in line.split()]

def read_input(file):
    try:
        with open(file, 'r') as f:
            sents = list(map(process_line, f.readlines()))
    except AttributeError:
        with open(file, 'r') as f:
            sents = list(map(str.split, f.readlines()))
    finally:
        return sents

if __name__ == '__main__':
    input_file, q_file, e_file, out_file, extra_file = sys.argv[1:]

    e_counts, q_counts = readMLE(e_file), readMLE(q_file)
    START = '<START>'

    tag_set = [tags[0] for tags in q_counts.keys() if len(tags)==1] + [START]
    tok_set = set(tok for tok, tag in e_counts)
    tags_for_word = defaultdict(list) #possible tags for each word (or word-signature), according to train data
    tag_counts = defaultdict(int)
    tok_counts = defaultdict(int)
    for (tok, tag), n in e_counts.items():
        tags_for_word[tok].append(tag)
        tag_counts[tag] += n
        tok_counts[tok] += n

    def get_e_prob_raw(tok, tag):
        return e_counts[(tok, tag)]/tag_counts[tag]

    def em_lmbda(n):
        return 1 - np.exp(-n/5)

    def get_e_prob(tok, tag):
        if tok_counts[tok] <= 10 and not tok.startswith('^'):
            lmbda = em_lmbda(tok_counts[tok])
            for sign in signatures:
                if sign.pattern.match(tok):
                    return lmbda*get_e_prob_raw(tok, tag) + (1-lmbda)*get_e_prob_raw('^'+sign.name, tag)
            else:
                lmbda*get_e_prob_raw(tok, tag) + (1-lmbda)*get_e_prob_raw('*UNK*', tag)
        return get_e_prob_raw(tok, tag)

    e_probs = {k: get_e_prob(*k) for k in list(e_counts.keys())}

    tag_list_counts = defaultdict(int)
    for tags, count in q_counts.items():
        tag_list_counts[tags[:-1]] += count

    def _div(x, y):
        if x==y==0:
            return 0
        return x/y
    lambda1, lambda2, lambda3 = 0.9, 0.09, 0.01
    def get_q(t1, t2, t3):
        if tag_list_counts[(t1, t2)] > 0:
            return (_div(lambda1*q_counts[(t1, t2, t3)], tag_list_counts[(t1, t2)]) + \
                    _div(lambda2*q_counts[(t2, t3)], tag_list_counts[(t2,)]) + \
                    _div(lambda3*q_counts[(t3,)],tag_list_counts[()]))
        else:
            return (_div(lambda2*q_counts[(t2, t3)], tag_list_counts[(t2,)]) + \
                    _div(lambda3*q_counts[(t3,)],tag_list_counts[()])) /(1 - lambda1)       

    q_probs = {tags: get_q(*tags) for tags in product(tag_set, repeat=3)}

    def score(token, tag_pp, tag_p, tag):
        return e_probs[(token, tag)]*q_probs[(tag_pp, tag_p, tag)]

    tag2idx = {tag: i for i, tag in enumerate(tag_set)}
    ids_and_tags_for_word = {w: [(tag2idx[t], t) for t in tags] for w, tags in tags_for_word.items()}
    def viterbi_decode(sentence, scoring=score):
        N, k = len(sentence), len(tag_set)
        scores = np.zeros((N, k+1, k+1), dtype=np.float)
        pointers = np.zeros((N, k+1, k+1), dtype=np.int)
        #a function that returns possible tags for the location in the sentence
        tags = lambda i: ids_and_tags_for_word[sentence[i]] if i>=0 else [(k, START)] 
        for i in range(N):
            token = sentence[i]
            scores[i] = -np.inf
            for itag, tag in tags(i):
                for itag_p, tag_p in tags(i-1):
                    for itag_pp, tag_pp in tags(i-2):
                        score = scores[i-1, itag_pp, itag_p] + np.log(scoring(token, tag_pp, tag_p, tag))
                        if score > scores[i, itag_p, itag]:
                            scores[i, itag_p, itag] = score
                            pointers[i, itag_p, itag] = itag_pp
        ys = np.zeros(N, dtype=np.int)
        #ys[-2:] = np.unravel_index(np.argmax(scores[-1,:-1,:-1]), (k,k))[-N:]
        if N==1:
            ys[0] = np.unravel_index(np.argmax(scores[-1]), (k+1, k+1))[-1]
        elif N==2:
            ys[-2:] = np.unravel_index(np.argmax(scores[-1, :-1, :]), (k, k+1))[-N:]
        else:
            ys[-2:] = np.unravel_index(np.argmax(scores[-1, :-1, :-1]), (k, k))[-N:]
            for i in range(N - 3, -1, -1):
                ys[i] = pointers[i + 2, ys[i + 1], ys[i + 2]]
        # for i in range(N-3, -1, -1):
        #     ys[i] = pointers[i+2, ys[i+1], ys[i+2]]
        return [tag_set[i] for i in ys]


    def replace_if_unknown(tok):
        if tok in tok_set:
            return tok
        for sign in signatures:
            if sign.pattern.match(tok):
                return '^' + sign.name
        else:
            return '*UNK*'

    def process_sentence(sent):
        sent = list(sent)
        if sign_dict['Aa'].pattern.match(sent[0]):
            sent[0] = sent[0].lower()
        return [replace_if_unknown(token) for token in sent]

    input_sents = read_input(input_file)

    output_tags = [viterbi_decode(process_sentence(sent)) for sent in input_sents]

    with open(out_file, 'w') as file:
        for sent, tags in zip(input_sents, output_tags):
            print(*[tok+'/'+tag for tok, tag in zip(sent, tags)], file=file)




