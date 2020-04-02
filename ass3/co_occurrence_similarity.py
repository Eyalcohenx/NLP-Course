from collections import Counter
import warnings
import numpy as np
from contextlib import contextmanager
from collections import defaultdict
from time import time

times_dict = defaultdict(float)


def get_second(element):
    return element[1]


# timing the code with this
@contextmanager
def timethis(label):
    t0 = time()
    yield
    times_dict[label] += time() - t0


# words list
words = ["car", "bus", "hospital", "hotel", "gun", "bomb", "horse", "fox", "table", "bowl", "guitar", "piano"]

warnings.filterwarnings("error")
# nouns set to be used when checking a word
noun_set = {"NN", "NNS", "NNP", "NNPS", "PRP", "PRP$", "WP", "WP$"}
# list of function words
function_words = {'ie', 'sometime', 'whither', 'hers', 'till', 'herewith', 'we', 'forty', 'four', 'via', 'out', 'alone',
                  'somehow', 'whereby', 'used', 'which', 'an', 'six', 'if', 'yet', 'about', 'latter', 'hereupon',
                  'million', 'for', 'amongst', 'and', 'where', 'since', 'another', 'besides', 'whereafter', 'together',
                  'neither', 'away', 'thereby', 'the', 'two', 'from', 'either', 'seven', 'other', 'little', 'nine',
                  'one', 'someone', 'anyway', 'however', 'by', 'therein', 'more', 'thence', 'has', 'well', 'once', 're',
                  'round', 'whence', 'eight', 'again', 'being', 'with', 'back', 'onto', 'before', 'both', 'their', 'be',
                  'mine', 'along', 'on', 'towards', 'namely', 'indeed', 'ten', 'did', 'lots', '"', 'can', 'as',
                  'throughout', 'lot', 'because', 'though', '(', 'must', 'to', 'whether', 'I', 'cannot', 'below', 'his',
                  'hereinafter', 'down', 'despite', 'sometimes', 'beforehand', ')', 'thus', 'latterly', 'nevertheless',
                  '.', 'somewhere', 'twenty', 'thereafter', 'END', 'too', 'he', 'all', 'until', 'ought', 'fifty',
                  'around', 'toward', 'five', 'became', 'without', 'hereunder', 'least', 'him', 'done', 'need', 'only',
                  'her', "''", 'why', 'any', 'else', 'them', 'eg', 'each', 'whose', 'would', 'nothing', 'next',
                  'should', 'per', 'twelve', 'does', 'noone', 'anyone', 'beyond', 'former', "'s", 'under', 'this', ';',
                  'everyone', 'that', 'between', '``', 'nowhere', 'none', 'your', 'others', 'these', 'something',
                  'wherever', 'but', 'its', 'amoungst', 'not', 'somewhat', 'top', 'heretofore', 'himself', 'elsewhere',
                  'otherwise', 'my', 'third', 'same', 'will', 'three', 'itself', 'whereas', 'further', 'thereupon',
                  'you', '[', 'oftentimes', '-', 'ever', 'somebody', 'are', 'wherein', 'some', 'of', 'no', 'outside',
                  'yourselves', 'less', 'could', 'upon', 'nobody', 'through', 'it', 'except', 'there', 'half', 'anyhow',
                  'whole', 'most', 'already', 'into', 'furthermore', 'thereof', 'at', 'beside', 'etc', 'have', 'unless',
                  'was', 'our', 'herein', 'yours', 'might', 'nineteen', 'START', 'whoever', 'nor', 'us', 'enough', ',',
                  'dare', 'inside', 'they', 'rather', 'past', 'always', 'thereon', 'several', 'ours', 'themselves',
                  'formerly', 'near', 'is', 'above', 'hereby', 'over', 'instead', 'i', 'how', 'whereupon', 'been', 'in',
                  'during', 'meanwhile', 'first', 'everybody', 'within', 'so', 'second', 'were', 'what', 'often', ']',
                  'anywhere', 'those', 'hereafter', 'every', 'who', 'now', 'yourself', 'still', 'while', 'everywhere',
                  'or', 'than', 'even', 'hence', 'off', 'thru', 'hundred', 'yes', 'perhaps', 'am', 'herself', 'do',
                  'when', 'afterwards', 'across', 'many', 'anything', 'thirty', 'fifteen', 'anybody', 'then', 'whom',
                  'a', 'everything', 'against', 'never', 'after', 'few', 'whyever', 'she', 'behind', 'very', 'although',
                  'up', 'also', 'hereabouts', 'last', 'whatever', 'thereabouts', 'had', 'here', 'me', 'myself',
                  'thousand', 'mostly', 'may', 'among', 'such', 'theirs', 'whenever', 'almost', 'ourselves', 'shall',
                  'much', 'moreover', 'therefore'}


# function to read lines efficiently, reading lines one at a time to save memory
def myreadlines(f, newline):
    buf = ""
    while True:
        while newline in buf:
            pos = buf.index(newline)
            yield buf[:pos]
            buf = buf[pos + len(newline):]
        chunk = f.read(4096)
        if not chunk:
            yield buf
            break
        buf += chunk


# start and end dummies, just for the window pass, we will ignore them when getting to the comparisons
START_WORD_INFO = ["START", "START", "START", "START", "START", "START", "START", "START", "START", "START"]
END_WORD_INFO = ["END", "END", "END", "END", "END", "END", "END", "END", "END", "END"]

# constants to make it more clear what every index in the word info means
unigrams = Counter()
LEMMATIZED_LOCATION = 2
CONTEXT_LOCATION = 7
FROM_LOCATION = 6
POSTAG_LOCATION = 4
threshold = 75

# the corpus file name
file = "wikipedia.sample.trees.lemmatized"

# creating words list to insert to the counter
with open(file, encoding="utf8") as f:
    words_list, contexts_list = zip(
        *[(word_info.split("\t")[LEMMATIZED_LOCATION], word_info.split("\t")[CONTEXT_LOCATION]) for word_info in f if
          word_info != "\n"])
contexts = Counter()
contexts.update(contexts_list)
unigrams.update(words_list)
unigrams = Counter(dict(filter(lambda x: x[1] >= threshold, unigrams.items())))
num_of_words = sum(unigrams.values())
words_set = set([x[0] for x in unigrams.items()])
# counters for each task tuple/window/sentence
u_att_counter_tuple = defaultdict(Counter)
total_att_counter_tuple = Counter()
att_and_words_sets_tuple = defaultdict(set)
att_set_tuple = set()

u_att_counter_words = defaultdict(Counter)
total_att_counter_words = Counter()
att_and_words_sets_words = defaultdict(set)

u_att_counter_windows = defaultdict(Counter)
total_att_counter_windows = Counter()
att_and_words_sets_windows = defaultdict(set)


# function to get the head noun
def get_head_noun(sent, word_info):
    for word_info1 in sent:
        if word_info1[FROM_LOCATION] == word_info[0]:
            if word_info1[POSTAG_LOCATION] in noun_set:
                return word_info1
    for word_info1 in sent:
        if word_info1[FROM_LOCATION] == word_info[0]:
            if get_head_noun(sent, word_info1)[POSTAG_LOCATION] in noun_set:
                return get_head_noun(sent, word_info1)
    return word_info


# function to get the features in the dependency task
def get_feature_dependency_edge(sent, word_info):
    to_word_info = word_info
    context_type = word_info[CONTEXT_LOCATION]
    to_word = word_info[LEMMATIZED_LOCATION]
    from_word_location = int(word_info[FROM_LOCATION]) - 1
    from_word_info = sent[from_word_location]
    from_word = sent[from_word_location][LEMMATIZED_LOCATION]
    to_list = []
    from_list = []
    # checking if the son is preposition and if so going down in the tree
    if to_word_info[POSTAG_LOCATION] == "IN":  # this means its preposition word
        if "mod" in to_word_info[CONTEXT_LOCATION] or "neg" in to_word_info[CONTEXT_LOCATION]:
            # this means its modifing it
            if to_word_info != get_head_noun(sent, to_word_info):
                to_list.append(to_word_info[CONTEXT_LOCATION])
                to_list.append(to_word)
                to_word_info = get_head_noun(sent, to_word_info)
            to_word = to_word_info[LEMMATIZED_LOCATION]
            context_type = to_word_info[CONTEXT_LOCATION]
    # checking if the dad is preposition and if so going up in the tree
    if from_word_info[POSTAG_LOCATION] == "IN":
        if "mod" in from_word_info[CONTEXT_LOCATION] or "neg" in from_word_info[CONTEXT_LOCATION]:
            from_list.append(from_word_info[CONTEXT_LOCATION])
            from_list.append(from_word)
            while from_word_info[POSTAG_LOCATION] not in noun_set and from_word_info[CONTEXT_LOCATION] != "ROOT":
                from_word_info = sent[int(from_word_info[FROM_LOCATION]) - 1]
                from_word = from_word_info[LEMMATIZED_LOCATION]
    to_tuple = tuple([from_word, context_type, "TO-ME"] + to_list)
    from_tuple = tuple([to_word, context_type, "FROM-ME"] + from_list)
    return to_word, from_word, to_tuple, from_tuple


# main loop, reading the file line by line and processing data for each of the tasks (sentence, window and dependency)
with open(file, encoding="utf8") as f:
    for string_sent in myreadlines(f, "\n\n"):
        if string_sent != "":
            sent = [x.split("\t") for x in string_sent.split("\n")]
            words_in_sent = [word_info[LEMMATIZED_LOCATION] for word_info in sent if
                             word_info[LEMMATIZED_LOCATION] in words_set and word_info[
                                 LEMMATIZED_LOCATION] not in function_words]
            start_sent_end = [START_WORD_INFO, START_WORD_INFO] + sent + [END_WORD_INFO, END_WORD_INFO]
            windows = its = zip(
                *[iter(start_sent_end), iter(start_sent_end[1:]), iter(start_sent_end[2:]), iter(start_sent_end[3:]),
                  iter(start_sent_end[4:])])
            for window in windows:
                word_info = window[2]
                word1 = word_info[LEMMATIZED_LOCATION]
                if word1 in words_set:
                    with timethis('words count'):
                        # code for the words that appear in the same sentence
                        u_att_counter_words[word1].update(words_in_sent)
                        total_att_counter_words.update(words_in_sent)
                    with timethis('dependency'):
                        # code for words that appear with the same context skipping the prepositions words
                        if word_info[CONTEXT_LOCATION] != "ROOT":
                            context_type = word_info[CONTEXT_LOCATION]
                            to_word, from_word, to_tuple, from_tuple = get_feature_dependency_edge(sent, word_info)
                            if to_word not in function_words and from_word not in function_words:
                                if to_word in words_set:
                                    u_att_counter_tuple[to_word].update([to_tuple])
                                    att_and_words_sets_tuple[to_tuple].add(to_word)
                                    att_set_tuple.add(to_tuple)
                                    total_att_counter_tuple.update([to_tuple])
                                if from_word in words_set:
                                    u_att_counter_tuple[from_word].update([from_tuple])
                                    att_and_words_sets_tuple[from_tuple].add(from_word)
                                    att_set_tuple.add(from_tuple)
                                    total_att_counter_tuple.update([from_tuple])
                    with timethis('windows'):
                        # code for the window based data collection
                        edges = [window[0], window[1], window[2], window[3]]
                        current_word = word_info[LEMMATIZED_LOCATION]
                        for edge_info in edges:
                            if edge_info[LEMMATIZED_LOCATION] not in function_words and edge_info[
                                LEMMATIZED_LOCATION] in words_set:
                                word2 = edge_info[LEMMATIZED_LOCATION]
                                u_att_counter_windows[current_word].update([word2])
                                total_att_counter_windows.update([word2])
                                att_and_words_sets_windows[word2].add(current_word)

num_of_u_att_tuple = sum(total_att_counter_tuple.values())
num_of_u_att_words = sum(total_att_counter_words.values())
num_of_u_att_windows = sum(total_att_counter_windows.values())


# probability functions, all of the functions here have been sanity checked as we saw in class
def p_u_att_tuple(word1, att):
    return u_att_counter_tuple[word1][att] / num_of_u_att_tuple


def p_u_att_words(word1, word_att):
    return u_att_counter_words[word1][word_att] / num_of_u_att_words


def p_u_att_windows(word1, word_att):
    return u_att_counter_windows[word1][word_att] / num_of_u_att_windows


def p_u_tuple(word):
    return sum(u_att_counter_tuple[word].values()) / num_of_u_att_tuple


def p_u_words(word):
    return sum(u_att_counter_words[word].values()) / num_of_u_att_words


def p_u_windows(word):
    return sum(u_att_counter_windows[word].values()) / num_of_u_att_windows


def p_att_tuple(att):
    return total_att_counter_tuple[att] / num_of_u_att_tuple


def p_att_words(att):
    return total_att_counter_words[att] / num_of_u_att_words


def p_att_windows(att):
    return total_att_counter_windows[att] / num_of_u_att_windows


# pmi functions for each tasks
def log_PMI_u_att_tuple(u, att):
    temp = p_u_tuple(u) * p_att_tuple(att)
    if temp == 0:
        return 0
    pmi = p_u_att_tuple(u, att) / temp
    if pmi == 0:
        return 0
    return np.log(pmi)


def log_PMI_u_att_words(u, att):
    temp = p_u_words(u) * p_att_words(att)
    if temp == 0:
        return 0
    pmi = p_u_att_words(u, att) / temp
    if pmi == 0:
        return 0
    return np.log(pmi)


def log_PMI_u_att_windows(u, att):
    temp = p_u_windows(u) * p_att_windows(att)
    if temp == 0:
        return 0
    pmi = p_u_att_windows(u, att) / temp
    if pmi == 0:
        return 0
    return np.log(pmi)

# the efficient code we saw in class for each of the tasks
def most_similar_efficient_att_tuple(u):
    highest_pmi_contexts = []
    D = defaultdict(float)
    for att in u_att_counter_tuple[u].most_common(100):
        log_pmi_u_att_0_tuple = log_PMI_u_att_tuple(u, att[0])
        highest_pmi_contexts.append((att[0], log_pmi_u_att_0_tuple))
        for v in att_and_words_sets_tuple[att[0]]:
            D[tuple([u, v])] += (log_pmi_u_att_0_tuple * log_PMI_u_att_tuple(v, att[0]))
    ret = list(D.items())
    ret.sort(key=get_second, reverse=True)
    highest_pmi_contexts.sort(key=get_second, reverse=True)
    return [x[1] for x, y in ret if x[1] != u][:20], [x[0] for x in highest_pmi_contexts if x[0][0] != u][:20]


def most_similar_efficient_att_words(u):
    highest_pmi_contexts = []
    D = defaultdict(float)
    for att in u_att_counter_words[u].most_common(100):
        log_pmi_u_att_0_words = log_PMI_u_att_words(u, att[0])
        highest_pmi_contexts.append((att[0], log_pmi_u_att_0_words))
        for v in u_att_counter_words[att[0]]:
            D[tuple([u, v])] += (log_pmi_u_att_0_words * log_PMI_u_att_words(v, att[0]))
    ret = list(D.items())
    ret.sort(key=get_second, reverse=True)
    highest_pmi_contexts.sort(key=get_second, reverse=True)
    return [x[1] for x, y in ret if x[1] != u][:20], [x[0] for x in highest_pmi_contexts if x[0] != u][:20]


def most_similar_efficient_att_windows(u):
    highest_pmi_contexts = []
    D = defaultdict(float)
    for att in u_att_counter_windows[u].most_common(100):
        log_pmi_u_att_0_windows = log_PMI_u_att_windows(u, att[0])
        highest_pmi_contexts.append((att[0], log_pmi_u_att_0_windows))
        for v in att_and_words_sets_windows[att[0]]:
            D[tuple([u, v])] += (log_pmi_u_att_0_windows * log_PMI_u_att_windows(v, att[0]))
    ret = list(D.items())
    ret.sort(key=get_second, reverse=True)
    highest_pmi_contexts.sort(key=get_second, reverse=True)
    return [x[1] for x, y in ret if x[1] != u][:20], [x[0] for x in highest_pmi_contexts if x[0] != u][:20]


tup_conts = []
two_words_cots = []
wind_conts = []

# printing part of the code from here...
for word in words:
    with timethis('tuple code'):
        tuple_similars, tuple_contexts = most_similar_efficient_att_tuple(word)
    with timethis('words code'):
        two_word_similars, two_words_contexts = most_similar_efficient_att_words(word)
    with timethis('windows code'):
        window_similars, window_contexts = most_similar_efficient_att_windows(word)
    tup_conts.append(tuple_contexts)
    two_words_cots.append(two_words_contexts)
    wind_conts.append(window_contexts)
    print(word)
    for i in range(20):
        print(str(window_similars[i]) + ", " + str(two_word_similars[i]) + ", " + str(tuple_similars[i]))
    print("*********")
print("\n")

for j in range(12):
    print(words[j])
    for i in range(20):
        print(str(wind_conts[j][i]) + ", " + str(two_words_cots[j][i]) + ", " + str(tup_conts[j][i]).replace(",", ";"))
    print("*********")
print("\n")

for most_common in unigrams.most_common(50):
    print(str(most_common[0]) + " " + str(most_common[1]))

print("\n")
for most_common in total_att_counter_tuple.most_common(50):
    print(str(most_common[0]) + " " + str(most_common[1]))

print("\nnum of words above 75 count: " + str(len([x[0] for x in unigrams.items() if x[1] > 75])))
print("\n")
print(times_dict)
