import codecs
from collections import Counter, defaultdict
from itertools import permutations, combinations

import spacy
from spacy import displacy
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsClassifier

WORD_INDEX = 0
WORD_TEXT = 1
WORD_LEMMA = 2
WORD_TAG = 3
WORD_POS = 4
HEAD_ID = 5
WORD_DEP = 6
WORD_ENT_IOB = 7
WORD_ENT = 8

SENT_ID = 0
SENT_TOKENS = 1
SENT_TEXT = 2
ENTS = 4
ENT1 = 5
ENT2 = 6
RELATION = 7

nlp = spacy.load('en_core_web_lg')

'''
    PERSON	People, including fictional.
    NORP	Nationalities or religious or political groups.
    FAC	Buildings, airports, highways, bridges, etc.
    ORG	Companies, agencies, institutions, etc.
    GPE	Countries, cities, states.
    LOC	Non-GPE locations, mountain ranges, bodies of water.
    PRODUCT	Objects, vehicles, foods, etc. (Not services.)
    EVENT	Named hurricanes, battles, wars, sports events, etc.
    WORK_OF_ART	Titles of books, songs, etc.
    LAW	Named documents made into laws.
    LANGUAGE	Any named language.
    DATE	Absolute or relative dates or periods.
    TIME	Times smaller than a day.
    PERCENT	Percentage, including ”%“.
    MONEY	Monetary values, including unit.
    QUANTITY	Measurements, as of weight or distance.
    ORDINAL	“first”, “second”, etc.
    CARDINAL	Numerals that do not fall under another type.
'''

person_ents = {"PERSON"}
places_ents = {"GPE"}

COMBINATIONS = 5

irrelevant_poss = {}


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


class CombinedClassifier:
    def __init__(self, train_data):
        self.place_person_counter, self.person_place_counter = get_pos_lists_between(
            [x for x, y in train_data if y == 1])
        self.min_matches = 1
        self.most_common = 20
        self.person_place_counter_most_common = self.person_place_counter.most_common(self.most_common)
        print(self.person_place_counter_most_common)
        self.place_person_counter_most_common = self.place_person_counter.most_common(self.most_common)
        print(self.place_person_counter_most_common)
        self.person_place_true_true, self.place_person_true_true, self.person_place_true_false, self.place_person_true_false = self.check_for_mistakes(
            train_data)
        place_person_counter_false, person_place_counter_false = get_pos_lists_between(
            [x for x, y in train_data if y == 0])
        print(place_person_counter_false.most_common(self.most_common))
        print(person_place_counter_false.most_common(self.most_common))



    @staticmethod
    def patterns_match_train_patterns(sent_patterns, gold_patterns):
        gold_patterns = [x[0] for x in gold_patterns]
        matched_counter = 0
        for sent_pattern in sent_patterns:
            if sent_pattern in gold_patterns:
                matched_counter += 1
        return matched_counter

    def predict_no_y(self, predict_data):
        predictions = []
        for sent_data in predict_data:
            sent_data_no_relation = sent_data
            if len(sent_data) == 8:
                sent_data_no_relation = sent_data[:-1]
            try:
                if check_if_person_and_location(sent_data_no_relation):
                    if ents_have_no_words_between(sent_data_no_relation):
                        predictions.append(0)
                    else:
                        patterns_list = get_patterns(sent_data_no_relation)
                        if len(intersection(patterns_list, self.person_place_true_false)) > len(intersection(patterns_list, self.person_place_true_true)):
                            predictions.append(0)
                        else:
                            predictions.append(1)
                elif check_if_location_and_person(sent_data_no_relation):
                    if ents_have_no_words_between(sent_data_no_relation):
                        predictions.append(0)
                    else:
                        patterns_list = get_patterns(sent_data_no_relation)
                        if len(intersection(patterns_list, self.place_person_true_false)) > len(intersection(patterns_list, self.place_person_true_true)):
                            predictions.append(0)
                        else:
                            predictions.append(1)
                else:
                    predictions.append(0)
            except AttributeError:
                predictions.append(0)
        return predictions

    def check_for_mistakes(self, predict_data):
        predictions = []
        for sent_data in [x for x, y in predict_data]:
            sent_data_no_relation = sent_data
            if len(sent_data) == 8:
                sent_data_no_relation = sent_data[:-1]
            try:
                if check_if_person_and_location(sent_data_no_relation):
                    if ents_have_no_words_between(sent_data_no_relation):
                        predictions.append(0)
                    else:
                        predictions.append(1)
                    # if self.patterns_match_train_patterns(get_patterns(sent_data_no_relation),
                    #                                       self.person_place_counter.most_common(
                    #                                               self.most_common)) > self.min_matches:
                    #     predictions.append(1)
                    # else:
                    #     predictions.append(0)
                elif check_if_location_and_person(sent_data_no_relation):
                    if ents_have_no_words_between(sent_data_no_relation):
                        predictions.append(0)
                    else:
                        predictions.append(1)
                    # if self.patterns_match_train_patterns(get_patterns(sent_data_no_relation),
                    #                                       self.place_person_counter.most_common(
                    #                                               self.most_common)) > self.min_matches:
                    #     predictions.append(1)
                    # else:
                    #     predictions.append(0)
                else:
                    predictions.append(0)
            except AttributeError:
                predictions.append(0)

        model_true = []
        model_false = []
        for sent_data, prediction in zip(predict_data, predictions):
            if prediction == 1:
                if sent_data[1] == 1:
                    model_true.append((sent_data[0], 1))
                else:
                    model_false.append((sent_data[0], 0))
        place_person_counter1, person_place_counter1 = get_pos_lists_between([x for x, y in model_true])
        place_person_counter2, person_place_counter2 = get_pos_lists_between([x for x, y in model_false])
        return [x[0] for x in person_place_counter1.most_common(self.most_common)], [x[0] for x in
                                                                                     place_person_counter1.most_common(
                                                                                         self.most_common)], [x[0] for x
                                                                                                              in
                                                                                                              person_place_counter2.most_common(
                                                                                                                  self.most_common)], [
                   x[0] for x in place_person_counter2.most_common(self.most_common)]


def ents_have_no_words_between(sent_data):
    ent1 = ent_text_to_ent(sent_data[ENT1], sent_data)
    ent2 = ent_text_to_ent(sent_data[ENT2], sent_data)
    if len(get_pos_between_ents(ent1, ent2, sent_data, {})) == 0:
        return True
    return False


def process_line(line):
    sent_id, sent = line.strip().split("\t")
    sent = sent.replace("-LRB-", "(")
    sent = sent.replace("-RRB-", ")")
    sent = nlp(sent)
    sent_data = []
    for word in sent:
        head_id = str(word.head.i + 1)  # we want ids to be 1 based
        if word == word.head:  # and the ROOT to be 0.
            assert (word.dep_ == "ROOT"), word.dep_
            head_id = 0  # root
        sent_data.append(
            [word.i + 1, word.text, word.lemma_, word.tag_, word.pos_, int(head_id), word.dep_, word.ent_iob_,
             word.ent_type_])
    noun_chunks = [np for np in sent.noun_chunks]
    named_entities = [ne for ne in sent.ents]
    return sent_id, sent_data, noun_chunks, named_entities


def process_line_full_train(line):
    sent_id, ent1, relation, ent2, sent = line.strip().split("\t")
    sent = sent[1:]
    sent = sent[:-1]
    sent = nlp(sent)
    noun_chunks = [n for n in sent.noun_chunks]
    named_entities = [ne for ne in sent.ents]
    tokens = [x for x in sent]
    return sent_id, tokens, sent.text, noun_chunks, named_entities, ent1, ent2, relation


def process_line_full_test(line):
    sent_id, sent = line.strip().split("\t")
    sent = sent.replace("-LRB-", "(")
    sent = sent.replace("-RRB-", ")")
    sent = nlp(sent)
    noun_chunks = [n for n in sent.noun_chunks]
    named_entities = [ne for ne in sent.ents]
    tokens = [x for x in sent]
    return sent_id, tokens, sent.text, noun_chunks, named_entities


def process_file_test(fname):
    data = []
    for line in codecs.open(fname, encoding="utf8"):
        data.append(process_line_full_test(line))
    return data


def process_file_train(fname):
    data = []
    for line in codecs.open(fname, encoding="utf8"):
        data.append(process_line_full_train(line))
    return data


def find_sub_list(sl, l):
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            return ind, ind + sll - 1


def find_sub_list_inside(sl, l):
    found_start = False
    start = None
    end = None
    start_ent = sl[0]
    end_ent = sl[-1]
    for i, word in enumerate(l):
        if start_ent in word and not found_start:
            start = i
            found_start = True
        if found_start and end_ent in word:
            end = i
    if start < end:
        temp = start
        start = end
        end = temp
    return start, end


def sum_of_words_between(ent1, ent2, sent_data):  # TODO: fix the last and start word
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities = sent_data
    try:
        start1, end1 = find_sub_list(ent1.split(), sent_text.split())
        start2, end2 = find_sub_list(ent2.split(), sent_text.split())
    except TypeError:
        start1, end1 = find_sub_list_inside(ent1.split(), sent_text.split())
        start2, end2 = find_sub_list_inside(ent2.split(), sent_text.split())
    sum = np.zeros(sent_tokens[0].vector.shape)
    if end1 <= start2:
        if (start2 - 1) - (end1 + 1) > 0:
            if end1 + 1 == start2 - 1:
                sum += sent_tokens[end1 + 1].vector
            for i in range(end1 + 1, start2 - 1):
                sum += sent_tokens[i].vector
        return sum
    if end2 <= start1:
        if (start1 - 1) - (end2 + 1) > 0:
            if end2 + 1 == start1 - 1:
                sum += sent_tokens[end2 + 1].vector
            for i in range(end2 + 1, start1 - 1):
                sum += sent_tokens[i].vector
        return sum
    return sum


def sum_of_words_before_ent1_and_after_ent2(ent1, ent2, sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities = sent_data
    try:
        start1, end1 = find_sub_list(ent1.split(), sent_text.split())
        start2, end2 = find_sub_list(ent2.split(), sent_text.split())
    except TypeError:
        start1, end1 = find_sub_list_inside(ent1.split(), sent_text.split())
        start2, end2 = find_sub_list_inside(ent2.split(), sent_text.split())
    sum_start = np.zeros(sent_tokens[0].vector.shape)
    sum_end = np.zeros(sent_tokens[0].vector.shape)
    if start1 < start2:
        if start1 != 0:
            for i in range(0, start1 - 1):
                sum_start += sent_tokens[i].vector
        if end2 != len(sent_tokens):
            for i in range(end2 + 1, len(sent_tokens)):
                sum_end += sent_tokens[i].vector
    elif start2 < start1:
        if start1 != 0:
            for i in range(0, start2):
                sum_start += sent_tokens[i].vector
        if end2 != len(sent_tokens):
            for i in range(end1, len(sent_tokens)):
                sum_end += sent_tokens[i].vector
    return sum_start + sum_end


def sum_of_ent(ent, sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities = sent_data
    if sent_id == "sent70":
        temp = 1
    start, end = find_sub_list(ent.split(), sent_text.split())
    sum = np.zeros(sent_tokens[0].vector.shape)
    if start == end:
        sum = sent_tokens[start].vector
        return sum
    for i in range(start, end):
        sum += sent_tokens[i].vector
    return sum


def sum_of_ent_func(ent, sent_data):
    return ent_text_to_ent(ent, sent_data).vector


def create_features(sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2, relation = sent_data
    sum_of_ent1 = sum_of_ent_func(ent1, sent_data[:-1])
    sum_of_ent2 = sum_of_ent_func(ent2, sent_data[:-1])
    sent_data = sent_data[:-3]
    sum_of_words_betweeno = sum_of_words_between(ent1, ent2, sent_data)
    sum_of_words_bet_and_aft = sum_of_words_before_ent1_and_after_ent2(ent1, ent2, sent_data)
    x = np.concatenate((sum_of_ent1, sum_of_ent2, sum_of_words_betweeno))
    y = 0
    if relation == "Live_In":
        y = 1
    return x, y


def create_features_no_y(sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    sum_of_ent1 = sum_of_ent_func(ent1, sent_data)
    sum_of_ent2 = sum_of_ent_func(ent2, sent_data)
    sent_data = sent_data[:-2]
    sum_of_words_betweeno = sum_of_words_between(ent1, ent2, sent_data)
    sum_of_words_bet_and_aft = sum_of_words_before_ent1_and_after_ent2(ent1, ent2, sent_data)
    x = np.concatenate((sum_of_ent1, sum_of_ent2, sum_of_words_betweeno))
    return x


def check_num_of_zero_and_ones(data):
    counter = Counter()
    counter.update([y for x, y in data])
    return counter


def check_if_person_and_location(sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    ent1_label = ent_text_to_ent(ent1, sent_data).label_
    ent2_label = ent_text_to_ent(ent2, sent_data).label_
    if ent1_label in person_ents and ent2_label in places_ents:
        return True
    return False


def check_if_location_and_person(sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    ent1_label = ent_text_to_ent(ent1, sent_data).label_
    ent2_label = ent_text_to_ent(ent2, sent_data).label_
    if ent1_label in person_ents and ent2_label in places_ents:
        return True
    return False


def check_if_there_is_closer_person_ent(sent_data):
    try:
        sent_id, sent_tokens, sent_text, noun_chunks, named_entities, person_ent, place_ent = sent_data
        ent_list = [x.text for x in named_entities]
        closest = person_ent
        min_dist = np.abs(ent_list.index(person_ent) - ent_list.index(place_ent))
        for i, ent in enumerate(named_entities):
            if ent.label_ in person_ents and np.abs(i - ent_list.index(place_ent)) < min_dist:
                min_dist = np.abs(i - ent_list.index(place_ent))
                closest = ent.text
        if closest == person_ent:
            return True
        return False
    except ValueError:
        return False


def check_if_there_is_closer_place_ent(sent_data):
    try:
        sent_id, sent_tokens, sent_text, noun_chunks, named_entities, person_ent, place_ent = sent_data
        ent_list = [x.text for x in named_entities]
        closest = place_ent
        min_dist = np.abs(ent_list.index(place_ent) - ent_list.index(person_ent))
        for i, ent in enumerate(named_entities):
            if ent.label_ in places_ents and np.abs(i - ent_list.index(person_ent)) < min_dist:
                min_dist = np.abs(i - ent_list.index(person_ent))
                closest = ent.text
        if closest == place_ent:
            return True
        return False
    except ValueError:
        return False


def create_features_small(sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2, relation = sent_data
    sent_data = sent_data[:-1]
    x = []
    y = 0
    if not check_if_person_and_location(sent_data):
        x.append(1)
    else:
        x.append(0)
    if check_if_there_is_closer_person_ent(sent_data):
        x.append(1)
    else:
        x.append(0)
    if check_if_there_is_closer_place_ent(sent_data):
        x.append(1)
    else:
        x.append(0)
    if relation == "Live_In":
        y = 1
    return x, y


def create_features_small_no_y(sent_data):
    x = []
    if not check_if_person_and_location(sent_data):
        x.append(1)
    else:
        x.append(0)
    if check_if_there_is_closer_person_ent(sent_data):
        x.append(1)
    else:
        x.append(0)
    if check_if_there_is_closer_place_ent(sent_data):
        x.append(1)
    else:
        x.append(0)
    return x


def ent_text_to_ent(ent, sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    for ent_i in named_entities:
        if ent == ent_i.text:
            return ent_i
    for ent_i in named_entities:
        if ent in ent_i.text:
            return ent_i
    for ent_i in named_entities:
        if ent_i.text in ent:
            return ent_i


def expand_data(train_data, create_features_func):
    train_data_processed = []
    train_data_unannotated = process_file_test("data/Corpus.TRAIN.txt")
    for sent_data_unanonotated in train_data_unannotated:
        sent_id, sent_tokens, sent_text, noun_chunks, named_entities = sent_data_unanonotated
        for ent1, ent2 in permutations(sent_data_unanonotated[ENTS], 2):
            lives_in_flag = "not_lives_in"
            for sent_data in train_data:
                if sent_data_unanonotated[SENT_ID] == sent_data[SENT_ID] and ent1.text == sent_data[
                    ENT1] and ent2.text == sent_data[ENT2] and sent_data[RELATION] == "Live_In":
                    # print(ent1.label_, ent2.label_)
                    lives_in_flag = "Live_In"
            appended_data = sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1.text, ent2.text, lives_in_flag
            train_data_processed.append(create_features_func(appended_data))
    return train_data_processed


def train_mlp_classifier(train_data):
    train_data_processed = expand_data(train_data, create_features)
    mlp = MLPClassifier()
    mlp.fit([x for x, y in train_data_processed], [y for x, y in train_data_processed])
    return mlp


def train_svm_classifier(train_data):
    train_data_processed = expand_data(train_data, create_features)
    svm = LinearSVC(max_iter=1000)
    svm.fit([x for x, y in train_data_processed], [y for x, y in train_data_processed])
    return svm


def train_lasso_classifier(train_data):
    train_data_processed = expand_data(train_data, create_features)
    lasso = LassoCV(max_iter=10000, cv=5)
    lasso.fit([x for x, y in train_data_processed], [y for x, y in train_data_processed])
    return lasso


def train_knn_classifier(train_data):
    train_data_processed = expand_data(train_data, create_features_small)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit([x for x, y in train_data_processed], [y for x, y in train_data_processed])
    return knn


def process_dev_data(dev_data, dev_data_with_y):
    full_data = []
    for dat in dev_data:
        sent_id, tokens, text, noun_chunks, named_entities = dat
        for ent1, ent2 in permutations(named_entities, 2):
            x = sent_id, tokens, text, noun_chunks, named_entities, ent1.text, ent2.text
            y = 0
            for dat_with_y in dev_data_with_y:
                if dat_with_y[SENT_ID] == sent_id and dat_with_y[ENT1] == ent1.text and dat_with_y[
                    ENT2] == ent2.text and dat_with_y[RELATION] == "Live_In":
                    # print(ent1.label_, ent2.label_)
                    y = 1
            full_data.append((x, y))
    return full_data


def get_pos_lists_between(train_data):
    place_person_list = []
    person_place_list = []
    for sent_data in train_data:
        sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
        he_lives = ent_text_to_ent(ent1, sent_data)
        in_this = ent_text_to_ent(ent2, sent_data)
        if he_lives.start < in_this.start:
            person_place_list.append(get_pos_between_ents(he_lives, in_this, sent_data, irrelevant_poss))
        else:
            place_person_list.append(get_pos_between_ents(in_this, he_lives, sent_data, irrelevant_poss))
    place_person_counter = Counter()
    person_place_counter = Counter()
    for place_person in place_person_list:
        for comb in combinations(place_person, COMBINATIONS):
            place_person_counter[str(comb)] += 1
    for person_place in person_place_list:
        for comb in combinations(person_place, COMBINATIONS):
            person_place_counter[str(comb)] += 1
    return place_person_counter, person_place_counter


def get_pos_list_between(sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    he_lives = ent_text_to_ent(ent1, sent_data)
    in_this = ent_text_to_ent(ent2, sent_data)
    return get_pos_between_ents(in_this, he_lives, sent_data, irrelevant_poss)


def get_patterns(sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    return [str(x) for x in combinations(get_pos_between_ents(ent_text_to_ent(ent1, sent_data), ent_text_to_ent(ent2, sent_data), sent_data, irrelevant_poss), COMBINATIONS)]


def get_pos_between_ents(ent1, ent2, sent_data, irrelevant_poss):
    if ent1.start < ent2.start:
        if ent1.end >= ent2.start - 1:
            if ent1.end == ent2.start - 1 and sent_data[SENT_TOKENS][ent1.end].pos_ not in irrelevant_poss:
                return [sent_data[SENT_TOKENS][ent1.end].pos_]
            else:
                return []
        else:
            pos_list = []
            for i in range(ent1.end, ent2.start - 1):
                if sent_data[SENT_TOKENS][i].pos_ not in irrelevant_poss:
                    pos_list.append(sent_data[SENT_TOKENS][i].pos_)
            return pos_list
    else:
        if ent2.end >= ent1.start - 1:
            if ent2.end == ent1.start - 1 and sent_data[SENT_TOKENS][ent2.end].pos_ not in irrelevant_poss:
                return [sent_data[SENT_TOKENS][ent2.end].pos_]
            else:
                return []
        else:
            pos_list = []
            for i in range(ent2.end, ent1.start - 1):
                if sent_data[SENT_TOKENS][i].pos_ not in irrelevant_poss:
                    pos_list.append(sent_data[SENT_TOKENS][i].pos_)
            return pos_list


def main():
    train_data_with_y = process_file_train("data/DEV.annotations")
    train_data = process_file_test("data/Corpus.DEV.txt")
    train_data_processed = process_dev_data(train_data, train_data_with_y)
    classifier = CombinedClassifier(train_data_processed)
    # y_pred = classifier.predict([x for x, y in train_data_processed])
    # print(len([y for y in y_pred if y == 1]))
    # y_true = [y for x, y in train_data_processed]
    # print(len([y for y in y_true if y == 1]))
    # print("TRAIN Scores:")
    # print("recall: ", recall_score(y_true, y_pred))
    # print("precision: ", precision_score(y_true, y_pred))
    # print("F1: ", f1_score(y_true, y_pred))
    dev_data_with_y = process_file_train("data/DEV.annotations")
    dev_data = process_file_test("data/Corpus.DEV.txt")
    dev_data_processed = process_dev_data(dev_data, dev_data_with_y)
    y_pred = classifier.predict_no_y([x for x, y in dev_data_processed])
    y_true = [y for x, y in dev_data_processed]
    print("DEV Scores:")
    print("recall: ", recall_score(y_true, y_pred))
    print("precision: ", precision_score(y_true, y_pred))
    print("F1: ", f1_score(y_true, y_pred))


if __name__ == "__main__":
    main()


def ners_are_between(left_list, right_list, phrase_between, id, tokens, text, noun_chuncks):
    found_left = False
    left_ent = []
    found_right = False
    right_ent = []
    found_phrase = False
    phrase_index = None
    for i, tok in enumerate(tokens):
        if not found_left:
            if tok.ent_type_ in left_list:
                found_left = True
        elif not found_phrase:
            if i < len(tokens) - len(phrase_between):
                found = True
                for j, phrase_word in enumerate(phrase_between):
                    if phrase_word != tokens[j + i].lemma_:
                        found = False
                if found:
                    found_phrase = True
                    phrase_index = i
        elif not found_right:
            if tok.ent_type_ in right_list:
                found_right = True
    if found_left and found_right and found_phrase:
        if phrase_index != 0:
            still_inside = True
            for i in range(phrase_index - 1, 0, -1):
                if tokens[i].ent_type_ in left_list and still_inside:
                    if tokens[i].ent_iob_ == 'B':
                        still_inside = False
                    left_ent.append(tokens[i])
        left_ent.reverse()
        started = False
        for i in range(phrase_index + len(phrase_between), len(tokens)):
            if tokens[i].ent_type_ in right_list:
                if tokens[i].ent_iob_ == 'B':
                    started = True
                elif started:
                    if tokens[i].ent_iob_ != 'I':
                        break
                right_ent.append(tokens[i])
        return left_ent, right_ent
    return None

    # phrases_between = [["live", "in"], ["is", "in"], ["locate", "at"], ["change"], ["turn"], ["work", "in"]]
    # for id, tokens, text, noun_chuncks in data:
    #     for phrase in phrases_between:
    #         if ners_are_between(person_ents, places_ents, phrase, id, tokens, text, noun_chuncks):
    #             left_ent, right_ent = ners_are_between(person_ents, places_ents, phrase, id, tokens, text, noun_chuncks)
    #             print([x.text for x in left_ent], [x.text for x in right_ent])

    # total_relations = len([y for x, y in test_data_processed if y == 1])
    # for i, y, y_tag in zip(range(len(y_pred)), [y for x, y in test_data_processed], y_pred):
    #     if y == y_tag:
    #         counter_same += 1
    #     if y == y_tag == 1:
    #         counter_same_and_one += 1
    #         print(test_data[i][5], test_data[i][7], test_data[i][6], test_data[i][2])
    # print("accuracy: ", counter_same_and_one / total_relations)
    # print("recall: ", counter_same / len(test_data_processed))


class Node:
    def __init__(self, data):
        id, text, lemma, tag, pos, head_id, dep, ent_iob, ent_type = data
        self.id = id
        self.text = text
        self.lemma = lemma
        self.tag = tag
        self.pos = pos
        self.head_id = head_id
        self.dep = dep
        self.ent_iob = ent_iob
        self.ent = ent_type
        self.children = []

    def add_children(self, sent_data):
        for word in sent_data:
            if word[HEAD_ID] == self.id:
                self.children.append(Node(word))
        for child in self.children:
            child.add_children(sent_data)


class DependencyTree:
    def __init__(self, sent_data_full):
        sent_id, sent_data, noun_chunks, named_entities, text = sent_data_full
        self.text = text
        self.sent_id = sent_id
        self.sent_data = sent_data
        self.noun_chunks = noun_chunks
        self.named_entities = named_entities
        root_data = None
        for word in self.sent_data:
            if word[HEAD_ID] == 0:
                root_data = word
        assert root_data is not None
        self.root = Node(root_data)
        self.root.add_children(sent_data)
