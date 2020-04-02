# Yehoshua Stern 314963927
# Eyal Cohen 207947086

import codecs
from itertools import permutations
from Model import RelationExtractionModel
from create_features import ent_text_to_ent
import spacy
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import sys

try:
    # en - large gets better results
    nlp = spacy.load('en_core_web_lg')
except:
    nlp = spacy.load('en')

SENT_ID = 0
SENT_TOKENS = 1
SENT_TEXT = 2
ENTS = 4
ENT1 = 5
ENT2 = 6
RELATION = 7

person_ents = set()
places_ents = set()


def process_line_full_train(line):
    sent_id, ent1, relation, ent2, sent = line.strip().split("\t")
    sent = sent[1:]
    sent = sent[:-1]
    sent = nlp(sent)
    noun_chunks = [n for n in sent.noun_chunks]
    named_entities = [ne for ne in sent.ents]
    tokens = [x for x in sent]
    return sent_id, tokens, sent.text, noun_chunks, named_entities, ent1, ent2, relation


def process_file_train(fname):
    data = []
    for line in codecs.open(fname, encoding="utf8"):
        data.append(process_line_full_train(line))
    return data


def process_file_test(fname):
    data = []
    for line in codecs.open(fname, encoding="utf8"):
        data.append(process_line_full_test(line))
    return data


def process_line_full_test(line):
    sent_id, sent = line.strip().split("\t")
    sent = sent.replace("-LRB-", "(")
    sent = sent.replace("-RRB-", ")")
    sent = nlp(sent)
    noun_chunks = [n for n in sent.noun_chunks]
    named_entities = [ne for ne in sent.ents]
    tokens = [x for x in sent]
    return sent_id, tokens, sent.text, noun_chunks, named_entities


def remove_dot(ent):
    if ent.endswith("."):
        return ent[:-1]
    return ent


def process_dev_data(dev_data, dev_data_with_y):
    full_data = []
    for dat in dev_data:
        sent_id, tokens, text, noun_chunks, named_entities = dat
        for ent1, ent2 in permutations(named_entities, 2):
            x = sent_id, tokens, text, noun_chunks, named_entities, ent1.text, ent2.text
            y = 0
            for dat_with_y in dev_data_with_y:
                sent_id_d, tokens_d, sent_text_d, noun_chunks_d, named_entities_d, ent1_d, ent2_d, relation_d = dat_with_y
                ent1_d = remove_dot(ent1_d)
                ent2_d = remove_dot(ent2_d)
                ent1_a = remove_dot(ent1.text)
                ent2_a = remove_dot(ent2.text)
                if sent_id_d == sent_id and ent1_d == ent1_a and ent2_d == ent2_a and relation_d == "Live_In":
                    person_ents.add(ent1.label_)
                    places_ents.add(ent2.label_)
                    y = 1
            full_data.append((x, y))
    return full_data


def remove_non_relevant_samples(data):
    filtered_data = []
    for sent_data, y in data:
        sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
        ent1_token = ent_text_to_ent(ent1, sent_data)
        ent2_token = ent_text_to_ent(ent2, sent_data)
        if ent1_token.label_ in person_ents and ent2_token.label_ in places_ents:
            filtered_data.append((sent_data, y))
    return filtered_data


def main():
    train_data_with_y = process_file_train(sys.argv[1])
    train_data = process_file_test(sys.argv[2])
    train_data_processed = process_dev_data(train_data, train_data_with_y)

    model = RelationExtractionModel(train_data_processed, places_ents, person_ents)
    train_data_filtered = remove_non_relevant_samples(train_data_processed)
    print("train data size before filtering: ", len(train_data_processed), "train data size after filtering: ",
          len(train_data_filtered))

    y_pred = model.predict([x for x, y in train_data_filtered])
    y_true = [y for x, y in train_data_filtered]
    print("TRAIN Scores:")
    print("recall: ", recall_score(y_true, y_pred))
    print("precision: ", precision_score(y_true, y_pred))
    print("F1: ", f1_score(y_true, y_pred))

    dev_data_with_y = process_file_train(sys.argv[3])
    dev_data = process_file_test(sys.argv[4])
    dev_data_processed = process_dev_data(dev_data, dev_data_with_y)
    dev_data_filtered = remove_non_relevant_samples(dev_data_processed)
    print("dev data size before filtering: ", len(dev_data_processed), "dev data size after filtering: ",
          len(dev_data_filtered))
    y_pred = model.predict([x for x, y in dev_data_filtered])
    y_true = [y for x, y in dev_data_filtered]
    print("DEV Scores:")
    print("recall: ", recall_score(y_true, y_pred))
    print("precision: ", precision_score(y_true, y_pred))
    print("F1: ", f1_score(y_true, y_pred))
    file = open("model", "wb")
    pickle.dump(model, file)
    file.close()


if __name__ == "__main__":
    main()

# BEST RUN:
# train data size before filtering:  7200 train data size after filtering:  2961
# TRAIN Scores:
# recall:  0.9245283018867925
# precision:  0.7368421052631579
# F1:  0.8200836820083681
# dev data size before filtering:  7558 dev data size after filtering:  3529
# DEV Scores:
# recall:  0.5803571428571429
# precision:  0.4676258992805755
# F1:  0.5179282868525895

# scores pos_path_tags=true pos_tags=true comb=3 class_weight=80:20
# TRAIN Scores:
# recall:  0.9223300970873787
# precision:  0.7307692307692307
# F1:  0.815450643776824
# DEV Scores:
# recall:  0.6
# precision:  0.41007194244604317
# F1:  0.48717948717948717

# scores pos_path_tags=true pos_tags=true comb=2
# recall:  0.8737864077669902
# precision:  0.6382978723404256
# F1:  0.737704918032787
# DEV Scores:
# recall:  0.6105263157894737
# precision:  0.38666666666666666
# F1:  0.47346938775510206

# scores pos_path_tags=true pos_tags=true comb=5
# recall:  0.9223300970873787
# precision:  0.6884057971014492
# F1:  0.7883817427385892
# DEV Scores:
# recall:  0.5789473684210527
# precision:  0.40145985401459855
# F1:  0.47413793103448276

# scores pos_path_tags=true pos_tags=true comb=4
# TRAIN Scores:
# recall:  0.9223300970873787
# precision:  0.7307692307692307
# F1:  0.815450643776824
# DEV Scores:
# recall:  0.5473684210526316
# precision:  0.3939393939393939
# F1:  0.4581497797356828

# scores pos_path_tags=true pos_tags=true comb=3 class_weight = 50:50
# TRAIN Scores:
# recall:  0.8543689320388349
# precision:  0.9565217391304348
# F1:  0.9025641025641026
# DEV Scores:
# recall:  0.4105263157894737
# precision:  0.46987951807228917
# F1:  0.4382022471910112

# scores pos_path_tags=true pos_tags=true comb=3 class_weight 90:10
# TRAIN Scores:
# recall:  0.941747572815534
# precision:  0.6024844720496895
# F1:  0.7348484848484848
# DEV Scores:
# recall:  0.631578947368421
# precision:  0.36585365853658536
# F1:  0.4633204633204633

# scores pos_path_tags=true pos_tags=true comb=3 class_weight 90:10
# TRAIN Scores:
# recall:  0.8932038834951457
# precision:  0.8214285714285714
# F1:  0.8558139534883721
# DEV Scores:
# recall:  0.5052631578947369
# precision:  0.41739130434782606
# F1:  0.45714285714285713

# train data size before filtering:  7200 train data size after filtering:  2961
# TRAIN Scores:
# recall:  0.9223300970873787
# precision:  0.7307692307692307
# F1:  0.815450643776824
# dev data size before filtering:  7558 dev data size after filtering:  3529
# DEV Scores:
# recall:  0.5473684210526316
# precision:  0.3939393939393939
# F1:  0.4581497797356828
