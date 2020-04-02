import codecs
import pickle
import sys
from itertools import permutations

import spacy

try:
    # en - large gets better results
    nlp = spacy.load('en_core_web_lg')
except:
    nlp = spacy.load('en')


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


def process_test_data(dev_data):
    full_data = []
    for dat in dev_data:
        sent_id, tokens, text, noun_chunks, named_entities = dat
        for ent1, ent2 in permutations(named_entities, 2):
            x = sent_id, tokens, text, noun_chunks, named_entities, ent1.text, ent2.text
            full_data.append(x)
    return full_data


def main():
    if sys.argv[1].endswith(".txt"):
        train_data = process_file_test(sys.argv[1])
        test_data_processed = process_test_data(train_data)
        file = open("model", "rb")
        model = pickle.load(file)
        file.close()
        y_pred = model.predict(test_data_processed)
        output_file = open(sys.argv[2], "w")
        for sent_data, y in zip(test_data_processed, y_pred):
            if y == 1:
                sent_id, tokens, text, noun_chunks, named_entities, ent1, ent2 = sent_data
                print(sent_id + "\t" + ent1 + "\t" + "Live_In" + "\t" + ent2 + "\t" + "( " + text + " )",
                      file=output_file)
        output_file.close()
    else:
        print("this program accepts the txt file, not the processed file")


if __name__ == "__main__":
    main()
