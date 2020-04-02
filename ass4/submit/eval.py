# Yehoshua Stern 314963927
# Eyal Cohen 207947086

import codecs
import sys

from sklearn.metrics import precision_score, recall_score, f1_score


def process_line(line):
    line_splitted = line.strip().split("\t")
    sent_id = line_splitted[0]
    ent1 = line_splitted[1]
    relation = line_splitted[2]
    ent2 = line_splitted[3]
    return sent_id, ent1, ent2, relation


def process_file(fname):
    data = []
    for line in codecs.open(fname, encoding="utf8"):
        data.append(process_line(line))
    return data


def remove_dot(ent):
    if ent.endswith("."):
        return ent[:-1]
    return ent


def create_y_true__y_predicted(gold_data, predicted_data):
    y_true = []
    y_predicted = []
    seen_trues = []
    for gold_sent_data in gold_data:
        sent_id, ent1, ent2, relation = gold_sent_data
        if relation == "Live_In":
            y_true.append(1)
        else:
            y_true.append(0)
        y_to_be_added_to_predicted = 0
        for predicted_sent in predicted_data:
            sent_id_g, ent1_g, ent2_g, relation_g = predicted_sent
            ent1_g = remove_dot(ent1_g)
            ent2_g = remove_dot(ent2_g)
            ent1 = remove_dot(ent1)
            ent2 = remove_dot(ent2)
            if sent_id == sent_id_g and ent1 == ent1_g and ent2 == ent2_g and relation == relation_g:
                y_to_be_added_to_predicted = 1
                seen_trues.append(predicted_sent)
        y_predicted.append(y_to_be_added_to_predicted)
    for predicted_sent in predicted_data:
        if predicted_sent not in seen_trues:
            y_true.append(0)
            y_predicted.append(1)
    return y_true, y_predicted


def main():
    data_gold = process_file(sys.argv[1])
    data_output = process_file(sys.argv[2])
    y_true, y_pred = create_y_true__y_predicted(data_gold, data_output)
    print("Scores:")
    print("recall: ", recall_score(y_true, y_pred))
    print("precision: ", precision_score(y_true, y_pred))
    print("F1: ", f1_score(y_true, y_pred))


if __name__ == "__main__":
    main()
