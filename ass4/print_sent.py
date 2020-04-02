from itertools import permutations

import spacy
from spacy import displacy

nlp = spacy.load('en')

SENT_TOKENS = 1

person_ents = {"PERSON"}
places_ents = {"GPE"}

sent = nlp(
    "Israel television rejected a skit by comedian Tuvia Tzafir that attacked public apathy by depicting an Israeli family watching TV while a fire raged outside .")

for i in sent:
    print(i.pos_, end=" ")


def check_if_person_and_location(sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    ent1_label = ent_text_to_ent(ent1, sent_data).label_
    ent2_label = ent_text_to_ent(ent2, sent_data).label_
    if ent1_label in person_ents and ent2_label in places_ents:
        return True
    return False


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




def get_start_ent1_and_end_ent2(ent1, ent2, sent_data):
    if ent1.end == ent2.start - 1:
        return []
    else:
        pos_list = []
        for i in range(ent1.end, ent2.start - 1):
            pos_list.append(sent_data[SENT_TOKENS][i].pos_)
        return pos_list
