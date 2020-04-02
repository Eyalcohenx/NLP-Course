from itertools import combinations
import numpy as np
import spacy
import networkx as nx

try:
    # en - large gets better results
    nlp = spacy.load('en_core_web_lg')
except:
    nlp = spacy.load('en')
SENT_TOKENS = 1
COMBINATIONS = 3
irrelevant_poss = {}

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


def word_b4_first_ent_and_word_after_second_ent(sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    ent1 = ent_text_to_ent(ent1, sent_data)
    ent2 = ent_text_to_ent(ent2, sent_data)

    word_b4 = "START"
    word_after = "END"

    # checking who comes first and switching if needed
    if ent1.start > ent2.start:
        temp = ent1
        ent1 = ent2
        ent2 = temp

    # adding the start and end word
    if ent1.start - 1 >= 0:
        word_b4 = sent_tokens[ent1.start - 1].text
    if ent2.end < len(sent_tokens):
        word_after = sent_tokens[ent2.end].text

    return word_b4, word_after


def bow_between_ents(ent1_token, ent2_token, sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    # checking who comes first and switching if needed
    if ent1_token.start > ent2_token.start:
        temp = ent1_token
        ent1_token = ent2_token
        ent2_token = temp
    if ent1_token.end == ent2_token.end - 1:
        return sent_tokens[ent1_token.end].vector
    bow = np.zeros(sent_tokens[0].vector.shape)
    for i in range(ent1_token.end, ent2_token.end - 1):
        bow += sent_tokens[i].vector
    return bow


def ents_between(ent1_token, ent2_token, sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    ents_list = [ent.text for ent in named_entities]
    # checking who comes first and switching if needed
    if ent1_token.start > ent2_token.start:
        temp = ent1_token
        ent1_token = ent2_token
        ent2_token = temp
    counter = 0
    for i in range(ents_list.index(ent1_token.text), ents_list.index(ent2_token.text)):
        counter += 1
    counter -= 1
    return counter


def get_path_length_between_ents(ent1_token, ent2_token, sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    doc = nlp(sent_text)
    # print('sentence:'.format(doc))  # Load spacy's dependency tree into a networkx graph
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_), '{0}'.format(child.lower_)))

    graph = nx.Graph(edges)  # Get the length and path
    entity1 = ent1_token.root.lower_
    entity2 = ent2_token.root.lower_
    # print(nx.shortest_path(graph, source=entity1, target=entity2))
    try:
        return nx.shortest_path_length(graph, source=entity1, target=entity2)
    except nx.exception.NetworkXNoPath:
        return -1


def get_path_between_ents(ent1_token, ent2_token, sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    doc = nlp(sent_text)
    # print('sentence:'.format(doc))  # Load spacy's dependency tree into a networkx graph
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_), '{0}'.format(child.lower_)))

    graph = nx.Graph(edges)  # Get the length and path
    entity1 = ent1_token.root.lower_
    entity2 = ent2_token.root.lower_
    try:
        return nx.shortest_path(graph, source=entity1, target=entity2)
    except nx.exception.NetworkXNoPath:
        return []


def get_path_pos_combinations_between_ents(ent1_token, ent2_token, sent_data):
    path = get_path_between_ents(ent1_token, ent2_token, sent_data)
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    words_list = [x.text.lower() for x in sent_tokens]
    pos_list = []
    for word in path:
        pos_list.append(sent_tokens[words_list.index(word.lower())].pos_)
    return [str(x) for x in combinations(pos_list, COMBINATIONS)]


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


def get_patterns(sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    return [str(x) for x in combinations(
        get_pos_between_ents(ent_text_to_ent(ent1, sent_data), ent_text_to_ent(ent2, sent_data), sent_data,
                             irrelevant_poss), COMBINATIONS)]


def get_links_combinations_between_ents(ent1_token, ent2_token, sent_data):
    path = get_path_between_ents(ent1_token, ent2_token, sent_data)
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    words_list = [x.text.lower() for x in sent_tokens]
    links_list = []
    edges = []
    for token in sent_tokens:
        edges.append((token.head.text.lower(), token.text.lower(), token.dep_))
    from_to_list = [x[0] + "_" + x[1] for x in edges]
    dep_list = [x[2] for x in edges]
    for i in range(len(path) - 1):
        current_in_path = path[i]
        next_in_path = path[i + 1]
        if current_in_path + "_" + next_in_path in from_to_list:
            links_list.append(dep_list[from_to_list.index(current_in_path + "_" + next_in_path)])
        elif next_in_path + "_" + current_in_path in from_to_list:
            links_list.append(dep_list[from_to_list.index(next_in_path + "_" + current_in_path)])
        else:
            raise Exception
    return [str(x) for x in combinations(links_list, 4)]


def create_features(sent_data):
    features = dict()
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    ent1_token = ent_text_to_ent(ent1, sent_data)
    ent2_token = ent_text_to_ent(ent2, sent_data)
    # adding the label types
    features["ent1"] = ent1_token.label_
    features["ent2"] = ent2_token.label_
    features["ent1_ent2"] = ent1_token.label_ + "_" + ent2_token.label_
    # adding the entity words
    # features.append(ent1_token.vocab)
    # features.append(ent2_token.vocab)
    # adding the wordb4 and after
    word_b4, word_after = word_b4_first_ent_and_word_after_second_ent(sent_data)
    features["word_b4"] = word_b4
    features["word_after"] = word_after
    # adding bow between ents
    # features["bow"] = bow_between_ents(ent1_token, ent2_token, sent_data)
    # number of features between
    features["num_of_ents_between"] = ents_between(ent1_token, ent2_token, sent_data)
    # get path between ents
    features["path_lengh_in_tree"] = get_path_length_between_ents(ent1_token, ent2_token, sent_data)
    # pos combination on path in tree between ents
    for pattern in get_path_pos_combinations_between_ents(ent1_token, ent2_token, sent_data):
        features["dep_combination:" + pattern] = 1
    # add part of speech patterns between entities
    for pattern in get_patterns(sent_data):
        features[pattern] = 1
    # add dependency edges
    # for pattern in get_links_combinations_between_ents(ent1_token, ent2_token, sent_data):
    #     features["dependency_edge_pattern:" + pattern] = 1
    return features


def create_features_data(train_data_processed):
    if len(train_data_processed) > 0:
        if len(train_data_processed[0]) == 2:
            return [(create_features(x), y) for x, y in train_data_processed]
        else:
            return [create_features(x) for x in train_data_processed]

