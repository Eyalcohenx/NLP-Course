# Yehoshua Stern 314963927
# Eyal Cohen 207947086

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from create_features import create_features_data, ent_text_to_ent, ents_between


def words_between_ents(sent_data):
    sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
    words_in_sent = [x.text for x in sent_tokens]
    ent1_token = ent_text_to_ent(ent1, sent_data)
    ent2_token = ent_text_to_ent(ent2, sent_data)
    words_between = []

    if ent1_token.start > ent2_token.start:
        return words_between

    for i in range(ent1_token.end, ent2_token.start):
        words_between.append(words_in_sent[i])

    return words_between


class RelationExtractionModel:
    def __init__(self, train_data_processed, places_ents, persons_ents, rule0=False, rule1=False, rule2=False):
        self.vec = DictVectorizer()
        train_data_featured = create_features_data(train_data_processed)
        X_data = [x for x, y in train_data_featured]
        self.vec.fit(X_data)
        X_data = self.vec.transform(X_data)
        self.lr = LogisticRegression(max_iter=300, solver="lbfgs", multi_class="auto", class_weight={1: 0.8, 0: 0.2})
        self.lr.fit(X_data, [y for x, y in train_data_featured])
        self.places_ents = places_ents
        self.persons_ents = persons_ents
        self.rule0 = rule0
        self.rule1 = rule1
        self.rule2 = rule2

    def predict(self, train_data_filtered_xs):
        train_data_filtered_featured_xs = create_features_data(train_data_filtered_xs)
        lr_output = self.lr.predict(self.vec.transform(train_data_filtered_featured_xs))
        if self.rule0:
            for i, sent_data in enumerate(train_data_filtered_xs):
                if not self.rule0_func(sent_data):
                    lr_output[i] = 0
        if self.rule1:
            for i, sent_data in enumerate(train_data_filtered_xs):
                if not self.rule1_func(sent_data):
                    lr_output[i] = 0
        if self.rule2:
            for i, sent_data in enumerate(train_data_filtered_xs):
                if not self.rule2_func(sent_data):
                    lr_output[i] = 0
        return lr_output

    # rule to check that the first entity in the person ents and the second entity in the places ents
    def rule0_func(self, sent_data):
        sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
        ent1_token = ent_text_to_ent(ent1, sent_data)
        ent2_token = ent_text_to_ent(ent2, sent_data)
        if ent1_token.label_ in self.persons_ents and ent2_token.label_ in self.places_ents:
            return True
        return False

    # this rule is if we have no ents between the two given ents and the first one is place and the second one is person
    # and they have 's in the middle it means the person live in the place
    def rule1_func(self, sent_data):
        sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
        ent1_token = ent_text_to_ent(ent1, sent_data)
        ent2_token = ent_text_to_ent(ent2, sent_data)
        if ent1_token.label_ in self.places_ents and ent2_token.label_ in self.persons_ents:
            if "'s" in words_between_ents(sent_data) and ents_between(ent1_token, ent2_token, sent_data) == 0:
                return True
        return False

    # this rule is checking if the words "live in" appear between the entities and if so we return true
    def rule2_func(self, sent_data):
        sent_id, sent_tokens, sent_text, noun_chunks, named_entities, ent1, ent2 = sent_data
        ent1_token = ent_text_to_ent(ent1, sent_data)
        ent2_token = ent_text_to_ent(ent2, sent_data)
        if ent1_token.label_ in self.persons_ents and ent2_token.label_ in self.places_ents:
            if "live" in words_between_ents(sent_data) and "in" in words_between_ents(sent_data) and ents_between(
                    ent1_token, ent2_token, sent_data) == 0:
                return True
        return False
