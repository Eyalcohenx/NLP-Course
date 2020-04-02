import numpy as np
from numpy import dot
from numpy.linalg import norm
from bisect import insort

from contextlib import contextmanager
from collections import defaultdict
from time import time

from vecs_asarray import bow5_vecs, deps_vecs

times_dict = defaultdict(float)


@contextmanager
def timethis(label):
    t0 = time()
    yield
    times_dict[label] += time() - t0


bow5_words_file = "bow5.words"
deps_words_file = "deps.words"
bow5_contexts_file = "bow5.contexts"
deps_contexts_file = "deps.contexts"

# creating lists to save data in them
words = ["car", "bus", "hospital", "hotel", "gun", "bomb", "horse", "fox", "table", "bowl", "guitar", "piano"]

similarities_car_bow5_words = [(0.0, "$JUNK$")]
similarities_bus_bow5_words = [(0.0, "$JUNK$")]
similarities_hospital_bow5_words = [(0.0, "$JUNK$")]
similarities_hotel_bow5_words = [(0.0, "$JUNK$")]
similarities_gun_bow5_words = [(0.0, "$JUNK$")]
similarities_bomb_bow5_words = [(0.0, "$JUNK$")]
similarities_horse_bow5_words = [(0.0, "$JUNK$")]
similarities_fox_bow5_words = [(0.0, "$JUNK$")]
similarities_table_bow5_words = [(0.0, "$JUNK$")]
similarities_bowl_bow5_words = [(0.0, "$JUNK$")]
similarities_guitar_bow5_words = [(0.0, "$JUNK$")]
similarities_piano_bow5_words = [(0.0, "$JUNK$")]

similarities_bow5_words = [similarities_car_bow5_words, similarities_bus_bow5_words, similarities_hospital_bow5_words,
                           similarities_hotel_bow5_words, similarities_gun_bow5_words,
                           similarities_bomb_bow5_words, similarities_horse_bow5_words, similarities_fox_bow5_words,
                           similarities_table_bow5_words, similarities_bowl_bow5_words,
                           similarities_guitar_bow5_words, similarities_piano_bow5_words]


# cosine similatiry function, using the numpy library
def cos(x, y):
    return dot(x, y) / (norm(x) * norm(y))


# reading the file line by line and timing it, saving the list sorted by using insort.
with timethis("bow5 words"):
    with open(bow5_words_file, encoding="utf8") as f:
        counter = 0
        for line in f:
            splitted = line.split()
            vec = np.asarray(splitted[1:]).astype(np.float)
            for i in range(12):
                if splitted[0] != words[i]:
                    insort(similarities_bow5_words[i], (cos(bow5_vecs[i], vec), splitted[0]))

similarities_car_deps_words = [(0.0, "$JUNK$")]
similarities_bus_deps_words = [(0.0, "$JUNK$")]
similarities_hospital_deps_words = [(0.0, "$JUNK$")]
similarities_hotel_deps_words = [(0.0, "$JUNK$")]
similarities_gun_deps_words = [(0.0, "$JUNK$")]
similarities_bomb_deps_words = [(0.0, "$JUNK$")]
similarities_horse_deps_words = [(0.0, "$JUNK$")]
similarities_fox_deps_words = [(0.0, "$JUNK$")]
similarities_table_deps_words = [(0.0, "$JUNK$")]
similarities_bowl_deps_words = [(0.0, "$JUNK$")]
similarities_guitar_deps_words = [(0.0, "$JUNK$")]
similarities_piano_deps_words = [(0.0, "$JUNK$")]

similarities_deps_words = [similarities_car_deps_words, similarities_bus_deps_words, similarities_hospital_deps_words,
                           similarities_hotel_deps_words, similarities_gun_deps_words,
                           similarities_bomb_deps_words, similarities_horse_deps_words, similarities_fox_deps_words,
                           similarities_table_deps_words,
                           similarities_bowl_deps_words,
                           similarities_guitar_deps_words, similarities_piano_deps_words]

print("\n")
# the code pretty much the same from here, we just do the same thing 4 times the only difference is the dot product
# when we do contexts
with timethis("deps words"):
    with open(deps_words_file, encoding="utf8") as f:
        counter = 0
        for line in f:
            splitted = line.split()
            vec = np.asarray(splitted[1:]).astype(np.float)
            for i in range(12):
                if splitted[0] != words[i]:
                    insort(similarities_deps_words[i], (cos(deps_vecs[i], vec), splitted[0]))

for i in range(12):
    print(words[i])
    temp1 = similarities_deps_words[i][-20:]
    temp1.sort(reverse=True)
    temp1 = [x[1] for x in temp1]
    temp2 = similarities_bow5_words[i][-20:]
    temp2.sort(reverse=True)
    temp2 = [x[1] for x in temp2]
    for j in range(20):
        print(str(temp2[j]) + ", " + str(temp1[j]))
    print("*********")

similarities_car_bow5_words = [(0.0, "$JUNK$")]
similarities_bus_bow5_words = [(0.0, "$JUNK$")]
similarities_hospital_bow5_words = [(0.0, "$JUNK$")]
similarities_hotel_bow5_words = [(0.0, "$JUNK$")]
similarities_gun_bow5_words = [(0.0, "$JUNK$")]
similarities_bomb_bow5_words = [(0.0, "$JUNK$")]
similarities_horse_bow5_words = [(0.0, "$JUNK$")]
similarities_fox_bow5_words = [(0.0, "$JUNK$")]
similarities_table_bow5_words = [(0.0, "$JUNK$")]
similarities_bowl_bow5_words = [(0.0, "$JUNK$")]
similarities_guitar_bow5_words = [(0.0, "$JUNK$")]
similarities_piano_bow5_words = [(0.0, "$JUNK$")]

similarities_bow5_words = [similarities_car_bow5_words, similarities_bus_bow5_words, similarities_hospital_bow5_words,
                           similarities_hotel_bow5_words, similarities_gun_bow5_words,
                           similarities_bomb_bow5_words, similarities_horse_bow5_words, similarities_fox_bow5_words,
                           similarities_table_bow5_words, similarities_bowl_bow5_words,
                           similarities_guitar_bow5_words, similarities_piano_bow5_words]

print("\n")

with timethis("bow5 contexts"):
    with open(bow5_contexts_file, encoding="utf8") as f:
        counter = 0
        for line in f:
            splitted = line.split()
            vec = np.asarray(splitted[1:]).astype(np.float)
            for i in range(12):
                if splitted[0] != words[i]:
                    insort(similarities_bow5_words[i], (np.dot(bow5_vecs[i], vec), splitted[0]))

similarities_car_deps_words = [(0.0, "$JUNK$")]
similarities_bus_deps_words = [(0.0, "$JUNK$")]
similarities_hospital_deps_words = [(0.0, "$JUNK$")]
similarities_hotel_deps_words = [(0.0, "$JUNK$")]
similarities_gun_deps_words = [(0.0, "$JUNK$")]
similarities_bomb_deps_words = [(0.0, "$JUNK$")]
similarities_horse_deps_words = [(0.0, "$JUNK$")]
similarities_fox_deps_words = [(0.0, "$JUNK$")]
similarities_table_deps_words = [(0.0, "$JUNK$")]
similarities_bowl_deps_words = [(0.0, "$JUNK$")]
similarities_guitar_deps_words = [(0.0, "$JUNK$")]
similarities_piano_deps_words = [(0.0, "$JUNK$")]

similarities_deps_words = [similarities_car_deps_words, similarities_bus_deps_words, similarities_hospital_deps_words,
                           similarities_hotel_deps_words, similarities_gun_deps_words,
                           similarities_bomb_deps_words, similarities_horse_deps_words, similarities_fox_deps_words,
                           similarities_table_deps_words,
                           similarities_bowl_deps_words,
                           similarities_guitar_deps_words, similarities_piano_deps_words]

print("\n")

with timethis("deps contexts"):
    with open(deps_contexts_file, encoding="utf8") as f:
        counter = 0
        for line in f:
            splitted = line.split()
            vec = np.asarray(splitted[1:]).astype(np.float)
            for i in range(12):
                if splitted[0] != words[i]:
                    insort(similarities_deps_words[i], (np.dot(deps_vecs[i], vec), splitted[0]))

for i in range(12):
    print(words[i])
    temp1 = similarities_deps_words[i][-20:]
    temp1.sort(reverse=True)
    temp1 = [x[1] for x in temp1]
    temp2 = similarities_bow5_words[i][-20:]
    temp2.sort(reverse=True)
    temp2 = [x[1] for x in temp2]
    for j in range(20):
        print(str(temp2[j]) + ", " + str(temp1[j]))
    print("*********")
