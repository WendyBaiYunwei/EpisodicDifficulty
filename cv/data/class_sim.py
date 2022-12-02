# class sim
# get average embedding
    # prepare data and container
        # prepare label
    # fill the containor
        # get average embedding
        # label to average embedding
# get sim ratio of the average embeddings
    # prepare data and container
    # fill the containor

from scipy.spatial import distance
import json
import pickle
import numpy as np
from collections import defaultdict

with open('./data/embedding_new.pkl', 'rb') as f:
    allEmebddings = pickle.load(f)

with open('./data/label_list.json', 'rb') as f:
    allLabels = json.load(f)

with open('./data/label_map.json', 'r') as f:
    labelMap = json.load(f)

def getAvgEmbeddings(allEmebddings, allLabels):
    embeddings = np.zeros((64, 512))
    labelToEmbeddings = defaultdict(list)
    for i in range(len(allLabels)):
        oneEmb = allEmebddings[i]
        oneLab = allLabels[i]
        labelToEmbeddings[oneLab].append(oneEmb)

    for i in range(64):
        label = labelMap[i]
        embeddings = labelToEmbeddings[label]
        avgEmbedding = np.mean(embeddings)
        # print(avgEmbedding.shape)
        embeddings[i] = avgEmbedding
    return embeddings

embeddings = getAvgEmbeddings(allEmebddings, allLabels)
pairWiseSim = defaultdict(list) #(class1, class2) -> sim

for i in range(64):
    embedding1 = embeddings[i]
    class1 = labelMap[i]
    for j in range(64):
        if i < j:
            class2 = labelMap[j]
            embedding2 = embeddings[j]
            e1 = embedding1.flatten()
            e2 = embedding2.flatten()
            sim = 1 - distance.cosine(e1, e2)
            pairWiseSim[class1].append((sim, class2))
            pairWiseSim[class2].append((sim, class1))

for oneClass in pairWiseSim:
    pairWiseSim[oneClass].sort(reverse=True)

#with open('class_sim.json', 'r') as f:
#    pairWiseSim = json.load(f)

class_matrix = {}
for first_class in pairWiseSim:
    class_matrix[first_class] = {}

for first_class in pairWiseSim:
    for i in range(len(pairWiseSim[first_class])):
        sim, second_class = pairWiseSim[first_class][i]
        class_matrix[first_class][second_class] = sim
        class_matrix[second_class][first_class] = sim

with open('./class_sim_matrix_test.json', 'w') as f:
    json.dump(class_matrix, f)
