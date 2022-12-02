# read the data
# arrange the data
    # container
    # fill container
        # key: considered datapoint
        # values: neighbours of the same class, sorted based on sim
            # get pairwise sim
# create json

from skimage.transform import resize
from skimage import io
import json
import pickle
from torchvision import models
import torch
import torch.nn as nn
import numpy as np

    # pic = resize(pic, (224, 224))
    # pic = np.swapaxes(pic, 0, 2)
    # pic = np.swapaxes(pic, 1, 2)
    # pic = np.expand_dims(pic, axis = 0)
    # pic = torch.from_numpy(pic)
    # return enc(pic.float().cuda())

def getSim(img1, img2):
    # img1 = getEmbedding(img1)
    # img2 = getEmbedding(img2)
    return np.mean((img1 - img2) ** 2)

def arrange(imgs, labels, names):
    result = {}
    for i in range(len(imgs)):
        if i % 100 == 0:
            print(i)
        sims = []
        for j in range(len(imgs)):
            if labels[i] == labels[j]:
                sim = getSim(imgs[i], imgs[j])
                sims.append((sim, names[j]))
        sims.sort(key=lambda x:x[0])
        result[names[i]] = sims
    return result

if __name__ == '__main__':
    with open('label_test.json', 'r') as f:
        labels = json.load(f)
    with open('name_list_test.json', 'r') as f:
        names = json.load(f)
    with open('embedding_new_test.pkl', 'rb') as f:
        imgs = pickle.load(f)

    result = arrange(imgs, labels, names)
    with open('embedding_sim_test.pkl', 'wb') as outfile:
        pickle.dump(result, outfile)