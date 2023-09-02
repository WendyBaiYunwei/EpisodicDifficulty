# load
# arrange imgs and positions according to labels
# for each class
# calc the centroid point
# for each point in the class
# get dist from the centroid
# save the img root of the nearest
# save the dict
import json
import pickle
from collections import defaultdict

def arrange(imgs, labels):
    res = {}
    for i, label in enumerate(labels):
        res[label] = imgs[i]
    return res

if __name__ == '__main__':
    with open('name_list_val.json', 'r') as f:
        imgs = json.load(f)
    with open('label_val.json', 'r') as f:
        labels = json.load(f)

    result = arrange(imgs, labels)
    with open('centroid_by_class_val.json', 'w') as outfile:
        json.dump(result, outfile)