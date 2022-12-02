from img2vec_pytorch import Img2Vec
from PIL import Image
import os
import glob
import pickle
import numpy as np
import json

# Initialize Img2Vec with GPU
img2vec = Img2Vec(cuda=False)


def mini_imagenet_folders():
    train_folder = './val'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]

    return metatrain_folders

#generate name list and label list
append = '_val'
name_list = []
for folder in mini_imagenet_folders():
    name_list = name_list + os.listdir(folder)
print(len(name_list))
with open('./data/name_list'+append+'.json','w') as f:
    json.dump(name_list, f)

train_folder = './val'
label_list = []
for label in os.listdir(train_folder):
    if os.path.isdir(os.path.join(train_folder, label)):
        for i in range(600):
            label_list.append(label)
print(len(label_list))
with open('./data/label'+append+'.json','w') as f:
    json.dump(label_list, f)


vectors = []
i = 0
for folder in mini_imagenet_folders():
    image_list = []
    for f in glob.iglob(folder+"/*"):
        image_list.append(Image.open(f))
    vector = img2vec.get_vec(image_list)
    vectors.append(vector)
    print(i)
    i += 1
vectors = np.array(vectors)

vectors = np.reshape(vectors, (-1, 512))
print(vectors.shape)
with open('./data/embedding_new'+append+'.pkl',"wb") as f:
    pickle.dump(vectors, f)
'''
# Read in an image (rgb format)
img = Image.open('test.jpg')
# Get a vector from img2vec, returned as a torch FloatTensor
vec = img2vec.get_vec(img, tensor=True)
print(vec.shape)
# Or submit a list
#vectors = img2vec.get_vec(list_of_PIL_images)
'''