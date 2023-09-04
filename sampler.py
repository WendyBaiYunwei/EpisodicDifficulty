from config import get_config
import torch
from skimage import io
import random
import json
import torchvision.transforms as transforms
import numpy as np
import pickle
import os

# one shot sampler
class Sampler():  
    def __init__(self, difficulty_level, batch_size, shot_size, class_size):
        params = get_config()
        random.seed(params['seed'])
        np.random.seed(params['seed'])
        torch.manual_seed(params['seed'])
        torch.cuda.manual_seed_all(params['seed'])
        self.dir = params['data_dir']
        normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
        self.transform = transforms.Compose([transforms.ToTensor(),normalize])
        self.difficulty_level = difficulty_level
        self.batch_size = batch_size
        self.shot_size = shot_size
        self.class_size = class_size
        self.max_difficulty = params['max_difficulty']
        self.order_on = True
        self.OFFSET = 9
        with open(os.path.join(self.dir, 'embedding_new.pkl'), 'rb') as f:
            self.embeddings = pickle.load(f)
        with open(os.path.join(self.dir, 'imgNameToIdx.json'), 'r') as f:
            self.name_to_idx =  json.load(f)
        with open(os.path.join(self.dir, 'embedding_sim.pkl'), 'rb') as f:
            self.sorted_adj = pickle.load(f)
        with open(os.path.join(self.dir, 'centroid_by_class.json'), 'r') as f:
            self.get_img_by_class = json.load(f)
        with open(os.path.join(self.dir, 'label_map.json'), 'r') as f:
            self.idx_to_label_name = json.load(f)
        with open(os.path.join(self.dir, 'buffer.pkl'),'rb') as f:
            self.buffer = pickle.load(f)

    def get_img_by_root(self, root):
        query = self.transform(io.imread(root))
        return query

    def get_query_set(self, ss, labels):
        query_set = []
        query_labels = []
        mix = []
        for i in range(len(ss)):
            oneshot = ss[i]
            if self.order_on: # one pass scheduler
                start_i = int(len(self.sorted_adj[oneshot]) * ((self.difficulty_level - 1)/self.max_difficulty))
                end_i = int(len(self.sorted_adj[oneshot]) * (self.difficulty_level/self.max_difficulty))
            else:
                start_i = 0
                end_i = len(self.sorted_adj[oneshot])
            neighbours = self.sorted_adj[oneshot][start_i:end_i]
            neighbours = [nb[1] for nb in neighbours]
            names = random.sample(neighbours, k = self.batch_size)

            for name in names:
                folder = name[:self.OFFSET]
                name = os.path.join(self.dir, 'train', folder, name)
                query = self.get_img_by_root(name)
                mix.append((query, labels[i], name))

        random.shuffle(mix)
        query_set = torch.stack([img_label[0] for img_label in mix])
        query_labels = torch.stack([img_label[1] for img_label in mix])
        names = [img_label[2] for img_label in mix]
        return query_set, query_labels, names

    def get_support_set_info(self, class_size, shot_size):
        buffer_i_start = int(len(self.buffer) * ((self.difficulty_level - 1) / self.max_difficulty)) - 1
        buffer_i_end = int(len(self.buffer) * (self.difficulty_level / self.max_difficulty)) - 1
        buffer_i = random.randint(buffer_i_start, buffer_i_end)
        if self.order_on:
            _, class_idxes, support_set_list = self.buffer[buffer_i]
        else:
            _, class_idxes, support_set_list = random.sample(self.buffer, k=1)[0]

        class_labels = [self.idx_to_label_name[idx] for idx in class_idxes]

        ss_names = []
        ss_imgs = []
        ss_labels = []
        for i in range(class_size):
            label = class_labels[i]
            for support in support_set_list[i]:
                name = support[1]
                ss_names.append(name)
                folder = name[:self.OFFSET]
                name = os.path.join(self.dir, 'train', folder, name)
                img = self.get_img_by_root(name)
                ss_imgs.append(img)
                idx = self.idx_to_label_name.index(label)
                torch_idx = torch.tensor(idx, dtype=torch.long)
                ss_labels.append(torch_idx)
        return torch.stack(ss_imgs), torch.stack(ss_labels), ss_names

    def get_batch(self):
        support_set, support_set_labels, ss_roots = self.\
            get_support_set_info(self.class_size, self.shot_size)
        query_set, query_set_label, query_names = self.get_query_set(ss_roots, support_set_labels)
        q_labels = []
        
        for ql in query_set_label:
            for i, ss_label in enumerate(support_set_labels):
                if ss_label == ql:
                    q_labels.append(torch.tensor(i // self.shot_size, dtype=torch.long))
                    break

        q_labels = torch.stack(q_labels)
        return support_set, query_set, q_labels, ss_roots, query_names