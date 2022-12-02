import copy
import torch
import torch.utils.data
from utils import padding
import pickle

class OrderedTrainDataLoader:
    def __init__(self, loaders, sorted_adj, support, query, pad_idx):
        # original
        self.loaders = loaders
        self.filenames = sorted(loaders.keys())
        self.loaders_ins = self.instantiate_all(loaders)
        # current indices
        self.index = -1
        self.indices = self.reset_indices(loaders)
        # max indices
        self.max_indices = self.get_batch_cnt(loaders)
        # arg
        self.support = support
        self.query = query
        self.pad_idx = pad_idx

        self.sorted_adj = sorted_adj

    def __len__(self):
        return len(self.loaders)

    def instantiate_one(self, loader):
        return list(copy.deepcopy(loader))

    def instantiate_all(self, loader):
        new_loader = {}
        for filename in loader:
            new_loader[filename] = {}
            for value in loader[filename]:
                new_loader[filename][value] = self.instantiate_one(loader[filename][value])
        return new_loader

    def reset_indices(self, loader):
        indices = {}
        for filename in loader:
            indices[filename] = {}
            for value in loader[filename]:
                indices[filename][value] = 0
        return indices

    def get_batch_cnt(self, loader):
        batch_cnt = {}
        for filename in loader:
            batch_cnt[filename] = {}
            for value in loader[filename]:
                batch_cnt[filename][value] = len(loader[filename][value])
        return batch_cnt

    def get_batch_idx(self, filename, value):
        if self.indices[filename][value] >= self.max_indices[filename][value]:
            self.loaders_ins[filename][value] = self.instantiate_one(self.loaders[filename][value])
            self.indices[filename][value] = 0

        return self.indices[filename][value]

    def get_filename(self):
        self.index = (self.index + 1) % len(self)
        return self.filenames[self.index]  

    def get_query_by_ss(self, support, type, difficulty_level, filename):
        # assume size-1 support
        support = str(support)
        support_tokens = support.replace(' ', '').replace('\n', '').split(',')
        support_len = len(support_tokens)
        remaining = []
        for token in support_tokens:
            if token != '0':
                remaining.append(token)
            else:
                break
        list_str = ', '.join(remaining)
        list_str = list_str[len('tensor(['):] + ']'
        all_nbs = self.sorted_adj[filename][type][list_str]
        if difficulty_level < len(all_nbs):
            queries = all_nbs[difficulty_level] # from similar to different
        else:
            if len(all_nbs) == 0:
                return None
            queries = all_nbs[-1]
        front = torch.LongTensor(queries[1:])
        padding = torch.zeros(support_len - len(queries) + 1, dtype=torch.long)
        queries = torch.cat([front, padding]).reshape(1, -1)
        return queries

    def combine_batch(self, neg_data, neg_target, pos_data, pos_target, difficulty_level, filename):
        neg_data, pos_data = padding(neg_data, pos_data, pad_idx=self.pad_idx)
        # combine support data and query data
        support_data = torch.cat([neg_data[0:self.support], pos_data[0:self.support]], dim=0)

        # print('neg_data', neg_data, self.support)
        neg_query = self.get_query_by_ss(neg_data[0:self.support], 'neg', difficulty_level, filename)
        pos_query = self.get_query_by_ss(pos_data[0:self.support], 'pos', difficulty_level, filename)
        if neg_query == None or pos_query == None:
            return None, None
        query_data = torch.cat([neg_query, pos_query], dim=0)
        data = torch.cat([support_data, query_data], dim=0)
        # combine support target and query target
        support_target = torch.cat([neg_target[0:self.support], pos_target[0:self.support]], dim=0)
        query_target = torch.cat([neg_target[self.support:], pos_target[self.support:]], dim=0)
        target = torch.cat([support_target, query_target], dim=0)
        return data, target

    def get_batch(self, difficulty_level):
        filename = self.get_filename()
        neg_idx = self.get_batch_idx(filename, 'neg')
        pos_idx = self.get_batch_idx(filename, 'pos')
        neg_data, neg_target = self.loaders_ins[filename]['neg'][neg_idx]
        pos_data, pos_target = self.loaders_ins[filename]['pos'][pos_idx]

        self.indices[filename]['neg'] += 1
        self.indices[filename]['pos'] += 1
        # incomplete batch
        if min(len(neg_data), len(pos_data)) < self.support + self.query: 
            return self.get_batch(difficulty_level)
        data, target = self.combine_batch(neg_data, neg_target, pos_data, pos_target, difficulty_level, filename)
        if data == None:
            return self.get_batch(difficulty_level)
        return data, target
