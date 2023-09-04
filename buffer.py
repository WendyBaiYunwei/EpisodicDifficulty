import random
import numpy as np
import json
import pickle
import time

with open('./data/label_map.json', 'r') as f:
    idx_to_label_name = json.load(f)

with open('./data/centroid_by_class.json') as f:
    get_centroid_by_class = json.load(f)

with open('./data/embedding_sim.pkl', 'rb') as f:
    sorted_adj = pickle.load(f)

with open('./data/embedding_new.pkl', 'rb') as f:
    embeddings = pickle.load(f).astype(np.float16)

with open('./data/imgNameToIdx.json', 'r') as f:
    name_to_idx = json.load(f)

TTL_CLASSES = 64
def get_probs(counts):
    probabilities = []
    for class_i in range(TTL_CLASSES): 
        count = counts[class_i]
        probabilities.append(1/(count**2))

    normalizing_k = 1 / sum(probabilities)
    return normalizing_k * np.array(probabilities)

def get_support_set_info(class_size, shot_size, trail_num):
    counts = [1 for i in range(TTL_CLASSES)] # class balancer, in place to prevent oversampling from certain classes.
    monte_carlo_trails = []
    for i in range(trail_num):
        if i % 25000 == 0:
            print(i)
        idxes = [i for i in range(TTL_CLASSES)]
        probabilities = get_probs(counts)
        class_idxes = np.random.choice(idxes,\
            replace = False, size = class_size, p=probabilities) 
        for class_i in class_idxes:
            counts[class_i] += 1
        class_labels = [idx_to_label_name[idx] for idx in class_idxes]
        avg_embedding_list = []
        support_set_list = []
        for label_i, label in enumerate(class_labels):
            centroid_image_name = get_centroid_by_class[label]
            support_set = random.sample(sorted_adj[centroid_image_name], k=shot_size)
            avg_embedding = np.zeros(np.shape(embeddings)[1])
            for image_pair in support_set:
                _, image_name = image_pair
                class_i = class_idxes[label_i]
                idx = name_to_idx[image_name]
                embedding = embeddings[idx].reshape(avg_embedding.shape)
                avg_embedding += embedding
            avg_embedding /= shot_size
            avg_embedding_list.append(avg_embedding)
            support_set_list.append(support_set)
        total_sim = 0
        for i in range(class_size):
            for j in range(i + 1, class_size):
                total_sim += np.linalg.norm(avg_embedding_list[i] / np.linalg.norm(avg_embedding_list[i]) -
                                            avg_embedding_list[j] / np.linalg.norm(avg_embedding_list[j]))
        monte_carlo_trails.append([total_sim, class_idxes, support_set_list])

    monte_carlo_trails.sort(reverse=True)
    with open('./data/buffer.pkl','wb') as f:
        pickle.dump(monte_carlo_trails, f)


if __name__ == '__main__':
    start = time.time()
    get_support_set_info(5, 1, 150000) # 5 way 1 shot, 150000 episodes of support set in buffer
    end = time.time()
    print(end - start)
