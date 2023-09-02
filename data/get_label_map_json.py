import json

with open('label_test.json', 'r') as f: # compress label
    lb = json.load(f)
    labels = []
    for classI in range(20):
        curLabel = lb[classI*600]
        labels.append(curLabel)
    del lb[:]

with open('label_map_test.json', 'w') as f:
    json.dump(labels, f)