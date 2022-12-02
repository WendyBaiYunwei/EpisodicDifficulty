from collections import defaultdict
adj = defaultdict(list)

for list in [[1, 2, 3],[-4, 5, 6]]:
    adj['100'].append(list)
    adj['100'].sort(key=lambda x:x[0])

print(adj)