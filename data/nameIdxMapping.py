# 2 mappers
# get image names and index
# fill the two mappers
# save

from collections import defaultdict
import json

imgNameToIdx = defaultdict(list)
idxToImgname = defaultdict(list)

def fillMapper(mapper, keys, values):
    for i in range(len(keys)):
        mapper[keys[i]].append(values[i])
    return mapper

with open('./name_list_test.json', 'rb') as f:
    allNames = json.load(f)

indexs =  [i for i in range(len(allNames))]
imgNameToIdx = fillMapper(imgNameToIdx, allNames, indexs)
idxToImgname = fillMapper(idxToImgname, indexs, allNames)

with open('imgNameToIdx_test.json', 'w') as f:
    json.dump(imgNameToIdx, f)

with open('idxToImgname_test.json', 'w') as f:
    json.dump(idxToImgname, f)