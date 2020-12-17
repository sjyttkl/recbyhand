import random,copy
from mxnet import nd
from utils import osUtils
from tqdm import tqdm

def readData(path):
    entity_list,relation_list=set(),set()
    pairs=[]
    for h, r, t in tqdm(osUtils.readTriple(path)):
        entity_list.add(h)
        entity_list.add(t)
        relation_list.add(r)
        pairs.append([h,r,t])
    return list(entity_list),list(relation_list),pairs


class DataIter():

    def __init__(self,entity_list,relation_list):
        self.entity_list=entity_list
        self.relation_list=relation_list

    def __corrupt(self,datasets):
        corrupted_datasets = []
        for triple in datasets:
            corrupted_triple = copy.deepcopy(triple)
            seed = random.random()
            if seed > 0.5:# 替换head
                rand_head = triple[0]
                while rand_head == triple[0]:
                    rand_head = random.sample(self.entity_list, 1)[0]
                corrupted_triple[0] = rand_head
            else:# 替换tail
                rand_tail = triple[1]
                while rand_tail == triple[1]:
                    rand_tail = random.sample(self.entity_list, 1)[0]
                corrupted_triple[1] = rand_tail
            corrupted_datasets.append(corrupted_triple)
        return corrupted_datasets

    def iter(self,pairs,batchSize):
        for i in range(len(pairs)//batchSize):
            dataset = random.sample(pairs, batchSize)
            corrupted_datasets=self.__corrupt(dataset)
            yield nd.array(dataset),nd.array(corrupted_datasets)