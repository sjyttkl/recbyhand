import random
import copy
from mxnet import nd
from tqdm import tqdm #产生进度条的库
from utils import osUtils #自己的util库
from data_set import filepaths #自己记录文件地址的py文件

def readData(path):
    entity_list,relation_list=set(),set()
    pairs=[]
    for h, r, t in tqdm(osUtils.readTriple(path)):
        entity_list.add(h)
        entity_list.add(t)
        relation_list.add(r)
        pairs.append([h,r,t])
    #返回实体集合列表，关系集合列表，与三元组列表
    return list(entity_list),list(relation_list),pairs

class DataIter():

    def __init__(self,entity_list,relation_list):
        self.entity_list=entity_list
        self.relation_list=relation_list

    def iter(self,pairs,batchSize):
        #传入三元组与batch size
        for i in range(len(pairs)//batchSize):
            dataset = random.sample(pairs, batchSize)
            corrupted_datasets=self.__corrupt(dataset)
            yield nd.array(dataset),nd.array(corrupted_datasets)
            #每次迭代返回正例采样与负例采样得到的array

    def __corrupt(self,datasets):#负例采样方法
        corrupted_datasets = []
        for triple in datasets:
            corrupted_triple = copy.deepcopy(triple)
            seed = random.random()
            if seed > 0.5:#替换head
                rand_head = triple[0]
                while rand_head == triple[0]:#如果采样得到自己则继续循环
                    rand_head = random.sample(self.entity_list, 1)[0]
                corrupted_triple[0] = rand_head
            else:#替换tail
                rand_tail = triple[1]
                while rand_tail == triple[1]:
                    rand_tail = random.sample(self.entity_list, 1)[0]
                corrupted_triple[1] = rand_tail
            corrupted_datasets.append(corrupted_triple)
        return corrupted_datasets

if __name__ == '__main__':
    entity, relationShips, pairs = readData(filepaths.FB15K_237.TEST)
    print(pairs[:5])

    dataLoader = DataIter(entity,relationShips)
    for datas in dataLoader.iter(pairs,batchSize=2):
        print(datas)
        import sys
        sys.exit()