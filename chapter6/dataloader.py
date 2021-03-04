import random
import copy
from mxnet import nd
from tqdm import tqdm #产生进度条的库
from utils import osUtils #自己的util库
from data_set import filepaths #自己记录文件地址的py文件

def readKgData(path):
    print('读取知识图谱三元组...')
    entity_set,relation_set=set(),set()
    pairs=[]
    for h, r, t in tqdm(osUtils.readTriple(path)):
        entity_set.add(int(h))
        entity_set.add(int(t))
        relation_set.add(int(r))
        pairs.append([int(h),int(r),int(t)])
    #返回实体集合列表，关系集合列表，与三元组列表
    return list(entity_set),list(relation_set),pairs

def readRecData(path,test_ratio=0.2):
    print('读取用户评分三元组...')
    user_set,item_set=set(),set()
    pairs=[]
    for u, i, r in tqdm(osUtils.readTriple(path)):
        user_set.add(int(u))
        item_set.add(int(i))
        pairs.append((int(u),int(i),int(r)))

    test_set=random.sample(pairs,int(len(pairs)*test_ratio))
    train_set=list(set(pairs)-set(test_set))

    #返回用户集合列表，物品集合列表，与用户，物品，评分三元组列表
    return list(user_set),list(item_set),train_set,test_set

def setForTopKevaluation(testSet):
    all_testItems = set()
    user_items=dict()
    for u,v,r in testSet:
        all_testItems.add(v)
        if u not in user_items:
            user_items[u]={
                'pos':set(),
                'neg':set()
            }
        if r=='1':
            user_items[u]['pos'].add(v)
        else:
            user_items[u]['neg'].add(v)
    return all_testItems,user_items

class DataIter():

    def __init__(self,entity_list,relation_list):
        self.entity_list=entity_list
        self.relation_list=relation_list

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

    def iter(self,recPairs,kgPairs,batchSize):
        #传入评分三元组，知识图谱三元组，batch size
        for i in range(len(recPairs)//batchSize):
            recDataSet = random.sample(recPairs,batchSize)
            kgDataset = random.sample(kgPairs, batchSize)
            kgDataset_corrupted_datasets = self.__corrupt(kgDataset)
            yield nd.array(recDataSet,dtype=int),nd.array(kgDataset,dtype=int),nd.array(kgDataset_corrupted_datasets,dtype=int)
            #每次迭代返回一批量的评分三元组，以及知识图谱正例采样与负例采样得到的array

if __name__ == '__main__':
    users, items, train_set, test_set = readRecData(filepaths.Ml_100K.RATING)
    all_testItems, topKtestSet = setForTopKevaluation(test_set)
    print(topKtestSet)
