from tqdm import tqdm
import numpy as np
import collections
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score,recall_score,accuracy_score
from chapter6 import dataloader,pathSim
from data_set import filepaths as fp
from sklearn.decomposition import NMF
#NFM矩阵分解
def getNFM(m,n_dim):
    nmf = NMF(n_components=n_dim)
    user_vectors = nmf.fit_transform(m)
    item_vectors = nmf.components_
    return user_vectors,item_vectors.T

#将事实按照关系分开，代表不同的元路径
def splitParis(kgPairs,movie_set):
    print('split paris')
    r_Pairs={}
    for h,r,t in tqdm(kgPairs):
        if h in movie_set:
            h, r, t = int(h), int(r), int(t)
            if r not in r_Pairs:r_Pairs[r]=[]
            r_Pairs[r].append([h,1,t])
    return r_Pairs

#得到所有元路径下的实体邻接表
def getAdjacencyListOfAllRelations(r_pairs):
    print('得到所有元路径下的实体邻接表...')
    r_al={}
    for r in tqdm(r_pairs):
        r_al[r]=pathSim.getAdjacencyList(r_pairs[r])
    return r_al

#得到所有元路径下的电影实体相似度矩阵
def getSimMatrixOfAllRelations(r_al,movie_set):
    print('计算实体相似度矩阵...')
    r_simMatrix={}
    for r in tqdm(r_al):
        r_simMatrix[r]=\
            pathSim.getSimMatrixFromAl(r_al[r],max(movie_set)+1)
    return r_simMatrix

class PER(torch.nn.Module):

    def __init__(self, kgPairs,user_set,movie_set,ratingParis,n_dim=100):
        super(PER, self).__init__()
        r_Paris = splitParis(kgPairs,movie_set)
        r_al = getAdjacencyListOfAllRelations(r_Paris)
        self.sim_matrix=getSimMatrixOfAllRelations(r_al,movie_set)
        print("计算用户偏好扩散矩阵...")
        self.sortedUserItemSims,self.r_map=self.init_userItemSims(user_set,ratingParis)
        print('初始化用户物品在每个元路径下的embedding...')
        self.embeddings = self.init_embedding(n_dim)

        self.r_linear = torch.nn.Linear(len(self.r_map),1,bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,u,v):
        r_preds=[]
        for i in self.r_map:
            r_pred=torch.sum(self.embeddings[i]['user'](u) * self.embeddings[i]['item'](v),axis=1,keepdim=True)
            r_preds.append(r_pred)
        r_preds=torch.cat(r_preds,1)
        r_preds=self.r_linear(r_preds)
        logit = self.sigmoid(r_preds).squeeze(1)
        return logit

    def init_embedding( self, n_dim ):
        embeddings = collections.defaultdict( dict )
        for r in self.r_map:
            user_vectors, item_vectors = self.__init_one_pre_emd( self.sortedUserItemSims[self.r_map[r]], n_dim )
            embeddings[r]['user'] = torch.nn.Embedding.from_pretrained( user_vectors,max_norm=1 )
            embeddings[r]['item'] = torch.nn.Embedding.from_pretrained( item_vectors,max_norm=1 )
        return embeddings

    def __init_one_pre_emd( self, mat, n_dim):
        user_vectors,item_vectors = getNFM( mat, n_dim)
        return torch.FloatTensor(user_vectors),torch.FloatTensor( item_vectors )

    def init_userItemSims(self, user_set, ratingParis):
        userItemAl = pathSim.getAdjacencyList(ratingParis,r_col=2)

        userItemSims = collections.defaultdict(dict)
        for r in self.sim_matrix:
            for u in userItemAl:
                userItemSims[r][u] = \
                    np.sum(self.sim_matrix[r][[i for i in userItemAl[u] if userItemAl[u][i] ==1]],axis=0)

        userItemSimMatrixs = {}
        for r in tqdm(userItemSims):
            userItemSimMatrix = []
            for u in user_set:
                userItemSimMatrix.append(userItemSims[r][int(u)].tolist())
            userItemSimMatrixs[r] = np.mat(userItemSimMatrix)

        r_map = {k: v for k, v in enumerate(sorted([r for r in userItemSims]))}
        return userItemSimMatrixs,r_map

def doEva(net,d):
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net(u,i)
    y_pred = np.array([1 if i >= 0.6 else 0 for i in out])
    y_true = r.detach().numpy()
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc =accuracy_score(y_true,y_pred)
    return p,r,acc


def train(epochs=10,batchSize=2048,lr=0.01):
    all_entity_set, relation_set, KgPairs = dataloader.readKgData(fp.Ml_1M.KG)
    user_set, movie_set, trainRatingPairs, testRatingPairs = dataloader.readRecData(fp.Ml_1M.RATING, test_ratio=0.1)

    net = PER(KgPairs, user_set, movie_set,trainRatingPairs)
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=1e-3)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-3)
    criterion = torch.nn.BCELoss()

    for e in range(epochs):
        all_lose = 0
        for u,v,r in tqdm(DataLoader(trainRatingPairs, batch_size=batchSize, shuffle=True)):
            r = torch.FloatTensor(r.detach().numpy())
            optimizer.zero_grad()
            result = net(u, v)
            loss = criterion(result, r)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch{},avg_loss={}'.format(e, all_lose / (len(trainRatingPairs) // batchSize)))
        p, r, acc = doEva(net, testRatingPairs)
        print('p:{},r:{},acc:{}'.format(round(p, 3), round(r, 3), round(acc, 3)))


if __name__ == "__main__":
    train()






