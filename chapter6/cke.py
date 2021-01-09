from mxnet import nd,autograd,gluon
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
from utils import evaluate #自己的测试库
from chapter6 import dataloader #自己建立的读取数据的py文件
from data_set import filepaths as fp #自己记录文件地址的py文件

class CKE(nn.Block):
    def __init__(self, n_user, n_entity, n_relation, e_dim=100, margin=1):
        super().__init__()
        self.margin=margin
        self.u_emb = nn.Embedding(n_user, e_dim) #用户向量
        self.e_emb = nn.Embedding(n_entity, e_dim) #实体向量
        self.r_emb = nn.Embedding(n_relation, e_dim) #关系向量

        self.wr = nn.Embedding(n_relation,e_dim)# 超平面法向量
        self.BCEloss = gluon.loss.SigmoidBCELoss(from_sigmoid=True)

    def __Htransfer(self, e, wr):
        norm_wr = wr/wr.norm(axis=1,keepdims=True)
        return e - nd.sum(e * norm_wr, 1, True) * norm_wr

    def __hinge_loss(self, dist_correct, dist_corrupt):
        a = dist_correct - dist_corrupt + self.margin
        return nd.maximum(a, 0)

    def kg_predict(self,x):
        r_index=x[:, 1]
        h = self.e_emb(x[:, 0])
        r = self.r_emb(r_index)
        t = self.e_emb(x[:, 2])
        wr = self.wr(r_index)
        score = self.__Htransfer(h,wr) + r - self.__Htransfer(t,wr)
        return nd.sum(score**2,axis=1,keepdims=True)**0.5

    def rec_predict(self,x):
        u = self.u_emb(x[:,0])
        i = self.e_emb(x[:,1])
        y=nd.sigmoid(nd.sum(u*i,axis=1))
        return y

    def net(self,X):
        x_rec,x_correct,x_corrupt=X
        y_ture = nd.array(x_rec[:, 2], dtype=np.float32)
        y_pred=self.rec_predict(x_rec)
        a=self.BCEloss(y_pred, y_ture)
        rec_loss = sum(a)
        y_correct=self.kg_predict(x_correct)
        y_corrupt=self.kg_predict(x_corrupt)
        kg_loss = sum(self.__hinge_loss(y_correct,y_corrupt))
        return rec_loss+kg_loss

#预测
def doEvaluation(net,testSet):
    pred = net.rec_predict(nd.array(testSet))
    y_true = [int(t[2]) for t in testSet]
    predictions = [1 if i >= 0.5 else 0 for i in pred]
    p = evaluate.precision(y_true=y_true, y_pred=predictions)
    r = evaluate.recall(y_true=y_true, y_pred=predictions)
    acc = evaluate.accuracy_score(y_true,y_pred=predictions)
    return p,r,acc

def train(net,dataLoad,recPairs,kgPairs,testSet,epochs=5,lr=0.01,batchSize=1024):
    from tqdm import tqdm
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    for e in range(epochs):
        l=0
        for X in tqdm(dataLoad.iter(recPairs,kgPairs,batchSize)):
            with autograd.record():
                loss = net.net(X)
            loss.backward()
            trainer.step(batchSize)
            l+=sum(loss).asscalar()
        print("Epoch {}, average loss:{}".format(e,round(l/len(recPairs),3)))
        p,r,acc=doEvaluation(net,testSet)
        print("p:{},r:{},acc:{}".format(round(p,3),round(r,3),round(acc,3)))

if __name__ == '__main__':
    #加载数据
    entitys, relationShips, kgPairs = dataloader.readKgData(fp.Ml_100K.KG)
    users, items, train_set,test_set = dataloader.readRecData(fp.Ml_100K.RATING)
    #初始化模型
    net = CKE(len(users),len(entitys), len(relationShips))
    net.collect_params().initialize(mx.init.Xavier())
    # 构建迭代器
    dataLoad = dataloader.DataIter(entitys, relationShips)
    #训练模型
    train(net, dataLoad, train_set, kgPairs, test_set)