from mxnet import nd,autograd,gluon
import mxnet as mx
from chapter5 import dataloader#自己刚建的读取数据的py文件
from data_set import filepaths#自己记录文件地址的py文件
from utils import mxnetUtils #自己的util库


class RESCAL(gluon.nn.Block):
    def __init__(self,n_entity, n_relation, dim=200, margin=1):
        super().__init__()
        self.margin=margin #(式5-14)中的 m
        self.n_entity=n_entity #实体的数量
        self.n_relation=n_relation #关系的数量
        self.dim = dim #embedding的长度
        # 随机初始化实体的embedding
        self.e = gluon.nn.Embedding(self.n_entity,dim)
        # 随机初始化关系矩阵的embedding
        self.r = gluon.nn.Embedding(self.n_relation,dim*dim)

    def batch_norm(self):
        for param in self.params:
            param=mxnetUtils.normlize(param)

    def net(self,X):
        x_correct,x_corrupt=X
        y_correct=self.predict(x_correct)
        y_corrupt=self.predict(x_corrupt)
        return self.__hinge_loss(y_correct,y_corrupt)

    def predict(self,x):
        h=self.e(x[:, 0])
        r=self.r(x[:, 1])
        t=self.e(x[:, 2])
        t=t.reshape(-1,self.dim,1)
        r=r.reshape(-1,self.dim,self.dim)
        tr=nd.batch_dot(r,t)
        tr=tr.reshape(-1,self.dim)
        score = nd.sum(h*tr,-1)
        return -score

    def __hinge_loss(self, dist_correct, dist_corrupt):
        a=dist_correct - dist_corrupt + self.margin
        return nd.maximum(a, 0)

def train(net,dataLoad,pairs,epochs=20,lr=0.01,batchSize=1024):
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    for e in range(epochs):
        l=0
        for X in dataLoad.iter(pairs,batchSize):
            with autograd.record():
                loss= net.net(X)
            loss.backward()
            trainer.step(batchSize)
            net.batch_norm()
            l+=sum(loss).asscalar()
        print("Epoch {}, average loss:{}".format(e,l/len(pairs)))

if __name__ == '__main__':
    entity, relationShips, pairs = dataloader.readData(filepaths.FB15K_237.TEST)
    net=RESCAL(len(entity),len(relationShips))
    net.collect_params().initialize(mx.init.Xavier())

    dataLoad = dataloader.DataIter(entity, relationShips)
    train(net,dataLoad,pairs)