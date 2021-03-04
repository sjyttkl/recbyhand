from mxnet import nd,autograd,gluon
import mxnet as mx
from chapter5 import dataloader#自己刚建的读取数据的py文件
from data_set import filepaths#自己记录文件地址的py文件
from utils import mxnetUtils #自己的util库

class TransE(gluon.nn.Block):
    def __init__(self,n_entity, n_relation, embedding_dim=200, margin=1):
        super().__init__()
        self.margin=margin #(式5-5)中的 m
        self.n_entity=n_entity #实体的数量
        self.n_relation=n_relation #关系的数量
        self.embedding_dim = embedding_dim #embedding的长度

        # 随机初始化实体的embedding
        self.e = gluon.nn.Embedding(self.n_entity,embedding_dim)
        # 随机初始化关系的embedding
        self.r = gluon.nn.Embedding(self.n_relation,embedding_dim)

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
        score= h + r - t
        return nd.sum(score**2,axis=1,keepdims=True)**0.5

    def __hinge_loss(self, dist_correct, dist_corrupt):
        a=dist_correct - dist_corrupt + self.margin
        return nd.maximum(a, 0)


class TransH(gluon.nn.Block):
    def __init__(self,n_entity, n_relation, embedding_dim=200, margin=1):
        super().__init__()
        self.margin=margin
        self.n_entity=n_entity
        self.n_relation=n_relation
        self.embedding_dim = embedding_dim
        self.e = gluon.nn.Embedding(self.n_entity,embedding_dim)
        self.r = gluon.nn.Embedding(self.n_relation,embedding_dim)

        # 随机初始化法向量的embedding
        self.wr = gluon.nn.Embedding(self.n_relation,embedding_dim)


    def batch_norm(self):
        for param in self.params:
            param=mxnetUtils.normlize(param)

    def __Htransfer(self, e, wr):
        norm_wr = wr/wr.norm(axis=1,keepdims=True)
        return e - nd.sum(e * norm_wr, 1, True) * norm_wr

    def __hinge_loss(self, dist_correct, dist_corrupt):
        a=dist_correct - dist_corrupt + self.margin
        return nd.maximum(a, 0)

    def predict(self,x):
        r_index=x[:, 1]
        h = self.e(x[:, 0])
        r = self.r(r_index)
        t = self.e(x[:, 2])
        wr = self.wr(r_index)
        score = self.__Htransfer(h,wr) + r - self.__Htransfer(t,wr)
        return nd.sum(score**2,axis=1,keepdims=True)**0.5

    def net(self,X):
        x_correct,x_corrupt=X
        y_correct=self.predict(x_correct)
        y_corrupt=self.predict(x_corrupt)
        return self.__hinge_loss(y_correct,y_corrupt)


class TransR(gluon.nn.Block):
    def __init__(self,n_entity, n_relation, k_dim=200,r_dim=100, margin=1):
        super().__init__()
        self.margin=margin
        self.n_entity=n_entity
        self.n_relation=n_relation
        self.k_dim=k_dim
        self.r_dim=r_dim
        self.e = gluon.nn.Embedding(self.n_entity,k_dim)
        self.r = gluon.nn.Embedding(self.n_relation,r_dim)
        self.wr = gluon.nn.Embedding(self.n_relation,k_dim*r_dim)

    def batch_norm(self):
        for param in self.params:
            param=mxnetUtils.normlize(param)

    def __Rtransfer(self, e, wr):
        e=e.reshape(-1,1,self.k_dim)
        twr=wr.reshape(-1,self.k_dim,self.r_dim)
        result=nd.batch_dot(e,twr)
        result=result.reshape(-1,self.r_dim)
        return result

    def __hinge_loss(self, dist_correct, dist_corrupt):
        a=dist_correct - dist_corrupt + self.margin
        return nd.maximum(a, 0)

    def predict(self,x):
        r_index=x[:, 1]
        h = self.e(x[:, 0])
        r = self.r(r_index)
        t = self.e(x[:, 2])
        wr = self.wr(r_index)
        score = self.__Rtransfer(h,wr) + r - self.__Rtransfer(t,wr)
        return nd.sum(score**2,axis=1,keepdims=True)**0.5

    def net(self,X):
        x_correct,x_corrupt=X
        y_correct=self.predict(x_correct)
        y_corrupt=self.predict(x_corrupt)
        return self.__hinge_loss(y_correct,y_corrupt)


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
    return net

if __name__ == '__main__':
    entity, relationShips, pairs = dataloader.readData(filepaths.FB15K_237.TEST)
    net=TransH(len(entity),len(relationShips))
    net.collect_params().initialize(mx.init.Xavier())
    dataLoad = dataloader.DataIter(entity, relationShips)
    net= train(net,dataLoad,pairs,epochs=20)


