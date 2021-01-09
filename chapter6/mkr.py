from mxnet import nd,autograd,gluon
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
from utils import evaluate #自己的测试库
from chapter6 import dataloader #自己建立的读取数据的py文件
from data_set import filepaths as fp #自己记录文件地址的py文件

class CrossCompress(nn.Block):

    def __init__(self,e_dim):
        super(CrossCompress, self).__init__()
        self.e_dim=e_dim
        self.weight_vv = nd.random_normal(shape=(self.e_dim, 1))
        self.weight_ev = nd.random_normal(shape=(self.e_dim, 1))
        self.weight_ve = nd.random_normal(shape=(self.e_dim, 1))
        self.weight_ee = nd.random_normal(shape=(self.e_dim, 1))
        self.bias_v = nd.zeros(self.e_dim)
        self.bias_e = nd.zeros(self.e_dim)
        params = [self.weight_vv, self.weight_ev, self.weight_ve, self.weight_ee, self.bias_v, self.bias_e]
        for param in params:
            param.attach_grad()

    def forward(self, v, e):
        # [batch_size, dim]
        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = v.reshape(-1, self.e_dim, 1)
        e = e.reshape(-1, 1, self.e_dim)
        # [batch_size, dim, dim]
        c_matrix = nd.batch_dot(v, e)
        c_matrix_transpose = nd.transpose(c_matrix, axes=(0, 2, 1))
        # [batch_size * dim, dim]
        c_matrix = c_matrix.reshape((-1, self.e_dim))
        c_matrix_transpose = c_matrix_transpose.reshape((-1, self.e_dim))
        # [batch_size, dim]
        v_output = nd.dot(c_matrix, self.weight_vv) + nd.dot(c_matrix_transpose, self.weight_ev)
        e_output = nd.dot(c_matrix, self.weight_ve) + nd.dot(c_matrix_transpose, self.weight_ee)
        v_output = v_output.reshape(-1, self.e_dim) + self.bias_v
        e_output = e_output.reshape(-1, self.e_dim) + self.bias_e
        return v_output, e_output

#加入dropout层，可替换掉MKR类中的user_dense1等。
class DenseLayer(nn.Block):
    def __init__(self,e_dim,dropout_prob):
        super(DenseLayer, self).__init__()
        self.dense=nn.Dense(e_dim, activation='relu')
        self.drop=nn.Dropout(dropout_prob)

    def forward(self,x):
        return self.drop(self.dense(x))

class MKR(nn.Sequential):
    def __init__(self, n_user, n_entity, n_relation, e_dim=100, margin=1):
        super().__init__()
        self.e_emb=e_dim
        self.margin=margin
        self.u_emb = nn.Embedding(n_user, e_dim)
        self.e_emb = nn.Embedding(n_entity, e_dim)
        self.r_emb = nn.Embedding(n_relation, e_dim)

        self.user_dense1 = nn.Dense(e_dim, activation='relu')
        self.user_dense2 = nn.Dense(e_dim, activation='relu')
        self.user_dense3 = nn.Dense(e_dim, activation='relu')
        self.tail_dense1 = nn.Dense(e_dim, activation='relu')
        self.tail_dense2 = nn.Dense(e_dim, activation='relu')
        self.tail_dense3 = nn.Dense(e_dim, activation='relu')
        self.cc_unit1 = CrossCompress(e_dim)
        self.cc_unit2 = CrossCompress(e_dim)
        self.cc_unit3 = CrossCompress(e_dim)

        self.BCEloss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

    def __hinge_loss(self, dist_correct, dist_corrupt):
        a=dist_correct - dist_corrupt + self.margin
        return nd.maximum(a, 0)

    def kg_predict(self,h,r,t):
        score= h + r - t
        return nd.sum(score**2,axis=1,keepdims=True)**0.5

    def rec_predict(self,u,v):
        y=nd.sigmoid(nd.sum(u*v,axis=1))
        return y

    def do_predict(self,x):
        u = self.u_emb(x[:, 0])
        v = self.e_emb(x[:, 1])
        u = self.user_dense1(u)
        u = self.user_dense2(u)
        u = self.user_dense3(u)
        v, h = self.cc_unit1(v, v)
        v, h = self.cc_unit2(v, h)
        v, h = self.cc_unit3(v, h)
        return self.rec_predict(u,v)

    def net(self,X):
        x_rec, x_pos, x_neg=X
        u = self.u_emb(x_rec[:, 0])
        v = self.e_emb(x_rec[:, 1])
        h_pos = self.e_emb(x_pos[:, 0])
        r_pos = self.r_emb(x_pos[:, 1])
        t_pos = self.e_emb(x_pos[:, 2])
        h_neg = self.e_emb(x_neg[:, 0])
        r_neg = self.r_emb(x_neg[:, 1])
        t_neg = self.e_emb(x_neg[:, 2])

        u=self.user_dense1(u)
        u=self.user_dense2(u)
        u=self.user_dense3(u)
        t_pos=self.tail_dense1(t_pos)
        t_pos=self.tail_dense2(t_pos)
        t_pos=self.tail_dense3(t_pos)
        v,h_pos=self.cc_unit1(v,h_pos)
        v,h_pos=self.cc_unit2(v,h_pos)
        v,h_pos=self.cc_unit3(v,h_pos)

        y_ture = nd.array(x_rec[:, 2], dtype=np.float32)

        rec_pred = self.rec_predict(u,v)
        rec_loss = sum(self.BCEloss(rec_pred, y_ture))

        kg_pos = self.kg_predict(h_pos,r_pos,t_pos)
        kg_neg = self.kg_predict(h_neg,r_neg,t_neg)
        kg_loss = sum(self.__hinge_loss(kg_pos,kg_neg))
        return rec_loss+kg_loss

#预测
def doEvaluation(net,testSet):
    pred = net.do_predict(nd.array(testSet))
    y_true = [int(t[2]) for t in testSet]
    predictions = [1 if i >= 0.5 else 0 for i in pred]
    p = evaluate.precision(y_true=y_true, y_pred=predictions)
    r = evaluate.recall(y_true=y_true, y_pred=predictions)
    acc = evaluate.accuracy_score(y_true,y_pred=predictions)
    return p,r,acc

def train(net,dataLoad,recPairs,kgPairs,testSet,epochs=5,lr=0.001,batchSize=1024):
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
        p, r, acc = doEvaluation(net, testSet)
        print("p:{},r:{},acc:{}".format(round(p, 3), round(r, 3), round(acc, 3)))




if __name__ == '__main__':
    entitys, relationShips, kgPairs = dataloader.readKgData(fp.Ml_100K.KG)
    users, items, train_set,test_set = dataloader.readRecData(fp.Ml_100K.RATING)

    dataLoader = dataloader.DataIter(entitys,relationShips)

    net = MKR(len(users),len(entitys), len(relationShips))
    net.collect_params().initialize(mx.init.Xavier())

    dataLoad = dataloader.DataIter(entitys, relationShips)

    train(net, dataLoad, train_set, kgPairs, test_set)