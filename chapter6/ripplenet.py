import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,precision_score,recall_score
from data_set import filepaths as fp
import collections
import numpy as np

kg_file=fp.Ml_100K.KG
rating_file=fp.Ml_100K.RATING
ripple_net_model_path=fp.Model.C5_RIPPLE_NET
dim = 16#实体和关系的向量维度
n_memory=32#每一波记录的节点数量
n_hop=2#水波扩撒的波数
lr = 0.02#学习率
batch_size=2048#批次数量
n_epoch=20#迭代次数
l2_weight = 1e-7#正则系数
kge_weight=0.01#知识图谱嵌入损失函数系数
show_loss=True#是否看损失函数收敛过程

item_update_mode='plus_transform'
#物品向量的更新模式，总共有四种：
#replace:直接用新一波预测的物品向量替代
#plus：与t-1波次的物品向量对应位相加
#replace_transform:用一个映射矩阵映射将预测的物品向量映射一下
#plus_transform：用一个映射矩阵映射将预测的物品向量映射一下后与t-1波次的物品向量对应位相加

using_all_hops=True#最终用户向量的产生方式，是否采用全部波次的输出向量相加。否则采用最后一波产生的输出向量作为用户向量
need_evaluation=True#训练过程中是否要验证
formal_train=False#是否是正式训练，若是，则会保存或覆盖原有模型
test_ratio = 0.2#测试集比例

def load_data(rating_file=rating_file,kg_file=kg_file):
    train_data, test_data, user_history_dict = __load_rating(rating_file)
    n_entity, n_relation, kg = __load_kg(kg_file)
    return train_data, test_data, n_entity, n_relation, kg, user_history_dict

def __load_kg(kg_file):
    print('reading KG file ...')
    kg_np = np.loadtxt(kg_file, dtype=np.int32)
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[str(head)].append([int(tail), int(relation)])

    return n_entity, n_relation, kg

def __load_rating(rating_file):
    print('reading rating file ...')
    rating_np = np.loadtxt(rating_file, dtype=np.int32)
    n_ratings = rating_np.shape[0]
    test_indices = np.random.choice(n_ratings, size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(set(range(n_ratings)) - set(test_indices))

    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if str(user) not in user_history_dict:
                user_history_dict[str(user)] = []
            user_history_dict[str(user)].append(item)

    train_indices = [i for i in train_indices if str(rating_np[i][0]) in user_history_dict]
    test_indices = [i for i in test_indices if str(rating_np[i][0]) in user_history_dict]

    train_data = rating_np[train_indices]
    test_data = rating_np[test_indices]
    return train_data, test_data, user_history_dict

class RippleNet(nn.Module):
    def __init__(self,n_entity,n_relation,dim=dim,n_hop=n_hop,
                 kge_weight=kge_weight,l2_weight=l2_weight,
                 n_memory=n_memory,item_update_mode=item_update_mode,using_all_hops=using_all_hops):
        super(RippleNet, self).__init__()

        self._parse_args(n_entity, n_relation,dim,n_hop,kge_weight,l2_weight,n_memory,item_update_mode,using_all_hops)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim * self.dim)
        self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        self.criterion = nn.BCELoss()

    def _parse_args(self, n_entity, n_relation,dim,n_hop,kge_weight,l2_weight,n_memory,item_update_mode,using_all_hops):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.n_hop = n_hop
        self.kge_weight = kge_weight
        self.l2_weight = l2_weight
        self.n_memory = n_memory
        self.item_update_mode = item_update_mode
        self.using_all_hops = using_all_hops

    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):

        # [batch size, dim]
        item_embeddings = self.entity_emb(items)
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, item_embeddings = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)

        scores = self.predict(item_embeddings, o_list)

        return_dict = self._compute_loss(
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
        return_dict["scores"] = scores

        return return_dict

    def forward_predict(self,items: torch.LongTensor,memories):
        memories_h,memories_r,memories_t=memories
        item_embeddings = self.entity_emb(items)
        h_emb_list,r_emb_list,t_emb_list = [],[],[]
        for i in range(self.n_hop):
            h_emb_list.append(self.entity_emb(memories_h[i]))
            r_emb_list.append(self.relation_emb(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim))
            t_emb_list.append(self.entity_emb(memories_t[i]))
        o_list, item_embeddings = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, item_embeddings)
        scores = self.predict(item_embeddings, o_list)
        return scores

    def _compute_loss(self, scores, labels, h_emb_list, t_emb_list, r_emb_list):
        base_loss = self.criterion(scores, labels.float())

        kge_loss = 0
        for hop in range(self.n_hop):
            # [batch size, n_memory, 1, dim]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=2)
            # [batch size, n_memory, dim, 1]
            t_expanded = torch.unsqueeze(t_emb_list[hop], dim=3)
            # [batch size, n_memory, dim, dim]
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
            kge_loss += torch.sigmoid(hRt).mean()
        kge_loss = -self.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += (h_emb_list[hop] * h_emb_list[hop]).sum()
            l2_loss += (t_emb_list[hop] * t_emb_list[hop]).sum()
            l2_loss += (r_emb_list[hop] * r_emb_list[hop]).sum()

        l2_loss = self.l2_weight * l2_loss

        loss = base_loss + kge_loss + l2_loss
        return dict(base_loss=base_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)

    def _key_addressing(self, h_emb_list, r_emb_list, t_emb_list, item_embeddings):
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=3)
            # [batch_size, n_memory, dim]
            Rh = torch.squeeze(torch.matmul(r_emb_list[hop], h_expanded))
            # [batch_size, dim, 1]
            v = torch.unsqueeze(item_embeddings, dim=2)
            # [batch_size, n_memory]
            probs = torch.squeeze(torch.matmul(Rh, v))
            # [batch_size, n_memory]
            probs_normalized = F.softmax(probs, dim=1)
            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)
            # [batch_size, dim]
            o = (t_emb_list[hop] * probs_expanded).sum(dim=1)
            item_embeddings = self._update_item_embedding(item_embeddings, o)
            o_list.append(o)
        return o_list, item_embeddings

    def _update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = self.transform_matrix(o)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = self.transform_matrix(item_embeddings + o)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)
        return torch.sigmoid(scores)

    def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        predictions = [1 if i >= 0.5 else 0 for i in scores]

        p = precision_score(y_true=labels, y_pred=predictions)
        r = recall_score(y_true=labels, y_pred=predictions)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        return auc,p,r

def get_feed_dict(n_hop,data, ripple_set, start, end):
    items = torch.LongTensor(data[start:end, 1])
    labels = torch.LongTensor(data[start:end, 2])
    memories_h, memories_r, memories_t = [], [], []
    for i in range(n_hop):
        memories_h.append(torch.LongTensor([ripple_set[user][i][0] for user in data[start:end, 0]]))
        memories_r.append(torch.LongTensor([ripple_set[user][i][1] for user in data[start:end, 0]]))
        memories_t.append(torch.LongTensor([ripple_set[user][i][2] for user in data[start:end, 0]]))
    return items, labels, memories_h, memories_r,memories_t

def get_single_user_ripple_set(single_user_history_dict,kg):
    user_ripple=[]
    for h in range(n_hop):
        memories_h = []
        memories_r = []
        memories_t = []
        if h == 0:
            tails_of_last_hop = single_user_history_dict
        else:
            tails_of_last_hop = user_ripple[-1][2]
        for entity in tails_of_last_hop:
            for tail_and_relation in kg.get(str(entity),[]):
                memories_h.append(entity)
                memories_r.append(tail_and_relation[1])
                memories_t.append(tail_and_relation[0])
        if len(memories_h) == 0:
            user_ripple.append(user_ripple[-1])
        else:
            replace = len(memories_h) < n_memory
            indices = np.random.choice(len(memories_h), size=n_memory, replace=replace)
            memories_h = [memories_h[i] for i in indices]
            memories_r = [memories_r[i] for i in indices]
            memories_t = [memories_t[i] for i in indices]
            user_ripple.append((memories_h, memories_r, memories_t))
    return user_ripple

def get_ripple_set(kg, user_history_dict):
    print('constructing ripple set ...')
    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = {}
    for user in user_history_dict:
        user_ripple=get_single_user_ripple_set(user_history_dict[user],kg)
        ripple_set[int(user)]=user_ripple
    return ripple_set

def doEvaluation(n_hop, model, train_data,test_data,ripple_set, batch_size,step):
    # evaluation
    train_auc, train_p, train_r = evaluation(n_hop, model, train_data, ripple_set, batch_size)
    test_auc, test_p, test_r = evaluation(n_hop, model, test_data, ripple_set, batch_size)

    print('epoch %d    train auc: %.4f  test auc: %.4f'
          % (step, train_auc, test_auc))
    print('train_p: %.4f    train_r: %.4f   test_p: %.4f    test_r: %.4f'
          % (train_p, train_r, test_p, test_r))


def evaluation(n_hop,model, data, ripple_set, batch_size):
    start = 0
    auc_list,ps,rs = [],[],[]
    model.eval()
    while start < data.shape[0]:
        auc,p,r = model.evaluate(*get_feed_dict(n_hop,data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        ps.append(p)
        rs.append(r)
        start += batch_size
    model.train()
    return float(np.mean(auc_list)),float(np.mean(ps)),float(np.mean(rs))


def train(data_info,n_epoch=n_epoch,batch_size=batch_size,n_hop=n_hop,show_loss=show_loss,need_evaluation=need_evaluation,model_path=ripple_net_model_path):
    train_data, test_data, n_entity, n_relation, kg, user_history_dict = data_info
    model = RippleNet(n_entity, n_relation)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr)

    for step in range(n_epoch):
        print('step:{}'.format(step))
        np.random.shuffle(train_data)
        start = 0
        ripple_set=get_ripple_set(kg,user_history_dict)
        while start < train_data.shape[0]:
            return_dict = model(*get_feed_dict(n_hop,train_data, ripple_set, start, start + batch_size))
            loss = return_dict["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            start += batch_size
            if show_loss:
                print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss.item()))
        if need_evaluation:
            doEvaluation(n_hop, model, train_data, test_data, ripple_set, batch_size, step)
            print('change ripple set')
            ripple_set = get_ripple_set(kg, user_history_dict)
            doEvaluation(n_hop, model, train_data,test_data,ripple_set, batch_size,step)
    if formal_train:
        torch.save(model, model_path)

if __name__=='__main__':
    train(load_data())