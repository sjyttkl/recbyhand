import os

class FB15K_237():
    # 下载地址: https://www.microsoft.com/en-us/download/details.aspx?id=52312
    __BASE = os.path.join(os.path.split(os.path.realpath(__file__))[0], './FB15K-237')
    O_TRAIN=os.path.join(__BASE,'train.txt')
    O_VALID=os.path.join(__BASE,'valid.txt')
    O_TEST=os.path.join(__BASE,'test.txt')

    EID2INDEX=os.path.join(__BASE,'eid2index.tsv')
    RID2INDEX=os.path.join(__BASE,'rid2index.tsv')

    TRAIN=os.path.join(__BASE,'train.tsv')
    VALID=os.path.join(__BASE,'valid.tsv')
    TEST=os.path.join(__BASE,'test.tsv')

class Kg_Co_occurrenceMatrix():
    BASE = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'kg_Co_occurrenceMatrix')

class Ml_100K():
    #下载地址：https://github.com/rexrex9/kb4recMovielensDataProcess
    __BASE = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'ml-100k')
    KG=os.path.join(__BASE,'kg_index.tsv')
    RATING = os.path.join(__BASE,'rating_index.tsv')
    RATING5 = os.path.join(__BASE, 'rating_index_5.tsv')

class Ml_1M():
    #下载地址：https://github.com/rexrex9/kb4recMovielensDataProcess
    __BASE = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'ml-1m')
    KG=os.path.join(__BASE,'kg_index.tsv')
    RATING = os.path.join(__BASE,'rating_index.tsv')
    RATING5 = os.path.join(__BASE, 'rating_index_5.tsv')

class Model():
    __BASE = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model')
    C5_RIPPLE_NET = os.path.join(__BASE,'c5_ripple_net.model')

class WIKI_VOTE():
    __BASE = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'wiki-vote')
    WIKI_VOTE = os.path.join(__BASE,'Wiki-Vote.txt')