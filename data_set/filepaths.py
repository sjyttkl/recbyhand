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