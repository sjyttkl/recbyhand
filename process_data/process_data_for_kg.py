from data_set import filepaths
from utils import utils


def __processEntitysAndRelationSets(entitys,relations,path):
    for h, r, t in utils.readTriple(path):
        entitys.add(h)
        entitys.add(t)
        relations.add(r)

def __generateTrainningDate(eid2index,rid2index,opath,path):
    with open(path,'w+') as f:
        for h,r,t in utils.readTriple(opath):
            f.write(eid2index[h]+'\t'+rid2index[r]+'\t'+eid2index[t]+'\n')

def processFBDate():
    entitys,relations=set(),set()

    __processEntitysAndRelationSets(entitys, relations, filepaths.FB15K_237.O_TRAIN)
    __processEntitysAndRelationSets(entitys, relations, filepaths.FB15K_237.O_TEST)
    __processEntitysAndRelationSets(entitys, relations, filepaths.FB15K_237.O_VALID)

    eid2index=[(eid,index) for index,eid in enumerate(entitys)]
    rid2index=[(rid,index) for index,rid in enumerate(relations)]

    eid2indexDict,rid2indexDict=dict(),dict()

    with open(filepaths.FB15K_237.EID2INDEX,'w+') as f:
        for eid,index in eid2index:
            f.write(eid+'\t'+str(index)+'\n')
            eid2indexDict[eid]=str(index)

    with open(filepaths.FB15K_237.RID2INDEX,'w+') as f:
        for rid,index in rid2index:
            f.write(rid+'\t'+str(index)+'\n')
            rid2indexDict[rid]=str(index)

    __generateTrainningDate(eid2indexDict,rid2indexDict, filepaths.FB15K_237.O_TRAIN, filepaths.FB15K_237.TRAIN)
    __generateTrainningDate(eid2indexDict,rid2indexDict, filepaths.FB15K_237.O_VALID, filepaths.FB15K_237.VALID)
    __generateTrainningDate(eid2indexDict,rid2indexDict, filepaths.FB15K_237.O_TEST, filepaths.FB15K_237.TEST)

if __name__=='__main__':
    processFBDate()