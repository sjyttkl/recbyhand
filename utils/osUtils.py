


def readTriple(path):
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            lines=line.strip().split()
            if len(lines)!=3:continue
            yield lines


def readFile(path):
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            lines=line.strip().split()
            if len(lines)==0:continue
            yield lines