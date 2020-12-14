


def readTriple(path):
    with open(path,'r') as f:
        for line in f.readlines():
            lines=line.strip().split()
            if len(lines)!=3:continue
            yield lines