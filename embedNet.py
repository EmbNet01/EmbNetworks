import numpy as np
import sys
import copy
from distutils.util import strtobool



def computePart(array,N,pr):
    partition = []
    for i in range(N):
        block=[]
        for j in range(N):
            if(array[j]==i+1):
                block.append(j+1)
        if(block!=[]):
            partition.append(block)
    return partition


def reduceModelColumn(A,partition):
    dim = len(partition)
    res = np.zeros((len(A),dim))
    for j in range(len(A)):
        index = 0
        for elem in partition:
            for el in elem:
                res[j][index] = res[j][index] +  A[j][el-1]
            index=index+1
    return res


if len(sys.argv) < 8 or len(sys.argv)>8:
    print("""
            You have to insert the following parameters:
             - path input network
             - number of nodes
             - directed
             - weighted
             - path partition A
             - path partition A^T
             - path output embedding""")
    sys.exit(1)

# Access arguments
sourcePath = sys.argv[1]
N = int(sys.argv[2])
directed = bool(strtobool(sys.argv[3]))
weighted = bool(strtobool(sys.argv[4]))
partPath = sys.argv[5]
partPathT = sys.argv[6]
destPath = sys.argv[7]

M = np.zeros((N,N),dtype=int);
fid = open(sourcePath,"r");
line = fid.readline();
while(line[0]=="%"):
    line = fid.readline();

while line!="":
    
    token = line.split();
    i= int(token[0])
    j= int(token[1])
    w=1
    if(weighted==True):
        w = float(token[2])
    M[i][j] = w
    if(directed==False):
        M[j][i] = w
    line = fid.readline();

fid.close();

print(weighted)
print(M)

f = open(partPath,"r")
Mcross = copy.deepcopy(M)

line=f.readline()
line=f.readline()
f.close()
line = line.split(",")
red = len(line)-1
perc = ((N-red)/N)
partitionAll = []

ppart = np.zeros(N,dtype=int)
index=1
for elem in line:
    elem = elem.split(" ")
    elem = elem[1:len(elem)-1]
    for el in elem:
        inde = int(el.replace("x",""))
        ppart[inde-1] = index;
    index=index+1

Peta =  computePart(ppart,N,0)
redModelA= reduceModelColumn(Mcross,Peta)

if(directed == True):

    f = open(partPathT,"r")
    Mcross = copy.deepcopy(np.transpose(M))

    line=f.readline()
    line=f.readline()
    f.close()
    line = line.split(",")
    red = len(line)-1
    perc = ((N-red)/N)
    partitionAll = []

    ppart = np.zeros(N,dtype=int)
    index=1
    for elem in line:
        elem = elem.split(" ")
        elem = elem[1:len(elem)-1]
        for el in elem:
            inde = int(el.replace("x",""))
            ppart[inde-1] = index;
        index=index+1

    Peta =  computePart(ppart,N,0)
    redModelA= np.concatenate((redModelA, reduceModelColumn(Mcross,Peta)), axis=-1)


np.savetxt(destPath, redModelA, delimiter=" ", fmt='%d')