import numpy as np
import copy
import re
import networkx as nx
import math
import random
import matplotlib.pyplot as plt
import sys
import statistics
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


networksW = ["lesmisW","newZW","BibleW","HSW","SITCW","BarbellW"]
networksDW = ["FBDW","USairportDW","AdvogatoDW","HallDW","cshiringDW","BarbellDW"]
networksD = ["AnybeatD","FilmTrustD","FAAD","EcoliD","BarbellD","uniEmailD"]
networks = ["BrazilAir","EUAir","USAir","actor","film","Barbell"]
syntheticD = ["syntD0","syntD5","syntD10","syntD15"]
syntheticW = ["syntW0","syntW5","syntW10","syntW15"]
syntheticDW = ["syntDW0","syntDW5","syntDW10","syntDW15"]


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

'''
def reduceModelColumn2(A,partition):
    partition = sorted(partition, key=len)
    N = len(A)
    res = np.zeros((N,len(partition)))
    index = 0
    for elem in partition:
        for el in elem:
            res[:,index] = res[:,index] +  A[:,el-1]
        index=index+1
    return res
'''

def crossFold(redModel,k,vector,label,N):
    under = int(math.floor(N/k))
    over = int(math.ceil(N/k))
    llo = np.ones(k,dtype=int ) *over
    index = k-1
    while(np.sum(llo) > N):
        llo[index] = under
        index = index -1
    totF1 = 0
    vectortestGeneral = vector
    num = 0
    for it in range(k):
        redModelAMethod = copy.deepcopy(redModel)
        vectortest = vectortestGeneral[num:num+llo[it]]
        vectorsize = llo[it]
        num = num + llo[it]
        vectortest = sorted(vectortest,reverse=True)
        labelTest = np.zeros(vectorsize,dtype=int)
        for i in range(len(vectortest)):
            labelTest[i] = label[vectortest[i]]
        
        elements = list(range(N))
        
        for elem in vectortest:
            elements.remove(elem)
        labelCross = np.zeros(N-vectorsize,dtype=int)
        for i in range(N-vectorsize):
            labelCross[i] = label[elements[i]]



        tests = []
        ppartred = copy.deepcopy(ppart)
        redModelAMethodRed = np.zeros((len(elements),redModelAMethod.shape[1]))
        for elem in vectortest:
            tests.append(redModelAMethod[elem,:])
            ppartred = np.delete(ppartred,elem)
        index = 0
        for elem in elements:
            redModelAMethodRed[index,:]=redModelAMethod[elem,:]
            index = index+1            
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(redModelAMethodRed, labelCross)


        pred = np.zeros(vectorsize);
        for i in range(len(tests)):
            pred[i] = neigh.predict([tests[i]])
        totF1 = totF1 + f1_score(labelTest,pred,average="micro")
    return totF1/k

    
def regr(redModel,vector,pr):
    t = int(len(vector)*0.20)
    vectortest=vector[:t]
    vectortest = sorted(vectortest,reverse=True)
    vectorsize = len(vectortest)
    pgTest = np.zeros(vectorsize)
    for i in range(len(vectortest)):
        pgTest[i] = pr[vectortest[i]]

    elements = list(range(N))
    for elem in vectortest:
        elements.remove(elem)
    pgCross = np.zeros(N-vectorsize)
    for i in range(N-vectorsize):
        pgCross[i] = pr[elements[i]]

    tests = []
    redModelRed = np.zeros((len(elements),redModel.shape[1]))
    for elem in vectortest:
        tests.append(redModel[elem,:])
    index=0
    for elem in elements:
        redModelRed[index,:]=redModel[elem,:]
        index = index+1
    if(name in networksD or name in networksDW or name in networksW):
        reg = Ridge(alpha=1).fit(redModelRed, pgCross)    
    else:
        reg = LinearRegression(positive=True).fit(redModelRed, pgCross)
    pgpred = np.zeros(vectorsize);
    for i in range(len(tests)):
        pgpred[i] = reg.predict([tests[i]])
    pgvalues = list(pr.values())
    tot = 0
    for i in range(len(pgpred)):
        tot = tot + pow(pgpred[i]-pgTest[i],2)
    return ((tot/len(pgpred))/(np.average(pgvalues)))



undirected=False
Binary=False
net = sys.argv[1]
just = False
Fix=False

name=""
if(net=="FilmTrustD"):
    name="FilmTrustD"
    undirected=False
    N=874
    PATH="FilmTrustD/FilmTrustD"
    Binary=True
    just=True
elif(net=="BrazilAir"):
    name="BrazilAir"
    undirected=True
    N=131
    DIR="BrazilAir"
    PATH="BrazilAir/BrazilAir"
    Binary=True
    just=True
elif(net=="USAir"):
    name="USAir"
    undirected=True
    N=1190
    DIR="USAir"
    PATH="USAir/USAir"
    Binary=True
    just=True    
elif(net=="EUAir"):
    name="EUAir"
    undirected=True
    N=399
    DIR="EUAir"
    PATH="EUAir/EUAir"
    Binary=True
    just=True
elif(net=="actor"):
    name="actor"
    undirected=True
    N=7779
    DIR="actor"
    PATH = "actor/actor"
    Binary = True
    just=True
elif(net=="film"):
    name="film"
    undirected=True
    N=27312
    DIR="film"
    PATH="film/film"
    Binary=True
    just=True
elif(net=="AnybeatD"):
    name="AnybeatD"
    undirected=False
    N=12645
    PATH="AnybeatD/AnybeatD"
    Binary=True
    just=True
elif(net=="Barbell"):
    name="Barbell"
    undirected=True
    N=30
    DIR = "Barbell"
    PATH = "Barbell/barbell"
    Binary=True
    just=True
elif(net=="BarbellDW"):
    name="BarbellDW"
    undirected=False
    N=29
    PATH="BarbellDW/BarbellDW"
    Binary=False
    just=True
elif(net=="BarbellW"):
    name="BarbellW"
    undirected=True
    N=30
    PATH="BarbellW/BarbellW"
    Binary=False
    just=True
elif(net=="BarbellD"):
    name="BarbellD"
    undirected=False
    N=29
    PATH="BarbellD/BarbellD"
    Binary=True
    just=True
elif(net=="syntD0"):
    name="syntD0"
    undirected=False
    N=180
    PATH="synt"
    Binary=True
    just=True
elif(net=="syntD5"):
    name="syntD5"
    undirected=False
    N=180
    PATH="synt"
    Binary=True
    just=True
elif(net=="syntD10"):
    name="syntD10"
    undirected=False
    N=180
    PATH="synt"
    Binary=True
    just=True
elif(net=="syntD15"):
    name="syntD15"
    undirected=False
    N=180
    PATH="synt"
    Binary=True
    just=True
elif(net=="syntW0"):
    name="syntW0"
    undirected=True
    N=180
    PATH="synt"
    Binary=False
    just=True
elif(net=="syntW5"):
    name="syntW5"
    undirected=True
    N=180
    PATH="synt"
    Binary=False
    just=True
elif(net=="syntW10"):
    name="syntW10"
    undirected=True
    N=180
    PATH="synt"
    Binary=False
    just=True
elif(net=="syntW15"):
    name="syntW15"
    undirected=True
    N=180
    PATH="synt"
    Binary=False
    just=True
elif(net=="syntDW15"):
    name="syntDW15"
    undirected=False
    N=180
    PATH="synt"
    Binary=False
    just=True 
elif(net=="syntDW10"):
    name="syntDW10"
    undirected=False
    N=180
    PATH="synt"
    Binary=False
    just=True   
elif(net=="syntDW5"):
    name="syntDW5"
    undirected=False
    N=180
    PATH="synt"
    Binary=False
    just=True  
elif(net=="syntDW0"):
    name="syntDW0"
    undirected=False
    N=180
    PATH="synt"
    Binary=False
    just=True  
elif(net=="BeachW"):
    name="BeachW"
    undirected=True
    N=43
    PATH="BeachW/BeachW"
    Binary=False
    just=True
elif(net=="lesmisW"):
    name="lesmisW"
    undirected=True
    N=77
    PATH="lesmisW/lesmisW"
    Binary=False
    just=True
elif(net=="newZW"):
    name="newZW"
    undirected=True
    N=1511
    PATH="newZW/newZW"
    Binary=False
    just=True 
elif(net=="BibleW"):
    name="BibleW"
    undirected=True
    N=1773
    PATH="BibleW/BibleW"
    Binary=False
    just=True
elif(net=="SITCW"):
    name="SITCW"
    undirected=True
    N=774
    PATH="SITCW/SITCW"
    Binary=False
    just=True
elif(net=="HSW"):
    name="HSW"
    undirected=True
    N=866
    PATH="HSW/HSW"
    Binary=False
    just=True
elif(net=="FBDW"):
    name = "FBDW"
    N = 1899
    DIR = "FBDW"
    PATH = "FBDW/FBDW"
    undirected=False
    Binary=False
    just=True
elif(net=="USairportDW"):
    name="USairportDW"
    N=500
    PATH = "USairportDW/USairportDW"
    DIR = "USairportDW"
    undirected=False
    Binary=False
    just = True
elif(net=="cshiringDW"):
    name="cshiringDW"
    N=2037
    PATH = "cshiringDW/cshiringDW"
    DIR = "cshiringDW"
    undirected=False
    Binary=False
    just = True
elif(net=="AdvogatoDW"):
    name="AdvogatoDW"
    N= 6541
    DIR= "AdvogatoDW"
    PATH = "AdvogatoDW/AdvogatoDW"
    undirected=False
    Binary=False
    just = True
elif(net=="HallDW"):
    name="HallDW"
    undirected=False
    N=217
    PATH = "HallDW/HallDW"
    Binary = False
    just=True
elif(net=="FAAD"):
    name="FAAD"
    undirected=False
    N=1226
    PATH = "FAAD/FAAD"
    Binary = True
    just=True
elif(net=="EcoliD"):
    name="EcoliD"
    undirected=False
    N=423
    PATH = "EcoliD/EcoliD"
    Binary = True
    just=True
elif(net=="uniEmailD"):
    name="uniEmailD"
    undirected=False
    N=1133
    PATH = "uniEmailD/uniEmailD"
    Binary = True
    just=True
elif(net=="jdkD"):
    name="jdkD"
    undirected=False
    N=6488
    PATH = "jdkD/jdkD"
    Binary = True
    just=True
elif(net=="wekaD"):
    name="wekaD"
    undirected=False
    N=2124
    PATH = "wekaD/wekaD"
    Binary = True
    just=True

if(name not in syntheticD and name not in syntheticDW and name not in syntheticW):
    M = np.zeros((N,N),dtype=int);
    fid = open(PATH,"r");
    line = fid.readline();
    while(line[0]=="%"):
        line = fid.readline();

    while line!="":
        
        token = line.split();
        i= int(token[0])
        j= int(token[1])
        w=1
        if(Binary==False):
            w = float(token[2])
        M[i][j] = w
        if(undirected==True):
            M[j][i] = w
        line = fid.readline();

    fid.close();    
    if(net=="USairportDW"):
        M = M/(np.max(M))

    if(undirected):
        temp = nx.Graph();
    else:
        temp = nx.DiGraph();


    G = nx.from_numpy_array(M,create_using=temp)
    pr=5
    ei = None
    if(sys.argv[2]=="regr"):
        if(name == "PhysD" or name=="FBDW"):
            if(Binary==False):
                ei=nx.eigenvector_centrality(G,max_iter=5000,weight='weight')
            else:
                ei=nx.eigenvector_centrality(G,max_iter=5000)
        else:
            if(Binary==False):
                ei=nx.eigenvector_centrality(G,max_iter=1000,weight='weight')
            else:
                ei=nx.eigenvector_centrality(G,max_iter=1000)
        if(Binary==False):
            bt=nx.betweenness_centrality(G,weight='weight')
        else:
            bt=nx.betweenness_centrality(G)


    if(name in networks and name!="Barbell"):
        fLabel = open(name+"/"+net+"Label")
        line = fLabel.readline()
        label = np.zeros(N)
        index = 0
        while(line!=""):
            spline = line.split(" ")
            label[int(spline[0])] = int(spline[1])
            line = fLabel.readline()
        fLabel.close()



    f = open("embed/"+sys.argv[1]+"BE","r")
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
        blockAll = []
        blockDegree = []
        blockPG = []
        elem = elem.split(" ")
        elem = elem[1:len(elem)-1]
        for el in elem:
            inde = int(el.replace("x",""))
            ppart[inde-1] = index;
        index=index+1

    Peta =  computePart(ppart,N,pr)
    redModelA= reduceModelColumn(Mcross,Peta)

    if(name in networksDW or name in networksD): 
        f = open("embed/"+sys.argv[1]+"TBE","r")            
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
            blockAll = []
            blockDegree = []
            blockPG = []
            elem = elem.split(" ")
            elem = elem[1:len(elem)-1]
            for el in elem:
                inde = int(el.replace("x",""))
                ppart[inde-1] = index;
            index=index+1

        Peta =  computePart(ppart,N,pr)
        redModelA= np.concatenate((redModelA, reduceModelColumn(Mcross,Peta)), axis=-1) 
    if(name in networksW or name in networks):
        fLT = open("embed/"+name+"LT.csv","r")
        dim = 100
        mLT = np.zeros((N,dim))
        line = fLT.readline()
        indexLT = 0
        while(line!=""):
            spline = line.split(" ")
            spline = spline[:dim]
            mLT[indexLT,:] = np.array([float(x) for x in spline])
            indexLT = indexLT+1
            line = fLT.readline()

        fLT.close()

    if(name in networks and name!="film"):
        fSEGK = open("embed/"+name+"SEGK"+".txt","r")
        dim = 20
        mSEGK = np.zeros((N,dim))
        line = fSEGK.readline()
        indexSEGK = 0
        while(line!=""):
            spline = line.split(" ")
            spline = spline[:dim+1]
            mSEGK[int(spline[0]),:] = np.array([float(x) for x in spline[1:]])
            line = fSEGK.readline()

        fSEGK.close()

    if(name in networks and name!="actor" and name!="film"):
        fS2V = open("embed/"+name+"S2V","r")
        dim = 128
        mS2V = np.zeros((N,dim))
        line = fS2V.readline()
        line = fS2V.readline()
        indexS2V = 0
        while(line!=""):
            spline = line.split(" ")
            spline = spline[:dim+1]
            mS2V[int(spline[0]),:] = np.array([float(x) for x in spline[1:]])
            line = fS2V.readline()

        fS2V.close()

    if(name in networks):
        fGAS = open("embed/"+name+"GAS","r")
        dim = 128
        mGAS = np.zeros((N,dim))
        line = fGAS.readline()
        line = fGAS.readline()
        indexLT = 0
        while(line!=""):
            spline = line.split(",")
            spline = spline[1:]
            mGAS[indexLT,:] = np.array([float(x) for x in spline])
            indexLT = indexLT+1
            line = fGAS.readline()

        fGAS.close()

    if(name in networks):
        fRiders = open("embed/"+name+"Rid","r")
        line = fRiders.readline()
        spline = line.split(",")
        dim = len(spline)-1
        mRid = np.zeros((N,dim))
        line = fRiders.readline()

        indexLT = 0
        while(line!=""):
            spline = line.split(",")
            spline = spline[1:]
            mRid[indexLT,:] = np.array([float(x) for x in spline])
            indexLT = indexLT+1
            line = fRiders.readline()

        fRiders.close()

    if( name in networksD or name in networks):
        mDRNE = np.load("embed/"+name+"DRNE"+".npy")

    #cancellabile
    if(name=="BarbellDW"):
        fEMB = open("embed/"+name+"EBR","r")
        dim = 29
        mEMB = np.zeros((N,dim))
        line = fEMB.readline()
        line = fEMB.readline()
        indexEMB = 0
        while(line!=""):
            spline = line.split(" ")
            spline = spline[:dim+1]
            mEMB[int(float(spline[0])),:] = np.array([float(x) for x in spline[1:]])
            line = fEMB.readline()

        fEMB.close()
    if(name=="BarbellD"):
        fEMB = open("embed/"+name+"EBR","r")
        dim = 29
        mEMB = np.zeros((N,dim))
        line = fEMB.readline()
        line = fEMB.readline()
        indexEMB = 0
        while(line!=""):
            spline = line.split(" ")
            spline = spline[:dim+1]
            mEMB[int(float(spline[0])),:] = np.array([float(x) for x in spline[1:]])
            line = fEMB.readline()

        fEMB.close()   
    if(name=="BarbellW"):
        fEMB = open("embed/"+name+"EBR","r")
        dim = 30
        mEMB = np.zeros((N,dim))
        line = fEMB.readline()
        line = fEMB.readline()
        indexEMB = 0
        while(line!=""):
            spline = line.split(" ")
            spline = spline[:dim+1]
            mEMB[int(float(spline[0])),:] = np.array([float(x) for x in spline[1:]])
            line = fEMB.readline()

        fEMB.close()
    if(name in networksW or name in networksD or name in networksDW):
        fEMB = open("embed/"+name+"EBR","r")
        if(N>=128):
            dim = 128
        else:
            dim = N
        mEMB = np.zeros((N,dim))
        line = fEMB.readline()
        line = fEMB.readline()
        indexEMB = 0
        while(line!=""):
            spline = line.split(" ")
            spline = spline[:dim+1]
            mEMB[int(float(spline[0])),:] = np.array([float(x) for x in spline[1:]])
            line = fEMB.readline()
        fEMB.close()      




if(sys.argv[2]=="regr"):

    BDEregr=np.zeros(2)
    mLTregr=np.zeros(2)
    mSEGKregr=np.zeros(2)
    mDRNEregr=np.zeros(2)
    mS2Vregr=np.zeros(2)
    mGASregr=np.zeros(2)
    mRidregr=np.zeros(2)
    mEMBregr=np.zeros(2) 
    t = 50
    for i in range(t):
        print("Cross " + str(i+1))
        random.seed(i)
        vector = random.sample(range(N), N)
        BDEregr[0] = BDEregr[0] + regr(redModelA,vector,ei)
        BDEregr[1] = BDEregr[1] + regr(redModelA,vector,bt)
        if(name in networks or name in networksW):
            mLTregr[0] = mLTregr[0] + regr(mLT,vector,ei)
            mLTregr[1] = mLTregr[1] + regr(mLT,vector,bt)
        if(name in networks or name in networksD):
            mDRNEregr[0] = mDRNEregr[0] + regr(mDRNE,vector,ei)                               
            mDRNEregr[1] = mDRNEregr[1] + regr(mDRNE,vector,bt)
        if(name in networksD or name in networksDW or name in networksW):
            mEMBregr[0] = mEMBregr[0] + regr(mEMB,vector,ei)
            mEMBregr[1] = mEMBregr[1] + regr(mEMB,vector,bt)
        if(name in networks and name!="film"):
            mSEGKregr[0] = mSEGKregr[0] + regr(mSEGK,vector,ei)
            mSEGKregr[1] = mSEGKregr[1] + regr(mSEGK,vector,bt)
        if(name in networks and name!="film" and name!="actor"):
            mS2Vregr[0] = mS2Vregr[0] + regr(mS2V,vector,ei)
            mS2Vregr[1] = mS2Vregr[1] + regr(mS2V,vector,bt)
        if(name in networks):
            mGASregr[0] = mGASregr[0] + regr(mGAS,vector,ei)
            mGASregr[1] = mGASregr[1] + regr(mGAS,vector,bt)
        if(name in networks):
            mRidregr[0] = mRidregr[0] + regr(mRid,vector,ei)
            mRidregr[1] = mRidregr[1] + regr(mRid,vector,bt)

    f = open("results/"+name+"Regression.csv","w")
    f.write("Method;Eigen;Between\n")

    f.write("BDEEmb;"+str(BDEregr[0]/t).replace(".",",")+";"+str(BDEregr[1]/t).replace(".",",")+"\n")
    
    if(name in networksW or name in networks):
        f.write("GWEmb;"+str(mLTregr[0]/t).replace(".",",")+";"+str(mLTregr[1]/t).replace(".",",")+"\n")
    if(name in networksD or name in networks):
        f.write("DRNEEmb;"+str(mDRNEregr[0]/t).replace(".",",")+";"+str(mDRNEregr[1]/t).replace(".",",")+"\n")
    if(name in networksD or name in networksDW or name in networksW):
        f.write("EMBEmb;"+str(mEMBregr[0]/t).replace(".",",")+";"+str(mEMBregr[1]/t).replace(".",",")+"\n")
    if(name in networks and name!="film"):
        f.write("SEGKEmb;"+str(mSEGKregr[0]/t).replace(".",",")+";"+str(mSEGKregr[1]/t).replace(".",",")+"\n")
    if(name in networks and name!="film" and name!="actor"):
        f.write("S2VEmb;"+str(mS2Vregr[0]/t).replace(".",",")+";"+str(mS2Vregr[1]/t).replace(".",",")+"\n")
    if(name in networks):
        f.write("GASEmb;"+str(mGASregr[0]/t).replace(".",",")+";"+str(mGASregr[1]/t).replace(".",",")+"\n")
    if(name in networks):
        f.write("RidEmb;"+str(mRidregr[0]/t).replace(".",",")+";"+str(mRidregr[1]/t).replace(".",",")+"\n")
    f.close()



if(sys.argv[2]=="cla" and (name in syntheticD or name in syntheticDW)):
        
        typ = sys.argv[3]    
        fclass = open("results/"+typ+"Classification.csv","w")
        BDEtotalscore = 0
        DRNEtotalscore = 0
        S2Vtotalscore = 0
        N2Btotalscore = 0
        EMBtotalscore = 0

        typ = sys.argv[3]
        if("W" in name):
            pert = int(sys.argv[3].replace("syntDW",""))   
        else:
            pert = int(sys.argv[3].replace("syntD",""))   

        for k in range(20):
            print("Instance "+str(k))
            fLabel = open("synt/"+"syntcirclesample"+str(k)+"Dpert"+str(pert)+"Label")
            line = fLabel.readline()
            label = np.zeros(N)
            index = 0
            while(line!=""):
                spline = line.split(" ")
                label[int(spline[0])] = int(spline[1])
                line = fLabel.readline()
            fLabel.close()


            M = np.zeros((N,N),dtype=int);
            if("W" in name):
                fid = open("synt/"+"syntcirclesample"+str(k)+"DWpert"+str(pert),"r")
            else:    
                fid = open("synt/"+"syntcirclesample"+str(k)+"Dpert"+str(pert),"r")
            line = fid.readline();
            while(line[0]=="%"):
                line = fid.readline();

            while line!="":
                
                token = line.split();
                i= int(token[0])
                j= int(token[1])
                w=1
                if(Binary==False):
                    w=float(token[2]) 
                M[i][j] = w
                if(undirected==True):
                    M[j][i] = w
                line = fid.readline();

            fid.close();    


            if("W" in name):
                f = open("embed/"+"syntcirclesample"+str(k)+"DWpert"+str(pert)+"BE","r")
            else:    
                f = open("embed/"+"syntcirclesample"+str(k)+"Dpert"+str(pert)+"BE","r")
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
                blockAll = []
                blockDegree = []
                blockPG = []
                elem = elem.split(" ")
                elem = elem[1:len(elem)-1]
                for el in elem:
                    inde = int(el.replace("x",""))
                    ppart[inde-1] = index;
                index=index+1

            Peta =  computePart(ppart,N,0)
            redModelA= reduceModelColumn(Mcross,Peta)

            if("W" in name):
                f = open("embed/"+"syntcirclesample"+str(k)+"DWTpert"+str(pert)+"BE","r")
            else:    
                f = open("embed/"+"syntcirclesample"+str(k)+"DTpert"+str(pert)+"BE","r")              
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
                blockAll = []
                blockDegree = []
                blockPG = []
                elem = elem.split(" ")
                elem = elem[1:len(elem)-1]
                for el in elem:
                    inde = int(el.replace("x",""))
                    ppart[inde-1] = index;
                index=index+1

            Peta =  computePart(ppart,N,0)
            redModelA= np.concatenate((redModelA, reduceModelColumn(Mcross,Peta)), axis=-1) 

            mDRNE = np.load("embed/"+"syntcirclesample"+str(k)+"Dpert"+str(pert)+"DRNE"+".npy")
            
            if("W" in name):
                fEMB = open("embed/synt"+"circle"+"sample"+str(k)+"DWpert"+str(pert)+"EBR","r")
            else:
                fEMB = open("embed/synt"+"circle"+"sample"+str(k)+"Dpert"+str(pert)+"EBR","r")                
            dim = 128
            mEMB = np.zeros((N,dim))
            line = fEMB.readline()
            line = fEMB.readline()
            indexEMB = 0
            while(line!=""):
                spline = line.split(" ")
                spline = spline[:dim+1]
                mEMB[int(float(spline[0])),:] = np.array([float(x) for x in spline[1:]])
                line = fEMB.readline()

            fEMB.close()       

            BDEscore=0
            mDRNEscore=0
            mEMBscore=0
            tt = 20
            for i in range(tt):
                random.seed(i+100)
                print("Cross " + str(i+1))
                vector = random.sample(range(N), N)
                
                BDEscore = BDEscore + crossFold(redModelA,5,vector,label,N)
                if("W" not in name):
                    mDRNEscore = mDRNEscore + crossFold(mDRNE,5,vector,label,N)
                mEMBscore = mEMBscore + crossFold(mEMB,5,vector,label,N)
            BDEtotalscore = BDEtotalscore+BDEscore/tt
            DRNEtotalscore = DRNEtotalscore+mDRNEscore/tt
            EMBtotalscore = EMBtotalscore+mEMBscore/tt

        if("W" in name):
            fclass.write("BE;EMB\n")
            fclass.write(str(BDEtotalscore/20)+";"+str(EMBtotalscore/20)+"\n")
        else:
            fclass.write("BE;DRNE;EMB\n")
            fclass.write(str(BDEtotalscore/20)+";"+str(DRNEtotalscore/20)+";"+str(EMBtotalscore/20)+"\n")
        fclass.close() 



if(sys.argv[2]=="cla" and (name in syntheticW)):
        
        typ = sys.argv[3]
        pert = int(sys.argv[3].replace("syntW",""))    
        fclass = open("results/synt"+"W"+str(pert)+"Classification.csv","w")
        BDEtotalscore = 0
        LTtotalscore = 0
        EMBtotalscore = 0
        DWtotalscore = 0
        N2Btotalscore = 0
        HPtotalscore = 0
        for k in range(20):
            print("instance "+str(k))
            fLabel = open("synt/"+"synt"+"circlesample"+str(k)+"Wpert"+str(pert)+"Label")
            line = fLabel.readline()
            label = np.zeros(N)
            index = 0
            while(line!=""):
                spline = line.split(" ")
                label[int(spline[0])] = int(spline[1])
                line = fLabel.readline()
            fLabel.close()


            M = np.zeros((N,N),dtype=int);
            fid = open("synt/"+"synt"+"circlesample"+str(k)+"Wpert"+str(pert)+"","r");
            line = fid.readline();
            while(line[0]=="%"):
                line = fid.readline();

            while line!="":
                
                token = line.split();
                i= int(token[0])
                j= int(token[1])
                w=1
                if(Binary==False):
                    w=float(token[2]) 
                M[i][j] = w
                if(undirected==True):
                    M[j][i] = w
                line = fid.readline();

            fid.close();    



            f = open("embed/"+"synt"+"circlesample"+str(k)+"Wpert"+str(pert)+"BE","r")
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
                blockAll = []
                blockDegree = []
                blockPG = []
                elem = elem.split(" ")
                elem = elem[1:len(elem)-1]
                for el in elem:
                    inde = int(el.replace("x",""))
                    ppart[inde-1] = index;
                index=index+1

            Peta =  computePart(ppart,N,0)
            redModelA= reduceModelColumn(Mcross,Peta)
            

            fLT = open("embed/"+"synt"+"circlesample"+str(k)+"Wpert"+str(pert)+"LT.csv","r")
            dim = 100
            mLT = np.zeros((N,dim))
            line = fLT.readline()
            indexLT = 0
            while(line!=""):
                spline = line.split(" ")
                spline = spline[:dim]
                mLT[indexLT,:] = np.array([float(x) for x in spline])
                indexLT = indexLT+1
                line = fLT.readline()

            fLT.close()


            fN2B = open("embed/synt"+"circle"+"sample"+str(k)+"Wpert"+str(pert)+"N2B","r")
            dim = 128
            mN2B = np.zeros((N,dim))
            line = fN2B.readline()
            line = fN2B.readline()
            indexN2B = 0
            while(line!=""):
                spline = line.split(" ")
                #print(spline)
                spline = spline[:dim+1]
                mN2B[int(float(spline[0])),:] = np.array([float(x) for x in spline[1:]])
                line = fN2B.readline()

            fN2B.close()
            
            
            fEMB = open("embed/synt"+"circle"+"sample"+str(k)+"Wpert"+str(pert)+"EBR","r")
            dim = 128
            mEMB = np.zeros((N,dim))
            line = fEMB.readline()
            line = fEMB.readline()
            indexEMB = 0
            while(line!=""):
                spline = line.split(" ")
                spline = spline[:dim+1]
                mEMB[int(float(spline[0])),:] = np.array([float(x) for x in spline[1:]])
                line = fEMB.readline()

            fEMB.close()            
            BDEscore=0
            mLTscore=0
            mDRNEscore=0
            mN2Bscore=0
            mEMBscore=0
            tt = 20
            for i in range(tt):
                random.seed(i+100)
                print("Cross " + str(i+1))
                vector = random.sample(range(N), N)
                
                BDEscore = BDEscore + crossFold(redModelA,5,vector,label,N)
                mEMBscore = mEMBscore + crossFold(mEMB,5,vector,label,N)
                mLTscore = mLTscore + crossFold(mLT,5,vector,label,N)
                mN2Bscore = mN2Bscore + crossFold(mN2B,5,vector,label,N)

            BDEtotalscore = BDEtotalscore+BDEscore/tt
            LTtotalscore = LTtotalscore+mLTscore/tt
            N2Btotalscore = N2Btotalscore+mN2Bscore/tt
            EMBtotalscore = EMBtotalscore+mEMBscore/tt


        fclass.write("BE;LT;N2B;EMB\n")
        fclass.write(str(BDEtotalscore/20)+";"+str(LTtotalscore/20)+";"+str(N2Btotalscore/20)+";"+str(EMBtotalscore/20)+"\n")
        fclass.close()

if(sys.argv[2]=="cla" and (name in networks)):
    degree = np.zeros((N,1));
    for i in range(N):
        degree[i] = np.sum(M[i,:])

    redModelA = np.append(redModelA,degree,axis=1)
    if(name!="film"):
        mLT = np.append(mLT,degree,axis=1)
    if(name!="film"):
        mSEGK = np.append(mSEGK,degree,axis=1)
    mDRNE = np.append(mDRNE,degree,axis=1)
    if(name!="film" and name!="actor"):
        mS2V = np.append(mS2V,degree,axis=1)
    mGAS = np.append(mGAS,degree,axis=1)
    mRid = np.append(mRid,degree,axis=1)

    BDEscore=0
    mLTscore=0
    mR2Vscore=0 
    mSEGKscore=0
    mDRNEscore=0
    mS2Vscore=0
    mGASscore=0
    mRidscore=0
    tt = 50
    for i in range(tt):
        random.seed(i+100)
        print("Cross " + str(i+1))
        vector = random.sample(range(N), N)
        
        BDEscore = BDEscore + crossFold(redModelA,5,vector,label,N)
        if(name!="film"):
            mLTscore = mLTscore + crossFold(mLT,5,vector,label,N)
        if(name!="film"):
            mSEGKscore = mSEGKscore + crossFold(mSEGK,5,vector,label,N)
        mDRNEscore = mDRNEscore + crossFold(mDRNE,5,vector,label,N)
        if(name!="film" and name!="actor"):
            mS2Vscore = mS2Vscore + crossFold(mS2V,5,vector,label,N)
        mGASscore = mGASscore + crossFold(mGAS,5,vector,label,N)
        mRidscore = mRidscore + crossFold(mRid,5,vector,label,N)
        
    f = open("results/"+name+"Classification.csv","w")
    f.write("Method;F1score\n")

    f.write("BDEEmb;"+str(BDEscore/tt).replace(".",",")+"\n")
    f.write("GWEmb;"+str(mLTscore/tt).replace(".",",")+"\n")
    f.write("SEGKEmb;"+str(mSEGKscore/tt).replace(".",",")+"\n")
    f.write("DRNEEmb;"+str(mDRNEscore/tt).replace(".",",")+"\n")
    f.write("S2VEmb;"+str(mS2Vscore/tt).replace(".",",")+"\n")
    f.write("GASEmb;"+str(mGASscore/tt).replace(".",",")+"\n")
    f.write("RidEmb;"+str(mRidscore/tt).replace(".",",")+"\n")

    f.close()

if(sys.argv[2]=="viz"): 


    initV = "random"

    if(name=="Barbell"):
        seed=5
        perp = 3
        initV = "random"

        colors = ["b","b","b","b","b","b","b","b","b","r","g","c","m","y","k","k","y","m","c","g","r","b","b","b","b","b","b","b","b","b"]

        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(redModelA)
        plt.figure(1)
        print("BE TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],colors[i]+'o')
        plt.savefig("results/"+"BE_TSNE.pdf")


        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(mLT)
        plt.figure(2)
        print("GW TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],colors[i]+'o')
        plt.savefig("results/"+"GW_TSNE.pdf")


        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(mSEGK)
        plt.figure(3)
        print("SEGK TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],colors[i]+'o')
        plt.savefig("results/"+"SEGK_TSNE.pdf")


        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(mDRNE)
        plt.figure(4)
        print("DRNE TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],colors[i]+'o')
        plt.savefig("results/"+"DRNE_TSNE.pdf")


        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(mS2V)
        plt.figure(5)
        print("S2V TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],colors[i]+'o')
        plt.savefig("results/"+"S2V_TSNE.pdf")


        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(mGAS)
        plt.figure(6)
        print("GAS TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],colors[i]+'o')
        plt.savefig("results/"+"GAS_TSNE.pdf")


        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(mRid)
        plt.figure(7)
        print("Riders TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],colors[i]+'o')
        plt.savefig("results/"+"Rid_TSNE.pdf")


        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(redModelA)
        plt.figure(8)
        print("BE PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],colors[i]+'o')
        plt.savefig("results/"+"BE_PCA.pdf")


        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(mLT)
        plt.figure(9)
        print("GW PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],colors[i]+'o')
        plt.savefig("results/"+"GW_PCA.pdf")


        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(mSEGK)
        plt.figure(10)
        print("SEGK PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],colors[i]+'o')
        plt.savefig("results/"+"SEGK_PCA.pdf")


        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(mDRNE)
        plt.figure(11)
        print("DRNE PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],colors[i]+'o')
        plt.savefig("results/"+"DRNE_PCA.pdf")


        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(mS2V)
        plt.figure(12)
        print("S2V PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],colors[i]+'o')
        plt.savefig("results/"+"S2V_PCA.pdf")


        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(mGAS)
        plt.figure(13)
        print("GAS PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],colors[i]+'o')
        plt.savefig("results/"+"GAS_PCA.pdf")


        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(mRid)
        plt.figure(14)
        print("Riders PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],colors[i]+'o')
        plt.savefig("results/"+"Rid_PCA.pdf")
    if(name=="BarbellDW"):
        perp = 3
        seed = 77
        colors = ["r","b","b","b","b","b","b","b","b","b","b","b","b","b","b","r","b","b","b","b","c","m","y","k","g","k","y","c","m"]
    
        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(redModelA)
        plt.figure(1)
        print("BE TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"BEBarbellDW_TSNE.pdf")

        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(mEMB)
        plt.figure(2)
        print("EMB TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"EMBBarbellDW_TSNE.pdf")        

        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(redModelA)
        plt.figure(3)
        print("BE PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"BEBarbellDW_PCA.pdf")
        
        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(mEMB)
        plt.figure(4)
        print("EMB PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"EMBBarbellDW_PCA.pdf")
    elif(name=="BarbellD"):
        perp = 2
        seed = 77
        colors = ["r","b","b","b","b","b","b","b","b","b","b","b","b","b","b","r","b","b","b","b","c","m","y","k","g","k","y","c","m"]
    
        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(redModelA)
        plt.figure(1)
        print("BE TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"BEBarbellD_TSNE.pdf")


        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(mEMB)
        plt.figure(2)
        print("EMB TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"EMBBarbellD_TSNE.pdf")        

        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(mDRNE)
        plt.figure(3)
        print("DRNE TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"DRNEBarbellD_TSNE.pdf")  

        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(redModelA)
        plt.figure(4)
        print("BE PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"BEBarbellD_PCA.pdf")
        
        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(mEMB)
        plt.figure(5)
        print("EMB PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"EMBBarbellD_PCA.pdf")

        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(mDRNE)
        plt.figure(6)
        print("DRNE PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"DRNEBarbellD_PCA.pdf")
    elif(name=="BarbellW"):
        perp = 4
        seed = 77
        colors = ["r","b","b","b","b","b","b","b","b","b","b","b","b","b","b","r","b","b","b","b","c","m","y","k","g","g","k","y","m","c"]
    
        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(redModelA)
        plt.figure(1)
        print("BE TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"BEBarbellW_TSNE.pdf")


        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(mEMB)
        plt.figure(2)
        print("EMB TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"EMBBarbellW_TSNE.pdf")        

        X_emb = TSNE(n_components=2, learning_rate='auto',init=initV,n_iter=10000,perplexity=perp,random_state=seed).fit_transform(mLT)
        plt.figure(3)
        print("GW TSNE")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"GWBarbellW_TSNE.pdf") 

        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(redModelA)
        plt.figure(5)
        print("BE PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"BEBarbellW_PCA.pdf")
        
        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(mEMB)
        plt.figure(6)
        print("EMB PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"EMBBarbellW_PCA.pdf")

        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(mLT)
        plt.figure(7)
        print("GW PCA")
        for i in range(N):
            plt.plot(X_emb[i,0],X_emb[i,1],color=colors[i],marker="o")
        plt.savefig("results/"+"GWBarbellW_PCA.pdf")
