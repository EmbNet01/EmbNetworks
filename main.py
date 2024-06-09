#!/usr/bin/env python
# coding: utf-8

# In[82]:


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


def crossFold(redModel,k,vector,label,N):
    under = int(math.floor(N/k))
    over = int(math.ceil(N/k))
    llo = np.ones(k,dtype=int ) *over
    index = k-1
    while(np.sum(llo) > N):
        llo[index] = under
        index = index -1
    totAcc = 0
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
if(net=="BrazilAir"):
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
elif(net=="Barbell"):
    name="Barbell"
    undirected=True
    N=30
    DIR = "Barbell"
    PATH = "Barbell/barbell"
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




if(sys.argv[2]=="regr" or sys.argv[2]=="cla" or sys.argv[2]=="viz" ):
    M = np.zeros((N,N),dtype=int);
    fid = open(PATH,"r");
    line = fid.readline();
    while(line[0]=="%"):
        line = fid.readline();

    while line!="":
        
        token = line.split();
        i= int(token[0])
        j= int(token[1])
        '''
        if(Binary==False):
            value = float(token[2])
            M[i-1][j-1] = value
        else:
            if(just==True):
                #print("JUST")
                M[i][j] = 1
            else:
                M[i-1][j-1] = 1
                
        if(undirected==True):
            if(Binary==False):
                M[j-1][i-1] = value
            else:
                if(just==True):
                    M[j][i] = 1
                else:
                    M[j-1][i-1] = 1
        '''
        M[i][j] = 1
        M[j][i] = 1
        line = fid.readline();

    fid.close();    


    temp = nx.Graph();

    G = nx.from_numpy_matrix(M,create_using=temp)
    pr=nx.pagerank(G, alpha=0.85)
    ei=nx.eigenvector_centrality(G,max_iter=1000)
    cl=nx.closeness_centrality(G)
    bt=nx.betweenness_centrality(G)




    if(name!="Barbell"):
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


    if(name!="film"):
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




    if(name!="film"):
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



    mDRNE = np.load("embed/"+name+"DRNE"+".npy")
    if(name == "actor"):
        for i in range(21):
            mDRNE = np.vstack([mDRNE, np.zeros((1,64))])


    if(name!="actor" and name!="film"):
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



if(sys.argv[2]=="regr"):
    BDEregr=np.zeros(2)
    mLTregr=np.zeros(2)
    mR2Vregr=np.zeros(2)
    mSEGKregr=np.zeros(2)
    mDRNEregr=np.zeros(2)
    mS2Vregr=np.zeros(2)
    mGASregr=np.zeros(2)
    mRidregr=np.zeros(2)
    t = 50
    for i in range(t):
        print("Cross " + str(i+1))
        random.seed(i)
        vector = random.sample(range(N), N)
        #print("Shape")
        #print(redModelA.shape)
        BDEregr[0] = BDEregr[0] + regr(redModelA,vector,ei)
        BDEregr[1] = BDEregr[1] + regr(redModelA,vector,bt)
        if(name!="film"):
            mLTregr[0] = mLTregr[0] + regr(mLT,vector,ei)
            mLTregr[1] = mLTregr[1] + regr(mLT,vector,bt)
        if(name!="film"):
            mSEGKregr[0] = mSEGKregr[0] + regr(mSEGK,vector,ei)
            mSEGKregr[1] = mSEGKregr[1] + regr(mSEGK,vector,bt)
        mDRNEregr[0] = mDRNEregr[0] + regr(mDRNE,vector,ei)                               
        mDRNEregr[1] = mDRNEregr[1] + regr(mDRNE,vector,bt)
        if(name!="film" and name!="actor"):
            mS2Vregr[0] = mS2Vregr[0] + regr(mS2V,vector,ei)
            mS2Vregr[1] = mS2Vregr[1] + regr(mS2V,vector,bt)
        mGASregr[0] = mGASregr[0] + regr(mGAS,vector,ei)
        mGASregr[1] = mGASregr[1] + regr(mGAS,vector,bt)
        mRidregr[0] = mRidregr[0] + regr(mRid,vector,ei)
        mRidregr[1] = mRidregr[1] + regr(mRid,vector,bt)
        

    f = open("results/"+name+"Regression.csv","w")
    f.write("Method;Eigen;Between\n")

    f.write("BDEEmb;"+str(BDEregr[0]/t).replace(".",",")+";"+str(BDEregr[1]/t).replace(".",",")+"\n")
    f.write("GWEmb;"+str(mLTregr[0]/t).replace(".",",")+";"+str(mLTregr[1]/t).replace(".",",")+"\n")
    f.write("SEGKEmb;"+str(mSEGKregr[0]/t).replace(".",",")+";"+str(mSEGKregr[1]/t).replace(".",",")+"\n")
    f.write("DRNEEmb;"+str(mDRNEregr[0]/t).replace(".",",")+";"+str(mDRNEregr[1]/t).replace(".",",")+"\n")
    f.write("S2VEmb;"+str(mS2Vregr[0]/t).replace(".",",")+";"+str(mS2Vregr[1]/t).replace(".",",")+"\n")
    f.write("GASEmb;"+str(mGASregr[0]/t).replace(".",",")+";"+str(mGASregr[1]/t).replace(".",",")+"\n")
    f.write("RidEmb;"+str(mRidregr[0]/t).replace(".",",")+";"+str(mRidregr[1]/t).replace(".",",")+"\n")

    f.close()


from sklearn.neighbors import KNeighborsClassifier

if(sys.argv[2]=="cla"):    
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



if(sys.argv[2]=="synt"): 

    houseBlock = np.ones((4,4))-np.eye(4) 
    starBlock = np.zeros((7,7))
    starBlock[0,:] = np.ones((1,7))
    starBlock[:,0] = np.ones((1,7))
    starBlock[0][0] = 0
    arrowBlock =np.array([[0,1,1,1,1,0],[1,0,0,1,0,0],[1,0,0,0,1,0],[1,1,0,0,0,1],[1,0,1,0,0,1],[0,0,0,1,1,0]])

    seq = []
    n1 = 0
    n2 = 0
    n3 = 0
    seq = ['arrow', 'house', 'star', 'house', 'arrow', 'star', 'star', 'house', 'house', 'arrow', 'star', 'star', 'house', 'arrow', 'house', 'arrow', 'house', 'house', 'arrow', 'star', 'house', 'star', 'house', 'house', 'arrow', 'star', 'arrow', 'arrow', 'house', 'house']
    for i in range(len(seq)):
        if(seq[i] == "house"):
            n1 = n1+1
        elif(seq[i]=="arrow"):
            n2 = n2+1
        elif(seq[i]=="star"):
            n3 = n3+1

    nhouse = n1
    nstar = n3
    narrow = n2
    dimhouse = 4
    dimstar = 7
    dimarrow = 6
    Kcircle = (nhouse+nstar+narrow)*2
    Khouse = nhouse*dimhouse
    Kstar = nstar*dimstar
    Karrow = narrow*dimarrow
    K = Kcircle+Karrow+Kstar+Khouse
    Hmat = np.zeros((K,K))
    NSynt = len(Hmat)

    for i in range(Kcircle):
        if(i == 0):
            Hmat[i][Kcircle-1] = 1
            Hmat[i][i+1] = 1
            Hmat[Kcircle-1][i] = 1
            Hmat[i+1][i] = 1
        elif(i == Kcircle-1):
            Hmat[i][i-1] = 1
            Hmat[i][0] = 1
            Hmat[i-1][i] = 1
            Hmat[0][i] = 1
        else:
            Hmat[i][i+1] = 1
            Hmat[i][i-1] = 1
            Hmat[i+1][i] = 1
            Hmat[i-1][i] = 1     

    tickx = 0
    ticky = 0
    weight = 1
    for elem in seq:
        if(elem == 'house'):
                Hmat[tickx*2][Kcircle+ticky] = 1
                Hmat[tickx*2][Kcircle+ticky+1] = 1
                Hmat[Kcircle+ticky][tickx*2] = 1
                Hmat[Kcircle+ticky+1][tickx*2] = 1
                Hmat[Kcircle+ticky:Kcircle+(ticky+dimhouse),Kcircle+ticky:Kcircle+(ticky+dimhouse)] = houseBlock*weight
                ticky = ticky+dimhouse
        elif(elem == 'star'):
                Hmat[tickx*2][Kcircle+ticky] = 1
                Hmat[Kcircle+ticky][tickx*2] = 1
                Hmat[Kcircle+ticky:Kcircle+(ticky+dimstar),Kcircle+ticky:Kcircle+(ticky+dimstar)] = starBlock*weight
                ticky = ticky+dimstar
        elif(elem == 'arrow'):
                Hmat[tickx*2][Kcircle+ticky] = 1
                Hmat[Kcircle+ticky][tickx*2] = 1
                Hmat[Kcircle+ticky:Kcircle+(ticky+dimarrow),Kcircle+ticky:Kcircle+(ticky+dimarrow)] = arrowBlock*weight
                ticky = ticky+dimarrow
        tickx = tickx+1

    role = np.zeros(NSynt)
    indexTop=0
    for i in range(Kcircle):
        
        if(i%2==0):
            if(seq[indexTop]=='house'):
                role[i] = 0
            elif(seq[indexTop]=='star'):
                role[i] = 10
            elif(seq[indexTop]=='arrow'):
                role[i] = 11
        else:
            if(seq[indexTop]=='house'):
                role[i] = 1
            elif(seq[indexTop]=='star'):
                role[i] = 12
            elif(seq[indexTop]=='arrow'):
                role[i] = 13
            indexTop = indexTop+1
    i = Kcircle
    indexTop = 0
    indexS = 1
    belong = np.zeros(NSynt)
    while(i<NSynt):
        if(seq[indexTop]=='house'):
            role[i] = 2
            belong[i] = indexS 
            role[i+1] = 2
            belong[i+1] = indexS      
            role[i+2] = 3
            belong[i+2] = indexS
            role[i+3] = 3
            belong[i+3] = indexS
            i = i+4
        elif(seq[indexTop]=='star'):
            role[i] = 4
            belong[i] = indexS 
            role[i+1] = 5
            belong[i+1] = indexS 
            role[i+2] = 5
            belong[i+2] = indexS 
            role[i+3] = 5
            belong[i+3] = indexS 
            role[i+4] = 5
            belong[i+4] = indexS 
            role[i+5] = 5
            belong[i+5] = indexS 
            role[i+6] = 5
            belong[i+6] = indexS 
            i=i+7
        elif(seq[indexTop]=='arrow'):
            role[i] = 6
            belong[i] = indexS 
            role[i+1] = 7
            belong[i+1] = indexS 
            role[i+2] = 7
            belong[i+2] = indexS 
            role[i+3] = 8
            belong[i+3] = indexS 
            role[i+4] = 8
            belong[i+4] = indexS 
            role[i+5] = 9
            belong[i+5] = indexS 
            i=i+6
        indexS = indexS+1
        indexTop = indexTop +1


    dim = 100
    pert = [0.0,0.10,0.20,0.30]
    eps = [1,1,2,2]
    epsN = [1,1,2,2]
    acc = [[],[],[],[],[],[],[]];
    accMin = [[],[],[],[],[],[],[]];
    accMax = [[],[],[],[],[],[],[]];
    for kk in range(4):
        onlyBDE = False
        noCross = False
        minBDEscore = 1
        minLTscore = 1
        minSEGKscore = 1
        minDRNEscore = 1
        minS2Vscore = 1
        minGASscore = 1
        minRidscore = 1

        maxBDEscore = 0
        maxLTscore = 0
        maxSEGKscore = 0
        maxDRNEscore = 0
        maxS2Vscore = 0
        maxGASscore = 0
        maxRidscore = 0    
        
        BDEtotalScore = 0
        GWtotalScore = 0
        DRNEtotalScore = 0
        SEGKtotalScore = 0
        S2VtotalScore = 0
        GAStotalScore = 0
        RidtotalScore = 0
        
        
        nIst = 10
        for k in range(nIst):
            print("Istanza " +str(k+1))
            M = np.copy(Hmat)
            random.seed(k+1)
            numbChang = int(pert[kk]*(np.sum(Hmat)/2))
            num =0
            undirected=True

            while num<numbChang:
                rn1 = random.randint(0, K-1)
                rn2 = random.randint(0, K-1)
                if(M[rn1][rn2]==0  and belong[rn1] == belong[rn2]):
                    M[rn1][rn2] = 1
                    M[rn2][rn1] = 1
                    num=num+1

            nameSynt = "embed/Synt/Circle"+str(nhouse)+"_"+str(nstar)+"_"+str(narrow)+"seed"+str(k+1)+"pert"+str(int(pert[kk]*100))
            
            fsynt = open(nameSynt+"BE","r")
            line = fsynt.readline();
            line = line.split(",")
            ppart = np.zeros(NSynt,dtype=int)
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
            PetaSynt =  computePart(ppart,NSynt,[])
            redModelASynt= reduceModelColumn(M,PetaSynt)

            if(onlyBDE==True):
                nameSynt = "embed/Synt/Circle"+str(nhouse)+"_"+str(nstar)+"_"+str(narrow)+"seed"+str(k+1)+"pert"+str(int(pert[kk]*100))
                dim=100
            if(onlyBDE==False):
                fLT = open(nameSynt+"_"+"LT.csv","r")
                mLT = np.zeros((NSynt,100))
                line = fLT.readline()
                indexLT = 0
                while(line!=""):
                    spline = line.split(" ")
                    spline = spline[:100]
                    mLT[indexLT,:] = np.array([float(x) for x in spline])
                    indexLT = indexLT+1
                    line = fLT.readline()

                fLT.close()

                fSEGK = open(nameSynt+"SEGK"+".txt","r")
                #fSEGK = open(name+"SEGK"+"34.txt","r")
                mSEGK = np.zeros((NSynt,20))
                line = fSEGK.readline()
                indexSEGK = 0
                while(line!=""):
                    spline = line.split(" ")
                    spline = spline[:20+1]
                    if(len(spline)>1):
                        mSEGK[int(spline[0]),:] = np.array([float(x) for x in spline[1:]])
                    line = fSEGK.readline()

                fSEGK.close()

                mDRNE = np.load(nameSynt+"DRNE"+".npy")

                fS2V = open(nameSynt+"S2V"+"","r")
                mS2V = np.zeros((NSynt,128))
                line = fS2V.readline()
                line = fS2V.readline()
                indexS2V = 0
                while(line!=""):
                    spline = line.split(" ")
                    spline = spline[:128+1]
                    if(len(spline)>1):
                    #print(len(spline))
                        mS2V[int(spline[0]),:] = np.array([float(x) for x in spline[1:]])
                    line = fS2V.readline()

                fS2V.close()
                
                fGAS = open(nameSynt+"GAS","r")
                mGAS = np.zeros((NSynt,128))
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
                
                
                fRiders = open(nameSynt+"Rid","r")
                line = fRiders.readline()
                spline = line.split(",")
                dim = len(spline)-1
                mRid = np.zeros((NSynt,dim))
                line = fRiders.readline()

                indexLT = 0
                while(line!=""):
                    spline = line.split(",")
                    #print(spline)
                    #print(len(spline))
                    spline = spline[1:]
                    #print(len(spline))
                    mRid[indexLT,:] = np.array([float(x) for x in spline])
                    indexLT = indexLT+1
                    line = fRiders.readline()

                fRiders.close()


            degree = np.zeros((NSynt,1));
            for i in range(NSynt):
                degree[i] = np.sum(M[i,:])

            redModelASynt = np.append(redModelASynt,degree,axis=1)
            if(onlyBDE==False):
                mLT = np.append(mLT,degree,axis=1)
                mSEGK = np.append(mSEGK,degree,axis=1)
                mDRNE = np.append(mDRNE,degree,axis=1)
                mS2V = np.append(mS2V,degree,axis=1)
                mGAS = np.append(mGAS,degree,axis=1)
                mRid = np.append(mRid,degree,axis=1)


            nCross = 10
            BDEscore=0
            mLTscore=0
            #mR2Vscore=0 
            mSEGKscore=0
            mDRNEscore=0
            mS2Vscore=0
            mGASscore=0
            mRidscore=0

            if(noCross==False):
                for i in range(nCross):
                    print("Cross " + str(i+1))
                    vector = random.sample(range(NSynt), NSynt)
                    BDEadd = crossFold(redModelASynt,5,vector,role,NSynt)
                    BDEscore = BDEscore + BDEadd
                    if(BDEadd<minBDEscore):
                        minBDEscore = BDEadd
                    if(BDEadd > maxBDEscore):
                        maxBDEscore = BDEadd
                    if(onlyBDE==False):
                        LTadd = crossFold(mLT,5,vector,role,NSynt)
                        mLTscore = mLTscore + LTadd
                        #mR2Vscore = mR2Vscore + crossFold(mR2V,5,vector)
                        SEGKadd = crossFold(mSEGK,5,vector,role,NSynt)
                        mSEGKscore = mSEGKscore + SEGKadd
                        DRNEadd = crossFold(mDRNE,5,vector,role,NSynt)
                        mDRNEscore = mDRNEscore + DRNEadd
                        S2Vadd = crossFold(mS2V,5,vector,role,NSynt)
                        mS2Vscore = mS2Vscore + S2Vadd
                        GASadd = crossFold(mGAS,5,vector,role,NSynt)
                        mGASscore = mGASscore + GASadd 
                        Ridadd = crossFold(mRid,5,vector,role,NSynt)
                        mRidscore = mRidscore + Ridadd 
                        
                        
                        if(LTadd<minLTscore):
                            minLTscore = LTadd
                        if(LTadd > maxLTscore):
                            maxLTscore = LTadd
                        
                        if(SEGKadd<minSEGKscore):
                            minSEGKscore = SEGKadd
                        if(SEGKadd > maxSEGKscore):
                            maxSEGKscore = SEGKadd
                        
                        if(DRNEadd<minDRNEscore):
                            minDRNEscore = DRNEadd
                        if(DRNEadd > maxDRNEscore):
                            maxDRNEscore = DRNEadd
                            
                        if(S2Vadd<minS2Vscore):
                            minS2Vscore = S2Vadd
                        if(S2Vadd > maxS2Vscore):
                            maxS2Vscore = S2Vadd
                        
                        if(GASadd<minGASscore):
                            minGASscore = GASadd
                        if(GASadd > maxGASscore):
                            maxGASscore = GASadd
                            
                        if(Ridadd<minRidscore):
                            minRidscore = Ridadd
                        if(Ridadd > maxRidscore):
                            maxRidscore = Ridadd
                        
                        
            BDEtotalScore = BDEtotalScore + BDEscore/nCross
            GWtotalScore = GWtotalScore + mLTscore/nCross
            SEGKtotalScore = SEGKtotalScore + mSEGKscore/nCross
            DRNEtotalScore = DRNEtotalScore + mDRNEscore/nCross
            S2VtotalScore = S2VtotalScore + mS2Vscore/nCross
            GAStotalScore = GAStotalScore + mGASscore/nCross
            RidtotalScore = RidtotalScore + mRidscore/nCross


        acc[0].append(BDEtotalScore/nIst)
        acc[1].append(GWtotalScore/nIst)
        acc[2].append(SEGKtotalScore/nIst)
        acc[3].append(DRNEtotalScore/nIst)
        acc[4].append(S2VtotalScore/nIst)
        acc[5].append(GAStotalScore/nIst)
        acc[6].append(RidtotalScore/nIst)
        
        accMin[0].append(minBDEscore)
        accMin[1].append(minLTscore)
        accMin[2].append(minSEGKscore)
        accMin[3].append(minDRNEscore)
        accMin[4].append(minS2Vscore)
        accMin[5].append(minGASscore)
        accMin[6].append(minRidscore)
        
        accMax[0].append(maxBDEscore)
        accMax[1].append(maxLTscore)
        accMax[2].append(maxSEGKscore)
        accMax[3].append(maxDRNEscore)
        accMax[4].append(maxS2Vscore)
        accMax[5].append(maxGASscore)
        accMax[6].append(maxRidscore)


    BDEperf = acc[0]
    GWperf = acc[1]
    SEGKperf = acc[2]
    DRNEperf = acc[3]
    S2Vperf = acc[4]
    GASperf = acc[5]
    Ridperf = acc[6]
    x = [0,0.10,0.20,0.30]
    plt.figure(1)

    plt.xlabel("Perturbation")
    plt.ylabel("F1-score")
    plt.plot(x,BDEperf,marker='o',label=r'$\varepsilon$-BE')
    plt.plot(x,GWperf,marker='o',label="Graphwave")
    plt.plot(x,SEGKperf,marker='o',label = "SEGK")
    plt.plot(x,DRNEperf,marker='o',label="DRNE")
    plt.plot(x,S2Vperf,marker='o',label="struc2vec")
    plt.plot(x,GASperf,marker='o',label="GAS")
    plt.plot(x,Ridperf,marker='o',label=r'RID$\varepsilon$Rs')


    plt.legend(bbox_to_anchor=(0.5, 1.15),loc="upper center",ncol = 4)
    plt.grid(True)
    plt.xticks([0,0.1,0.2,0.3])
    plt.savefig("results/"+"SyntAvg.pdf")


    BDEperfMax = accMax[0]
    GWperfMax = accMax[1]
    SEGKperfMax = accMax[2]
    DRNEperfMax = accMax[3]
    S2VperfMax = accMax[4]
    GASperfMax = accMax[5]
    RidperfMax = accMax[6]
    x = [0,0.10,0.20,0.30]
    plt.figure(2)

    plt.xlabel("Perturbation")
    plt.ylabel("F1-score")
    plt.plot(x,BDEperfMax,marker='o',label=r'$\varepsilon$-BE')
    plt.plot(x,GWperfMax,marker='o',label="Graphwave")
    plt.plot(x,SEGKperfMax,marker='o',label = "SEGK")
    plt.plot(x,DRNEperfMax,marker='o',label="DRNE")
    plt.plot(x,S2VperfMax,marker='o',label="struc2vec")
    plt.plot(x,GASperfMax,marker='o',label="GAS")
    plt.plot(x,RidperfMax,marker='o',label=r'RID$\varepsilon$Rs')

    plt.legend(bbox_to_anchor=(0.5, 1.15),loc="upper center",ncol = 4)
    plt.xticks([0,0.1,0.2,0.3])
    plt.grid(True)
    plt.savefig("results/"+"SyntMax.pdf")


    BDEperfMin = accMin[0]
    GWperfMin = accMin[1]
    SEGKperfMin = accMin[2]
    DRNEperfMin = accMin[3]
    S2VperfMin = accMin[4]
    GASperfMin = accMin[5]
    RidperfMin = accMin[6]
    x = [0,0.10,0.20,0.30]
    plt.figure(3)

    plt.xlabel("Perturbation")
    plt.ylabel("F1-score")
    plt.plot(x,BDEperfMin,marker='o',label=r'$\varepsilon$-BE')
    plt.plot(x,GWperfMin,marker='o',label="Graphwave")
    plt.plot(x,SEGKperfMin,marker='o',label = "SEGK")
    plt.plot(x,DRNEperfMin,marker='o',label="DRNE")
    plt.plot(x,S2VperfMin,marker='o',label="struc2vec")
    plt.plot(x,GASperfMin,marker='o',label="GAS")
    plt.plot(x,RidperfMin,marker='o',label=r'RID$\varepsilon$Rs')

    plt.legend(bbox_to_anchor=(0.5, 1.15),loc="upper center",ncol = 4)
    plt.grid(True)
    plt.xticks([0,0.1,0.2,0.3])
    plt.savefig("results/"+"SyntMin.pdf")