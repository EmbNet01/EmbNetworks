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