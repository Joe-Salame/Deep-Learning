from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import math

# Données de test
mean1 = [4, 4]
cov1 = [[1, 0], [0, 1]] #
data1 = np.transpose(np.random.multivariate_normal(mean1, cov1, 128))
mean2 = [-4, -4]
cov2 = [[4, 0], [0, 4]] #
data2 = np.transpose(np.random.multivariate_normal(mean2, cov2, 128))
data=np.concatenate((data1, data2), axis=1)
oracle=np.concatenate((np.zeros(128),np.ones(128)))
test1=np.transpose(np.random.multivariate_normal(mean1, cov1, 64))
test2=np.transpose(np.random.multivariate_normal(mean2, cov2,64))
test=np.concatenate((test1,test2), axis=1)
oracle=np.concatenate((np.zeros(128),np.ones(128)))

def kppv(test,data,oracle,k):
    clas=np.zeros(len(test[0]))#initialisatsion d'un matrice clas contenant des 0 pour memorise la prediction de la classe de chaque point
            
    for i in range(0,len(test[0])):
        result = [] #initialisatsion d'un tableau result pour memoriser les distances euclidiennes d'un point a predire avec tout les points des donnees d'apprentissages
        for j in range(0,len(data[0])):
            d=euclidean_distance(test[0][i],data[0][j],test[1][i],data[1][j])#calcule de la distance euclidienne entre les deux points
            result.append((d,oracle[j]))#memoriser le resultat du calcul de distance precedent avec le classe du point du donne d'apprentissage dans le tableau result
        result.sort()#trier le tableau result pour prendre le K plus petits distances
        cl0 = 0#conteur pour le class 0
        cl1 = 0#conteur pour le class 1
        
        for x in result[0:k]: #on itere dans les k points dans le tableau result pour calucler le nombre de chaque class entre ces k points
            if x[1] == 1:
                cl1+=1
            else:
                cl0+=1
        
        if cl1>cl0:#si le nombre de class1>de classe 0 donc on predire class 1
            clas[i]=1
        else:#sinon classe on predire class 0
            clas[i]=0
        
    
    return clas

def euclidean_distance(x1, x2, y1, y2):#fonction pour calculer la distiance euclidienne entre deux points
    distance = np.sqrt(((x1-x2)**2)+((y1-y2)**2))
    return distance 

def affiche_classe(x,clas,K):   
    for k in range(0,K):
        ind=(clas==k)
        plt.plot(x[0,ind],x[1,ind],"o")
    plt.show()

K=3
clas=kppv(test,data,oracle,K)
affiche_classe(test,clas,2)