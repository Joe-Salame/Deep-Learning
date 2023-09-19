import numpy as np
import matplotlib.pyplot as plt
import random
import math 

# Données de test
mean1 = [4, 4]
cov1 = [[1, 0], [0, 1]] #
data1 = np.transpose(np.random.multivariate_normal(mean1, cov1, 128))
mean2 = [-4, -4]
cov2 = [[4, 0], [0, 4]] #
data2 = np.transpose(np.random.multivariate_normal(mean2, cov2, 128))
data=np.concatenate((data1, data2), axis=1)
oracle=np.concatenate((np.zeros(128)-1,np.ones(128)))

def active(x,w):
    res = 0
    for i in range(0,2):
        res += x[0,i]*w[0,i+1] #Calcule la somme des produits entre les lignes x(i) avec les lignes w(i+1)(i+1 pour w car la premiere ligne contient le seuil)
    
    res += w[0,0] #Apres le calucle des produits entre x(i) w(i+1),on fait l'addition de seuil avec le resultat res
    return res #On renvoi le resultat final de res
    
    

def perceptron(x,w,active):#Pour faire la prediction
    y = active(x,w) #y obtient le resultat de la fonction d'avtivation 
    if y < 0:
        return -1 #Si l'activation donne un nombre neguative donc renvoi -1(la prediction)
    else:
        return 1 #sinon renvoi 1    

def apprentissage(data,oracle,active,perceptron):
    w = np.array([[0.5, 1.2, 0.8]]) #initialisation des parametres b avec 2 weights dans un tableau
    alpha = 0.1 #initialisation du taux d'apprentissage alpha
    x = np.array([[0,0]]) #initialisation d'un tableau (x) vide qui prend les deux entrees pour faire les calcules si la prediction est faute 
    
    error = [] #initialisation d'un tableau d'error pour le calcul de l'erreur cumulee 
    
    
    for j in range(0,100):
        error += [0] #on ajoute une case au tableau error avec chaque iteration
        for i in range(len(data[0])):
            x[0,0] = data[0][i] #memoriser l'entree dans x
            x[0,1] = data[1][i]
            h = perceptron(x,w,active) #l'appel de la fonction perceptron pour faire la prediction
            error[j]+=(oracle[i]-h)**2 #calculer l'erreur
            
            if h != oracle[i]: # si la prediction est faute on fait la mise a jour des parameteres
                w[0,1] +=  alpha * (oracle[i] - h ) * x[0,0] #mise a jour de w1
                w[0,2] +=  alpha * (oracle[i] - h ) * x[0,1]#mise a jour de w2
                w[0,0] +=  alpha * (oracle[i] - h )  #mise a jour de b(seuil)
            
    return w , error#renvoiyer les parameteres apres la mise a jour avec l'erreur cumulee pour le passage complet de l'ensemble d'apprentissage

def affiche_classe(x,clas,K,w):
    t=[np.min(x[0,:]),np.max(x[0,:])]
    z=[(-w[0,0]-w[0,1]*np.min(x[0,:]))/w[0,2],(-w[0,0]-w[0,1]*np.max(x[0,:]))/w[0,2]]
    plt.plot(t,z,color="#0F62FF");
    ind=(clas==-1)
    plt.title('Model classification')
    plt.plot(x[0,ind],x[1,ind],"o",color="#A56EFF")
    ind=(clas==1)
    plt.plot(x[0,ind],x[1,ind],"o",color="#4ED0CE")
    plt.show()

w,mdiff=apprentissage(data,oracle,active,perceptron)#l'appel de la fonction apprentissage pour apprendre

plt.title('Errors plot')
plt.plot(mdiff,color="#0F62FF")
plt.show()
affiche_classe(data,oracle,2,w)