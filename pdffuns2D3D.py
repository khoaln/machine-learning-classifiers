import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys
import pdb

def norm2D(mu,Sgm,X1,X2):

    [n,m]=np.shape(X1)
    p=np.zeros(np.shape(X1))
    for i in np.arange(0, n):
        for j in np.arange(0, m):
            x=np.array((X1[i][j],X2[i][j]))
            p[i][j] = 1 / ((2 * np.pi) * np.sqrt(np.linalg.det(Sgm))) * \
                np.exp(-1 / 2 * np.dot(np.dot((x - mu).T,np.linalg.inv(Sgm)),(x - mu)))

    return p

def norm3D(mu,Sgm,X1,X2,X3):
    
    [n,m]=np.shape(X1)
    p=np.zeros(np.shape(X1))
    for i in np.arange(0, n):
        for j in np.arange(0, m):
            x=np.array((X1[i][j],X2[i][j],X3[i][j]))
            p[i][j]=1 / (np.power(2*np.pi,3/2) * np.sqrt(np.linalg.det(Sgm))) * \
                np.exp(-1 / 2 * np.dot(np.dot((x - mu).T,np.linalg.inv(Sgm)),(x - mu)))

    return p

def Parzen2D(h1,N,Xi,x1vector,x2vector):
    hN=h1/np.sqrt(N)
    
    # apply norm2D to compute the estimated density function values
    Sgm=np.array([[hN*hN, 0], [0, hN*hN]])
    pN=np.zeros(np.shape(x1vector))
    for i in np.arange(0, N):
        mu=Xi[i]
        p=norm2D(mu,Sgm,x1vector,x2vector)
        pN=pN+p
    return pN/N

def Parzen3D(h1,N,Xi,x1vector,x2vector,x3vector):
    hN=h1/np.sqrt(N)
    
    # apply norm3D to compute the estimated density function values
    Sgm=np.array([[hN*hN, 0, 0], [0, hN*hN, 0], [0, 0, hN*hN]])
    pN=np.zeros(np.shape(x1vector))
    for i in np.arange(0, N):
        mu=Xi[i]
        p=norm3D(mu,Sgm,x1vector,x2vector,x3vector)
        pN=pN+p
    return pN/N

def KNN2D(kN,N,Xi,x1vector,x2vector):
    [n,m]=np.shape(x1vector)
    p=np.zeros(np.shape(x1vector))
    for i in np.arange(0, n):
        for j in np.arange(0, m):
            x=np.array((x1vector[i][j],x2vector[i][j]))
            # Determine Vn
            Varray=np.zeros(N)
            for k in np.arange(0,N):
                Varray[k]=np.pi*np.square(np.dot((x-Xi[k]).T,x-Xi[k]))
            Varray.sort()
            VN=Varray[kN-1]
            if VN==0:
                p[i][j]=np.inf
            else:
                p[i][j]=(kN/N)/VN
            
    return p

def KNN3D(kN,N,Xi,x1vector,x2vector,x3vector):
    [n,m]=np.shape(x1vector)
    p=np.zeros(np.shape(x1vector))
    for i in np.arange(0, n):
        for j in np.arange(0, m):
            x=np.array((x1vector[i][j],x2vector[i][j],x3vector[i][j]))
            # Determine Vn
            Varray=np.zeros(N)
            for k in np.arange(0,N):
                Varray[k]=np.pi*np.square(np.dot((x-Xi[k]).T,x-Xi[k]))
            Varray.sort()
            VN=Varray[kN-1]
            if VN==0:
                p[i][j]=np.inf
            else:
                p[i][j]=(kN/N)/VN
            
    return p
    
