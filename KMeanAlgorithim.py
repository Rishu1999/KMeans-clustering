from copy import  deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
#plt.style.use('ggplot')
#plt.rcParams['figure.figsize']=(8,6)
#importing the datasets
data=pd.read_csv('KmeansData.csv')
print("Input Data and shape")
print(data.shape)#(3000,2)
print(data.head())#first 5 rows
#getting the the value as  ploting ist
f1=data['V1'].values#core value
f2=data['V2'].values#core value
#print("np.mean(f1)=",np.mean(f1))
plt.scatter(f1,f2,c='blue',s=10)#for dot ploting of data
plt.show()
X=np.array(list(zip(f1,f2)))#X=input data
#[[2.07345,-3.241693]]
print(X)
#a=12,7
#b=5,3
#Eucladian Distance Calculator
def eu_dist(a,b,ax=1):
    return np.linalg.norm(a-b,axis=ax)
#number of cluster
k=3
#X cordinates of random centroids
C_x=np.random.randint(0,np.max(X)-20,size=k)#for genarate the 20 pixel before
print("C_x=",C_x)
#Y cordinates of random centroids
C_y=np.random.randint(0,np.max(X)-20,size=k)
print("C_y=",C_y)
C=np.array(list(zip(C_x,C_y)),dtype=np.float32)
print('initial centroids and (random positions):')
print(C)
print(C.shape)#(3,2)
#ploting along with centroids
plt.scatter(f1,f2,s=10,c='k')
plt.scatter(C_x,C_y,marker='*',s=500,c='r')
plt.show()

#to sto the value of centroids when its updates
C_old=np.zeros(C.shape)
print('C=\n',C)
print('C_old\n',C_old)
print('len(x)=',len(X))
#cluster labels(0,1,3)
clusters=np.zeros(len(X))#array of 3000 value prefilled with  zero
#zero filled numpy array of 3000 elementes
print('cluster:=',clusters)
#error function distance between two centroids
#and old centroids
error=eu_dist(C,C_old)
print("error before loop",error)
#loop will run till the error becomes become Zero
while error.all():#error !=0
    #assigning each value to its closest Cluster
    for i in range(len(X)):
        distance=eu_dist(X[i],C)
        cluster=np.argmin(distance)
        clusters[i]=cluster
        #sorting old centroids value
    C_old=deepcopy(C)
    #Finding the new centroids by taking average value
    for i in range(k):#beacuse w e have to find the new centrod value
        points=[X[j] for j in range(len(X)) if clusters[j]==i]
        C[i]=np.mean(points,axis=0)
    error=eu_dist(C,C_old)
    print("error in loop",error)
colors=['b','c','r']
fig,ax=plt.subplots()
for i in range(k):
    points=np.array([X[j] for j in range(len(X))if clusters[j]==i])
    ax.scatter(points[:,0],points[:,1],s=25,c=colors[i])
ax.scatter(C[:,0],C[:,1],marker='*',s=100,c='y')
print('final centroids:',C)
plt.show()






