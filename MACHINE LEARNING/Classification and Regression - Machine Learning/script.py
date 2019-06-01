#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


# In[2]:


def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    flat_y=y.ravel()

    arr = np.asarray(list(set(flat_y)))
    length=arr.size
    tmp = []
    
    means=np.zeros((X.shape[1],length))
   # print(means)
# 
    #### testing
    d = dict()
    c = dict()
#     tmp = enumerate(X)
#     print("1",X)
    for i in range(X.shape[0]):
        t = int(flat_y[i])
#             print(t)
        if t in d:
            d[t] += X[i]
            c[t] += 1
        else:
            d[t] = X[i].copy()
            c[t] = 1
#     print("2",X)
    t1 = []
    t2 = []
    for i in range(1,6):
        t1.append(d[i][0] / c[i])
        t2.append(d[i][1] / c[i])

#     print(t1,t2)
    means = np.asarray([t1,t2])
    covmat=np.cov(X,rowvar=0)
    #print(covmat)
#     print(means)
   
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    flat_y=y.ravel()
    arr = np.asarray(list(set(flat_y)))
#     print("l2",l)
    length=arr.size
    tmp = []
    
    means=np.zeros((X.shape[1],length))
    d = dict()
    c = dict()
    for i,x in enumerate(X):
        t = int(flat_y[i])
#             print(t)
        if t in d:
            d[t] += x
            c[t] += 1
        else:
            d[t] = x.copy()
            c[t] = 1
    t1 = []
    t2 = []
    for i in range(1,6):
        t1.append(d[i][0] / c[i])
        t2.append(d[i][1] / c[i])

    means = np.asarray([t1,t2])
    
    covmats=[np.zeros((X.shape[1],X.shape[1]))]*length

    for i in range(length):
        covmats[i]=np.cov(X[flat_y==arr[i]],rowvar=0)
#         print(l[i],covmats[i])
#     print("measn 2",means)
#     print("covmat 2",covmats)
#     print(X)
   
    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    l = Xtest.shape[0]
    l1 = means.shape[1]
    ypred = np.empty([Xtest.shape[0], 1])
    
    for data in range(l):
        i = 0
        j = 0
        for p in range(l1):
            m = means[:,p]
            x = Xtest[data]
            a = np.divide(np.dot(np.dot(np.transpose(np.subtract(x,m)),
                        inv(covmat)),
                np.subtract(x,m)),
            -2)
            b = np.multiply(np.power(det(covmat),0.5) ,
                    (44/7))           
            value = np.divide(np.exp(a),b)
            if value > i:
                i = value
                j = p
        ypred[data,0] = (j+1)
#     print(ypred)
    
    acc = 0;
    for i in range(Xtest.shape[0]):
        if ypred[i] == ytest[i]:
            acc += 1;
  
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    val= np.zeros((Xtest.shape[0],means.shape[1]))
    
    for i in range(means.shape[1]):
        inv = np.linalg.inv(covmats[i])
        determinant = np.linalg.det(covmats[i])
        tp = np.transpose(np.dot(inv, np.transpose(Xtest - means[:,i])))
        a = np.exp(-0.5*np.sum((Xtest - means[:,i])* tp,1))
        b = (np.sqrt(np.pi*2)*(np.power(determinant,2)))
        val[:,i] = a / b
    ypred = np.argmax(val,1)
    ypred = ypred + 1
    ytest = ytest.ravel()
    
    acc = 0;
    for i in range(Xtest.shape[0]):
        if ypred[i] == ytest[i]:
            acc += 1;
   
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD   
    w = np.dot(np.linalg.pinv(X), y)   
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD  
    iden = np.eye(X.shape[1])
    x_transpose = np.transpose(X)
    inverse_calc = np.linalg.inv(np.add(np.dot(x_transpose,X),(lambd * iden)))
    w = np.dot(np.dot(inverse_calc,x_transpose),y)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    lin_diff = np.subtract(ytest,np.dot(Xtest,w))
    mse=(np.sum(lin_diff*lin_diff))/Xtest.shape[0]
    
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 
    weight_matrix = np.asmatrix(w)
    square_error = y - np.dot(X,np.transpose(weight_matrix))
    error_transpose_error = (0.5 * (np.dot(np.transpose(square_error),square_error)))
    weight_matrix_reg = (0.5 * lambd * np.dot(weight_matrix,np.transpose(weight_matrix)))
    error_grad = (lambd * np.transpose(weight_matrix))-(np.dot(np.transpose(X),square_error))
    error_grad = np.reshape(np.array(error_grad),weight_matrix.shape[1])
    error = error_transpose_error + weight_matrix_reg

    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    Xp = np.zeros((x.shape[0],p+1))
    
    for i in range(x.shape[0]):
        for j in range(p+1):
            Xp[i,j]  = x[i] ** j
    return Xp


# In[3]:


# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()


# In[4]:


if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))


# In[5]:



# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    print("training data error:", mses3_train[i],"test data error:",mses3[i] , "lambda:" , lambd)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()


# In[6]:


# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# In[7]:


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(mses3)]# REPLACE THIS WITH lambda_opt estimated from Problem 3
print("lam vaue",lambda_opt)
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    print("training:" , mses5_train[p,1] , "test:" ,mses5[p,1] , p )
    print("training_0:" , mses5_train[p,0] , "test_0:" ,mses5[p,0] , p )
    

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()


# In[ ]:





# In[ ]:




