
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt


# In[ ]:


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0
    
    print("train_data size ",train_data.shape)
    print("test size  ",test_data.shape)
    print("validation size  ",validation_data.shape)

    return train_data, train_label, validation_data, validation_label, test_data, test_label


# In[ ]:


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# In[1]:


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data   
    train_data_with_bias = np.concatenate((train_data, (np.ones((n_data, 1)))), axis=1)
    mul = np.dot(train_data_with_bias, initialWeights)
    sigma_val = sigmoid(mul)
    sigma_val.resize(len(sigma_val),1)
    tmp1 = np.multiply(labeli, np.log(sigma_val))
    tmp2 = np.multiply(np.subtract(1, labeli), np.log(np.subtract(1, sigma_val)))
    error = - (1/n_data) * np.sum(tmp1 + tmp2)

    #error_grad calculation
    tmp1 = np.subtract(sigma_val, labeli)
    tmp2 = np.dot(train_data_with_bias.T, tmp1)
    error_grad = np.multiply(1/n_data, tmp2)
    error_grad = error_grad.flatten()
    return error, error_grad


# In[2]:


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    data_with_bias = np.concatenate((data, (np.ones((data.shape[0], 1)))), axis=1)
    tmp = np.dot(data_with_bias, W)
    label = np.argmax(tmp, axis=1)
    label.resize(len(data),1)
    return label


# In[3]:


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    initialWeights_b = params.reshape(n_feature + 1, n_class)
    train_data_with_bias = np.concatenate((train_data, (np.ones((n_data, 1)))), axis=1)
    tmp1 = np.exp(np.dot(train_data_with_bias, initialWeights_b))
    tmp2 = np.sum(tmp1, axis=1)
    tmp2.resize(len(tmp2),1)
    sigma_val = np.divide(tmp1,tmp2)
    tmp4 = np.multiply(labeli,np.log(sigma_val))
    error = -np.sum(tmp4)/n_data

    tmp1 = np.subtract(sigma_val,labeli)
    error_grad = np.dot(tmp1.T,train_data_with_bias).T/n_data
    error_grad = error_grad.flatten()

    return error, error_grad


# In[4]:


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    data_with_bias = np.insert(data, len(train_data.T), 1, axis=1)
    tmp1 = np.dot(data_with_bias,W)
    label = np.argmax(tmp1, axis=1)
    label.resize(len(label),1)
    return label


# In[ ]:


# """
# Script for Logistic Regression
# """
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

test_corrected = [0,0,0,0,0,0,0,0,0,0,0,0]
test_incorrect = [0,0,0,0,0,0,0,0,0,0,0,0]

for index,x in enumerate(predicted_label):
    v = x[0] 
    if v  == train_label[index]:
#         print(test_corrected,v)
        test_corrected[v] = test_corrected[v] + 1
    else:
        test_incorrect[v] = test_incorrect[v] + 1
print("test corect",test_corrected)
print("test incorrect",test_incorrect)

acc =  []

print('\n The Training Accuracy for the 10 digits are: ')
for x in range(10):
  acc.append(test_corrected[x] / (test_corrected[x] +test_incorrect[x] ))
print(acc)

  
# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

test_corrected = [0,0,0,0,0,0,0,0,0,0,0,0]
test_incorrect = [0,0,0,0,0,0,0,0,0,0,0,0]

for index,x in enumerate(predicted_label):
    v = x[0] 
    if v  == validation_label[index]:
#         print(test_corrected,v)
        test_corrected[v] = test_corrected[v] + 1
    else:
        test_incorrect[v] = test_incorrect[v] + 1
print("test corect",test_corrected)
print("test incorrect",test_incorrect)

acc =  []

print('\n The Validation Accuracy for the 10 digits are: ')

for x in range(10):
  acc.append(test_corrected[x] / (test_corrected[x] +test_incorrect[x] ))
print(acc)

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


test_corrected = [0,0,0,0,0,0,0,0,0,0,0,0]
test_incorrect = [0,0,0,0,0,0,0,0,0,0,0,0]

for index,x in enumerate(predicted_label):
    v = x[0] 
    if v  == test_label[index]:
#         print(test_corrected,v)
        test_corrected[v] = test_corrected[v] + 1
    else:
        test_incorrect[v] = test_incorrect[v] + 1
print("test corect",test_corrected)
print("test incorrect",test_incorrect)

acc =  []

print('\n The Testing Accuracy for the 10 digits are: ')

for x in range(10):
  acc.append(test_corrected[x] / (test_corrected[x] +test_incorrect[x] ))
print(acc)

# In[ ]:


"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

test_corrected = [0,0,0,0,0,0,0,0,0,0,0,0]
test_incorrect = [0,0,0,0,0,0,0,0,0,0,0,0]

for index,x in enumerate(predicted_label_b):
    v = x[0] 
    if v  == train_label[index]:
#         print(test_corrected,v)
        test_corrected[v] = test_corrected[v] + 1
    else:
        test_incorrect[v] = test_incorrect[v] + 1
print("test corect",test_corrected)
print("test incorrect",test_incorrect)

acc =  []
print('\n The Training Accuracy for the 10 digits are: ')

for x in range(10):
  acc.append(test_corrected[x] / (test_corrected[x] +test_incorrect[x] ))
print(acc)

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

test_corrected = [0,0,0,0,0,0,0,0,0,0,0,0]
test_incorrect = [0,0,0,0,0,0,0,0,0,0,0,0]

for index,x in enumerate(predicted_label_b):
    v = x[0] 
    if v  == validation_label[index]:
#         print(test_corrected,v)
        test_corrected[v] = test_corrected[v] + 1
    else:
        test_incorrect[v] = test_incorrect[v] + 1
print("test corect",test_corrected)
print("test incorrect",test_incorrect)

acc =  []

print('\n The Validation Accuracy for the 10 digits are: ')

for x in range(10):
  acc.append(test_corrected[x] / (test_corrected[x] +test_incorrect[x] ))
print(acc)

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

test_corrected = [0,0,0,0,0,0,0,0,0,0,0,0]
test_incorrect = [0,0,0,0,0,0,0,0,0,0,0,0]

for index,x in enumerate(predicted_label_b):
    v = x[0] 
    if v  == test_label[index]:
#         print(test_corrected,v)
        test_corrected[v] = test_corrected[v] + 1
    else:
        test_incorrect[v] = test_incorrect[v] + 1
print("test corect",test_corrected)
print("test incorrect",test_incorrect)

acc =  []

print('\n The Testing Accuracy for the 10 digits are: ')

for x in range(10):
  acc.append(test_corrected[x] / (test_corrected[x] +test_incorrect[x] ))
print(acc)





# mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
# print(mat.keys())
# print(mat["train0"])


index = np.random.choice(train_data.shape[0], 10000, replace=False)  

# Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
# Your code here.
# train_data = []
# # train_data.extend(mat["train8"][1800:])
# # train_data.extend(mat["train9"])
# train_data.extend(mat["train0"][:1000])
# train_data.extend(mat["train1"][:1000])
# train_data.extend(mat["train2"][:1000])
# train_data.extend(mat["train3"][:1000])
# train_data.extend(mat["train4"][:1000])
# train_data.extend(mat["train5"][:1000])
# train_data.extend(mat["train6"][:1000])
# train_data.extend(mat["train7"][:1000])
# train_data.extend(mat["train8"][:1000])
# train_data.extend(mat["train9"][:1000])
# train_data = np.asarray(train_data)
train_data = train_data[index]
print("train data",train_data.shape)

# train_label = []
# #mat["train8"][1800:]
# train_label.extend([0 for x in range(1000)])
# train_label.extend([1 for x in range(1000)])
# train_label.extend([2 for x in range(1000)])
# train_label.extend([3 for x in range(1000)])
# train_label.extend([4 for x in range(1000)])
# train_label.extend([5 for x in range(1000)])
# train_label.extend([6 for x in range(1000)])
# train_label.extend([7 for x in range(1000)])
# train_label.extend([8 for x in range(1000)])
# train_label.extend([9 for x in range(1000)])

# train_label.extend([8 for x in range(1800,len(mat["train8"]))])
# train_label.extend([9 for x in mat["train9"]])
# train_label = np.asarray(train_label)
train_label = train_label[index]

print("train label",train_label.shape)
print(train_label)




"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
train_label = train_label.ravel()
validation_label = validation_label.ravel()
test_label = test_label.ravel()
print('--------------SVM-------------------\n')

# Implementing the Linear kernel using inbuilt function
print('Linear SVM: \n')
model = svm.SVC(kernel='linear')
model.fit(train_data,train_label)
values_predicted = model.predict(train_data);
print('\n Training set Accuracy:' + str(100*np.mean((values_predicted == train_label).astype(float))) + '%')
values_predicted = model.predict(validation_data);
print('\n Validation set Accuracy:' + str(100*np.mean((values_predicted == validation_label).astype(float))) + '%')
values_predicted = model.predict(test_data);
print('\n Testing set Accuracy:' + str(100*np.mean((values_predicted == test_label).astype(float))) + '%')

# Kernel used is Rbf
# using Gamma = 1 and keeping other parameters default
print('RBF, gamma = 1  SVM: \n')
model = svm.SVC(kernel ='rbf',gamma=1)
model.fit(train_data,train_label)
values_predicted = model.predict(train_data);
print('\n Training set Accuracy:' + str(100*np.mean((values_predicted == train_label).astype(float))) + '%')
values_predicted = model.predict(validation_data);
print('\n Validation set Accuracy:' + str(100*np.mean((values_predicted == validation_label).astype(float))) + '%')
values_predicted = model.predict(test_data);
print('\n Testing set Accuracy:' + str(100*np.mean((values_predicted == test_label).astype(float))) + '%')

#Kernel used is RBF
# Using Gamma =Default and C = 1
print('RBF, gamma = default, C = 1 SVM: \n')
model = svm.SVC(kernel='rbf')
model.fit(train_data,train_label)
values_predicted = model.predict(train_data);
print('\n Training set Accuracy:' + str(100*np.mean((values_predicted == train_label).astype(float))) + '%')
values_predicted = model.predict(validation_data);
print('\n Validation set Accuracy:' + str(100*np.mean((values_predicted == validation_label).astype(float))) + '%')
values_predicted = model.predict(test_data);
print('\n Testing set Accuracy:' + str(100*np.mean((values_predicted == test_label).astype(float))) + '%')

#Kernel used is RBF
#Using gamma Default, 
#varying value of C (1, 10, 20,..100)
print('RBF, gamma = default, varying C values 10-100 SVM: \n')
for i in range(10,101,10):
    model = svm.SVC(C=i,kernel='rbf')
    model.fit(train_data,train_label)
    values_predicted = model.predict(train_data);
    print('C =' + str(i)+'\n')
    print('\n Training set Accuracy:' + str(100*np.mean((values_predicted == train_label).astype(float))) + '%')
    values_predicted = model.predict(validation_data);
    print('\n Validation set Accuracy:' + str(100*np.mean((values_predicted == validation_label).astype(float))) + '%')
    values_predicted = model.predict(test_data);
    print('\n Testing set Accuracy:' + str(100*np.mean((values_predicted == test_label).astype(float))) + '%')