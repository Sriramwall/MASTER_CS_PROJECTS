import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sigmoid_value = 1 / (1 + np.exp(-1 * np.asarray(z)))

    return  sigmoid_value # your code here


def preprocess():
    """ Input:
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

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    #print(mat.keys())
    # print(mat["train0"])

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    train_data = []
    train_data.extend(mat["train0"])
    train_data.extend(mat["train1"])
    train_data.extend(mat["train2"])
    train_data.extend(mat["train3"])
    train_data.extend(mat["train4"])
    train_data.extend(mat["train5"])
    train_data.extend(mat["train6"][:len(mat["train6"]) - 2000 ])
    train_data.extend(mat["train7"][:len(mat["train7"]) - 2000 ])
    train_data.extend(mat["train8"][:len(mat["train8"]) - 2000 ])
    train_data.extend(mat["train9"][:(50000-48051)])
    train_data = np.asarray(train_data)
    print(train_data.shape)

    train_label = []
    train_label.extend([0 for x in mat["train0"]])
    train_label.extend([1 for x in mat["train1"]])
    train_label.extend([2 for x in mat["train2"]])
    train_label.extend([3 for x in mat["train3"]])
    train_label.extend([4 for x in mat["train4"]])
    train_label.extend([5 for x in mat["train5"]])
    train_label.extend([6 for x in range(0, len(mat["train6"]) - 2000 )])
    train_label.extend([7 for x in range(0, len(mat["train7"]) - 2000 )])
    train_label.extend([8 for x in range(0, len(mat["train8"]) - 2000 )])
    train_label.extend([9 for x in range(0, (50000-48051))])
    train_label = np.asarray(train_label)
    print(train_label.shape)


    validation_data = []
    validation_data.extend(mat["train0"][:1000])
    validation_data.extend(mat["train1"][:1000])
    validation_data.extend(mat["train2"][:1000])
    validation_data.extend(mat["train3"][:1000])
    validation_data.extend(mat["train4"][:1000])
    validation_data.extend(mat["train5"][:1000])
    validation_data.extend(mat["train6"][:1000])
    validation_data.extend(mat["train7"][:1000])
    validation_data.extend(mat["train8"][:1000])
    validation_data.extend(mat["train9"][:1000])
    validation_data = np.asarray(validation_data)
    print("validation data",validation_data.shape)

    validation_label = []
    validation_label.extend([0 for x in range(1000)])
    validation_label.extend([1 for x in range(1000)])
    validation_label.extend([2 for x in range(1000)])
    validation_label.extend([3 for x in range(1000)])
    validation_label.extend([4 for x in range(1000)])
    validation_label.extend([5 for x in range(1000)])
    validation_label.extend([6 for x in range(1000)])
    validation_label.extend([7 for x in range(1000)])
    validation_label.extend([8 for x in range(1000)])
    validation_label.extend([9 for x in range(1000)])

    
    validation_label = np.asarray(validation_label)
    print("validation label",validation_label.shape)

    test_data = []
    test_data.extend(mat["test0"])
    test_data.extend(mat["test1"])
    test_data.extend(mat["test2"])
    test_data.extend(mat["test3"])
    test_data.extend(mat["test4"])
    test_data.extend(mat["test5"])
    test_data.extend(mat["test6"])
    test_data.extend(mat["test7"])
    test_data.extend(mat["test8"])
    test_data.extend(mat["test9"])
    test_data = np.asarray(test_data)
    print(test_data.shape)

    test_label = []
    test_label.extend([0 for x in mat["test0"]])
    test_label.extend([1 for x in mat["test1"]])
    test_label.extend([2 for x in mat["test2"]])
    test_label.extend([3 for x in mat["test3"]])
    test_label.extend([4 for x in mat["test4"]])
    test_label.extend([5 for x in mat["test5"]])
    test_label.extend([6 for x in mat["test6"]])
    test_label.extend([7 for x in mat["test7"]])
    test_label.extend([8 for x in mat["test8"]])
    test_label.extend([9 for x in mat["test9"]])
    test_label = np.asarray(test_label)
    print("test label",test_label.shape)

    similar_cols = np.all(train_data == train_data[0, :], axis=0)

    matches_true = np.where(True == similar_cols)
    global matches_false
    matches_false = np.where(False == similar_cols)

    t = np.asarray(matches_true[0])
    train_data = np.delete(train_data,t,axis=1)
    test_data = np.delete(test_data,t,axis=1)
    validation_data = np.delete(validation_data,t,axis=1)

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    # Feedforward Pass
    training_data_tmp = []
    for x in training_data:
      training_data_tmp.append(np.append(x,1))
    training_data_tmp = np.asarray(training_data_tmp)

    aj = training_data_tmp.dot(np.transpose(w1))
    z =  sigmoid(aj)

    z_bias = []
    for x in z:
      z_bias.append(np.append(x,1))
    z_bias = np.asarray(z_bias)

    bl = z_bias.dot(np.transpose(w2))
    ol = sigmoid(bl)

    y = np.zeros((len(training_data), n_class))


    for i in range(len(training_data)):
        y[i][(training_label[i])] = 1

    delta_output = ol - y
    deltaj_deltaw = (1 / len(train_data)) * (np.dot(delta_output.transpose(), z_bias) + np.multiply(lambdaval, w2))

    deltaProd = np.dot(delta_output, np.delete(w2, len(w2.T) - 1, axis = 1))
    grad_w1 = (1 / len(train_data)) * (np.dot(np.multiply(z * (1 - z), deltaProd).transpose(),
                     training_data_tmp) + (lambdaval * w1)) 

    reg_param = (lambdaval / float(2 * (y.shape[0]))) * (np.sum(w1 * w1) + np.sum(w2 * w2))
    loss_mat_input = np.add(np.multiply(np.subtract(1.0, y), np.log(np.subtract(1.0, ol))) , np.multiply(y, np.log(ol)))
    loss_matrix = np.sum(loss_mat_input) * (1 / float(y.shape[0]))
    obj_val = (-1 * loss_matrix) + reg_param

    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), deltaj_deltaw.flatten()), 0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    
    # Your code here

    data_tmp = []
    for x in data:
      data_tmp.append(np.append(x,1))

    data = np.asarray(data_tmp)
    z1 = data.dot(np.transpose(w1))
    hidden_out = sigmoid(z1)

    data_tmp = []
    for x in hidden_out:
      data_tmp.append(np.append(x,1))

    hidden_out = np.asarray(data_tmp)
    z2 = hidden_out.dot(np.transpose(w2))
    final_out = sigmoid(z2)

    labels = np.argmax(final_out, axis=1)

    return labels

"""**************Neural Network Script Starts here********************************"""

import datetime
print(datetime.datetime.now())

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 180

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 10

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

#CG - Minimization of scalar function of one or more variables using the conjugate gradient algorithm.
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
# print(nn_params)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
print(datetime.datetime.now())
# pickle file creation
# pickle_out = open("params.pickle","wb")
# # pickle.dump(n_hidden, pickle_out)
# pickle_out.dump(, pickle_out)
# pickle_out.close()
import pickle
data = {"selected_features" : matches_false, "n_hidden":n_hidden,"w1":w1,"w2":w2,"lambda":lambdaval}

with open('params.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)