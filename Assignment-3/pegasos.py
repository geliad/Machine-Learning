import json
import numpy as np


###### Q1.1 ######
def objective_function(X, y, w, lamb):
    """
    Inputs:
    - Xtrain: A 2 dimensional numpy array of data (number of samples x number of features)
    - ytrain: A 1 dimensional numpy array of labels (length = number of samples )
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm

    Return:
    - train_obj: the value of objective function in SVM primal formulation
    """
    # you need to fill in your solution here

    N = len(y)

    L2 = np.linalg.norm(w, ord=2)

    zeroes = np.zeros(N)
    wTX = np.dot(w.T, X.T)
    ywTx = np.multiply(y, wTX)
    ywTx_max = np.maximum(zeroes, 1 - ywTx)
    summation = np.sum(ywTx_max)

    obj_value = (lamb / 2.0) * np.power(L2, 2) + (1 / N) * summation
    return obj_value


###### Q1.2 ######
def pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations):
    """
    Inputs:
    - Xtrain: A list of num_train elements, where each element is a list of D-dimensional features.
    - ytrain: A list of num_train labels
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm
    - k: mini-batch size
    - max_iterations: the maximum number of iterations to update parameters

    Returns:
    - learnt w
    - traiin_obj: a list of the objective function value at each iteration during the training process, length of 500.
    """
    np.random.seed(0)
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    N = Xtrain.shape[0]
    D = Xtrain.shape[1]

    train_obj = []

    for iter in range(1, max_iterations + 1):
        A_t = np.floor(np.random.rand(k) * N).astype(int)  # index of the current mini-batch

        # Select X and y from indices
        X = Xtrain[A_t]
        y = ytrain[A_t]

        # Step 4
        wTx = np.dot(w.T, X.T)
        ywTx = np.multiply(y, wTx)
        A_t_plus = np.where(ywTx < 1)

        X_plus = X[A_t_plus[1]]
        y_plus = y[A_t_plus[1]]

        # Step 5
        learning_rate = 1.0 / (lamb * iter)

        # Step 6
        param_update1 = np.multiply(1 - learning_rate * lamb, w)
        param_update2 = np.sum(np.multiply(y_plus, X_plus.T).T, axis=0)
        param_update2 = (learning_rate / k) * param_update2.reshape(D, 1)
        w_t_half = param_update1 + param_update2

        # Step 7
        L2 = np.linalg.norm(w_t_half, ord=2)
        w = min(1, (1.0 / np.sqrt(lamb)) / L2) * w_t_half

        obj_value = objective_function(X, y, w, lamb)
        train_obj.append(obj_value)

    return w, train_obj


###### Q1.3 ######
def pegasos_test(Xtest, ytest, w, t=0.):
    """
    Inputs:
    - Xtest: A list of num_test elements, where each element is a list of D-dimensional features.
    - ytest: A list of num_test labels
    - w_l: a numpy array of D elements as a D-dimension vector, which is the weight vector of SVM classifier and learned by pegasos_train()
    - t: threshold, when you get the prediction from SVM classifier, it should be real number from -1 to 1. Make all prediction less than t to -1 and otherwise make to 1 (Binarize)

    Returns:
    - test_acc: testing accuracy.
    """
    # you need to fill in your solution here

    y = np.array(ytest).reshape(len(ytest), 1)
    predictions = np.dot(Xtest, w)
    predictions = np.where(predictions < t, -1, 1)

    test_acc = np.sum(predictions == y) / len(y)

    return test_acc


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""


def data_loader_mnist(dataset):
    with open(dataset, 'r') as f:
        data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    ## below we add 'one' to the feature of each sample, such that we include the bias term into parameter w
    Xtrain = np.hstack((np.ones((len(Xtrain), 1)), np.array(Xtrain))).tolist()
    Xvalid = np.hstack((np.ones((len(Xvalid), 1)), np.array(Xvalid))).tolist()
    Xtest = np.hstack((np.ones((len(Xtest), 1)), np.array(Xtest))).tolist()

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = -1.
        else:
            ytrain[i] = 1.
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = -1.
        else:
            ytest[i] = 1.

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def pegasos_mnist():
    test_acc = {}
    train_obj = {}

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset='mnist_subset.json')

    max_iterations = 500
    k = 100
    for lamb in (0.01, 0.1, 1):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k,
                                                                               max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    lamb = 0.1
    for k in (1, 10, 1000):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k,
                                                                               max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    return test_acc, train_obj


def main():
    test_acc, train_obj = pegasos_mnist()  # results on mnist
    print('mnist test acc \n')
    for key, value in test_acc.items():
        print('%s: test acc = %.4f \n' % (key, value))

    with open('pegasos.json', 'w') as f_json:
        json.dump([test_acc, train_obj], f_json)


if __name__ == "__main__":
    main()
