import h5py
import numpy as np
from DataScienceProjects.simple_neural_net import *

def initialization_results():

    W1 = [[ 0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388],
          [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
          [-0.01313865, 0.00884622, 0.00881318, 0.01709573, 0.00050034],
          [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]]
    b1 = [[ 0.], [ 0.], [ 0.], [ 0.]]
    W2 = [[-0.01185047, -0.0020565, 0.01486148, 0.00236716],
          [-0.01023785, -0.00712993, 0.00625245, -0.00160513],
          [-0.00768836, -0.00230031, 0.00745056, 0.01976111]]
    b2 = [[ 0.], [ 0.], [ 0.]]

    return {"W1":W1, "b1":b1, "W2":W2, "b2":b2}


def linear_forward_results():

    Z = [[ 3.26295337, -1.23429987]]

    return {"Z":Z}


def lin_act_forward_results():

    A_sigmoid = [[ 0.96890023, 0.11013289]]
    A_relu = [[ 3.43896131, 0.]]

    return {"A_sigmoid":A_sigmoid, "A_relu":A_relu}


def L_model_forward_results():

    AL = [[ 0.03921668, 0.70498921, 0.19734387, 0.04728177]]
    len_caches = 3

    return {"AL":AL, "len_caches":len_caches}


def cost_results():

    cost = 0.2797765635793423

    return {"cost":cost}


def linear_backward_results():

    dA_prev = [[-1.15171336, 0.06718465, -0.3204696, 2.09812712],
               [ 0.60345879, -3.72508701, 5.81700741, -3.84326836],
               [-0.4319552, -1.30987417, 1.72354705, 0.05070578],
               [-0.38981415, 0.60811244, -1.25938424, 1.47191593],
               [-2.52214926, 2.67882552, -0.67947465, 1.48119548]]
    dW = [[ 0.07313866, -0.0976715, -0.87585828, 0.73763362, 0.00785716],
          [ 0.85508818,  0.37530413, -0.59912655, 0.71278189, -0.58931808],
          [ 0.97913304, -0.24376494, -0.08839671, 0.55151192, -0.10290907]]
    db = [[-0.14713786], [-0.11313155], [-0.13209101]]

    return {"dA_prev":dA_prev, "dW":dW, "db":db}


def lin_act_backward_results():
    
    dA_prev_sigmoid = [[0.11017994, 0.01105339],
                       [0.09466817, 0.00949723],
                       [-0.05743092, -0.00576154]]
    dW_sigmoid = [[0.10266786, 0.09778551, -0.01968084]]
    db_sigmoid = [[-0.05729622]]

    dA_prev_relu = [[0.44090989, -0.],
                    [0.37883606, -0.],
                    [-0.2298228, 0.]]
    dW_relu = [[ 0.44513824,  0.37371418, -0.10478989]]
    db_relu = [[-0.20837892]]

    return {"dA_prev_sigmoid":dA_prev_sigmoid, "dW_sigmoid":dW_sigmoid, "db_sigmoid":db_sigmoid,
            "dA_prev_relu":dA_prev_relu, "dW_relu":dW_relu, "db_relu":db_relu}


def L_model_backward_results():

    dW1 = [[ 0.41010002, 0.07807203, 0.13798444, 0.10502167],
           [ 0., 0., 0., 0.],
           [ 0.05283652, 0.01005865, 0.01777766, 0.0135308 ]]
    db1 = [[-0.22007063], [ 0.], [-0.02835349]]
    dA1 = [[ 0.12913162, -0.44014127],
           [-0.14175655, 0.48317296],
           [ 0.01663708, -0.05670698]]

    return {"dW1":dW1, "db1":db1, "dA1":dA1}


def update_parameters_results():

    W1 = [[-0.59562069, -0.09991781, -2.14584584, 1.82662008],
          [-1.76569676, -0.80627147, 0.51115557, -1.18258802],
          [-1.0535704, -0.86128581, 0.68284052, 2.20374577]]
    b1 = [[-0.04659241], [-1.28888275], [ 0.53405496]]
    W2 = [[-0.55569196, 0.0354055, 1.32964895]]
    b2 = [[-0.84610769]]

    return {"W1":W1, "b1":b1, "W2":W2, "b2":b2}


def load_data():
    
    train_dataset = h5py.File('../data/test_data/train_catvnoncat.h5', "r")
    train_x= np.array(train_dataset["train_set_x"][:])
    train_y = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('../data/test_data/test_catvnoncat.h5', "r")
    test_x = np.array(test_dataset["test_set_x"][:])
    test_y = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])
    
    train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))
    
    return train_x, train_y, test_x, test_y, classes


def predict(X, y, parameters, accuracy=False):

    m = X.shape[1]
    n = len(parameters) // 2
    
    probs, caches = L_model_forward(X, parameters, hidden_active='relu', outer_active='sigmoid')

    p = np.where(probs > 0.5, 1, 0)

    if accuracy:
        return p, np.sum((p == y)/m)
    else:    
        return p


def predict_results():

    accuracy_train = 0.985645933014
    accuracy_test = 0.8

    return {"accuracy_train":accuracy_train, "accuracy_test":accuracy_test}


def predict_reg(X, parameters, hidden_active="relu", outer_active="sigmoid"):
    
    y_est, caches = L_model_forward(X, parameters, hidden_active=hidden_active, outer_active=outer_active)

    return y_est


