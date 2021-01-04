import numpy as np
import copy
import sys
import time

def initialize_parameters(layer_dims, seed, init_scale):
    """
    Randomly initialize the weights and biases for each layer

    Arguments:
    layer_dims -- list of layer dimensions
    seed -- random seed
    init_scale -- initialization scale factor
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    Wl -- weight matrix for layer l. Numpy array (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector for layer l. Numpy array (layer_dims[l], 1)
    """
    
    np.random.seed(seed)

    # Ensure that init_scale is an array
    init_scale = np.array(init_scale)

    # If init_scale is a single value, copy it for each layer
    if not init_scale.shape:
        init_scale = np.repeat(init_scale, len(layer_dims))

    parameters = {}
    L = len(layer_dims)

    # Loop through the layers to initialize weights and biases
    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * init_scale[l-1]
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear component of the forward propagation

    Arguments:
    A -- activation functions from previous layer. Numpy array (layer_dims[l-1], batch_length)
    W -- weight matrix. Numpy array (layer_dims[l], layer_dims[l-1])
    b -- bias vector. Numpy array (layer_dims[l], 1)
   
    Returns:
    Z -- the pre-activation parameter
    cache -- a dictionary containing "A", "W" and "b", used for backward propagation
    """

    Z = np.dot(W, A) + b

    cache = (A, W, b)

    # Assert that Z has the correct shape
    assert(Z.shape == (W.shape[0], A.shape[1]))

    return Z, cache


def sigmoid(Z):
    """
    Implement the sigmoid activation function

    Arguments:
    Z -- the pre-activation parameter
    
    Returns:
    A -- the post-activation parameter
    cache -- a dictionary containing "Z", used for backward propagation
    """

    A = 1.0 / (1.0 + np.exp(-Z))

    cache = (Z)

    return A, cache


def relu(Z):
    """
    Implement the RELU activation function

    Arguments:
    Z -- the pre-activation parameter
    
    Returns:
    A -- the post-activation parameter
    cache -- a dictionary containing "Z", used for backward propagation
    """

    A = np.where(Z>0, Z, 0)

    cache = (Z)
        
    return A, cache


def identity(Z):
    """
    Implement the identity activation function

    Arguments:
    Z -- the pre-activation parameter
    
    Returns:
    A -- the post-activation parameter
    cache -- a dictionary containing "Z", used for backward propagation
    """
    
    A = np.copy(Z)

    cache = (Z)
        
    return A, cache


def tanhact(Z):
    """
    Implement the tanh activation function

    Arguments:
    Z -- the pre-activation parameter
    
    Returns:
    A -- the post-activation parameter
    cache -- a dictionary containing "Z", used for backward propagation
    """
    
    A = np.tanh(Z)

    cache = (Z)
        
    return A, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Carry out the LINEAR->ACTIVATION layer for forward propagation 

    Arguments:
    A_prev -- activation for previous layer. Numpy array (layer_dims[l-1], batch_length)
    W -- weights matrix. Numpy array (layer_dims[l], layer_dims[l-1])
    b -- bias vector. Numpy array (layer_dims[l], 1)
    activation -- activation function for this layer, as string

    Returns:
    A -- the post-activation parameter for this layer
    cache -- a dictionary containing "linear cache" and "activation cache"
    """
    
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation=='sigmoid':
        A, activation_cache = sigmoid(Z)

    if activation=='relu':
        A, activation_cache = relu(Z)

    if activation=='identity':
        A, activation_cache = identity(Z)

    if activation=='tanh':
        A, activation_cache = tanhact(Z)

    # Assert that A has the correct shape
    assert(A.shape == (W.shape[0], A_prev.shape[1]))

    # Combine linear and activation caches
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters, hidden_active, outer_active):
    """
    Carry out forward propagation 

    Arguments:
    X -- input matrix (layer_dims[0], ], batch_length)
    parameters -- dictionary of current weights and biases
    hidden_active -- activation function for hidden layers, as string
    outer_active -- activation function for outer layer, as string

    Returns:
    AL -- the post-activation parameter for output layer. Numpy array (layer_dims[L], batch_length)
    cache -- a dictionary containing "linear cache" and "activation cache" for each layer
    """

    caches = []
    A = X
    L = len(parameters) // 2 # Divide by two since parameters have W and b

    for l in range(1, L):

        A_prev = A

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation=hidden_active)

        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation=outer_active)

    # Assert that AL has the correct shape
    assert(AL.shape == (parameters['b'+str(L)].shape[0], X.shape[1]))

    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y, W, cost_func, lambd):
    """
    Compute the cost function

    Arguments:
    AL -- the post-activation parameter for output layer. Numpy array (layer_dims[L], batch_length)
    Y -- input targets for current batch. Numpy array (layer_dims[L], batch_length)
    W -- list containing all weight matrices
    cost_func -- cost function to use, as string
    lambd -- lambda parameter(s) to regularize weights (currently only supports L2 regularization)

    Returns:
    cost -- the cost as a float
    """

    # Ensure lambd is an array
    lambd = np.array(lambd)

    # If lambd is a single value, copy it for each layer
    if not lambd.shape:
        lambd = np.repeat(lambd, len(W))

    m = Y.shape[1]
    num = Y.flatten().shape[0]

    if cost_func=="entropy":
        cost = -(np.dot(Y.flatten(), np.log(AL).flatten()) + np.dot((1.0 - Y.flatten()), np.log(1.0 - AL).flatten()))

    if cost_func=="MSE":
        cost = np.dot((AL - Y).flatten(), (AL - Y).flatten())

    # Compute the regularization for L2 norm
    reg = np.sum([lambd[i] * np.sum(np.power(W[i], 2.0)) for i in range(len(W))])

    # Add regularization term to cost
    cost = np.squeeze(cost) / num + reg / (2.0 * m)

    # Assert that cost is a single value
    assert(cost.shape == ())

    return cost


def linear_backward(dZ, cache, lambd):
    """
    Implement the linear component of the backward propagation

    Arguments:
    dZ -- gradient of the cost with respect to the current pre-activation parameter
    cache -- list containing activation from previous layer, current weight matrix and current bias
    lambd -- lambda parameter(s) to regularize weights (currently only supports L2 regularization)
    
    Returns:
    dA_prev -- the gradient of the cost with repsect to the previous post-activation parameter
    dW -- the gradient of the cost with respect to the current weights
    db -- the gradient of the cost with respect to the current biases
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T) / m + lambd * W / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    
    return dA_prev, dW, db


def sigmoid_backward(dA, activation_cache):
    """
    Compute the derivative of the cost with respect to the pre-activation parameter assuming a sigmoid activation function

    Arguments:
    dA -- gradient of the cost with respect to the current post-activation parameter. Numpy array (layer_dims[L], ], batch_length)
    activation cache -- contains the pre-activation parameter
    
    Returns:
    dZ -- the gradient of the cost with repsect to the previous pre-activation parameter
    """
    
    Z = activation_cache

    A = 1.0 / (1.0 + np.exp(-Z))
    
    dZ = dA * A * (1.0 - A)
    
    return dZ


def relu_backward(dA, activation_cache):
    """
    Compute the derivative of the cost with respect to the pre-activation parameter assuming a RELU activation function

    Arguments:
    dA -- gradient of the cost with respect to the current post-activation parameter. Numpy array (layer_dims[L], batch_length)
    activation cache -- contains the pre-activation parameter
    
    Returns:
    dZ -- the gradient of the cost with repsect to the previous pre-activation parameter
    """
    
    Z = activation_cache

    dZ = np.where(Z>0, dA, 0.0)
    
    return dZ


def identity_backward(dA, activation_cache):
    """
    Compute the derivative of the cost with respect to the pre-activation parameter assuming an identity activation function

    Arguments:
    dA -- gradient of the cost with respect to the current post-activation parameter. Numpy array (layer_dims[L], batch_length)
    activation cache -- contains the pre-activation parameter
    
    Returns:
    dZ -- the gradient of the cost with repsect to the previous pre-activation parameter
    """
    
    dZ = np.copy(dA)
    
    return dZ


def tanh_backward(dA, activation_cache):
    """
    Compute the derivative of the cost with respect to the pre-activation parameter assuming a tanh activation function

    Arguments:
    dA -- gradient of the cost with respect to the current post-activation parameter. Numpy array (layer_dims[L], batch_length)
    activation cache -- contains the pre-activation parameter
    
    Returns:
    dZ -- the gradient of the cost with repsect to the previous pre-activation parameter
    """
    
    Z = activation_cache
    
    dZ = 1.0 - np.power(np.tanh(Z), 2.0)
    
    return dZ


def linear_activation_backward(dA, cache, activation, lambd):
    """
    Carry out the LINEAR->ACTIVATION layer for backward propagation 

    Arguments:
    dA -- gradient of the cost with respect to the current post-activation parameter. Numpy array (layer_dims[L], batch_length)
    cache -- list containing the linear cache and the activation cache
    activation -- activation function for this layer, as string
    lambd -- lambda parameter(s) to regularize weights (currently only supports L2 regularization)

    Returns:
    dA_prev -- the derivative of the cost with respect to the post-activation parameter for the previous layer
    dW -- the derivative of the cost with respect to the weights for this layer
    db -- the derivative of the cost with respect to the biases for this layer
    """
    
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    if activation == "identity":
        dZ = identity_backward(dA, activation_cache)

    if activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
    
        
    dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    return dA_prev, dW, db


def calculate_loss_backward(AL, Y, cost_func):
    """
    Compute the cost function for backward propagation

    Arguments:
    AL -- the post-activation parameter for output layer. Numpy array (layer_dims[L], batch_length)
    Y -- input targets for current batch
    cost_func -- cost function to use, as string

    Returns:
    dAL -- the derivative of the cost with repect to the post-activation parameter in the outer later
    """
    
    m = AL.shape[1]
    
    if cost_func == "entropy":

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    if cost_func == "MSE":

        dAL = 2*(AL - Y)

    return dAL
    
    
def L_model_backward(AL, Y, caches, cost_func, lambd,  hidden_active, outer_active):
    """
    Carry out backward propagation 

    Arguments:
    AL -- the post-activation parameter for output layer (layer_dims[L], batch_length)
    Y -- input targets for current batch (layer_dims[L], batch_length)
    caches -- list of linear and activation caches for each layer
    cost_func -- cost function to use, as string
    lambd -- lambda parameter(s) to regularize weights (currently only supports L2 regularization)
    hidden_active -- activation function for hidden layers, as string
    outer_active -- activation function for outer layer, as string

    Returns:
    grads -- python dictionary containing the derivatives of the cost with respect to the various parameters:
                    dAl -- post-activation parameter for layer l. Numpy array (layer_dims[l], batch_length)
                    dWl -- weight matrix for layer l. Numpy array (layer_dims[l], layer_dims[l-1])
                    dbl -- bias vector for layer l. Numpy array (layer_dims[l], 1)
    """

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Ensure lambd is an array
    lambd = np.array(lambd)

    # If lambd is a single value, copy it for each layer
    if not lambd.shape:
        lambd = np.repeat(lambd, L)

    dAL = calculate_loss_backward(AL, Y, cost_func)

    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, outer_active, lambd[L-1])

    for l in reversed(range(L-1)):

        current_cache = caches[l]
        
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, hidden_active, lambd[l])
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update the parameters assuming standard gradient descent

    Arguments:
    parameters -- dictionary of current weights and biases
    grads -- python dictionary containing the derivatives of the cost with respect to the various parameters
    learning_rate -- the learning rate, alpha, parameter

    Returns:
    parameters -- dictionary of updated weights and biases
    """

    L = len(parameters) // 2 # Divide by two since parameters have W and b

    for l in range(L):

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)] 
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters


def update_momentum(parameters, grads, grads_v, learning_rate, beta, t):
    """
    Update the parameters assuming gradient descent with momentum

    Arguments:
    parameters -- dictionary of current weights and biases
    grads -- python dictionary containing the derivatives of the cost with respect to the various parameters
    grads_v -- python dictionary containing the derivatives of the cost with respect to the various parameters using momentum
    learning_rate -- the learning rate, alpha, parameter
    beta -- the momentum parameter
    t -- the number of steps in the rolling average, used for bias correction at the start of the roll

    Returns:
    parameters -- dictionary of updated weights and biases
    """
    
    L = len(parameters) // 2 # Divide by two since parameters have W and b

    for l in range(L):

        grads_v["dW" + str(l+1)] = beta * grads_v["dW" + str(l+1)] + (1.0 - beta) * grads["dW" + str(l+1)]
        grads_v["db" + str(l+1)] = beta * grads_v["db" + str(l+1)] + (1.0 - beta) * grads["db" + str(l+1)]
        v_corrected_dW = grads_v["dW" + str(l+1)] / (1.0 - beta**t)
        v_corrected_db = grads_v["db" + str(l+1)] / (1.0 - beta**t)
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected_dW
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected_db

    return parameters, grads_v


def update_rmsprop(parameters, grads,  grads_s, learning_rate,  beta, t, epsilon):
    """
    Update the parameters assuming gradient descent with RMS propagation

    Arguments:
    parameters -- dictionary of current weights and biases
    grads -- python dictionary containing the derivatives of the cost with respect to the various parameters
    grads_s -- python dictionary containing the derivatives of the cost with respect to the various parameters using RMS propagation
    learning_rate -- the learning rate, alpha, parameter
    beta -- the RMS propagation parameter
    t -- the number of steps in the rolling average, used for bias correction at the start of the roll
    epsilon -- smoothing term to avoid division by ~0

    Returns:
    parameters -- dictionary of updated weights and biases
    """
    
    L = len(parameters) // 2 # Divide by two since parameters have W and b

    for l in range(L):

        grads_s["dW" + str(l+1)] = beta * grads_s["dW" + str(l+1)] + (1.0 - beta) * np.power(grads["dW" + str(l+1)], 2)
        grads_s["db" + str(l+1)] = beta * grads_s["db" + str(l+1)] + (1.0 - beta) * np.power(grads["db" + str(l+1)], 2)
        s_corrected_dW = grads_s["dW" + str(l+1)] / (1.0 - beta**t)
        s_corrected_db = grads_s["db" + str(l+1)] / (1.0 - beta**t)
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)] / (np.sqrt(s_corrected_dW) + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)] / (np.sqrt(s_corrected_db) + epsilon)

    return parameters, grads_s


def update_adam(parameters, grads,  grads_v, grads_s, learning_rate, beta1, beta2, t, epsilon):
    """
    Update the parameters assuming gradient descent with Adaptive Moments

    Arguments:
    parameters -- dictionary of current weights and biases
    grads -- python dictionary containing the derivatives of the cost with respect to the various parameters
    grads_v -- python dictionary containing the derivatives of the cost with respect to the various parameters using momentum
    grads_s -- python dictionary containing the derivatives of the cost with respect to the various parameters using RMS propagation
    learning_rate -- the learning rate, alpha, parameter
    beta1 -- the momentum parameter
    beta2 -- the RMS propagation parameter
    t -- the number of steps in the rolling average, used for bias correction at the start of the roll
    epsilon -- smoothing term to avoid division by ~0

    Returns:
    parameters -- dictionary of updated weights and biases
    """
    
    L = len(parameters) // 2 # Divide by two since parameters have W and b

    for l in range(L):

        grads_v["dW" + str(l+1)] = beta1 * grads_v["dW" + str(l+1)] + (1.0 - beta1) * grads["dW" + str(l+1)]
        grads_v["db" + str(l+1)] = beta1 * grads_v["db" + str(l+1)] + (1.0 - beta1) * grads["db" + str(l+1)] 
        grads_s["dW" + str(l+1)] = beta2 * grads_s["dW" + str(l+1)] + (1.0 - beta2) * np.power(grads["dW" + str(l+1)], 2)
        grads_s["db" + str(l+1)] = beta2 * grads_s["db" + str(l+1)] + (1.0 - beta2) * np.power(grads["db" + str(l+1)], 2)
        v_corrected_dW = grads_v["dW" + str(l+1)] / (1.0 - beta1**t)
        v_corrected_db = grads_v["db" + str(l+1)] / (1.0 - beta1**t)
        s_corrected_dW = grads_s["dW" + str(l+1)] / (1.0 - beta2**t)
        s_corrected_db = grads_s["db" + str(l+1)] / (1.0 - beta2**t)
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected_dW/ (np.sqrt(s_corrected_dW) + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected_db / (np.sqrt(s_corrected_db) + epsilon)

    return parameters, grads_v, grads_s


def create_batches(X, Y, batch_size):
    """
    Determines the batches of X and Y given a requested batch size

    Arguments:
    X -- input features. Numpy array (layer_dims[0], m)
    Y -- input targets. Numpy array (layer_dims[L], m)
    batch_size -- integer giving requested batch size

    Returns:
    X -- randomly shuffled features
    Y -- randomly shuffled targets (shuffed so as to maintain correspondence with X)
    batch_lengths -- list of batch lengths, with the last batch containing the remainder
    m_tot -- total number of examples in dataset
    """
    
    m_tot = Y.shape[1]
    if batch_size:
        batch_num = int(m_tot/batch_size)
        batch_rem = m_tot % batch_size
        X = X.T
        Y = Y.T
        np.random.shuffle(X)
        np.random.shuffle(Y)
        X = X.T
        Y = Y.T
        batch_lengths = [batch_size] * batch_num
        if batch_rem:
            batch_lengths = [batch_size] * batch_num + [batch_rem]     
        
    else:
        # If batch_size is zero, all examples are in one batch: batch gradient descent
        batch_lengths = [m_tot]

    return X, Y, batch_lengths, m_tot


def get_W_all(caches):
    """
    Recover all of the weights from the caches

    Arguments:
    caches -- list of linear and activation caches for each layer

    Returns:
    W_all -- list of weight matrices for each layer
    """
    
    W_all = []
        
    for c in caches:

        linear_cache, activation_cache = c
        Adummy, W, bdummy = linear_cache

        W_all.append(W)

    return W_all
        

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, max_iter=2500, niter_nochange=10, rel_tol=1.0e-5, seed=None,
                  init_scale=0.01, hidden_active="relu", outer_active="sigmoid", cost_func="entropy", optimize='adam',
                  batch_size=32, lambd=0, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1.0e-8, print_cost=100):
    """
    Compute weights and biases for a neural network with L layers and an arbitrary number of nodes per layer

    Arguments:
    X -- input features. Numpy array (layer_dims[0], m)
    Y -- input targets. Numpy array (layer_dims[L], m)
    layer_dims -- list of dimensions of each layer. Currently needs to include the input and output layers
    learning_rate -- the learning rate, alpha, parameter
    max_iter -- maximum iteration to use
    niter_nochange -- number of iterations after which the program will terminate if the cost does not change
    rel_tol -- relative tolerance for convergence
    seed -- random seed
    init_scale -- initialization scale factor
    hidden_active -- activation function for hidden layers, as string
    outer_active -- activation function for outer layer, as string
    cost_func -- cost function to use, as string
    optimize -- optimization method to use, as string
    batch_size -- integer giving requested batch size
    lambd -- lambda parameter(s) to regularize weights (currently only supports L2 regularization)
    beta -- the momentum parameter if using momentum or RMS propagation parameter if using RMS propagation
    beta1 -- the momentum parameter if using adaptive moments
    beta2 -- the RMS propagation parameter if using adaptive moments
    epsilon -- smoothing term to avoid division by ~0 when using adaptive momements or RMS propagation
    print_cost -- how often to output the cost to screen (number of iterations)

    Returns:
    W_all -- list of weight matrices for each layer
    """
    
    time0 = time.time()
    
    np.random.seed(seed)
            
    cost = 0

    count_nochange = 0

    parameters = initialize_parameters(layers_dims, seed, init_scale)

    tcount = 1

    for i in range(max_iter):

        X, Y, batch_lengths, m_tot = create_batches(X, Y, batch_size)
        
        cost_prev = cost
        cost = 0

        b_prev = 0

        for t in range(len(batch_lengths)):
        
            AL, caches = L_model_forward(X[:,b_prev:batch_lengths[t]+b_prev], parameters,  hidden_active, outer_active)
            W_all = get_W_all(caches)
            cost_batch = compute_cost(AL, Y[:,b_prev:batch_lengths[t]+b_prev], W_all, cost_func, lambd)
            if cost_batch!=cost_batch:
                print("Invalid cost functin")
                sys.exit(1)
            cost += cost_batch * batch_lengths[t] / m_tot
            grads = L_model_backward(AL, Y[:,b_prev:batch_lengths[t]+b_prev], caches, cost_func, lambd, hidden_active, outer_active)

            if optimize=='momentum':
                if tcount==1:
                    grads_v = copy.deepcopy(grads)
                    grads_v = {k:np.zeros_like(grads_v[k]) for k in grads_v.keys()}
                parameters, grads_v = update_momentum(parameters, grads, grads_v, learning_rate * batch_lengths[t] / m_tot, beta, tcount)
            elif optimize=='rms_prop':
                if tcount==1:
                    grads_s = copy.deepcopy(grads)
                    grads_s = {k:np.zeros_like(grads_s[k]) for k in grads_s.keys()}
                parameters, grads_s = update_rmsprop(parameters, grads, grads_s, learning_rate * batch_lengths[t] / m_tot, beta, tcount, epsilon)
            elif optimize=='adam':
                if tcount==1:
                    grads_v = copy.deepcopy(grads)
                    grads_v = {k:np.zeros_like(grads_v[k]) for k in grads_v.keys()}
                    grads_s = copy.deepcopy(grads_v)
                parameters, grads_v, grads_s = update_adam(parameters, grads,  grads_v, grads_s, learning_rate * batch_lengths[t] / m_tot, beta1, beta2, tcount, epsilon)
            else:
                parameters = update_parameters(parameters, grads, learning_rate * batch_lengths[t] / m_tot)

            b_prev += batch_lengths[t]
            tcount += 1

        if print_cost and not i%print_cost:
            print ("Cost after iteration %i: %e" %(i, cost))

        if cost_prev:
            if np.abs(cost-cost_prev)/cost_prev<rel_tol:
                count_nochange += 1
                if count_nochange == niter_nochange:
                    print("Cost has not changed for %i iterations" %niter_nochange)
                    break
            else:
                count_nochange = 0
        
        if i==max_iter-1:
            print("Reached maximum number of iterations before converging")

    time1 = time.time()
    m, s = divmod(time1-time0, 60)
    print("Time for run = %i mins : %i secs" %(m, s))
        
    return parameters
