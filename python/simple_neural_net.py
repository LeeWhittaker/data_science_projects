import numpy as np
import copy
import sys
import time

def initialize_parameters(layer_dims, seed, init_scale):

    np.random.seed(seed)

    init_scale = np.array(init_scale)
    
    if not init_scale.shape:
        init_scale = np.repeat(init_scale, len(layer_dims))

    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * init_scale[l-1]
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters


def linear_forward(A, W, b):

    Z = np.dot(W, A) + b

    cache = (A, W, b)

    assert(Z.shape == (W.shape[0], A.shape[1]))

    return Z, cache


def sigmoid(Z):

    A = 1.0 / (1.0 + np.exp(-Z))

    cache = (Z)

    return A, cache


def relu(Z):

    A = np.where(Z>0, Z, 0)

    cache = (Z)
        
    return A, cache


def identity(Z):
    
    A = np.copy(Z)

    cache = (Z)
        
    return A, cache


def tanhact(Z):
    
    A = np.tanh(Z)

    cache = (Z)
        
    return A, cache


def linear_activation_forward(A_prev, W, b, activation):

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation=='sigmoid':
        A, activation_cache = sigmoid(Z)

    if activation=='relu':
        A, activation_cache = relu(Z)

    if activation=='identity':
        A, activation_cache = identity(Z)

    if activation=='tanh':
        A, activation_cache = tanhact(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters, hidden_active, outer_active):

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):

        A_prev = A

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation=hidden_active)

        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation=outer_active)

    assert(AL.shape == (parameters['b'+str(L)].shape[0], X.shape[1]))

    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y, W, cost_func, lambd):
    
    lambd = np.array(lambd)
    
    if not lambd.shape:
        lambd = np.repeat(lambd, len(W))

    m = Y.shape[1]
    num = Y.flatten().shape[0]

    if cost_func=="entropy":
        cost = -(np.dot(Y.flatten(), np.log(AL).flatten()) + np.dot((1.0 - Y.flatten()), np.log(1.0 - AL).flatten()))

    if cost_func=="MSE":
        cost = np.dot((AL - Y).flatten(), (AL - Y).flatten())

    reg = np.sum([lambd[i] * np.sum(np.power(W[i], 2.0)) for i in range(len(W))])
        
    cost = np.squeeze(cost) / num + reg / (2.0 * m)
    
    assert(cost.shape == ())

    return cost


def linear_backward(dZ, cache, lambd):

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

    Z = activation_cache

    A = 1.0 / (1.0 + np.exp(-Z))
    
    dZ = dA * A * (1.0 - A)
    
    return dZ


def relu_backward(dA, activation_cache):
    
    Z = activation_cache

    dZ = np.where(Z>0, dA, 0.0)
    
    return dZ


def identity_backward(dA, activation_cache):

    dZ = np.copy(dA)
    
    return dZ


def tanh_backward(dA, activation_cache):

    Z = activation_cache
    
    dZ = 1.0 - np.power(np.tanh(Z), 2.0)
    
    return dZ


def linear_activation_backward(dA, cache, activation, lambd):
    
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

    m = AL.shape[1]
    
    if cost_func == "entropy":

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    if cost_func == "MSE":

        dAL = 2*(AL - Y)

    return dAL
    
    
def L_model_backward(AL, Y, caches, cost_func, lambd,  hidden_active, outer_active):

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    lambd = np.array(lambd)
    
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

    L = len(parameters) // 2

    for l in range(L):

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)] 
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters


def update_momentum(parameters, grads, grads_v, learning_rate, beta, t):

    L = len(parameters) // 2

    for l in range(L):

        grads_v["dW" + str(l+1)] = beta * grads_v["dW" + str(l+1)] + (1.0 - beta) * grads["dW" + str(l+1)]
        grads_v["db" + str(l+1)] = beta * grads_v["db" + str(l+1)] + (1.0 - beta) * grads["db" + str(l+1)]
        v_corrected_dW = grads_v["dW" + str(l+1)] / (1.0 - beta**t)
        v_corrected_db = grads_v["db" + str(l+1)] / (1.0 - beta**t)
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected_dW
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected_db

    return parameters, grads_v


def update_rmsprop(parameters, grads,  grads_s, learning_rate,  beta, t, epsilon):

    L = len(parameters) // 2

    for l in range(L):

        grads_s["dW" + str(l+1)] = beta * grads_s["dW" + str(l+1)] + (1.0 - beta) * np.power(grads["dW" + str(l+1)], 2)
        grads_s["db" + str(l+1)] = beta * grads_s["db" + str(l+1)] + (1.0 - beta) * np.power(grads["db" + str(l+1)], 2)
        s_corrected_dW = grads_s["dW" + str(l+1)] / (1.0 - beta**t)
        s_corrected_db = grads_s["db" + str(l+1)] / (1.0 - beta**t)
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)] / (np.sqrt(s_corrected_dW) + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)] / (np.sqrt(s_corrected_db) + epsilon)

    return parameters, grads_s


def update_adam(parameters, grads,  grads_v, grads_s, learning_rate, beta1, beta2, t, epsilon):

    L = len(parameters) // 2

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
        batch_lengths = [m_tot]

    return X, Y, batch_lengths, m_tot


def get_W_all(caches):

    W_all = []
        
    for c in caches:

        linear_cache, activation_cache = c
        Adummy, W, bdummy = linear_cache

        W_all.append(W)

    return W_all
        

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, max_iter=2500, niter_nochange=10, rel_tol=1.0e-5, seed=None, init_scale=0.01, hidden_active="relu", outer_active="sigmoid", cost_func="entropy", update='adam', batch_size=32, lambd=0, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1.0e-8, print_cost=100):

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

            if update=='momentum':
                if tcount==1:
                    grads_v = copy.deepcopy(grads)
                    grads_v = {k:np.zeros_like(grads_v[k]) for k in grads_v.keys()}
                parameters, grads_v = update_momentum(parameters, grads, grads_v, learning_rate * batch_lengths[t] / m_tot, beta, tcount)
            elif update=='rms_prop':
                if tcount==1:
                    grads_s = copy.deepcopy(grads)
                    grads_s = {k:np.zeros_like(grads_s[k]) for k in grads_s.keys()}
                parameters, grads_s = update_rmsprop(parameters, grads, grads_s, learning_rate * batch_lengths[t] / m_tot, beta, tcount, epsilon)
            elif update=='adam':
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
                    break
            else:
                count_nochange = 0
        
        if i==max_iter-1:
            print("Reached maximum number of iterations before converging")

    time1 = time.time()
    m, s = divmod(time1-time0, 60)
    print("Time for run = %i mins : %i secs" %(m, s))
        
    return parameters
