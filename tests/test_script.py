from data_science_projects.simple_neural_net import *
from test_cases import *
from test_results import *
from pytest import approx
import copy

##########################################
print("\nTesting initialize_parameters...")

parameters = initialize_parameters([5,4,3], seed=3, init_scale=0.01)

results = initialization_results()

assert(np.allclose(parameters["W1"], results["W1"]))
assert(np.allclose(parameters["b1"], results["b1"]))
assert(np.allclose(parameters["W2"], results["W2"]))
assert(np.allclose(parameters["b2"], results["b2"]))

print("initialize_parameters OK\n")

###########################################

print("Testing linear_forward...")

A, W, b = linear_forward_test_case()
Z, linear_cache = linear_forward(A, W, b)

results = linear_forward_results()

assert(np.allclose(Z, results["Z"]))

print("linear_forward OK\n")

##########################################

print("Testing linear_activation_forward...")

A_prev, W, b = linear_activation_forward_test_case()
A_sigmoid, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
A_relu, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")

results = lin_act_forward_results()

assert(np.allclose(A_sigmoid, results["A_sigmoid"]))
assert(np.allclose(A_relu, results["A_relu"]))

print("linear_activation_forward OK\n")

##########################################

print("Testing L_model_forward...")

X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters, hidden_active='relu', outer_active='sigmoid')

results = L_model_forward_results()

assert(np.allclose(AL, results["AL"]))
assert(len(caches)==results["len_caches"])

print("L_model_forward OK\n")

##########################################

print("Testing cost...")

Y, AL = compute_cost_test_case()

W = np.array([0])

cost = compute_cost(AL, Y, W, cost_func='entropy', lambd=0)

results = cost_results()

assert(np.allclose(cost, results["cost"]))

print("cost OK\n")

##########################################

print("Testing linear_backward...")

dZ, linear_cache = linear_backward_test_case()

dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd=0)

results = linear_backward_results()

assert(np.allclose(dA_prev, results["dA_prev"]))
assert(np.allclose(dW, results["dW"]))
assert(np.allclose(db, results["db"]))

print("linear_backward OK\n")

##########################################

print("Testing linear_activation_backward...")

dAL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev_sigmoid, dW_sigmoid, db_sigmoid = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid", lambd=0)

dA_prev_relu, dW_relu, db_relu = linear_activation_backward(dAL, linear_activation_cache, activation = "relu", lambd=0)

results = lin_act_backward_results()

assert(np.allclose(dA_prev_sigmoid, results["dA_prev_sigmoid"]))
assert(np.allclose(dW_sigmoid, results["dW_sigmoid"]))
assert(np.allclose(db_sigmoid, results["db_sigmoid"]))

assert(np.allclose(dA_prev_relu, results["dA_prev_relu"]))
assert(np.allclose(dW_relu, results["dW_relu"]))
assert(np.allclose(db_relu, results["db_relu"]))

print("linear_activation_backward OK\n")

##########################################

print("Testing L_model_backward...")

AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches, cost_func='entropy', lambd=0, hidden_active='relu', outer_active='sigmoid')

results = L_model_backward_results()

assert(np.allclose(grads["dW1"], results["dW1"]))
assert(np.allclose(grads["db1"], results["db1"]))
assert(np.allclose(grads["dA1"], results["dA1"]))

print("L_model_backward OK\n")

##########################################

print("Testing update_parameters...")

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

results = update_parameters_results()

assert(np.allclose(parameters["W1"], results["W1"]))
assert(np.allclose(parameters["b1"], results["b1"]))
assert(np.allclose(parameters["W2"], results["W2"]))
assert(np.allclose(parameters["b2"], results["b2"]))

print("update_parameters OK\n")

##########################################

print("Running gradient checks...")

def get_W_all(caches):

    W_all = []

    for c in caches:

        linear_cache, activation_cache = c
        Adummy, W, bdummy = linear_cache

        W_all.append(W)

    return W_all


def part_L_model(parameters, n_x, lambd, hidden_active, outer_active, cost_func):

    AL, caches = L_model_forward(train_x[:n_x], parameters, hidden_active=hidden_active, outer_active=outer_active)

    W_all = get_W_all(caches)
    
    cost = compute_cost(AL, train_y, W_all, lambd=lambd, cost_func=cost_func)

    grads = L_model_backward(AL, train_y[:n_x], caches, lambd=lambd, hidden_active=hidden_active, outer_active=outer_active, cost_func=cost_func)

    return grads, cost


train_x, train_y, test_x, test_y, classes = load_data()

train_x = train_x.reshape(train_x.shape[0], -1).T / 255
test_x = test_x.reshape(test_x.shape[0], -1).T / 255

n_x = 100
layer_dims = [n_x, 10, 5, 1]
lambd = 100
hidden_active = 'relu'
outer_active = 'sigmoid'
cost_func = 'entropy'
epsilon = 1e-7
scaleb = 0.01

init_scale = 1.0/np.sqrt(layer_dims)

parameters = initialize_parameters(layer_dims, seed=1, init_scale=0.01)


for i in parameters.keys():
    if i=='b1' or i=='b2' or i=='b3' or i=='b4':
        parameters[i] = np.random.randn(parameters[i].shape[0], parameters[i].shape[1])*scaleb

grads, cost = part_L_model(parameters, n_x, lambd, hidden_active, outer_active, cost_func)

delta_grads = []

L2_norm_grads = 0
L2_norm_grads_epsilon = 0

for i in parameters.keys():
    for k1 in range(parameters[i].shape[0]):
        for k2 in range(parameters[i].shape[1]):           
            parameters_epsilon = copy.deepcopy(parameters)
            parameters_epsilon[i][k1, k2] += epsilon        
            grads_temp, cost_plus = part_L_model(parameters_epsilon, n_x, lambd, hidden_active, outer_active, cost_func)

            parameters_epsilon = copy.deepcopy(parameters)
            parameters_epsilon[i][k1, k2] += -epsilon        
            grads_temp, cost_minus = part_L_model(parameters_epsilon, n_x, lambd, hidden_active, outer_active, cost_func)            
                
            grad_test = (cost_plus - cost_minus)/(2.0 * epsilon)
            grad_true = grads['d'+i][k1,k2]
            
            delta_grads.append(grad_test - grad_true)

            L2_norm_grads += grad_true**2.0
            L2_norm_grads_epsilon += grad_test**2.0
            
test = np.sqrt(np.sum(np.array(delta_grads)**2.0))/(np.sqrt(L2_norm_grads) + np.sqrt(L2_norm_grads_epsilon))

assert(test<1e-6)

print("gradient checks OK\n")

##########################################

print("Running smoke test...\n")

n_x = train_x.shape[0]

layer_dims = [n_x, 20, 7, 5, 1]

init_scale = 1.0/np.sqrt(layer_dims)

parameters = L_layer_model(train_x, train_y, layer_dims, seed=1, init_scale=init_scale, print_cost=100, batch_size=None, optimize='momentum', beta=0.0)

pred_train, accuracy_train = predict(train_x, train_y, parameters, accuracy=True)

pred_test, accuracy_test = predict(test_x, test_y, parameters, accuracy=True)

results = predict_results()

assert(accuracy_train == approx(results["accuracy_train"]))
assert(accuracy_test == approx(results["accuracy_test"]))

print("\nsmoke test OK\n")
