import tensorflow as tf

def L_layer_model(X, Y, layer_dims, activations, cost, optimizer='adam', learning_rate=0.001,
                  epochs=10000, rel_tol=1.0e-5,
                  lambd=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-7, batch_size=32, seed=1, verbose=False):

    tf.random.set_seed(seed)
    Init = tf.keras.initializers.LecunNormal(seed=seed)

    model = tf.keras.Sequential()
    regularizer = tf.keras.regularizers.L2(lambd)
    
    if len(X.shape)>2:
        model.add(tf.keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])))

    for i in range(len(layer_dims)):
        model.add(tf.keras.layers.Dense(layer_dims[i], activation=activations[i],
                                        kernel_initializer=Init,
                                        kernel_regularizer=regularizer))

    if optimizer=='adam':
        Opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        
    if cost=='entropy':
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if cost=='mse':
        loss = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer=Opt,
                  loss=loss,
                  metrics=['accuracy'])

    loss = 0

    for i in range(epochs):

        loss_prev = loss
        if verbose:
            print('Epoch = ', i+1)
        loss = float(model.fit(X, Y, batch_size=batch_size, verbose=verbose).history['loss'][0])
    
        if abs(loss - loss_prev)/loss<rel_tol:
            break

    return model
   
