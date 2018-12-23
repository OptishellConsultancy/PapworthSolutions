
import tensorflow as tf
class SimpleMLPTrainer(object): 
    def __init__ (self,data,colsInRng,colsOutRng):
        print("Running simple Multi layer perception Trainer")
        self.Train(data,colsInRng,colsOutRng)
     # Parameters
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 1
    display_step = 1

    # Network Parameters
    n_hidden_1 = 64 # 1st layer number of neurons
    n_hidden_2 = 64 # 2nd layer number of neurons
    n_input = 8 # data input
    n_classes = 2 # MNIST total classes (0-9 digits)

    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }


    # Create model
    def multilayer_perceptron(self,x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']

        return out_layer

    def Train(self, data, colsInRng, colsOutRng):    
        # Construct model
        logits = self.multilayer_perceptron(self.X)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op)
        # Initializing...
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(self.training_epochs):
                avg_cost = 0.
       
                # Loop over all data
                for rows in data: 

                #Mapping data
                    x_dataIn  = []
                    x_dataOut = []
                    itemCount = 0
                    for item in rows:
                        if itemCount <= colsInRng:
                            x_dataIn.append(item)
                        else:
                            if itemCount <= colsOutRng: #Not necessary but is safe
                                x_dataOut.append(item)
                        itemCount += 1

                    # Run optimization op (backprop) and cost op (to get loss value)
                    c = sess.run([train_op, loss_op], feed_dict={self.X: x_dataIn, self.Y: x_dataOut})
                    # Compute average loss
                    avg_cost += c / len(data)
                # Display logs per epoch step
                
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            print("Optimization Finished!")

        # Test model
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy : {}".format(accuracy))
