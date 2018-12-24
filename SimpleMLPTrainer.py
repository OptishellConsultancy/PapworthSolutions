from __future__ import print_function 
import tensorflow as tf
class SimpleMLPTrainer(object): 
    def __init__ (self, dataDonorsSet, dataRecipientSet, dataRequiredResultSet,rows):
        print("Running simple Multi layer perception Trainer")
        self.Train(dataDonorsSet, dataRecipientSet, dataRequiredResultSet,rows)
     # Parameters
    learning_rate = 0.001
    training_epochs = 5000
    batch_size = 1
    display_step = 1

    # Network Parameters
    n_hidden_1 = 2 # 1st layer number of neurons
    n_hidden_2 = 5 # 2nd layer number of neurons
    n_input = 5 # data sets //5*2 donar and recipient
    n_classes = 2 #output classes

    # tf Graph input
    n_input = 5 # data sets //5*2 donar and recipient
    X = tf.placeholder("float", [2,n_input])
    Y = tf.placeholder("float", [n_classes])
   
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
    def multilayer_perceptron(self,X):
            
        out1 = tf.tanh(tf.add(tf.matmul(X, self.weights['h1']), self.biases['b1']))
        out2 = tf.tanh(tf.add(tf.matmul(out1, self.weights['h2']), self.biases['b2']))       
        out_layer = tf.matmul(out2, self.weights['out']) + self.biases['out']

        return out_layer

    def Train(self, dataDonorsSet, dataRecipientSet, dataRequiredResultSet, rows):    
        # Construct model
        logits = self.multilayer_perceptron(self.X)

        # Define loss and optimizer
        loss = tf.subtract(self.Y, logits)
        mse =  tf.reduce_mean(tf.square(loss))   
        train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)
  

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            err, target = 1, 0.01
            epoch = 0
            
            while err > target and epoch < self.training_epochs:
                rowCnt = 0
                while rowCnt < rows:
                    train_in = [dataDonorsSet[rowCnt],dataRecipientSet[rowCnt]]
                    train_out = dataRequiredResultSet[rowCnt]
                    #
                    #print(train_in)
                    #print(train_out)
                    err, _ = sess.run([mse, train], feed_dict={self.X: train_in, self.Y: train_out})
                    rowCnt += 1
                    #
                epoch += 1

            print("epoch: {}, mse: {}".format(epoch, err))