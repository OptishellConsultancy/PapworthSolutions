import tensorflow as tf

T, F = 1., -1.

train_in = [
 [T, T],
 [T, F],
 [F, T],
 [F, F],
]

train_out = [
 [F],
 [T],
 [T],
 [F],
]

X = tf.placeholder("float", [4,2])
Y = tf.placeholder("float", [4,1])

w1 = tf.Variable(tf.random_normal([2, 2]))
b1 = tf.Variable(tf.zeros([2]))

w2 = tf.Variable(tf.random_normal([2, 1]))
b2 = tf.Variable(tf.zeros([1]))

out1 = tf.tanh(tf.add(tf.matmul(X, w1), b1))
out2 = tf.tanh(tf.add(tf.matmul(out1, w2), b2))

error = tf.subtract(Y, out2)
mse = tf.reduce_mean(tf.square(error))

train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

err, target = 1, 0.01
epoch, max_epochs = 0, 20

while err > target and epoch < max_epochs:
   epoch += 1
   err, _ = sess.run([mse, train], feed_dict={X: train_in, Y: train_out})

print("epoch: {}, mse: {}".format(epoch, err))