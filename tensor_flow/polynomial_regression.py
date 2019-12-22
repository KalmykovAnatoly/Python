import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

learning_rate = 0.01
training_epochs = 40

x_train = np.linspace(-1, 1, 101)

num_coeffs = 6
trY_coeffs = [1, 2, 3, 4, 5, 6]
y_train = 0

for i in range(num_coeffs):
    y_train += trY_coeffs[i] * np.power(x_train, i)

y_train += np.random.randn(*x_train.shape) * 1.5

plt.scatter(x_train, y_train)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)


w = tf.Variable([0.] * num_coeffs, name="weights")
y_model = model(X, w)

cost = tf.pow(Y - y_model, 2)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

for epoch in range(training_epochs):
    for (x, y) in zip(x_train, y_train):
        session.run(train_op, feed_dict={X: x, Y: y})

w_val = session.run(w)

session.close()
y_train2 = 0
for i in range(num_coeffs):
    y_train2 += w_val[i] * np.power(x_train, i)

plt.plot(x_train, y_train2, 'r')
plt.show()
