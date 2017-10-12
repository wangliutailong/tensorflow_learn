import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
sess = tf.Session()
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([x[0] for x in iris.data])
batch_size = 50
learning_rate = 0.001
x_data = tf.placeholder(shape=[None,1], dtype = tf.float32)
y_target = tf.placeholder(shape=[None,1], dtype = tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
model_output = tf.add(tf.matmul(x_data, A),b)

# Add loss function
lasso_param = tf.constant(1.)
heavyside_step = tf.truediv(1., tf.add(1., tf.exp(tf.multiply(-100.,tf.subtract(A, lasso_param)))))
regularization_param = tf.multiply(heavyside_step, 99.)
loss = tf.add(tf.reduce_mean(tf.square(y_target - model_output)),regularization_param)

# ridge_param = tf.constant(1.)
# ridge_loss  = tf.reduce_mean(tf.square(A))
# loss = tf.expand_dims(tf.add(tf.reduce_mean(tf.square(y_target -model_output)), tf.multiply(ridge_param, ridge_loss)), 0)

# Init, optimizer
init = tf.global_variables_initializer()
sess.run(init)
my_opt=tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# Train
loss_vec = []
for i in range(1500):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target:rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss[0])
    if (i+1)%300==0:
        print('Step #''' + str(i+1) + ' A = ' + str(sess.run(A)) +' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

lin = np.linspace(0,1500,1500)
plt.plot(lin, loss_vec,'b')
plt.show()

[slope] = sess.run(A)
[y_i]   = sess.run(b)
best_fit = []
for i in x_vals:
    best_fit.append(slope*i + y_i)

plt.plot(x_vals, y_vals, 'o',label="original data")
plt.plot(x_vals, best_fit, 'r',label='Best fit')
plt.legend(loc='upper left')
plt.show()
