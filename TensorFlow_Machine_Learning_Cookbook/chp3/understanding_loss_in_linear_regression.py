import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
sess = tf.Session()

iris=datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([x[0] for x in iris.data])
batch_size=25
learning_rate = 0.05
iterations = 50
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A_l1 = tf.Variable(tf.random_normal(shape=[1,1]))
b_l1 = tf.Variable(tf.random_normal(shape=[1,1]))
A_l2 = tf.Variable(tf.random_normal(shape=[1,1]))
b_l2 = tf.Variable(tf.random_normal(shape=[1,1]))
model_output_l1 = tf.add(tf.matmul(x_data, A_l1), b_l1)
model_output_l2 = tf.add(tf.matmul(x_data, A_l2), b_l2)

loss_l1 = tf.reduce_mean(tf.abs(y_target - model_output_l1))
loss_l2 = tf.reduce_mean(tf.square(y_target - model_output_l2))

init = tf.global_variables_initializer()
sess.run(init)

my_opt_l1 = tf.train.GradientDescentOptimizer(learning_rate)
train_step_l1 = my_opt_l1.minimize(loss_l1)
loss_vec_l1 = []
my_opt_l2 = tf.train.GradientDescentOptimizer(learning_rate)
train_step_l2 = my_opt_l2.minimize(loss_l2)
loss_vec_l2 = []

print("L1 loss")
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step_l1, feed_dict={x_data: rand_x, y_target:rand_y})
    temp_loss_l1 = sess.run(loss_l1, feed_dict={x_data: rand_x,
                                                y_target: rand_y})
    loss_vec_l1.append(temp_loss_l1)
    if (i+1)%25==0:
        print('Step #' + str(i+1) + 'Loss = ' + str(temp_loss_l1) +  ' A = ' + str(sess.run(A_l1)) + ' b = ' + str(sess.run(b_l1)))

print("L2 loss")
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step_l2, feed_dict={x_data: rand_x, y_target:rand_y})
    temp_loss_l2 = sess.run(loss_l2, feed_dict={x_data: rand_x,
                                                y_target: rand_y})
    loss_vec_l2.append(temp_loss_l2)
    if (i+1)%25==0:
        print('Step #' + str(i+1) + 'Loss = ' + str(temp_loss_l2) +  ' A = ' + str(sess.run(A_l2)) + ' b = ' + str(sess.run(b_l2)))

plt.plot(loss_vec_l1, 'k-', label='L1 Loss')
plt.plot(loss_vec_l2, 'r--', label='L2 Loss')
plt.title('L1 and L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L1 Loss')
plt.legend(loc='upper right')
plt.show()
