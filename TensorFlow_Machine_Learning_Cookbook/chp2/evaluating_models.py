import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sess = tf.Session()

#Create graph, data, variables
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10.,100)
x_data = tf.placeholder(shape=[None,1],dtype=tf.float32)
y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)
batch_size = 25
train_indices = np.random.choice(len(x_vals), int(round(len(x_vals)*0.8)),replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test  = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test  = y_vals[test_indices]
A = tf.Variable(tf.random_normal(shape=[1,1]))

#Declare mode, loss function
my_output = tf.matmul(x_data, A)
loss = tf.reduce_mean(tf.square(my_output - y_target))
init = tf.initialize_all_variables()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

#Run
for i in range(100):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target:rand_y})
    if (i+1)%25==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data:rand_x, y_target: rand_y})))

#Evaluate the model
mse_test = sess.run(loss, feed_dict={x_data:np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
mse_train = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
print('MSE on test:'  + str(np.round(mse_test, 2)))
print('MSE on train:' + str(np.round(mse_train, 2)))


########################################################
# Classification example
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.Session()
batch_size = 25
x_vals = np.concatenate((np.random.normal(-1,1,50),np.random.normal(2,1,50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(shape=[1,None],dtype=tf.float32)
y_target = tf.placeholder(shape=[1,None],dtype=tf.float32)
train_indices = np.random.choice(len(x_vals), int(round(len(x_vals)*0.8)), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
b = tf.Variable(tf.random_normal(mean=10, shape=[1]))

#Model
my_output = tf.add(x_data, b)
init = tf.initialize_all_variables()
sess.run(init)
xentropy=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output,labels=y_target))
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)
for i in range(1800):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = [x_vals_train[rand_index]]
    rand_y = [y_vals_train[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target:rand_y})

print(sess.run(b))


y_prediction = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(x_data,b))))
correct_prediction = tf.equal(y_prediction, y_target)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_value_test = sess.run(accuracy, feed_dict={x_data: [x_vals_test], y_target: [y_vals_test]})
acc_value_train = sess.run(accuracy, feed_dict={x_data: [x_vals_train], y_target: [y_vals_train]})
print('Accuracy on train set: ' + str(acc_value_train))
print('Accuracy on test set: ' + str(acc_value_test))

A_result = sess.run(b)
bins = np.linspace(-5, 5, 100)
plt.hist(x_vals[0:50], bins, alpha=0.5, label='N(-1,1)',facecolor='blue')
plt.hist(x_vals[50:100], bins, alpha=0.5, label='N(2,1)',color='yellowgreen')
plt.plot((A_result, A_result), (0, 8), 'k--', linewidth=3, label='A = '+ str(np.round(A_result, 2)))
plt.legend(loc='upper right')
acc_value = (acc_value_test + acc_value_train)/2
plt.title('Binary Classifier, Accuracy=' + str(np.round(acc_value, 2)))
plt.show()

intercept = sess.run(b)
x_array_0 = x_vals[0:50]
y_array_0 = y_vals[0:50]
y_array_0_r = x_vals[0:50] + np.repeat(intercept, 50)
x_array_1 = x_vals[50:]
y_array_1 = y_vals[50:]
y_array_1_r = x_vals[50:] + np.repeat(intercept, 50)

x_sigmoid = tf.linspace(-3.,5.,500)
y_sigmoid = tf.nn.sigmoid(x_sigmoid)
x_sigmoid_array = sess.run(x_sigmoid)
y_sigmoid_array = sess.run(y_sigmoid)

plt.plot(x_array_0, y_array_0, 'rx', ms=10, mew=2, label='setosa''')
plt.plot(x_array_0, y_array_0_r, 'bx', ms=10, mew=2, label='setosa''')
plt.plot(x_array_1, y_array_1, 'ro', label='Non-setosa')
plt.plot(x_array_1, y_array_1_r, 'bo', label='Non-setosa')
plt.plot(x_sigmoid_array, y_sigmoid_array, 'k-.', label='Sigmoid')
plt.show()
