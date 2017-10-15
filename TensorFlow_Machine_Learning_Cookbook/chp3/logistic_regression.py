import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
sess = tf.Session()
iris = datasets.load_iris()
x_vals = np.array([x[2] for x in iris.data])
y_vals = np.array([x[0] for x in iris.data])
batch_size = 50
learning_rate = 0.001
x_data = tf.placeholder(shape=[None,1], dtype = tf.float32)
y_target = tf.placeholder(shape=[None,1], dtype = tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
model_output = tf.add(tf.matmul(x_data, A),b)
