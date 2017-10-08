import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()

x_vals = tf.linspace(-1.,1.,500)
target = tf.constant(0.)

#L2 norm loss function
l2_y_vals = tf.square(target - x_vals)
l2_y_out  = sess.run(l2_y_vals)

#L1 norm loss function
l1_y_vals = tf.abs(target - x_vals)
l1_y_out  = sess.run(l1_y_vals)

#Pseudo-Huber loss
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1),
                        tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)
phuber1_y_out = sess.run(phuber1_y_vals)
delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2),
                        tf.sqrt(1. + tf.square((target - x_vals)/delta2)) - 1.)
phuber2_y_out = sess.run(phuber2_y_vals)


#Plot
x_array = sess.run(x_vals)
plt.plot(x_array, l2_y_out, 'b--',label='L2 Loss')
plt.plot(x_array, l1_y_out, 'r--', label='L1 Loss')
plt.plot(x_array, phuber1_y_out,'k-.',label='P-Huber Loss (0.25)')
plt.plot(x_array, phuber2_y_out,'g:',label='P-Huber Loss (5.0)')
plt.ylim(-0.2,0.4)
plt.legend(loc='lower right',prop={'size':11})
plt.show()

#Start classificatioon loss function
#Redefine x_vals
x_vals = tf.linspace(-3.,5.,500)
target = tf.constant(1.)
targets = tf.fill([500,],1.)

#Hinge loss function
hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target,x_vals))
hinge_y_out  = sess.run(hinge_y_vals)

#Cross-entropy loss
xentropy_y_vals = -tf.multiply(target, tf.log(x_vals)) - tf.multiply(1. - target, tf.log(1. - x_vals))
xentropy_y_out  = sess.run(xentropy_y_vals)

#Sigmoid cross entropy loss function
xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_vals, labels=targets)
xentropy_sigmoid_y_out  = sess.run(xentropy_sigmoid_y_vals)
targets_0=tf.fill([500,],0.)
xentropy_sigmoid_y_vals_0 = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_vals, labels=targets_0)
xentropy_sigmoid_y_out_0  = sess.run(xentropy_sigmoid_y_vals_0)


#Weighted cross entropy loss function
weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(targets, x_vals,weight)
xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)

#Softmax cross entropy loss function
unscaled_logits = tf.constant([[1.,-3.,10.]])
target_dist = tf.constant([[0.1,0.02,0.88]])
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=unscaled_logits, labels=target_dist)
print(sess.run(softmax_xentropy))

x_array = sess.run(x_vals)
plt.plot(x_array, hinge_y_out, 'b-', label='Hinge Loss')
plt.plot(x_array, xentropy_y_out, 'r--', label='Cross Entropy Loss')
plt.plot(x_array, xentropy_sigmoid_y_out, 'k-.', label='Cross Entropy Sigmoid Loss')
plt.plot(x_array, xentropy_sigmoid_y_out_0, 'k--.', label='Cross Entropy Sigmoid Loss labels0')
plt.plot(x_array, xentropy_weighted_y_out, 'g:', label='Weighted Cross Enropy Loss (x0.5)')
plt.ylim(-1.5, 3)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()
