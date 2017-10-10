import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
sess = tf.Session()

x_vals=np.linspace(0,10,100)
y_vals=x_vals + np.random.normal(0,1,100)
x_vals_column=np.transpose(np.matrix(x_vals))
one_column = np.transpose(np.matrix(np.repeat(1,100)))
A = np.column_stack((x_vals_column, one_column))
b = np.transpose(np.matrix(y_vals))
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

# Find the Cholesky decomposition of our square matrix, AtA
# Ax=b
# LL'x=b
# Ly=b; L'x = y
# =>sol1; L'x = sol1
# =>x = sol2
tA_A = tf.matmul(tf.transpose(A_tensor),A_tensor)
L = tf.cholesky(tA_A)
tA_b = tf.matmul(tf.transpose(A_tensor), b)
sol1 = tf.matrix_solve(L, tA_b)
sol2 = tf.matrix_solve(tf.transpose(L),sol1)

solution_eval = sess.run(sol2)

slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
print('slope: ' + str(slope))
print('y_intercept: ' + str(y_intercept))

best_fit = []
for i in x_vals:
    best_fit.append(slope*i+y_intercept)
plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line',linewidth=3)
plt.legend(loc='upper left')
plt.show()
