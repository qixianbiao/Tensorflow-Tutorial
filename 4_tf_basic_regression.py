#Author: Xianbiao Qi
#Date:   2016.04.06

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle








#y = f(x), here we suppose f(x) = 2*x^2 + 3.1*x + 1.1 + noise, x belongs to [-1, 1]
#noise have mean 0 and std 0.2

trainx = 2*np.random.rand(100, 1)-1
noise = np.random.randn(100, 1)*0.2
trainy = 2.0*np.power(trainx, 2.0) + 3.1*trainx + 1.1 + noise
x = np.hstack((np.ones([100, 1], dtype=np.float32), trainx, np.power(trainx, 2), np.power(trainx, 3), np.power(trainx, 4)))
y = trainy #np.squeeze(trainy)
print x.shape, y.shape

valx = 2*np.random.rand(20, 1)-1
valy = 2.0*np.power(valx, 2.0) + 3.1*valx + 1.1
val_x = np.hstack((np.ones([20, 1], dtype=np.float32), valx, np.power(valx, 2), np.power(valx, 3), np.power(valx, 4)))
val_y = valy #np.squeeze(valy)
print val_x.shape, val_y.shape

#linear regression
#here, we map x into [1, x, x*^2, x*3, x*4]

def model(x, w):
    y = tf.matmul(x, weights['l1'])
    return y

#Main Function
if __name__ == '__main__':
    #Input and Output Definitation
    X = tf.placeholder(tf.float32, [None, 5])
    Y = tf.placeholder(tf.float32, [None, 1])

    #parameters definition and initiation
    weights = {
    'l1': tf.Variable(tf.truncated_normal([5, 1], stddev=0.01))
    }
    biases = {
    'l1': tf.Variable(tf.constant(0.001, shape=[1], dtype=tf.float32))
    }

    #loss funcation and optimization function
    pred_Y = model(X, weights)
    loss = tf.reduce_mean( tf.pow(pred_Y - Y, 2) )
    opt = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    #deploy and parameter initiation
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    #training
    epoch = 10
    batchsize = 5
    for i in range(epoch):
        [x, y] = shuffle(x,y,random_state=2000)
        for batch in range(0, x.shape[0], batchsize):
            ops, ll = sess.run([opt, loss], feed_dict={X:x[batch:batch+batchsize], Y:y[batch:batch+batchsize]})
            print('loss value: ', ll)
    predValues = sess.run(pred_Y, feed_dict={X:val_x})
    print("valx values are : ", valx.reshape([1, 20]))
    print("valy values are : ", valy.reshape([1, 20]))
    print("Predict values are : ", predValues.reshape([1, 20]))




'''
#MLP for regression

#y = f(x), here we suppose f(x) = 2*x^2 + 3.1*x + 1.1 + noise, x belongs to [-1, 1]
#noise have mean 0 and std 0.2

trainx = 2*np.random.rand(100, 1)-1
noise = np.random.randn(100, 1)*0.2
trainy = 2.0*np.power(trainx, 2.0) + 3.1*trainx + 1.1 + noise
x = np.hstack((np.ones([100, 1], dtype=np.float32), trainx, np.power(trainx, 2), np.power(trainx, 3), np.power(trainx, 4)))
y = trainy #np.squeeze(trainy)
print x.shape, y.shape

valx = 2*np.random.rand(20, 1)-1
valy = 2.0*np.power(valx, 2.0) + 3.1*valx + 1.1
val_x = np.hstack((np.ones([20, 1], dtype=np.float32), valx, np.power(valx, 2), np.power(valx, 3), np.power(valx, 4)))
val_y = valy #np.squeeze(valy)
print val_x.shape, val_y.shape

#linear regression
#here, we map x into [1, x, x*^2, x*3, x*4]

def model(x, w):
    h1 = tf.nn.relu(tf.matmul(x, weights['l1']))
    y = tf.matmul(h1, weights['l2'])
    return y

#Main Function
if __name__ == '__main__':
    #Input and Output Definitation
    X = tf.placeholder(tf.float32, [None, 5])
    Y = tf.placeholder(tf.float32, [None, 1])

    #parameters definition and initiation
    weights = {
    'l1': tf.Variable(tf.truncated_normal([5, 20], stddev=0.01)),
    'l2': tf.Variable(tf.truncated_normal([20, 1], stddev=0.01))
    }

    #loss funcation and optimization function
    pred_Y = model(X, weights)
    loss = tf.reduce_mean( tf.pow(pred_Y - Y, 2) )
    opt = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    #deploy and parameter initiation
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    #training
    epoch = 10
    batchsize = 5
    for i in range(epoch):
        [x, y] = shuffle(x,y,random_state=2000)
        for batch in range(0, x.shape[0], batchsize):
            ops, ll = sess.run([opt, loss], feed_dict={X:x[batch:batch+batchsize], Y:y[batch:batch+batchsize]})
            print('loss value: ', ll)
    predValues = sess.run(pred_Y, feed_dict={X:val_x})
    print("valx values are : ", valx.reshape([1, 20]))
    print("valy values are : ", valy.reshape([1, 20]))
    print("Predict values are : ", predValues.reshape([1, 20]))

'''
