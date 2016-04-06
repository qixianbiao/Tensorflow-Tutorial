#Author: Xianbiao Qi
#Date:   2016.04.06

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


#Model Definition
def model(X, weights, bias):
    h1 = tf.nn.relu( tf.matmul(X, weights['l1']) + biases['l1'] )
    pred_Y = tf.matmul(h1, weights['output']) + biases['output']
    return pred_Y



#Main Function
if __name__ == '__main__':
    #Training Data
    x = np.vstack([np.random.random([100, 5]) - 1.0, np.random.random([100, 5])+1.0])
    y = np.vstack([np.ones([100, 2])*[1.0, 0.0], np.ones([100, 2])*[0.0, 1.0]])
    val_x = np.vstack([np.random.random([100, 5]) - 1.0, np.random.random([100, 5])+1.0])
    val_y = np.vstack([np.ones([100, 2])*[1.0, 0.0], np.ones([100, 2])*[0.0, 1.0]])


    #parameters definition and initiation
    weights = {
    'l1': tf.Variable(tf.truncated_normal([5, 32], stddev=0.01)),
    'output': tf.Variable(tf.truncated_normal([32, 2], stddev=0.01))
    }
    biases = {
    'l1': tf.Variable(tf.constant(0.001, shape=[32], dtype=tf.float32)),
    'output': tf.Variable(tf.constant(0.001, shape=[2], dtype=tf.float32))
    }


    #Input and Output Definitation
    X = tf.placeholder(tf.float32, [None, 5])
    Y = tf.placeholder(tf.float32, [None, 2])

    #loss funcation and optimization function
    pred_Y = model(X, weights, biases)
    loss = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y) )
    opt = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    #deploy and parameter initiation
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    #training
    epoch = 10
    for i in range(epoch):
        [x, y] = shuffle(x,y,random_state=2000)
        for batch in range(0, x.shape[0], 50):
            ops, ll = sess.run([opt, loss], feed_dict={X:x[batch:batch+10], Y:y[batch:batch+10]})
            print('loss value: ', ll)
            acc = sess.run([accuracy], feed_dict={X:val_x, Y:val_y})
            print("acc is : ", acc)

