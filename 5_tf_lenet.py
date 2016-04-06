#Author: Xianbiao Qi
#Date:   2016.04.06

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print(mnist.train.num_examples, mnist.test.num_examples, mnist.validation.num_examples)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(np.min(mnist.train.images), np.max(mnist.train.images))


#Model Definition
def model(X, W, bias, p1, p2, p3):


    l1 = tf.nn.relu( tf.nn.conv2d(X, W['l1'], strides=[1, 1, 1, 1], padding='VALID') + biases['l1'] )
    l1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p1)

    l2 = tf.nn.relu( tf.nn.conv2d(l1, W['l2'], strides=[1, 1, 1, 1], padding='VALID') + biases['l2'] )
    l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p2)

    dim = l2.get_shape().as_list()
    l2 = tf.reshape(l2, [-1, dim[1]*dim[2]*dim[3]])
    l2 = tf.nn.dropout(l2, p3)

    fc1 = tf.nn.relu( tf.matmul(l2, weights['fc']) + biases['fc'] )



    pred_Y = tf.matmul(fc1, weights['output']) + biases['output']
    return pred_Y



#Main Function
if __name__ == '__main__':
    #Training Data
    x = mnist.train.images
    x  = x.reshape([-1, 28, 28, 1])
    y = mnist.train.labels
    val_x = mnist.test.images
    val_x = val_x.reshape([-1, 28, 28, 1])
    val_y = mnist.test.labels


    #parameters definition and initiation
    weights = {
    'l1': tf.Variable(tf.truncated_normal([5, 5, 1, 16], stddev=tf.sqrt(2.0/(5*5*1)))),
    'l2': tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=tf.sqrt(2.0/(5*5*16)))),
    'fc': tf.Variable(tf.truncated_normal([32*4*4, 128], stddev=np.sqrt(2.0/(32*4*4)))),
    'output': tf.Variable(tf.truncated_normal([128, 10], stddev=np.sqrt(1.0/128)))
    }
    biases = {
    'l1': tf.Variable(tf.constant(0.001, shape=[16], dtype=tf.float32)),
    'l2': tf.Variable(tf.constant(0.001, shape=[32], dtype=tf.float32)),
    'fc': tf.Variable(tf.constant(0.001, shape=[128], dtype=tf.float32)),
    'output': tf.Variable(tf.constant(0.001, shape=[10], dtype=tf.float32))
    }


    #Input and Output Definitation
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10])
    kp1 = tf.placeholder(tf.float32)
    kp2 = tf.placeholder(tf.float32)
    kp3 = tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)

    #loss funcation and optimization function
    pred_Y = model(X, weights, biases, kp1, kp2, kp3)
    loss = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y) )
    #opt = tf.train.MomentumOptimizer(0.0015, 0.9).minimize(loss)
    opt = tf.train.RMSPropOptimizer(lr, 0.9).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    #deploy and parameter initiation
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    #training
    epoch = 20
    batchsize = 128
    learningrate = np.linspace(0.003, 0.0003, epoch)
    for i in range(epoch):
        [x, y] = shuffle(x,y,random_state=2000)
        for batch in range(0, x.shape[0], batchsize):
            ops, ll = sess.run([opt, loss], feed_dict={X:x[batch:batch+batchsize], Y:y[batch:batch+batchsize], kp1:0.9, kp2:0.8, kp3:0.5, lr:learningrate[i]})
        print('loss value: ', ll)
        acc = sess.run([accuracy], feed_dict={X:val_x, Y:val_y, kp1:1.0, kp2:1.0, kp3:1.0})
        print("acc is : ", acc)
