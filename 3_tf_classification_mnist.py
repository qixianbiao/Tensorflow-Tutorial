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






#Part 1: basic framework


#Model Definition
def model(X, weights, bias):
    h1 = tf.nn.relu( tf.matmul(X, weights['l1']) + biases['l1'] )
    pred_Y = tf.matmul(h1, weights['output']) + biases['output']
    return pred_Y



#Main Function
if __name__ == '__main__':
    #Training Data
    x = mnist.train.images
    y = mnist.train.labels
    val_x = mnist.test.images
    val_y = mnist.test.labels


    #parameters definition and initiation
    weights = {
    'l1': tf.Variable(tf.truncated_normal([784, 64], stddev=0.01)),
    'output': tf.Variable(tf.truncated_normal([64, 10], stddev=0.01))
    }
    biases = {
    'l1': tf.Variable(tf.constant(0.001, shape=[64], dtype=tf.float32)),
    'output': tf.Variable(tf.constant(0.001, shape=[10], dtype=tf.float32))
    }


    #Input and Output Definitation
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])

    #loss funcation and optimization function
    pred_Y = model(X, weights, biases)
    loss = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y) )
    opt = tf.train.GradientDescentOptimizer(0.001).minimize(loss)  #try 0.01
    #opt = tf.train.MomentumOptimizer(0.0010, 0.9).minimize(loss)
    #opt = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
    #opt = tf.train.AdamOptimizer(0.001, 0.9).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    #deploy and parameter initiation
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    #training
    epoch = 20
    batchsize = 128
    for i in range(epoch):
        [x, y] = shuffle(x,y,random_state=2000)
        for batch in range(0, x.shape[0], batchsize):
            ops, ll = sess.run([opt, loss], feed_dict={X:x[batch:batch+batchsize], Y:y[batch:batch+batchsize]})
        print('loss value: ', ll)
        acc = sess.run([accuracy], feed_dict={X:val_x, Y:val_y})
        print("acc is : ", acc)







#Part 2: Initiation and more nodes, dropout
'''
#Model Definition
def model(X, weights, bias, p1):
    h1 = tf.nn.relu( tf.matmul(X, weights['l1']) + biases['l1'] )
    h1 = tf.nn.dropout(h1, p1)
    pred_Y = tf.matmul(h1, weights['output']) + biases['output']
    return pred_Y



#Main Function
if __name__ == '__main__':
    #Training Data
    x = mnist.train.images
    y = mnist.train.labels
    val_x = mnist.test.images
    val_y = mnist.test.labels


    #parameters definition and initiation
    weights = {
    'l1': tf.Variable(tf.truncated_normal([784, 256], stddev=np.sqrt(2.0/784))),
    'output': tf.Variable(tf.truncated_normal([256, 10], stddev=np.sqrt(1.0/256)))
    }
    biases = {
    'l1': tf.Variable(tf.constant(0.001, shape=[256], dtype=tf.float32)),
    'output': tf.Variable(tf.constant(0.001, shape=[10], dtype=tf.float32))
    }


    #Input and Output Definitation
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    kp = tf.placeholder(tf.float32)

    #loss funcation and optimization function
    pred_Y = model(X, weights, biases, kp)
    loss = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y) )
    #opt = tf.train.MomentumOptimizer(0.0015, 0.9).minimize(loss)
    opt = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    #deploy and parameter initiation
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    #training
    epoch = 20
    batchsize = 128
    for i in range(epoch):
        [x, y] = shuffle(x,y,random_state=2000)
        for batch in range(0, x.shape[0], batchsize):
            ops, ll = sess.run([opt, loss], feed_dict={X:x[batch:batch+batchsize], Y:y[batch:batch+batchsize], kp:0.6})
        print('loss value: ', ll)
        acc = sess.run([accuracy], feed_dict={X:val_x, Y:val_y, kp:1.0})
        print("acc is : ", acc)



'''












#Part 3: Adaptive Learning Rate:

'''
#Model Definition
def model(X, weights, bias, p1):
    h1 = tf.nn.relu( tf.matmul(X, weights['l1']) + biases['l1'] )
    h1 = tf.nn.dropout(h1, p1)
    pred_Y = tf.matmul(h1, weights['output']) + biases['output']
    return pred_Y



#Main Function
if __name__ == '__main__':
    #Training Data
    x = mnist.train.images
    y = mnist.train.labels
    val_x = mnist.test.images
    val_y = mnist.test.labels


    #parameters definition and initiation
    weights = {
    'l1': tf.Variable(tf.truncated_normal([784, 256], stddev=np.sqrt(2.0/784))),
    'output': tf.Variable(tf.truncated_normal([256, 10], stddev=np.sqrt(1.0/256)))
    }
    biases = {
    'l1': tf.Variable(tf.constant(0.001, shape=[256], dtype=tf.float32)),
    'output': tf.Variable(tf.constant(0.001, shape=[10], dtype=tf.float32))
    }


    #Input and Output Definitation
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    kp = tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)

    #loss funcation and optimization function
    pred_Y = model(X, weights, biases, kp)
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
            ops, ll = sess.run([opt, loss], feed_dict={X:x[batch:batch+batchsize], Y:y[batch:batch+batchsize], kp:0.6, lr:learningrate[i]})
        print('loss value: ', ll)
        acc = sess.run([accuracy], feed_dict={X:val_x, Y:val_y, kp:1.0})
        print("acc is : ", acc)

'''







#Part 4: Deeper MLP
'''
#Model Definition
def model(X, weights, bias, p1, p2):
    h1 = tf.nn.relu( tf.matmul(X, weights['l1']) + biases['l1'] )
    h1 = tf.nn.dropout(h1, p1)
    h2 = tf.nn.relu( tf.matmul(h1, weights['l2']) + biases['l2'] )
    h2 = tf.nn.dropout(h2, p2)
    pred_Y = tf.matmul(h2, weights['output']) + biases['output']
    return pred_Y



#Main Function
if __name__ == '__main__':
    #Training Data
    x = mnist.train.images
    y = mnist.train.labels
    val_x = mnist.test.images
    val_y = mnist.test.labels


    #parameters definition and initiation
    weights = {
    'l1': tf.Variable(tf.truncated_normal([784, 256], stddev=np.sqrt(2.0/784))),
    'l2': tf.Variable(tf.truncated_normal([256, 256], stddev=np.sqrt(2.0/256))),
    'output': tf.Variable(tf.truncated_normal([256, 10], stddev=np.sqrt(1.0/256)))
    }
    biases = {
    'l1': tf.Variable(tf.constant(0.001, shape=[256], dtype=tf.float32)),
    'l2': tf.Variable(tf.constant(0.001, shape=[256], dtype=tf.float32)),
    'output': tf.Variable(tf.constant(0.001, shape=[10], dtype=tf.float32))
    }


    #Input and Output Definitation
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    kp1 = tf.placeholder(tf.float32)
    kp2 = tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)

    #loss funcation and optimization function
    pred_Y = model(X, weights, biases, kp1, kp2)
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
            ops, ll = sess.run([opt, loss], feed_dict={X:x[batch:batch+batchsize], Y:y[batch:batch+batchsize], kp1:0.9, kp2:0.6, lr:learningrate[i]})
        print('loss value: ', ll)
        acc = sess.run([accuracy], feed_dict={X:val_x, Y:val_y, kp1:1.0, kp2:1.0})
        print("acc is : ", acc)
'''








