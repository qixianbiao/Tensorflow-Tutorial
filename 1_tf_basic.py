#Author: Xianbiao Qi
#Date:   2016.04.06

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Constant, Sequences and Random Values
#constant
aa = tf.zeros([2, 3], dtype=tf.float32, name='zeroconstant')
bb = tf.ones([2, 2], dtype=tf.float32, name='oneconstant')

#sequence
cc = tf.linspace(0.0, 1, 10, name='seq1')
dd = tf.range(10, name='seq2')

#random
ee = tf.random_normal([2,3], mean=10.0, stddev=1.0, name='random_normal')
ff = tf.truncated_normal([2,2], mean=10.0, stddev=1.0, name='truncnormal')

#deploy
sess = tf.Session()
[a, b, c, d, e, f] = sess.run([aa, bb, cc, dd, ee, ff])
print("aa: ", a)
print("bb: ", b)
print("cc: ", c)
print("dd: ", d)
print("ee: ", e)
print("ff: ", f)
sess.close()




#math
aa = tf.random_normal([2, 2], mean=0.0, stddev=1.0)
bb = tf.ones([2, 2], dtype=tf.float32)
cc = tf.random_normal([2, 1], mean=0.0, stddev=1.0)

dd = tf.add(aa, bb)
ee = tf.sub(aa, bb)
ff = tf.mul(aa, bb)
gg = tf.matmul(aa, cc)

#deploy
sess = tf.Session()
[a, b, c, d, e, f, g] = sess.run([aa, bb, cc, dd, ee, ff, gg])
print("aa: ", a)
print("bb: ", b)
print("cc: ", c)
print("dd: ", d)
print("ee: ", e)
print("ff: ", f)
print("gg: ", g)
sess.close()
#tf.abs, tf.neg, tf.sign, tf.inv, tf.square, tf.round
#tf.sqrt, tf.rsqrt




#Variables
weight = tf.Variable(tf.random_normal([3, 10], mean=0.0, stddev=0.01, dtype=tf.float32), name='nnweights')
bias = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[10]), name='nnbiases')

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.matmul(x, weight) + bias

sess = tf.Session()
sess.run(tf.initialize_all_variables())
xinput = np.random.rand(5, 3)
y_ = sess.run(y, feed_dict={x: xinput})
print y_
sess.close()






#neural network
from skimage import data
img = data.lena()
img = np.float32(img.reshape([1, img.shape[0], img.shape[1], img.shape[2]]))
print img.shape
print img.dtype
W = tf.Variable(tf.random_normal([5, 5, 3, 10], mean=0.0, stddev=0.01), name='nnweights2')
B = tf.Variable(tf.constant(0.01, shape=[10]), name='nnbiases2')
y1 = tf.nn.conv2d(img, W, strides=[1, 1, 1, 1], padding='SAME')
y2 = tf.nn.conv2d(img, W, strides=[1, 2, 2, 1], padding='SAME')
y3 = tf.nn.conv2d(img, W, strides=[1, 1, 1, 1], padding='VALID')
y4 = tf.nn.conv2d(img, W, strides=[1, 2, 2, 1], padding='VALID')


y5 = tf.nn.relu(tf.nn.conv2d(img, W, strides=[1, 1, 1, 1], padding='SAME') + B)
y6 = tf.nn.sigmoid(tf.nn.conv2d(img, W, strides=[1, 1, 1, 1], padding='SAME') + B)

y7 = tf.nn.dropout(y5, 0.8)
y8 = tf.nn.max_pool(y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
sess = tf.Session()
sess.run(tf.initialize_all_variables())
[y1_, y2_, y3_, y4_, y5_, y6_, y7_, y8_] = sess.run([y1, y2, y3, y4, y5, y6, y7, y8])
print y1_.shape
print y2_.shape
print y3_.shape
print y4_.shape
print y5_.shape
print y6_.shape
print y7_.shape
print y8_.shape




#Finally,
#In fact, there are more contents needed to be learned, such as,
#tf.image,  tf.train,  tf.TFRecordReader and so on, please read code and check the API document










#the following code is from https://github.com/pkmital/tensorflow_tutorials/blob/master/python/01_basics.py
#with a slight change, you can continue practicing more TF
'''

n_values = 30
x = tf.linspace(-3.0, 3.0, n_values)

sess = tf.Session()
result = sess.run(x)
print "result, ", result
print x.eval(session=sess)

sigma = 1.0
mean = 0.0
z = (tf.exp(tf.neg( tf.pow(x-mean, 2.0)/(2.0* tf.pow(sigma, 2.0)))*
(1.0/(sigma * tf.sqrt(2.0*3.1415)))))

assert  z.graph is tf.get_default_graph()

plt.plot(sess.run(z))

dim = z.get_shape().as_list()

print(dim)

dim1 = sess.run(tf.shape(z))
print dim1

print sess.run(tf.pack([tf.shape(z), tf.shape(z), [3], [4]]))

aa = np.random.random([3, 3])
bb = np.random.random([3, 3])
cc = sess.run(tf.pack([aa, bb]))
print cc.shape

dd = sess.run(tf.concat(0, [aa, bb]))
print dd.shape


z_2d = tf.matmul(tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))
plt.imshow(sess.run(z_2d))
print sess.run(z_2d).shape

x = tf.reshape(tf.sin(tf.linspace(-3.0, 3.0, n_values)),[n_values, 1])
y = tf.reshape(tf.ones_like(x), [1, n_values])
z = tf.mul(tf.matmul(x, y), z_2d)
plt.imshow(sess.run(z))

ops = tf.get_default_graph().get_operations()
print([op.name for op in ops])


def gabor(n_values=32, sigma=1.0, mean=0.0):
    x = tf.linspace(-3.0, 3.0, n_values)
    z = (tf.exp(tf.neg(tf.pow(x-mean, 2.0)/(2.0*tf.pow(sigma, 2.0))))*
    (1.0/(sigma * tf.sqrt(2.0*3.1415))))
    gauss_kernel = tf.matmul(tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))
    x = tf.reshape(tf.sin(tf.linspace(-3.0, 3.0, n_values)), [n_values, 1])
    y = tf.reshape(tf.ones_like(x), [1, n_values])
    gabor_kernel = tf.mul(tf.matmul(x, y), gauss_kernel)
    return gabor_kernel

plt.imshow(sess.run(gabor()))
print sess.run( gabor() )



def convolve(img, W):
    if len(W.get_shape()) == 2:
        dims  = W.get_shape().as_list() + [1, 1]
        W = tf.reshape(W, dims)

    if len(img.get_shape()) == 2:
        dims = [1] + img.get_shape().as_list() + [1]
        img = tf.reshape(img, dims)
    elif len(img.get_shape()) == 3:
        dims = [1] + img.get_shape().as_list()
        img = tf.reshape(img, dims)
        W = tf.concat(2, [W, W, W])

    convolved = tf.nn.conv2d(img, W, strides=[1, 1, 1, 1], padding='SAME')
    return convolved

from skimage import data
img = data.lena()

plt.imshow(img)
print(img.shape)

x = tf.placeholder(tf.float32, shape=img.shape)

out = convolve(x, gabor())

result = sess.run(tf.squeeze(out), feed_dict={x: img})
plt.imshow(result)

'''