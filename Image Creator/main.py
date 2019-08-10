import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
metrics=[]
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("data/mnist",one_hot=True)

img=tf.placeholder(tf.float32,[None,784])
num=tf.placeholder(tf.float32,[None,10])

w0=tf.Variable(np.random.randn(10,50).astype(np.float32))
b0=tf.Variable(np.random.randn(50).astype(np.float32))
w1=tf.Variable(np.random.randn(50,85).astype(np.float32))
b1=tf.Variable(np.random.randn(85).astype(np.float32))
w2=tf.Variable(np.random.randn(85,100).astype(np.float32))
b2=tf.Variable(np.random.randn(100).astype(np.float32))
w3=tf.Variable(np.random.randn(100,128).astype(np.float32))
b3=tf.Variable(np.random.randn(128).astype(np.float32))
w4=tf.Variable(np.random.randn(128,257).astype(np.float32))
b4=tf.Variable(np.random.randn(257).astype(np.float32))
w5=tf.Variable(np.random.randn(257,512).astype(np.float32))
b5=tf.Variable(np.random.randn(512).astype(np.float32))
w6=tf.Variable(np.random.randn(512,600).astype(np.float32))
b6=tf.Variable(np.random.randn(600).astype(np.float32))
w7=tf.Variable(np.random.randn(600,700).astype(np.float32))
b7=tf.Variable(np.random.randn(700).astype(np.float32))
w_out=tf.Variable(np.random.randn(700,784).astype(np.float32))
b_out=tf.Variable(np.random.randn(784).astype(np.float32))

y=tf.nn.tanh(tf.linalg.matmul(num,w0)+b0)
y=tf.nn.tanh(tf.linalg.matmul(y,w1)+b1)
y=tf.nn.tanh(tf.linalg.matmul(y,w2)+b2)
y=tf.nn.tanh(tf.linalg.matmul(y,w3)+b3)
y=tf.nn.tanh(tf.linalg.matmul(y,w4)+b4)
y=tf.nn.tanh(tf.linalg.matmul(y,w5)+b5)
y=tf.nn.tanh(tf.linalg.matmul(y,w6)+b6)
y=tf.nn.tanh(tf.linalg.matmul(y,w7)+b7)
y=tf.nn.sigmoid(tf.linalg.matmul(y,w_out)+b_out)

l2=tf.reduce_sum(tf.square(img-y))
optimizer1=tf.train.AdamOptimizer(1e-3).minimize(l2)
optimizer2=tf.train.RMSPropOptimizer(0.0045).minimize(l2)
saver=tf.train.Saver()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(50000):
    im,n=mnist.test.next_batch(256)
    if i%2==0:
      sess.run(optimizer1,feed_dict={img:im,num:n})
    if i%100==0:
      loss=sess.run(l2,feed_dict={img:im,num:n})
      metrics.append(loss)
      random_val=np.random.randint(0,10)
      rand_vec=np.zeros(10).astype(np.float32)
      rand_vec[random_val]=1
      print("Iter",i,"Minibatch Loss="+str(loss))
      fig=sess.run(y,feed_dict={num:(rand_vec,)}).reshape(28,28)
      plt.imshow(fig,cmap="gray")
      plt.show()
      print("Number is",random_val)
    else:
      sess.run(optimizer2,feed_dict={img:im,num:n})
  saver.save(sess,"data/model/model.cktp")
  plt.plot(metrics,"-g.")
