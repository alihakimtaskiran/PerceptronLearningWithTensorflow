import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist=input_data.read_data_sets("data/",one_hot=True)

X=tf.placeholder(tf.float32,[None,784])
Y=tf.placeholder(tf.float32,[None,10])

lr=1e-3
batch_size=512

W=tf.Variable(tf.random.truncated_normal([784,10],stddev=0.1),name="W")
b=tf.Variable(tf.zeros(10),name="b")

y=tf.nn.softmax(tf.linalg.matmul(X,W)+b)

xent=-tf.reduce_sum(Y*tf.math.log(y))

correct_pred=tf.equal(tf.argmax(Y,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

optimizer=tf.train.AdamOptimizer(lr).minimize(xent)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(15000):
    batch_x,batch_y=mnist.train.next_batch(batch_size)
    sess.run(optimizer,feed_dict={X:batch_x,Y:batch_y})
    if i%100==0:
      acc,loss=sess.run([accuracy,xent],feed_dict={X:batch_x,Y:batch_y})
      print("Iteration",i,"Acc="+str(acc),"Minibatch Loss="+str(loss))
  test_acc,test_loss=sess.run([accuracy,xent],feed_dict={X:mnist.test.images,Y:mnist.test.labels})
  print("Test Accuracy="+str(test_acc),"Test Loss="+str(test_loss))
