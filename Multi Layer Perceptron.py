import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist=input_data.read_data_sets("data/",one_hot=True)

X=tf.placeholder(tf.float32,[None,784])
Y=tf.placeholder(tf.float32,[None,10])

lr=1e-3
batch_size=512

layer0=784
layer1=256
layer2=128
layer3=64
layer4=32
layer_out=10

W0=tf.Variable(tf.random.truncated_normal([layer0,layer1],stddev=0.1),name="W0")
b0=tf.Variable(tf.zeros(layer1),name="b0")

W1=tf.Variable(tf.random.truncated_normal([layer1,layer2],stddev=0.1),name="W1")
b1=tf.Variable(tf.zeros(layer2),name="b1")

W2=tf.Variable(tf.random.truncated_normal([layer2,layer3],stddev=0.1),name="W2")
b2=tf.Variable(tf.zeros(layer3),name="b2")

W3=tf.Variable(tf.random.truncated_normal([layer3,layer4],stddev=0.1),name="W3")
b3=tf.Variable(tf.zeros(layer4),name="b3")

W_out=tf.Variable(tf.random.truncated_normal([layer4,layer_out],stddev=0.1),name="W_out")
b_out=tf.Variable(tf.zeros(10),name="b_out")


y0=tf.nn.tanh(tf.linalg.matmul(X,W0)+b0)
y1=tf.nn.tanh(tf.linalg.matmul(y0,W1)+b1)
y2=tf.nn.tanh(tf.linalg.matmul(y1,W2)+b2)
y3=tf.nn.tanh(tf.linalg.matmul(y2,W3)+b3)
y=tf.nn.softmax(tf.linalg.matmul(y3,W_out)+b_out)

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
