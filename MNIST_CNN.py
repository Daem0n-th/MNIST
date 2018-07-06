
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[4]:


mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)


# In[5]:


def init_weights(shape):
    random_dist=tf.truncated_normal(shape,stddev=0)
    return tf.Variable(random_dist)


# In[6]:


def init_bias(shape):
    init_bias_tensor=tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_tensor)


# In[7]:


def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')


# In[8]:


def subsampling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# In[9]:


def convolution_layer(input_x,shape):
    w=init_weights(shape)
    b=init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,w)+b)


# In[10]:


def dense_layer(input_layer,size):
    input_size=int(input_layer.get_shape()[1])
    w=init_weights([input_size,size])
    b=init_bias([size])
    return tf.matmul(input_layer,w)+b


# In[11]:


x_data=tf.placeholder(tf.float32,shape=[None,784])
y_data=tf.placeholder(tf.float32,shape=[None,10])
hold_prob=tf.placeholder(tf.float32)


# In[12]:


x_image=tf.reshape(x_data,shape=[-1,28,28,1])


# In[13]:


convo_1 = convolution_layer(x_image,[5,5,1,32])
pooling_1 = subsampling(convo_1)
convo_2 = convolution_layer(pooling_1,[5,5,32,64])
pooling_2 = subsampling(convo_2)
flat_layer = tf.reshape(pooling_2,[-1,7*7*64])
dense_1 = tf.nn.relu(dense_layer(flat_layer,1024))
drop_out = tf.nn.dropout(dense_1,keep_prob=hold_prob)
y_pred = dense_layer(drop_out,10)


# In[14]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data,logits=y_pred))


# In[15]:


optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)


# In[16]:


init=tf.global_variables_initializer()


# In[ ]:


steps = 5000
with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        batch_x , batch_y = mnist.train.next_batch(50)
        feed={x_data:batch_x,y_data:batch_y,hold_prob:0.5}
        sess.run(train,feed_dict=feed)
        if i%100==0:
            acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred,1),tf.argmax(y_data,1)),tf.float32))
            print("Accuracy:")
            print(sess.run(acc,feed_dict={x_data:mnist.test.images,y_data:mnist.test.labels,hold_prob:1.0}))

