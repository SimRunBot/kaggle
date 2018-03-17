__author__ = 'Simon'

import numpy as np
import pandas as pd
import tensorflow as tf

import time

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# batch norm functions
def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    # maintains moving averages by employing exponential decay
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        #calculates mean and variance for convolutional layers
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    # apply method adds shadow copies of trained variables and add ops that maintain a moving average of the trained variables in their shadow copies.
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    # (condition, true function, false function)
    # for test cases
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    # ( scale * (input - mean) / variance ) + offset
    return Ybn, update_moving_averages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

# DATA ----------

#import data
train = pd.read_csv("./data/mnist/train.csv")

# split into data and labels
label = train["label"].as_matrix()
# one hot encoding for labels
label = LabelEncoder().fit_transform(label)[:, None]
label_one_hot = OneHotEncoder().fit_transform(label).todense()
#drop label column from train data
data = train.drop(["label"],axis=1)
#convert pandas dataframe to numpy array
data = data.as_matrix()
# convert values to float
data = data.astype(np.float32)
# reshape array
data = np.reshape(data,(-1,28,28,1))

#seperate training and validation data
train_data,validation_data = data[:-10000],data[-10000:]
train_labels, valid_labels = label_one_hot[:-10000], label_one_hot[-10000:]

print(np.shape(train_data))

# training data placeholders
tf_data = tf.placeholder(tf.float32,shape=[None,28,28,1])
tf_labels = tf.placeholder(tf.float32, shape=[None,10])
# variable learning rate
lr = tf.placeholder(tf.float32)
# dropout probability fc and conv layer
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)
# test flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
# network architecture:
# 3 conv layers (K,L,M), 1 fc layer
# shape [5, 5, 1, K] = [ filter size(2dim) , input channels, output channels]

# filter channels
K = 24  # 6  4
L = 48  # 12 8
M = 64  # 24 12
# convolution weight matrices variables initialization
w1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1),name="w1")
b1 = tf.Variable(tf.constant(0.1, tf.float32, [K]),name="b1")
w2 = tf.Variable(tf.truncated_normal([5, 5, K, L],stddev=0.1),name="w2")
b2 = tf.Variable(tf.constant(0.1, tf.float32, [L]),name="b2")
w3 = tf.Variable(tf.truncated_normal([4, 4, L, M],stddev=0.1),name="w3")
b3 = tf.Variable(tf.constant(0.1, tf.float32, [M]),name="b3")
# neurons
N = 200
# fc layer weight matrices
w4 = tf.Variable(tf.truncated_normal([7*7*M,N],stddev=0.1),name="w4")
b4 = tf.Variable(tf.constant(0.1, tf.float32, [N]),name="b4")
w5 = tf.Variable(tf.truncated_normal([N, 10],stddev=0.1),name="w5")
b5 = tf.Variable(tf.constant(0.1, tf.float32, [10]),name="b5")

# the model
# input image batch, weights, stride, padding same as border, biases
# outputs := conv1 : 28x28, conv2 : 14x14, conv3 : 7x7 | because of stride
# batch normalization split up layers into 4 terms
conv1l = tf.nn.conv2d(tf_data, w1, strides=[1, 1, 1, 1], padding="SAME")
conv1bn, update_ema1 = batchnorm(conv1l,tst,iter,b1,convolutional=True)
conv1r = tf.nn.relu(conv1bn)
conv1 = tf.nn.dropout(conv1r,pkeep_conv,compatible_convolutional_noise_shape(conv1r))

conv2l = tf.nn.conv2d(conv1, w2, strides=[1, 2, 2, 1], padding="SAME")
conv2bn, update_ema2 = batchnorm(conv2l,tst,iter,b2,convolutional=True)
conv2r = tf.nn.relu(conv2bn)
conv2 = tf.nn.dropout(conv2r,pkeep_conv,compatible_convolutional_noise_shape(conv2r))

conv3l = tf.nn.conv2d(conv2, w3, strides=[1, 2, 2, 1], padding="SAME")
conv3bn, update_ema3 = batchnorm(conv3l,tst,iter,b3,convolutional=True)
conv3r = tf.nn.relu(conv3bn)
conv3 = tf.nn.dropout(conv3r,pkeep_conv,compatible_convolutional_noise_shape(conv3r))
# flatten values for fully connected layer
flatten = tf.reshape(conv3,shape=[-1, 7 * 7 * M])

fcl = tf.matmul(flatten,w4)
fcbn, update_ema4 = batchnorm(fcl,tst,iter,b4)
fcr = tf.nn.relu(fcbn)
fc = tf.nn.dropout(fcr, pkeep)
Ylogits = tf.matmul(fc,w5) + b5
Y_pred = tf.nn.softmax(Ylogits)

#group exponential moving average objects together
update_ema = tf.group(update_ema1,update_ema2,update_ema3,update_ema4)

# loss
# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=tf_labels)
cross_entropy = tf.reduce_mean(cross_entropy ) * 100

# accuracy
correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, learning rate is placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# config for preventing oom error on gpu
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

# i had to start using cpu for more conv filter channels because gpu too small
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
# add saving functionality for checkpoint creation
saver = tf.train.Saver()

# init plus config
init = tf.global_variables_initializer()
sess = tf.Session(config=config)
sess.run(init)

# need a better method to feed batches of data into model without sklearn
#



ss = ShuffleSplit(n_splits=5000,train_size=100)
ss.get_n_splits(train_data,train_labels)
error_history = [(0,np.nan,10)]

for step, (idx, _) in enumerate(ss.split(train_data,train_labels),start=1):

     # learning rate decay
    max_learning_rate = 0.002
    min_learning_rate = 0.0001
    decay_speed = 1600.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * np.exp(-step/decay_speed)

    fd ={tf_data:train_data[idx], tf_labels:train_labels[idx], lr:learning_rate, tst:False, pkeep:0.75, pkeep_conv: 1.0}
    sess.run(train_step,feed_dict=fd)
    sess.run(update_ema, feed_dict={tf_data:train_data[idx], tf_labels:train_labels[idx], tst:False, iter:step, pkeep:1.0, pkeep_conv: 1.0})

    if step%500 == 0:
        try:
            print (time.asctime( time.localtime(time.time()) ))
        except:
            print("no time")
        fd = {tf_data:validation_data, tf_labels:valid_labels, lr:learning_rate, tst: True, pkeep: 1.0, pkeep_conv: 1.0}
        valid_loss, valid_accuracy = sess.run([cross_entropy,accuracy], feed_dict=fd)
        error_history.append((step,valid_loss,valid_accuracy))
        print("Step %i \t Valid. Acc. = %f"%(step,valid_accuracy),end="\n")
        # Save the variables to disk.
        save_path = saver.save(sess, "/checkpoint/model.ckpt")
        print("Model saved in path: %s" % save_path)


# TEST DATA PREDICTION --------------
print("TEST DATA PREDICTION --------------")
test = pd.read_csv("./data/mnist/test.csv")
test_data = test.as_matrix()
test_data = test_data.astype(np.float32)
test_data = test_data.reshape(-1,28,28,1)

#test_pred = sess.run(Y_pred,feed_dict={tf_data:test_data, pkeep:1})
# ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[28000,6,29,29]
# need to feed in test data in 4 packs to avoid OOM error
print("batch testing")
#test_labels = np.argmax(test_pred,axis=1)
test1 = test_data[0:7000]
test2 = test_data[7000:14000]
test3 = test_data[14000:21000]
test4 = test_data[21000:]
test_predictions = []
test_predictions.append(np.argmax(sess.run(Y_pred,feed_dict={tf_data:test1, tst: True, pkeep: 1.0, pkeep_conv: 1.0}),axis=1))
test_predictions.append(np.argmax(sess.run(Y_pred,feed_dict={tf_data:test2, tst: True, pkeep: 1.0, pkeep_conv: 1.0}),axis=1))
test_predictions.append(np.argmax(sess.run(Y_pred,feed_dict={tf_data:test3, tst: True, pkeep: 1.0, pkeep_conv: 1.0}),axis=1))
test_predictions.append(np.argmax(sess.run(Y_pred,feed_dict={tf_data:test4, tst: True, pkeep: 1.0, pkeep_conv: 1.0}),axis=1))
# transforming list to array
test_predictions = np.array(test_predictions)
test_predictions = np.reshape(test_predictions,(28000))
print(test_predictions.shape)
# SUBMISSION ----------
print("SUBMISSION ----------")
#submission = pd.DataFrame(data={"ImageId":(np.arange(test_labels.shape[0])+1), "Label":test_labels})
submission = pd.DataFrame(data={"ImageId":(np.arange(len(test_predictions))+1), "Label":test_predictions})
submission.to_csv("bestCNN_submission.csv",index=False)
print("done")


# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file("./checkpoint/model.ckpt", tensor_name='', all_tensors=True)
try:
    print(chkp.print_tensors_in_checkpoint_file("./checkpoint/model.ckpt", tensor_name='', all_tensors=True))
except:
    print("no ")