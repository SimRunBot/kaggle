import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# PARAMETERS ----------

LABELS = 10 # Number of different types of labels (1-10)
WIDTH = 28 # width / height of the image
CHANNELS = 1 # Number of colors in the image (greyscale)

VALID = 10000 # Validation data size

STEPS = 15000 # 3500  # Number of steps to run
BATCH = 100 # Stochastic Gradient Descent batch size
PATCH = 5 # Convolutional Kernel size
DEPTH = 8 #8 # Convolutional Kernel depth size == Number of Convolutional Kernels
HIDDEN = 400 #100 # Number of hidden neurons in the fully connected layer

LR = 0.001 # Learning rate

# DATA ----------

#import data
train = pd.read_csv("./train.csv")


# split into data and labels
label = train["label"].as_matrix()
# one hot encoding for labels

label = LabelEncoder().fit_transform(label)[:, None]
label_one_hot = OneHotEncoder().fit_transform(label).todense()

data = train.drop(["label"],axis=1)
data = data.as_matrix() #convert pandas dataframe to numpy array

# print(data[0].dtype)
# reshape array
# convert values to float
data = data.astype(np.float32)
# print(data[0].dtype)
data = np.reshape(data,(-1,28,28,1))

#seperate training and validation data
train_data,validation_data = data[:-10000],data[-10000:]
train_labels, valid_labels = label_one_hot[:-VALID], label_one_hot[-VALID:]
# some python list indexing checks
# test = [1,2,3,4]
# print(test[-2])
# print(test[-2:])
# print(test[:-2])

# uncomment for shape checks
# with tf.Session() as sess:
#     print('train data shape = ' + str(train_data.get_shape()))
#
#     print('labels shape = ' + str(label_one_hot.get_shape()))


# MODEL -----------

# initialize input data placeholders
tf_data = tf.placeholder(tf.float32, shape=(None,WIDTH,WIDTH,CHANNELS))
tf_labels = tf.placeholder(tf.float32,shape=(None,LABELS))

# network architecture
# 4 layers: 2 conv 2 fully connected, output layer with one hot encoding
# WIDTH = 28| CHANNELS = 1| LABELS = 10| PATCH = 5| DEPTH = 8| HIDDEN = 100
# conv layer 1
w1 = tf.Variable(tf.truncated_normal([PATCH,PATCH,CHANNELS,DEPTH],stddev=0.1))
#bias
b1 = tf.Variable(tf.zeros([DEPTH]))

# conv layer 2
w2 = tf.Variable(tf.truncated_normal(([PATCH,PATCH,DEPTH,DEPTH*2])))
b2 = tf.Variable(tf.constant(1.0,shape=[2*DEPTH]))

# hidden layer 1 // means floor operation aka 9//4 = 2
w3 = tf.Variable(tf.truncated_normal([WIDTH // 4*WIDTH // 4* 2*DEPTH,HIDDEN],stddev=0.1))
b3 = tf.Variable(tf.constant(1.0,shape=[HIDDEN]))

# hidden layer 2
w4 = tf.Variable(tf.truncated_normal([HIDDEN,LABELS],stddev=0.1))
b4 = tf.Variable(tf.constant(1.0,shape=[LABELS]))

def logits(data):
    #conv1
    x = tf.nn.conv2d(data,w1,[1,1,1,1],padding="SAME")
    x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding="SAME")
    x = tf.nn.relu(x+b1)

    #conv2
    x = tf.nn.conv2d(x,w2,[1,2,2,1],padding="SAME")
    x = tf.nn.relu(x + b2)

    # fully connected layer 1
    x = tf.reshape(x,(-1,WIDTH//4*WIDTH//4*DEPTH*2))
    x = tf.matmul(x, w3)
    x = tf.nn.relu(x + b3)

    # fully connected layer 2
    x = tf.matmul(x,w4) + b4

    return x

# PREDICTION
tf_pred = tf.nn.softmax(logits(tf_data))
tf_acc = 100*tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf_pred, 1), tf.argmax(tf_labels, 1))))

# softmax cross entropy expects unscaled logits, computes softmax cross entropy loss between logits and labels
tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits(tf_data),labels=tf_labels))

# Learning Rate optimizer
#tf_opt = tf.train.GradientDescentOptimizer(LR)
#tf_opt = tf.train.AdamOptimizer(LR)
tf_opt = tf.train.RMSPropOptimizer(LR)
tf_step = tf_opt.minimize(tf_loss)

# TRAIN -------------

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

ss = ShuffleSplit(n_splits=STEPS,train_size=BATCH)
ss.get_n_splits(train_data,train_labels)
error_history = [(0,np.nan,10)]

for step, (idx, _) in enumerate(ss.split(train_data,train_labels),start=1):
    fd ={tf_data:train_data[idx], tf_labels:train_labels[idx]}
    session.run(tf_step,feed_dict=fd)
    if step%500 == 0:
        fd = {tf_data:validation_data, tf_labels:valid_labels}
        valid_loss, valid_accuracy = session.run([tf_loss,tf_acc], feed_dict=fd)
        error_history.append((step,valid_loss,valid_accuracy))
        print("Step %i \t Valid. Acc. = %f"%(step,valid_accuracy),end="\n")


# VISUALIZING ----------

# un"zip"s the history triple into three parts
# steps, loss, acc = zip(*error_history)
# fig =plt.figure()
# plt.title("Validation Loss/ Accuracy")
# ax_loss = fig.add_subplot(111)
# ax_acc = ax_loss.twinx()
# plt.xlabel("Training Steps")
# plt.xlim(0,max(steps))
#
#
# ax_loss.plot(steps, loss, '-o', color='C0')
# ax_loss.set_ylabel('Log Loss', color='C0');
# ax_loss.tick_params('y', colors='C0')
# ax_loss.set_ylim(0.01, 0.5)
#
# ax_acc.plot(steps, acc, '-o', color='C1')
# ax_acc.set_ylabel('Accuracy [%]', color='C1');
# ax_acc.tick_params('y', colors='C1')
# ax_acc.set_ylim(1,100)
#
# plt.show()

# TEST DATA PREDICTION --------------
print("TEST DATA PREDICTION --------------")
test = pd.read_csv("./test.csv")
test_data = test.as_matrix()
test_data = test_data.astype(float)
test_data = test_data.reshape(-1,WIDTH,WIDTH,CHANNELS)

test_pred = session.run(tf_pred,feed_dict={tf_data:test_data})
test_labels = np.argmax(test_pred,axis=1)


# PLOT AN EXAMPLE -------
# k = 0
# print("LABEL PREDICTION : %i"%test_labels[k])
# fig = plt.figure(figsize=(2,2))
# plt.axis("off")
# plt.imshow(test_data[k,:,:,0])
# plt.show()

# SUBMISSION ----------
print("SUBMISSION ----------")
submission = pd.DataFrame(data={"ImageId":(np.arange(test_labels.shape[0])+1), "Label":test_labels})
submission.to_csv("submission.csv",index=False)

