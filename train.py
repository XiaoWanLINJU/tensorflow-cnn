#! /usr/bin/env python
#-*-encoding:utf-8-*-
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
import parameter
#from tensorflow.contrib import learn



# Training parameters
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
#print("\nParameters:") 
#for attr, value in sorted(FLAGS.__flags.items()): 
#    print("{}={}".format(attr.upper(), value))
#print("")
# TODO: print all parameters in parameter file

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x, y = data_helpers.load_cntkfile(parameter.dada_path + '/mt03.cntk')# 
print parameter.dada_path + '/mt03.cntk'

# TODO: This is very crude, should use cross-validation
x_train, x_dev = x[:-300], x[-300:]
y_train, y_dev = y[:-300], y[-300:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

def weight_varible(shape):
    initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)

def conv2d(x, W):
    print x
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

def batch_mean_var_norm(x):
    batch_mean,  batch_var = tf.nn.moments(x, [0])# 0 means batch
    xn = (x - batch_mean) / tf.sqrt(batch_var + 1e-6)# epsilon
    return xn

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # paras
        W_conv1 = weight_varible([ 2, parameter.input_dim, parameter.input_channel, parameter.output_channel])
        # width, height, inputchannel, outputchannel
        b_conv1 = bias_variable([parameter.output_channel])#each outputchannel has a bias
        # conv layer-1
        input_x = tf.placeholder(tf.float32, [None, 2, parameter.input_dim, 1], name="input_x")
        # any batch, width, height, inputchannel
        xn = batch_mean_var_norm(input_x)
        h_conv1 = tf.nn.relu(conv2d(xn, W_conv1) + b_conv1)
#        h_pool1 = max_pool_2x2(h_conv1)
        # full connection
        # input layer's output channel has the dimention: 1*8*32, fully connected layer has 10 dimention
#        W_fc1 = weight_varible([2 * parameter.input_dim * parameter.output_channel, parameter.hidden_layer_size])
        # dimention stay same, maybe the padding 
#        b_fc1 = bias_variable([parameter.hidden_layer_size])

#        h_pool2_flat = tf.reshape(h_pool1, [-1, 2 * parameter.input_dim * parameter.output_channel])
#        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout
        keep_prob = tf.placeholder(tf.float32)
#        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # output layer: softmax
#        W_fc2 = weight_varible([parameter.hidden_layer_size, parameter.class_num])
        h_conv1_flat = tf.reshape(h_conv1,  [-1,  2 * parameter.input_dim * parameter.output_channel])
        b_fc2 = bias_variable([parameter.class_num])
        W_fc2 = weight_varible([2 * parameter.input_dim * parameter.output_channel,  parameter.class_num])
        #just one conv layer
        y_conv = tf.nn.softmax(tf.matmul(h_conv1_flat, W_fc2) + b_fc2)
        input_y = tf.placeholder(tf.float32, [None, parameter.class_num], name="input_y")# #no of classes is 2
        global_step = tf.Variable(0, name="global_step", trainable=False)
        
        # model training
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_conv, input_y)
        loss = tf.reduce_mean(cross_entropy)#average the cross entropy values across the batch dimension 
        train_step = tf.train.GradientDescentOptimizer(1e-1)
        train_op = train_step.minimize(loss, global_step=global_step)

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(input_y, 1))#element truth, index the maxvalue of dimention 1, version difference here with argmax
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", loss)
        acc_summary = tf.scalar_summary("accuracy", accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())


        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_each_epoch(x_batch, y_batch): 
            """
            A single training step
            """
            feed_dict = {
              input_x: np.reshape(np.asarray(x_batch), (parameter.batch_size, 2, parameter.input_dim, 1)),
              input_y: np.array(y_batch),
              keep_prob: 1.0
            }
            _, step, summaries = sess.run([train_op, global_step, train_summary_op], feed_dict=feed_dict)# return fetch, which is the first para in sess.run
            lo = loss.eval(feed_dict)
            ac = accuracy.eval(feed_dict)
            #op = y_conv.eval(feed_dict)
             
            # not as list, 
            time_str = datetime.datetime.now().isoformat()
            train_summary_writer.add_summary(summaries, step)
            print("{}: batch {}, trainingloss {:g}, trainingacc {:g}".format(time_str,  step,  lo,  ac))
            #print op
        def dev_each_epoch(x_dev, y_dev, step, writer=None):
            """
            Evaluates model on a dev set
            """
            lo = 0.0
            ac = 0.0
            feed_dict = {
                input_x: np.reshape(np.asarray(x_dev),  (len(x_dev), 2,  parameter.input_dim, 1)),
                input_y: y_dev,
                keep_prob: 1.0
            }
            summaries = sess.run( dev_summary_op,feed_dict)            
            lo += loss.eval(feed_dict)
            ac += accuracy.eval(feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: batch {}, devloss {:g}, devacc {:g}".format(time_str, step, lo, ac))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches for each train process
        batches = data_helpers.batch_iter(# list of list(each batch)
            list(zip(x_train, y_train)), parameter.batch_size, parameter.num_epoch)
        print len(x_train), parameter.batch_size, parameter.dev_epoch, (len(x_train) * 1.0 / parameter.batch_size)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            current_step = tf.train.global_step(sess, global_step)
            train_each_epoch(x_batch, y_batch)
           # TODO: user friendly show..
            if current_step % (parameter.dev_epoch * (len(x_train) * 1.0 / parameter.batch_size)) == 0:# dev as each epoch end
                #print("\nEvaluation:")
                #batches2 = data_helpers.batch_iter(
                #    list(zip(x_dev, y_dev)), parameter.batch_size, 1)
                dev_each_epoch(x_dev, y_dev, current_step, writer=dev_summary_writer)
                print("")
            if current_step % (parameter.checkpoint_every * (len(x_train) * 1.0 / parameter.batch_size))== 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                #print("Saved model checkpoint to {}\n".format(path))
