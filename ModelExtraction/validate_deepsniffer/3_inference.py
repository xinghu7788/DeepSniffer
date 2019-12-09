#!/usr/bin/env python

import time
import argparse
import os
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf

from data_processor_seg import layer_int_to_name_map, convert_decode_to_str
from data_feeder_seg import DataFeeder
#from hparams import hparams, hparams_debug_string #TODO: put all hyper-param in hparams
from util import infolog, plot, ValueWindow
log = infolog.log

# Some configs. (latency, r, w, r/w, i/o)
num_features = 5

# OP classes + blank + others (10)
num_classes = 10

# Hyper-parameters
num_epochs = 100   #10000
num_hidden = 128
num_layers = 1

batch_size = 1

def add_stats(cost, ler):
  with tf.variable_scope('stats') as scope:
    #tf.summary.histogram('linear_outputs', model.linear_outputs)
    #tf.summary.histogram('linear_targets', model.linear_targets)
    tf.summary.scalar('val_cost', cost)  # FIXME: now only feed with val data
    tf.summary.scalar('val_ler', ler)
    #tf.summary.scalar('learning_rate', model.learning_rate)
    return tf.summary.merge_all()

def next_infer_batch(sample_file, label_file):

    row = ''
    label_list =[]
    with open(label_file, 'r') as lfile:
        for row in lfile:
            line_list = row.split(' ')  # only one line for label sequence 
            label_list.append(line_list[0])
            print("label sequence")
            print(label_list)

    from getSample import convert_inputs_to_ctc_format
    infer_inputs, infer_targets, infer_seq_len = convert_inputs_to_ctc_format(sample_file, label_list)
    infer_inputs = np.reshape(infer_inputs,(1,infer_inputs.shape[0],5))
    return infer_inputs, infer_targets, infer_seq_len, label_list

def run_ctc(log_dir, args):
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
    #input_path = os.path.join(args.base_dir, args.input)
    log('Checkpoint path: %s' % checkpoint_path)
    #log('Loading training data from: %s' % input_path)
    log('Using model: %s' % args.model)
    #log(hparams_debug_string())

    # Set up DataFeeder:
    #dataset = DataFeeder(input_path)

    # Build the model
    graph = tf.Graph()
    with graph.as_default():
        # e.g: log filter bank or MFCC features
        # Has size [batch_size, max_step_size, num_features], but the
        # batch_size and max_step_size can vary along each step
        inputs = tf.placeholder(tf.float32, [None, None, num_features]) #batch size = 1
        #inputs = tf.placeholder(tf.float32, [None,num_features])
        # Here we use sparse_placeholder that will generate a
        # SparseTensor required by ctc_loss op.
        targets = tf.sparse_placeholder(tf.int32)

        # 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None])

        # Defining the cell
        # Can be:
        #   tf.nn.rnn_cell.RNNCell
        #   tf.nn.rnn_cell.GRUCell
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

        # Stacking rnn cells
        stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                            state_is_tuple=True)

        # The second output is the last state and we will no use that
        outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

        shape = tf.shape(inputs)
        batch_s, max_time_steps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, num_hidden])

        # Truncated normal with mean 0 and stdev=0.1
        # Tip: Try another initialization
        # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
        W = tf.Variable(tf.truncated_normal([num_hidden,
                                             num_classes],
                                            stddev=0.1))
        # Zero initialization
        # Tip: Is tf.zeros_initializer the same?
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, num_classes])

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))

        loss = tf.nn.ctc_loss(targets, logits, seq_len)
        cost = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer().minimize(cost)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)
        #optimizer = tf.train.MomentumOptimizer(learning_rate=0.005, momentum=0.9).minimize(cost)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

        # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              targets))

        stats = add_stats(cost, ler)

        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

    # Bookkeeping:
    time_window = ValueWindow(100)
    train_cost_window = ValueWindow(100)
    train_ler_window = ValueWindow(100)
    val_cost_window = ValueWindow(100)
    val_ler_window = ValueWindow(100)
  
    # Run!
    with tf.Session(graph=graph) as sess:
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        if args.restore_step:
            # Restore from a checkpoint if the user requested it.
            restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
            saver.restore(sess, restore_path)
            log('Resuming from checkpoint: %s' % (restore_path,))
        else:
            log('Starting new training run')


        for index_val in range(0,1):
            index_val = index_val + 1
            sample_file = args.sample_file
            label_file = args.label_file
            val_inputs, val_targets, val_seq_len, val_original  = next_infer_batch(sample_file, label_file)
            #print(val_inputs)
            print(val_targets)
            val_feed = {inputs: val_inputs,
                        targets: val_targets,
                        seq_len: val_seq_len}
            val_cost, val_ler = sess.run([cost, ler], feed_dict=val_feed)
            val_cost_window.append(val_cost)
            val_ler_window.append(val_ler)

            # Decoding
            d = sess.run(decoded[0], feed_dict=val_feed)
            # Replacing blank label to none
            str_decoded = ''
            for x in np.asarray(d[1]):
                if x in layer_int_to_name_map:
                    str_decoded = str_decoded + layer_int_to_name_map[x] + ' '
                else:
                    print("x=%d MAJOR ERROR? OUT OF PREDICTION SCOPE" % x)

            print('for Sample %s' % sample_file)
            print('Original val: %s' % val_original) # TODO
            print('Decoded val: %s' % str_decoded)

            message = "avg_train_cost = {:.3f}, avg_train_ler = {:.3f}, " \
                      "val_cost = {:.3f}, val_ler = {:.3f}, " \
                      "avg_val_cost = {:.3f}, avg_val_ler = {:.3f}"
            log(message.format(
                     train_cost_window.average, train_ler_window.average,
                    val_cost, val_ler,
                    val_cost_window.average, val_ler_window.average))


            # END OF TRAINING


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.getcwd())
    parser.add_argument('--input', default='training_data/train.txt')
    parser.add_argument('--model', default='first_ctc')
    parser.add_argument('--hparams', default='',
        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
    parser.add_argument('--summary_interval', type=int, default=1, help='Steps between running summary ops.')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Steps between writing checkpoints.')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    parser.add_argument('--sample_file', type=str)
    parser.add_argument('--label_file',type=str)
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.model

    log_dir = os.path.join(args.base_dir, 'predictors/logs_%s' % run_name)
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'inference.log'), run_name)
    #hparams.parse(args.hparams) #FIXME

    run_ctc(log_dir, args)

if __name__ == '__main__':
    main()

