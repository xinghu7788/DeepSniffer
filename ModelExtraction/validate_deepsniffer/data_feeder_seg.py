import os
import sys
import re
import numpy as np
import random
import math


class DataFeeder(object):

    def __init__(self, meta_filename):
        self.data_dir = os.path.dirname(meta_filename)
        self.meta_filename = meta_filename

        self.sample_list = []
        with open(meta_filename, 'r') as infile:
            for line in infile:
                sample = line.strip().split('|')
                index, feats_filename, feats_n_frames, labels_filename, label_n_frames, segs_filename, seg_length = sample
                index = int(index)
                feats_filename = os.path.join(self.data_dir, feats_filename)
                labels_filename = os.path.join(self.data_dir, labels_filename)
                segs_filename = os.path.join(self.data_dir, segs_filename) 
                self.sample_list.append((index,
                                        feats_filename,
                                        feats_n_frames,
                                        labels_filename,
                                        label_n_frames,
                                        segs_filename,
                                        seg_length))

        print('load %d samples' % len(self.sample_list))
        self.training_ratio = 0.8
        self.training_index = math.floor(len(self.sample_list) * 0.8)
        self.training_set = self.sample_list[:self.training_index]
        self.testing_set = self.sample_list[self.training_index:]
        self.n_training_set = len(self.training_set)
        print('training_set has %d samples' % len(self.training_set))
        print('testing_set has %d samples' % len(self.testing_set))

        self.shuffle = True


    

    def next_training_batch(self):
        while True:
            if self.shuffle:
                random.shuffle(self.training_set)
            for sample in self.training_set:
                index, feats_filename, feats_n_frames, labels_filename, label_n_frames, segs_filename, seg_length = sample
                print('training:', index, os.path.basename(feats_filename), os.path.basename(labels_filename))
                train_inputs = np.load(feats_filename)  # 3-D
                seg_table = np.load(segs_filename)
                train_targets = np.load(labels_filename)
                #print('input shape:', train_inputs.shape)
                #from roi_selection import scheduler_select_seg
                from roi_selection import all_select_seg
                train_inputs, train_targets_sparse, original = all_select_seg(train_inputs, train_targets, seg_table)
                train_seq_len = [train_inputs.shape[1]] 
                #train_targets = np.load(labels_filename)
                #print('target shape:', train_targets.shape)
                #TODO: cut the region of interest according to the seg_table
                yield train_inputs, train_targets_sparse, train_seq_len, index, original

    def next_testing_batch(self):
        while True:
            if self.shuffle:     # the testing set has no need to shuffle 
                random.shuffle(self.testing_set)
            for sample in self.testing_set:
                index, feats_filename, feats_n_frames, labels_filename, label_n_frames, segs_filename, seg_length = sample
                print('testing:', index, os.path.basename(feats_filename), os.path.basename(labels_filename))
                train_inputs = np.load(feats_filename)  # 3-D
                seg_table = np.load(segs_filename)
                train_targets = np.load(labels_filename)

                from roi_selection import all_select_seg            
                train_inputs, train_targets_sparse, original = all_select_seg(train_inputs, train_targets, seg_table)
                train_seq_len = [train_inputs.shape[1]] 
 

                yield train_inputs, train_targets_sparse, train_seq_len, index, original



