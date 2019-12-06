import os
import sys
import re
import numpy as np
import random
import math

def produce_fname(feats_filename):
    name_units = feats_filename.strip().split('_')
    feats,model,postfix=name_units
    index = postfix.strip().split('.')
    sfilename = model+'_sample'+index[0]+'.log'
    sfilepath = os.path.join(model,sfilename)
    lfilename = '_klayer_'+index[0]+'.log'
    #print(sfilename,lfilename)
    lfilepath = os.path.join(model,lfilename)
    print(lfilepath,sfilepath)
    return sfilepath, lfilepath


def val_list(meta_filename):
    data_dir = os.path.dirname(meta_filename)
    meta_filename = meta_filename

    sample_list = []
    with open(meta_filename, 'r') as infile:
        for line in infile:
            sample = line.strip().split('|')
            #index, feats_filename, feats_n_frames, labels_filename, label_n_frames = sample
            feats_filename = sample[1]
            sfile,lfile = produce_fname(feats_filename)
            sample_list.append((sfile,lfile))
            #print('load %d samples' % len(sample_list))
    training_ratio = 0.8
    training_index = math.floor(len(sample_list) * 0.8)
    training_set = sample_list[:training_index]
    testing_set = sample_list[training_index:]
    n_training_set = len(training_set)
    print('training_set has %d samples' % len(training_set))
    print('testing_set has %d samples' % len(testing_set))
    return testing_set


def main():
    #print("execute here")
    #produce_fname("feats_vgg_1022.npy")
    val_list('training_data/test_v3/train.txt')

if __name__ == '__main__':
        main()



