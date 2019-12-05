#!/usr/bin/env python
#coding='utf8'

''' convert *.log -> *.npy
'''

import os
import sys
import argparse
import re
import random
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from data_processor_seg import convert_input_to_ctc_format
from data_processor_seg import convert_label_to_ctc_format

#meta_index = 0
sample_file_list =[]
label_file_list =[]
def _process_all_sample(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the xxx dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have the dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples.
    '''

    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_single_sample on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    dir_list = os.listdir(in_dir)
    print(dir_list)

    for dir_name in dir_list:
        dir_path = os.path.join(in_dir, dir_name)

        filename_list = os.listdir(dir_path)
        re_sample_filename = re.compile(r'.+_sample(\d+)\.log')
        #print(root)
        #print(dirs)
        #print(files)

        #for dirname in dirs: 
        print("#dirname")
        #print(dirs)
        for filename in filename_list:
            print("#filename")
            print(filename)
            match = re_sample_filename.match(filename)
            if not match:
                continue
            print("match")
            index = match.group(1)
            index = dir_name + "_" + index
            print(index)

            input_log_filepath = os.path.join(dir_path, filename)

            futures.append(executor.submit(partial(_process_single_sample, input_log_filepath, out_dir, index)))

    return [future.result() for future in tqdm(futures)]


def _process_single_sample(input_log_filepath, out_dir, index):
    '''Preprocesses a single log file.

    Args:
    out_dir: The directory to write the spectrograms into

    Returns:
    A (index, feats_filename, n_frames) tuple to write to train.txt
    '''

    # Load the sample to a numpy array:
    try:
        train_inputs = convert_input_to_ctc_format(input_log_filepath)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return

    n_frames = train_inputs.shape[0]

    # 2D -> 3D
    train_inputs = np.reshape(train_inputs, (1,train_inputs.shape[0], train_inputs.shape[1]))

    print("index", index)
    print(train_inputs)


    # Write to disk:
    feats_filename = 'feats_%s.npy' % index
    np.save(os.path.join(out_dir, feats_filename), train_inputs, allow_pickle=False)

    # Return a tuple describing this training example:
    return (index, feats_filename, n_frames)


def _process_all_label(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the xxx dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have the dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples.
    '''

    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_single_label on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []



    dir_list = os.listdir(in_dir)
    print(dir_list)

    for dir_name in dir_list:
        dir_path = os.path.join(in_dir, dir_name)

        filename_list = os.listdir(dir_path)
        #re_label_filename = re.compile(r'.+_label(\d+)\.log')
        re_label_filename = re.compile('_klayer_(\d+)\.log')


        for filename in filename_list:
            print("#filename")
            print(filename)
            match = re_label_filename.match(filename)
            if not match:
                continue
            print("match")
            index = match.group(1)
            index = dir_name + "_" + index
            print(index)

            label_filepath = os.path.join(dir_path, filename)

            futures.append(executor.submit(partial(_process_single_label, label_filepath, out_dir, index))) # parallel processing
        #futures.append(_process_single_label(label_filepath, out_dir, index)) # serial processing

    return [future.result() for future in tqdm(futures)] # parallel processing
    #return futures  # serial processing


def _process_single_label(label_filepath, out_dir, index):
    '''Preprocesses a single log file.

    Args:
    out_dir: The directory to write the labels into

    Returns:
    A (index, label_filename, n_frames) tuple to write to train.txt
    '''
    #print('process label file:', label_filepath)
    # Load the file to a numpy array:
    try:
        train_targets, seg_table = convert_label_to_ctc_format(label_filepath)
        print("train_targets")
        print(train_targets)
        print("index_table")
        print(seg_table)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return
    #indices, values, shape = train_targets  # TODO: sparse tensor is error?!

    #n_frames = values.shape[0]
    n_frames = len(train_targets)
    line_number = seg_table[len(seg_table)-1]
    #print('indices:',indices)
    #print('values:', values)
    #print('shape:',shape)

    # Write to disk:
    #labels_filename = 'labels_%s.npy' % index
    labels_filename = 'klayer_%s.npy' % index
    seg_filename = 'seg_%s.npy' % index



    print("Labelindex", index)
    #print(train_targets)
    #np.save(os.path.join(out_dir, labels_filename), train_targets, allow_pickle=False)
    np.save(os.path.join(out_dir, labels_filename), train_targets, allow_pickle=True)
    np.save(os.path.join(out_dir, seg_filename), seg_table, allow_pickle=True)

    # Return a tuple describing this training example:
    return (index, labels_filename, n_frames, seg_filename, line_number)


def _write_all_metadata(sample_metadata, label_metadata, out_dir):
    '''
    wirte metadata to 'train.txt'
    '''
    # find common index in samples and labels
    meta_index = 0
    index_to_sample = {m[0]: m for m in sample_metadata if m is not None}
    index_to_label = {m[0]: m for m in label_metadata if m is not None}
    common_metadata = []
    for index, sample in index_to_sample.items():
        if index in index_to_label:
            label = index_to_label[index]
            common_metadata.append((meta_index, sample[1], sample[2], label[1], label[2], label[3], label[4]))
            meta_index = meta_index + 1

    random.shuffle(common_metadata)

    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in common_metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')

    print('Read sample files: %d, label files: %d, in common: %d' % \
            (len(sample_metadata), len(label_metadata), len(common_metadata)))

    sample_frames = sum([m[2] for m in common_metadata])
    label_frames = sum([m[4] for m in common_metadata])
    print('Wrote total %d frames of sample' % (sample_frames,))
    print('Wrote total %d frames of label' % (label_frames,))

    print('Max sample length:  %d' % max(m[2] for m in common_metadata))
    print('Max label length: %d' % max(m[4] for m in common_metadata))




def prepare_data(args):
    in_dir = os.path.join(args.base_dir, args.input)
    out_dir = os.path.join(args.base_dir, args.output)
    os.makedirs(out_dir, exist_ok=True)

    sample_metadata = _process_all_sample(in_dir, out_dir, args.num_workers)
    label_metadata = _process_all_label(in_dir, out_dir, args.num_workers)
    _write_all_metadata(sample_metadata, label_metadata, out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.getcwd())
    parser.add_argument('--input', default='sample_generator/test_set_new')
    parser.add_argument('--output', default='training_data/nas_p_cutConcat_slices')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    args = parser.parse_args()

    prepare_data(args)


if __name__ == '__main__':
    main()

