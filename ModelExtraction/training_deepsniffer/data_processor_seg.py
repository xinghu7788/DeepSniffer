from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

layer_name_to_int_map = {'conv':0, 'fc':1, 'pooling':2, 'bn':3, 'droptout':4, 'relu':5, 'concact':6, 'add':7}
layer_int_to_name_map = {0:'conv', 1:'fc', 2:'pooling', 3:'bn', 4:'droptout',5:'relu', 6:'concact', 7:'add',8:'unknown'}
layer_unknownop = 8
#layer_name_to_int_map = {'conv':9, 'fc':10, 'pooling':0, 'bn':3, 'relu':5, 'concact':25, 'add':20}
#layer_int_to_name_map = {9:'conv', 10:'fc', 0:'pooling', 3:'bn', 5:'relu', 25:'concact', 20:'add'}
#layer_unknownop = 8


#for incept
#avg = [8.72398045e-01, 1.77459282e+05, 9.00731648e+04, 6.83415541e+01, 6.29139157e+01]
#std = [7.57980755e+00, 2.24056676e+06, 5.11390743e+05, 1.46651702e+03, 7.84557961e+02]
#avg = [8.72398045e-01, 1.77459282e+05, 9.00731648e+04, 6.83415541e+01, 6.29139157e+01, 0]
#std = [7.57980755e+00, 2.24056676e+06, 5.11390743e+05, 1.46651702e+03, 7.84557961e+02, 0]

#for resnet vgg
#avg= [2.33467596e+00 ,7.09582091e+05, 3.77132234e+05, 1.46072860e+02,8.93658187e+02]
#std= [1.82229621e+01, 6.68823787e+06, 1.54423371e+06, 1.01381564e+03,1.76410238e+04]

#avg= [2.33467596e+00 ,7.09582091e+05, 3.77132234e+05, 1.46072860e+02,8.93658187e+02,1]
#std= [1.82229621e+01, 6.68823787e+06, 1.54423371e+06, 1.01381564e+03,1.76410238e+04,1]

def read_sample_file(fileName):
    outputList=[]
    with open(fileName,'r') as infile:
        start_maker=0
        first_op=1
        w_lastLayer=0
        for row in infile:
            SampleVector = []
            tmplist = row.split('\t')
            if(len(tmplist)<3):
                continue

            kernelName_index=0
            w_index=1
            r_index=2
            lat_index=3   # in sample file, the third column is execution time

            write_accesses = float(tmplist[w_index])
            read_accesses = float(tmplist[r_index])
            execute_time = float(tmplist[lat_index])

            if (tmplist[kernelName_index]=="void kernelPointwiseApply2<softPlusupdateOutput_functor<float>, float, float, unsigned int, int=-2, int=-2>(TensorInfo<softPlusupdateOutput_functor<float>, float>, TensorInfo<float, float>, float, float)"):
                if start_maker==0:
                    start_maker=1   # it is the beginning of the ROI
                    continue
                else:
                    start_maker=0   # it is the end of the ROI
                    break

            SampleVector.append(execute_time)    #execution time
            SampleVector.append(read_accesses)   # read accesses
            SampleVector.append(write_accesses)  # write accesses
            SampleVector.append(read_accesses/write_accesses)

            if first_op==1:
                SampleVector.append(1)
                first_op=0
                w_lastLayer = write_accesses 
            else:
                SampleVector.append(w_lastLayer/write_accesses)
                w_lastLayer = write_accesses

            outputList.append(SampleVector)

    return outputList
        # readout all the rows and summarize the max layer numbers
        # get the sample out for training. give the value to   


def convert_input_to_ctc_format(sample_filename):
    '''
        inputs are reading a sequence of the profiling log
        it is two dimensiones:  k:(latency, r, w, r/w , i/o)
        (k1,k2,...,kn)
    '''
    inputs = read_sample_file(sample_filename)  # a list of list in python
    # Transform in 3D array
    if inputs == []:
        raise Exception('sampe file is empty: %s' % sample_filename)
    train_inputs = np.array(inputs)
    #print('train_inputs.shape:', train_inputs.shape)
    #train_inputs = (train_inputs - avg) / std  # FIXME: nomalization is right?
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    #print('train_inputs.shape:', train_inputs.shape)

    return train_inputs


def convert_label_to_ctc_format(label_filename):
    '''
    '''
    target_layers = []
    index_table = []
    with open(label_filename, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line != '':
                array_list = line.split(' ')
                target_layers.append(array_list[0])
                index_table.append(array_list[1])

    if target_layers == []:
        raise Exception('label file is empty: %s' % label_filename)

    # Get only the index of the layer type. Eg. 0 for conv, 1 for ReLu;
    # FIXME: wrap a funtion
    targets = []
    for layer in target_layers:
        if layer in layer_name_to_int_map:
            targets.append(layer_name_to_int_map[layer])
        else:
            targets.append(layer_unknownop)   # for the unknown op
            
    targets = np.array(targets)
    # Creating sparse representation to feed the placeholder
    #train_targets = sparse_tuple_from([targets])
    #transfer the targets to sparse format before training.
    indexs = np.array(index_table)
    return targets, indexs

    #return train_targets, indexs


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.

        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths

def convert_decode_to_str(decoded_tensor):
    d = decoded_tensor[0]
    # Replacing blank label to none
    str_decoded = ''
    for x in np.asarray(d[1]):
        if x in layer_int_to_name_map:
            str_decoded = str_decoded + layer_int_to_name_map[x]+ ' '
        else:
            print("x = %d MAJOR ERROR? OUT OF PREDICTION SCOPE" % x)
    return str_decoded


#def _test():
    #dict1=read_sample_file_TimeFile(argv[1])
    #outlist=SampleGen_AllFile(argv[2],dict1)
    #_write_to_file(outlist,argv[3])
    #print(outlist)
    #list1 = read_sample_file("vgg_sample0.log")
#    lableList = [0,1,0,1,2,3,1,3,1,3]
#    convert_input_to_ctc_format("vgg_sample0.log",lableList)
    #print(list1)

#if __name__ == '__main__':

#    _test()

