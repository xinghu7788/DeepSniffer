from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
#from python_speech_features import mfcc

SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space
#avg = [8.72398045e-01, 1.77459282e+05, 9.00731648e+04, 6.83415541e+01, 6.29139157e+01]
#std = [7.57980755e+00, 2.24056676e+06, 5.11390743e+05, 1.46651702e+03, 7.84557961e+02]

indexDict = {'conv':0, 'fc':1, 'pooling':2, 'bn':3, 'dropout':4, 'relu':5, 'concact':6, 'add':7}
#indexDict = {'conv':9, 'fc':10, 'pooling':0, 'bn':3, 'relu':5, 'concact':25, 'add':20}
#layer_int_to_name_map = {9:'conv', 10:'fc', 0:'pooling', 3:'bn', 5:'relu', 25:'concact', 20:'add'}
#
#indexDict = {'conv':0, 'relu':1}
unknownop = 8
max_percentage = 0
def add_noise(value):
    import random
    percentage = random.randint(0, 2*max_percentage)
    value_noise = value*(percentage-max_percentage)/100 + value
    return value_noise


def SampleFinder(fileName):
    tmplist=[]
    outputList=[]
    SampleVector=[]
    start_maker=0
    first_op=1
    lat_index=3   # in sample file, the third column is execution time
    kernelName_index=0
    r_index=2
    w_index=1
    w_lastLayer=0

    #print(fileName)
    with open(fileName,'r') as infile:
        for row in infile:
            tmplist = []
            SampleVector = []
            tmplist = row.split('\t')
            if(len(tmplist)<3):
                continue
            if (tmplist[kernelName_index]=="void kernelPointwiseApply2<softPlusupdateOutput_functor<float>, float, float, unsigned int, int=-2, int=-2>(TensorInfo<softPlusupdateOutput_functor<float>, float>, TensorInfo<float, float>, float, float)"):
                if start_maker==0:
                    start_maker=1   # it is the beginning of the ROI
                    continue
                else:
                    start_maker=0   # it is the end of the ROI
                    break
            write_accesses = float(tmplist[w_index])
            read_accesses = float(tmplist[r_index])
            write_accesses = add_noise(write_accesses)
            read_accesses = add_noise(read_accesses)

            SampleVector.append(float(tmplist[lat_index]))   #execution time
            SampleVector.append(read_accesses)   # read accesses
            SampleVector.append(write_accesses)   # write accesses
            SampleVector.append(read_accesses/write_accesses)

            if first_op==1:
                SampleVector.append(1)
                first_op=0
                w_lastLayer = write_accesses 
            else:
                SampleVector.append(w_lastLayer/write_accesses)
                w_lastLayer = write_accesses

            #print(SampleVector)
            outputList.append(SampleVector)
    #print(outputList)   
    if outputList == []:
        return [];
    train_inputs = np.array(outputList)
    #print(train_inputs.shape)
    #print(train_inputs)
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs) # this normalization operation may need re-design()
    #print(train_seq_len)
    train_seq_len = [train_inputs.shape[1]]
    #print(train_seq_len)

    return outputList #train_inputs
        # readout all the rows and summarize the max layer numbers
        # get the sample out for training. give the value to   



def convert_inputs_to_ctc_format(sampleName, target_layers):
    #print(target_layers)

    #inputs are reading a sequence of the profiling log
    #it is two dimensiones:  k:(latency, r, w, r/w , i/o)
    #(k1,k2,...,kn)
    
    inputs = SampleFinder(sampleName)
    # Transform in 3D array
    if inputs == []:
        return [],[],[]
    train_inputs = np.array(inputs)
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    #train_inputs = (train_inputs - avg)/std
    #train_seq_len = [train_inputs.shape[1]]
    train_seq_len = [train_inputs.shape[0]] 
    # the sequence length of the kernel profiling log.


    # Get only the index of the layer type. Eg. 0 for conv, 1 for ReLu;
    targets = []#.split(',')
    # do we need to translate the char to int?
    for i in range(0, len(target_layers)):
        if target_layers[i] in indexDict:
            targets.append(indexDict[target_layers[i]])
        else:
            targets.append(unknownop)   # for the unknown op
            
    targets = np.array(targets)

    #targets = original.replace(' ', '  ')
    #targets = targets.split(' ')

    # Adding blank label
    #targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

    # Transform char into index
    #targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX  for x in targets])

    # Creating sparse representation to feed the placeholder
    train_targets = sparse_tuple_from([targets])
    #print("train_seq_len:")
    #print(train_seq_len)
    #print("train_inputs")
    #print(train_inputs)
  
    return train_inputs, train_targets, train_seq_len#, original


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



def main():
    #dict1=SampleFinder_TimeFile(argv[1])
    #outlist=SampleGen_AllFile(argv[2],dict1)
    #_write_to_file(outlist,argv[3])
    #print(outlist)
    #list1 = SampleFinder("vgg_sample0.log")
    lableList = [0,1,0,1,2,3,1,3,1,3]
    convert_inputs_to_ctc_format("vgg_sample0.log",lableList)
    #print(list1)

if __name__ == '__main__':

    main()

