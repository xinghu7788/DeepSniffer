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
from data_processor import convert_input_to_ctc_format
from data_processor import convert_label_to_ctc_format

#meta_index = 0

kernel_dict = {'void cudnn::detail::implicit_convolve_sgemm<float, int=1024, int=6, int=7, int=3, int=3, int=5, int=1, bool=1, bool=0, bool=1>(int, int, int, float const *, int, cudnn::detail::implicit_convolve_sgemm<float, int=1024, int=6, int=7, int=3, int=3, int=5, int=1, bool=1, bool=0, bool=1>*, float const *, kernel_conv_params, int, float, float, int, float const *, float const *, int, int)': 'conv', 'void add_tensor_kernel_v3<int=2, float, float, int=128, int=1, int=1, int=4, int=2>(cudnnTensorStruct, float*, cudnnTensorStruct, float const *, float, float)': 'conv', 'void cudnn::detail::bn_fw_inf_1C11_kernel_new<float, float, bool=1, int=1>(float, cudnn::detail::bn_fw_inf_1C11_kernel_new<float, float, bool=1, int=1>, cudnnTensorStruct, float const *, float, cudnnTensorStruct*, float, cudnn::detail::bn_fw_inf_1C11_kernel_new<float, float, bool=1, int=1> const *, cudnn::detail::bn_fw_inf_1C11_kernel_new<float, float, bool=1, int=1> const , cudnn::detail::bn_fw_inf_1C11_kernel_new<float, float, bool=1, int=1> const , cudnn::detail::bn_fw_inf_1C11_kernel_new<float, float, bool=1, int=1> const , cudnn::detail::bn_fw_inf_1C11_kernel_new<float, float, bool=1, int=1>)': 'bn', 'void kernelPointwiseApply1<ThresholdUpdateOutputIP<float>, float, unsigned int, int=-2>(TensorInfo<ThresholdUpdateOutputIP<float>, float>, float, float)': 'relu', 'void im2col4d_kernel<float, int>(im2col4d_params, cudnnConvolutionStruct, cudnnTensor4dStruct, float const *, float*, int)': 'conv','cudnn_convolve_sgemm_sm35_ldg_nn_64x16x64x16x16': 'conv', 'void kern_precompute_indices<bool=0>(int*, int, int, int, int, int, int)': 'conv', 'cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x8x256x8x32': 'conv', 'void cudnn::detail::implicit_convolve_sgemm<float, int=1024, int=5, int=5, int=3, int=3, int=3, int=1, bool=1, bool=0, bool=1>(int, int, int, float const *, int, cudnn::detail::implicit_convolve_sgemm<float, int=1024, int=5, int=5, int=3, int=3, int=3, int=1, bool=1, bool=0, bool=1>*, float const *, kernel_conv_params, int, float, float, int, float const *, float const *, int, int)': 'conv', 'cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x16x128x8x32': 'conv', 'void cudnn::detail::implicit_convolve_sgemm<float, int=128, int=5, int=5, int=3, int=3, int=3, int=1, bool=1, bool=0, bool=1>(int, int, int, float const *, int, cudnn::detail::implicit_convolve_sgemm<float, int=128, int=5, int=5, int=3, int=3, int=3, int=1, bool=1, bool=0, bool=1>*, float const *, kernel_conv_params, int, float, float, int, float const *, float const *, int, int)': 'conv', 'void kernelPointwiseApply2<TensorAddOp<float>, float, float, unsigned int, int=-2, int=-2>(TensorInfo<TensorAddOp<float>, float>, TensorInfo<float, float>, float, float)': 'add', 'cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_32x16x32x8x8': 'conv', 'void cudnn::detail::implicit_convolve_sgemm<float, int=128, int=6, int=7, int=3, int=3, int=5, int=1, bool=1, bool=0, bool=1>(int, int, int, float const *, int, cudnn::detail::implicit_convolve_sgemm<float, int=128, int=6, int=7, int=3, int=3, int=5, int=1, bool=1, bool=0, bool=1>*, float const *, kernel_conv_params, int, float, float, int, float const *, float const *, int, int)': 'conv', 'void cudnn::detail::precomputed_convolve_sgemm<float, int=128, int=5, int=5, int=3, int=3, int=3, int=1, bool=1>(int, int, int, float const *, int, cudnn::detail::precomputed_convolve_sgemm<float, int=128, int=5, int=5, int=3, int=3, int=3, int=1, bool=1>*, float const *, kernel_conv_params, int, float, float, int, float const *, float const *, int*)': 'conv', 
        'void cudnn::detail::precomputed_convolve_sgemm<float, int=128, int=6, int=7, int=3, int=3, int=5, int=1, bool=1>(int, int, int, float const *, int, cudnn::detail::precomputed_convolve_sgemm<float, int=128, int=6, int=7, int=3, int=3, int=5, int=1, bool=1>*, float const *, kernel_conv_params, int, float, float, int, float const *, float const *, int*)': 'conv', 
        'cudnn_convolve_sgemm_sm35_ldg_nn_32x16x64x8x16': 'conv','void MaxPoolForward<float, float>(int, float const *, int, int, int, int, int, int, int, int, int, int, int, int, int, int, float*, long*)': 'pooling','void flip_filter<float, float>(float*, float const *, int, int, int, int)': 'conv','void fft2d_r2c_16x16<float>(float2*, float const *, int, int, int, int, int, int, int, int)': 'conv','compute_gemm_pointers(float2**, float2 const *, int, float2 const *, int, float2 const *, int, int)': 'conv','void fermiPlusCgemmLDS128_batched<bool=1, bool=0, bool=0, bool=0, int=4, int=4, int=4, int=3, int=3, bool=1, bool=0>(float2**, float2**, float2**, float2*, float2 const *, float2 const *, int, int, int, int, int, int, __int64, __int64, __int64, float2 const *, float2 const *, float2, float2, int)': 'conv','void fft2d_c2r_16x16<float, bool=0>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)': 'conv','void gemv2T_kernel_val<float, float, float, int=128, int=16, int=2, int=4, bool=0>(int, int, float, float const *, int, float const *, int, float, float*, int)': 'fc', 'void cudnn::detail::explicit_convolve_sgemm<float, int, int=512, int=6, int=8, int=3, int=3, int=5, int=0, bool=1>(int, int, int, float const *, int, float const , int, cudnn::detail::explicit_convolve_sgemm<float, int, int=512, int=6, int=8, int=3, int=3, int=5, int=0, bool=1>*, kernel_conv_params, int, int, float, float, int, float const *, float const *)': 'conv', 'void add_tensor_kernel_v3<int=2, float, float, int=16, int=16, int=1, int=16, int=4>(cudnnTensorStruct, float*, cudnnTensorStruct, float const *, float, float)': 'conv', 'void cudnn::detail::explicit_convolve_sgemm<float, int, int=1024, int=5, int=5, int=3, int=3, int=3, int=0, bool=1>(int, int, int, float const *, int, float const , int, cudnn::detail::explicit_convolve_sgemm<float, int, int=1024, int=5, int=5, int=3, int=3, int=3, int=0, bool=1>*, kernel_conv_params, int, int, float, float, int, float const *, float const *)': 'conv', 'void cudnn::winograd::generateWinogradTilesKernel<int=0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)': 'conv','void cudnn::winograd::winograd3x3Kernel<float, float, int=1, int=4, int=8, bool=1>(cudnn::maxwell::winograd::KernelParams)': 'conv','void cudnn::detail::implicit_convolve_sgemm<float, int=512, int=6, int=8, int=3, int=3, int=5, int=1, bool=1, bool=0, bool=1>(int, int, int, float const *, int, cudnn::detail::implicit_convolve_sgemm<float, int=512, int=6, int=8, int=3, int=3, int=5, int=1, bool=1, bool=0, bool=1>*, float const *, kernel_conv_params, int, float, float, int, float const *, float const *, int, int)': 'conv','void cudnn::winograd::winograd3x3Kernel<float, float, int=1, int=4, int=8, bool=0>(cudnn::maxwell::winograd::KernelParams)': 'conv', 'void cudnn::detail::explicit_convolve_sgemm<float, int, int=1024, int=6, int=7, int=3, int=3, int=5, int=0, bool=1>(int, int, int, float const *, int, float const , int, cudnn::detail::explicit_convolve_sgemm<float, int, int=1024, int=6, int=7, int=3, int=3, int=5, int=0, bool=1>*, kernel_conv_params, int, int, float, float, int, float const *, float const *)': 'conv', 'void AvePoolForward<float, float, bool=1>(int, float const *, int, int, int, int, int, int, int, int, int, int, int, int, float*)': 'pooling', 'void CatArrayBatchedCopy<float, unsigned int, int=4>(float*, CatArrInputTensor<float, unsigned int>*, OutputTensorSizeStride<unsigned int, unsigned int=4>, int, unsigned int)': 'concact'}

unknown_kernel_dict={}

def _kernel_find(fileName):
    tmplist=[]
    outputList=[]
    SampleVector=[]
    kernel_label_list = []
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
            kernel_label = _mark_label_kernel(tmplist[kernelName_index])
            if kernel_label == '':
                kernel_label = 'conv' # the unknown op is conv in default
            kernel_label_list.append(kernel_label)
    print("kernel_layer:")
    print(kernel_label_list)
    return kernel_label_list


def _mark_label_kernel(kernel_name):
    if kernel_name in kernel_dict:
        return kernel_dict[kernel_name]
    else:
        print("NOTICE: unknown kernel!")
        
        if kernel_name in unknown_kernel_dict:
            unknown_kernel_dict[kernel_name]+=1
        else:
            unknown_kernel_dict[kernel_name]=1

        return ''


def _scan_all_samplefile(in_dir):

    dir_list = os.listdir(in_dir)
 
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
            labels_filename = '_klayer_%s.log' % index
            index = dir_name + "_" + index
            print(index)

            input_log_filepath = os.path.join(dir_path, filename)
            kernel_layer_list = _kernel_find(input_log_filepath)
            output_log_filepath = os.path.join(dir_path,labels_filename )
            fileObject = open(output_log_filepath, 'w')

            linenumber = 0
            previous_kernel_layer = ''

            for kernel_layer in kernel_layer_list:
                linenumber += 1
                if kernel_layer == previous_kernel_layer:
                    continue
                
                fileObject.write(kernel_layer)
                previous_kernel_layer = kernel_layer
                fileObject.write(' ')
                fileObject.write(str(linenumber-1))
                fileObject.write('\n')
            fileObject.close()


 

def record_kernels(args):
    in_dir = os.path.join(args.base_dir, args.input)
    #out_dir = os.path.join(args.base_dir, args.output)
    #os.makedirs(out_dir, exist_ok=True)
    _scan_all_samplefile(in_dir)
    print(kernel_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.getcwd())
    parser.add_argument('--input', default='sample_generator/inception')
    args = parser.parse_args()

    record_kernels(args)
    print(unknown_kernel_dict)


if __name__ == '__main__':
    main()

