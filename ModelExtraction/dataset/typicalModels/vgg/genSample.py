


from __future__ import unicode_literals
from __future__ import division
import os, sys, time
import fileinput
import re
import logging
import argparse
import operator
import math
import codecs
import numpy as np
import sys

def _split_line_by_comma(line):
    ret_list = []
    state = 0;
    token_start = 0
    token_end = 0
    for i, c in enumerate(line):
        if c == ',':
            if state == 1:
                pass
            else:
                if token_start != i:
                    ret_list.append(line[token_start:i])
                else:
                    ret_list.append('EMPTY')
                token_tart = i + 1
        if c == '"':
            if state == 1:
                state = 0
            else:
                state = 1
        else:
            pass
        if i == len(line)-1:
            if token_start !=i:
                ret_list.append(line[token_start:])
            else:
                ret_list.append('EMPTY')
    return ret_list

def _write_to_file(list_of_list, output_filename):
    with open(output_filename, 'w') as outfile:
        for l in list_of_list:
            #outfile.write('\t'.join(l) + '\n')
            outfile.writelines(["%s\t" % item for item in l])
            outfile.writelines("\n")

def SampleFinder_TimeFile(fileName):
    import csv
    out_list = []
    tmplist = []
    funcNameDict = {"a":[1,2]} 
    with open(fileName,newline='') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            #print(len(row))
            #tmplist = '~'.join(row)
            if (len(row)) == 17:
                funcNameTemp = row[16]
                funcName = funcNameTemp.split('[')
                if funcName[0] in funcNameDict:
                    funcNameDict[funcName[0]].append(row[1])
                else:
                    tmplist=[]
                    tmplist.append(row[1])
                    funcNameDict[funcName[0]]=tmplist
                
            
    return funcNameDict
            #print('\n')
            #break
        #for row in f_csv:
            #match = re.match(r'==\d+==\s', row)
            #if match:
            #    continue
            #tokens = _split_line_by_comma(row.strip())
            #out_list.append(tokens)
            #out_list.append(row)
            #_write_to_file(row, 'temp.txt')
    #print(out_list)


            
        # readout all the rows and summarize the max layer numbers
        # get the sample out for training. give the value to
def SampleGen_AllFile(mfileName,funcDict):
    import csv
    out_list = []
    tmplist = []
    indexDict = {}
    header =1 
    with open(mfileName,newline='') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            #print(len(row))
            #tmplist = '~'.join(row)
            #if (len(row)) == ??:
            #    tmplist = print(row[16].split(' '))
            #    print('\n')
 
            #tmplist = '~'.join(row)
           
            if (len(row)) == 9:
                if (row[5]==''):
                    continue
                if (header==1):
                    header=0
                    continue
                
                tmplist=[]
                tmplist.append(row[3])
                tmplist.append(row[4])
                tmplist.append(row[5])
                #print("hi, there")
                #print(tmplist[0])
                funcNameTemp = row[3]
                funcName = funcNameTemp.split('[')
                print(funcName[0])
                kernelName = funcName[0]+' '
                if kernelName in indexDict:
                    if kernelName in funcDict:
                        #print("find funcName in funcDict")
                        exeLatList = funcDict[kernelName]
                        if indexDict[kernelName]<len(exeLatList):
                            tmplist.append(exeLatList[indexDict[kernelName]])
                            indexDict[kernelName]=indexDict[kernelName]+1
                        else:
                            indexDict[kernelName]=0
                            tmplist.append(exeLatList[0])
                            print("conflicts between memory and latency traces\n")
                            print(kernelName)
                            print("+++++")
                            print(funcDict[kernelName])
                        #print(tmplist)
                    else:
                        print("MAJOR CONFLICT, no such a kernel in time sequence")   
                        print(kernelName)
                            
                else:
                    #print("not in indexDict")
                    if kernelName in funcDict:
                        #print("not in indexDict but in funcDict")
                        indexDict[kernelName]=0
                        exeLatList = funcDict[kernelName]
                        tmplist.append(exeLatList[0])
                        #print(tmplist)
                    else:
                        print("MAJOR CONFLICT, not such a kernel in time sequence")
                        print(kernelName)
            print(tmplist)
            out_list.append(tmplist)
    return out_list

def main(argv):
    dict1=SampleFinder_TimeFile(argv[1])
    outlist=SampleGen_AllFile(argv[2],dict1)
    _write_to_file(outlist,argv[3])
    #print(outlist)


if __name__ == '__main__':

    main(sys.argv)
    

