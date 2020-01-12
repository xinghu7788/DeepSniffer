#!/usr/bin/env python
#coding=utf8
import os
import re

dir = os.getcwd()
logdir = os.path.join(dir, ".")

filename_list = os.listdir(logdir)
#print(filename_list)

for fn in filename_list:
    filepath = os.path.join(logdir, fn)
    
    #print(filepath)
    with open(filepath) as infile:
        lines = infile.readlines()
    last_line = lines[-1]

    match = re.search(r'avg_val_ler = (\S+)', last_line)
    if match:
        n = match.group(1)
        print("%s\t%s" % (fn, n))
    else:
        print("%s\t%s" % (fn, "ERROR"))
