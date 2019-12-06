
import numpy as np
from file_logger import FileLogger

file_logger = FileLogger('result.csv',['current_epoch','train_cost','train_ler','val_cost','val_ler'])
def CalEpochInterval(fileName, interval):
    import csv
    with open(fileName, newline='') as f:
        f_csv = csv.reader(f)
        index = 0
        epoch = 0
        train_cost = 0
        train_ler = 0
        val_cost = 0
        val_ler = 0
        acc = 0
        headers = next(f_csv)
        for row in f_csv:
            listline = row[0].split(' ')
            print(listline)
            train_cost = train_cost + float(listline[1])
            train_ler = train_ler + float(listline[2])
            val_cost = val_cost + float(listline[3])
            val_ler = val_ler + float(listline[4])
            if (float(listline[4]) < 0.05):
                acc = acc +1
            index = index + 1
            if index == interval-1:
                file_logger.write([epoch,train_cost/interval,train_ler/interval,acc,val_ler/interval])
                train_cost=0
                train_ler=0
                val_cost=0
                val_ler=0
                index = 0
                acc = 0
                epoch = epoch + 1
                
if __name__=='__main__':
    CalEpochInterval('input.csv',1000)
