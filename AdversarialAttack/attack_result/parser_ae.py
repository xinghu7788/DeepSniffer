import numpy as np

length = 15
file_list = ['densenet.log','deepsniffer.log', 'vgg.log', 'resnet.log', 'mix.log']
#file_list = ['our.log']
for file_name in file_list:
    f = open(file_name)
    data = f.readlines()
    f.close()
    length = len(data)
    print(length)
    dif_suc_list = []
    dif_all_list = []
    suc_rate = []
    #dif_suc_list.append(float(d[-3]))
    #dif_all_list.append(float(d[-1]))
    for i in range(length):
        d = data[i] 
        d = d.split()
        dif_suc_list.append(float(d[-3]))
        dif_all_list.append(float(d[-1]))
        suc_rate.append(float(d[7]))
    dif_suc_array = np.sqrt(np.asarray(dif_suc_list))
    dif_all_array = np.sqrt(np.asarray(dif_all_list))
    suc_rate_array = np.asarray(suc_rate)
    print(file_name)
    print('attack_succ_rate:\t', np.mean(suc_rate_array))
    print('succ_distance:\t', np.mean(dif_suc_array))
    print('all_distance:\t', np.mean(dif_all_array))
    print('\n')

