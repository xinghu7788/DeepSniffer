

from data_processor_seg import sparse_tuple_from
add_op = 7
incept_op = 6

def scheduler_select_seg(train_inputs, train_targets, seg_table):
    import random
    flag = random.randint(0,300)
    if flag < 100:
        return hotspot_select_seg(train_inputs, train_targets, seg_table)
    elif flag <200:
        return random_select_seg(train_inputs, train_targets, seg_table)
    else:
        return end_select_seg(train_inputs, train_targets,seg_table)

def random_select_seg(train_inputs, train_targets, seg_table):

    #print("inputs, targets, seg_table")
    #print(train_inputs)
    #print(train_targets)
    #print(seg_table)

    length = len(seg_table)
    #print("length:")
    #print(length)
    
    import random
    seg_index_start = random.randint(0,length-3)
    #seg_index_end = random.randint(seg_index_start+1, length-1)
    r_seg_index_end = min(seg_index_start + 120, length-1)   #fixed the length as 100?
    seg_index_end = random.randint(seg_index_start+1, r_seg_index_end)
    input_start = int(seg_table[seg_index_start])

    if seg_index_end == length-1:
        input_end = int(seg_table[seg_index_end])+1
    else:
        input_end = int(seg_table[seg_index_end])
    
    print("seg_start, seg_end, input_start, input_end")
    print(seg_index_start,seg_index_end,input_start,input_end)

    # cut the roi out of train_inputs and train_targets
    roi_inputs = train_inputs[:,input_start:input_end,:] 
    #print("roi_inputs")
    #print(roi_inputs)

    roi_targets = train_targets[seg_index_start:seg_index_end]
    #print("roi_targets")
    #print(roi_targets)
    from data_processor_seg import sparse_tuple_from
    roi_targets_sparse = sparse_tuple_from([roi_targets])
    return roi_inputs, roi_targets_sparse, roi_targets

def all_select_seg(train_inputs, train_targets, seg_table):
    roi_inputs = train_inputs 
    roi_targets = train_targets
    from data_processor_seg import sparse_tuple_from
    roi_targets_sparse = sparse_tuple_from([roi_targets])
    return roi_inputs, roi_targets_sparse, roi_targets

def find_hotspot(train_targets, op_hot):
    hotop_pos_list = []
    op_index = 0
    for op_id in train_targets:
        op_index += 1
        #print("op_id")
        #print(op_id)
        if op_id != op_hot:
            continue
        hotop_pos_list.append(op_index-1)
    #print("positions:")
    #print(hotop_pos_list)
    if hotop_pos_list == []:
        return -1
    else:
        import random
        i = random.randint(0, len(hotop_pos_list)-1)
        return hotop_pos_list[i]

def hotspot_select_seg(train_inputs, train_targets, seg_table):

    length = len(seg_table)
    hot_pos = -1
    hot_pos_add = find_hotspot(train_targets,add_op)
    hot_pos_incept = find_hotspot(train_targets, incept_op)

    #print(hot_pos_add)
    if(hot_pos_add == -1):
        if(hot_pos_incept == -1):
            return(random_select_seg(train_inputs,train_targets,seg_table))
        else:
            hot_pos = hot_pos_incept
    else:
        hot_pos = hot_pos_add

    import random
    left_max_range = hot_pos
    right_max_range = length - hot_pos
    r_left = max(0,left_max_range-70)
    #r_right = min()
    left_range = random.randint(r_left, left_max_range)
    right_range = random.randint(2, right_max_range)
    if right_range-left_range>120:
        right_range = left_range + random.randint(0,120)

    seg_index_start = hot_pos - left_range
    seg_index_end = hot_pos + right_range-1
    input_start = int(seg_table[seg_index_start])
    input_end = int(seg_table[seg_index_end])
    #print(seg_index_start, seg_index_end, input_start, input_end)





    roi_inputs = train_inputs[:,input_start:input_end,:] 

    roi_targets = train_targets[seg_index_start:seg_index_end]

    from data_processor_seg import sparse_tuple_from
    roi_targets_sparse = sparse_tuple_from([roi_targets])
    return roi_inputs, roi_targets_sparse, roi_targets

def end_select_seg(train_inputs, train_targets, seg_table):

    length = len(seg_table)
    range_left = max(0, length-120)
    import random
    seg_index_start = random.randint(range_left,length-3)
    #seg_index_end = random.randint(seg_index_start+1, length-1)
    seg_index_end = length-1
    input_start = int(seg_table[seg_index_start])

    input_end = int(seg_table[seg_index_end])+1
    
    print("seg_start, seg_end, input_start, input_end")
    print(seg_index_start,seg_index_end,input_start,input_end)

    roi_inputs = train_inputs[:,input_start:input_end,:] 
    roi_targets = train_targets[seg_index_start:seg_index_end]
    #print("roi_targets")
    #print(roi_targets)
    from data_processor_seg import sparse_tuple_from
    roi_targets_sparse = sparse_tuple_from([roi_targets])
    return roi_inputs, roi_targets_sparse, roi_targets



