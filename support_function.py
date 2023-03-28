def check_function(idx, parameter_name, para_range):
    check = True
    for para_idx in len(idx):
        left = para_range[parameter_name[para_idx]][0]
        right = para_range[parameter_name[para_idx]][1]
        if idx[para_idx] < left | idx[para_idx] > right:
            check = False
    return check