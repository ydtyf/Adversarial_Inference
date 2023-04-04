import numpy as np

# 按照一定间距生成待模拟的参数空间
# Generate the parameter based on certain interval
# TODO:需要变成全形式生成
def generate_abc(parameter):
    # Set the range of values for a, b, and c, and the interval size

    # Generate a list of values for a, b, and c within the specified range and interval
    a_values = np.arange(parameter['0'][0], parameter['0'][1] + parameter['0'][2], parameter['0'][2])
    b_values = np.arange(parameter['1'][0], parameter['1'][1] + parameter['1'][2], parameter['1'][2])
    c_values = np.arange(parameter['2'][0], parameter['2'][1] + parameter['2'][2], parameter['2'][2])

    # Return the lists of values as a tuple
    return (a_values, b_values, c_values)
