import numpy as np

def generate_abc(interval = 1):
    # Set the range of values for a, b, and c, and the interval size
    min_val = -5
    max_val = 5

    # Generate a list of values for a, b, and c within the specified range and interval
    a_values = np.arange(0, 5 + interval, interval)
    b_values = np.arange(0, 0 + interval, interval)
    c_values = np.arange(0, 5 + interval, interval)

    # Return the lists of values as a tuple
    return (a_values, b_values, c_values)
