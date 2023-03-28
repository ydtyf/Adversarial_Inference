# Calculate the y values given the sample result of x values
import numpy as np

def generate_simu(x_values, para, the_seed=9):
    p0 = para['0']
    p1 = para['1']
    p2 = para['2']

    # Evaluate the function at each x value using numpy's vectorized operations
    y_values = p2 * x_values ** 2 + 0 * p1 + p0
    # If needed, I can add the noise also on the simulation data
    # num_samples = len(x_values)
    # np.random.seed(the_seed)
    # noise = np.random.normal(0, 2, num_samples)
    data = [list(pair) for pair in zip(x_values, y_values)]

    # return full scale data of x and y
    return data

def generate_simuy(x_values, para, the_seed=9):
    p0 = para['0']
    p1 = para['1']
    p2 = para['2']

    # Evaluate the function at each x value using numpy's vectorized operations
    y_values = p2 * x_values ** 2 + 0 * p1 + p0
    # If needed, I can add the noise also on the simulation data
    # num_samples = len(x_values)
    # np.random.seed(the_seed)
    # noise = np.random.normal(0, 2, num_samples)

    # return full scale data of x and y
    return y_values