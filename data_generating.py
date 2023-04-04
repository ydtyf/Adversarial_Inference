import numpy as np
from data_model import simu_data
# Set the random seed for reproducibility

def generate_tem(simu_para: simu_data, noise_var = 1):
    np.random.seed(simu_para.the_seed)
    p0 = simu_para.para['0']
    p1 = simu_para.para['1']
    p2 = simu_para.para['2']

    # Set the number of samples and the range of x values
    x_values = np.random.uniform(low=simu_para.x_min, high=simu_para.x_max, size=simu_para.num_samples)

    # Evaluate the function at each x value using numpy's vectorized operations
    y_values = p2 * x_values ** 2 + x_values * p1 + p0
    noise = np.random.normal(0, noise_var, simu_para.num_samples)
    y_values += noise
    data = [list(pair) for pair in zip(x_values, y_values)]

    return data

def generate_temxy(simu_para: simu_data, noise_var = 1):
    np.random.seed(simu_para.the_seed)
    p0 = simu_para.para['0']
    p1 = simu_para.para['1']
    p2 = simu_para.para['2']

    # Set the number of samples and the range of x values
    x_values = np.random.uniform(low=simu_para.x_min, high=simu_para.x_max, size=simu_para.num_samples)

    # Evaluate the function at each x value using numpy's vectorized operations
    y_values = p2 * x_values ** 2 + x_values * p1 + p0
    noise = np.random.normal(0, noise_var, simu_para.num_samples)
    y_values += noise

    return x_values, y_values
