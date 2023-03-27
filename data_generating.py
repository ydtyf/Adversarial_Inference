import numpy as np
# Set the random seed for reproducibility

def generate_tem(x_min, x_max, para, num_samples, the_seed = 7, noise_var = 1):
    np.random.seed(the_seed)
    p0 = para['0']
    p1 = para['1']
    p2 = para['2']

    # Set the number of samples and the range of x values
    # Generate a set of x values using numpy's linspace function
    x_values = np.random.uniform(low=x_min, high=x_max, size=num_samples)

    # Evaluate the function at each x value using numpy's vectorized operations
    y_values = p2 * x_values ** 2 + 0 * p1 + p0
    noise = np.random.normal(0, noise_var, num_samples)
    y_values += noise
    data = [list(pair) for pair in zip(x_values, y_values)]

    return data
