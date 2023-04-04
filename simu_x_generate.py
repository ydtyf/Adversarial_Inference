# generate the same distribution of the x values
import numpy as np

# 根据给定X分布生成一个相同的X分布
# Generate a new X distribution based on given sample
# TODO:注意，新生成的东西可能要考虑X之间的cor
def generate_simu(x_min, x_max, num_samples, the_seed = 8):
    np.random.seed(the_seed)
    # Set the number of simulation x and the range of x values
    # TODO this part requires further work on it, I need to make a real distribution sample
    x_values = np.random.uniform(low=x_min, high=x_max, size=num_samples)

    return x_values