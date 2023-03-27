from data_generating import generate_tem
from simulation_parameter_generating import generate_abc
import matplotlib.pyplot as plt
import itertools
from data_simulation import generate_simu
import numpy as np
import tensorflow as tf

class value_compare:
    sample = []
    simulation = []
    minimal = {}
    least_loss = float('inf')

    def __init__(self,size = 10):
        self.size = size
        pass

    def compare(self, para):
        labels = np.concatenate([np.ones(self.size), np.zeros(self.size)])  # create labels (0 for small data, 1 for large data)
        x_data = np.concatenate((np.array(self.sample).T, np.array(self.simulation).T), axis=1).T
        y_data = np.array(labels).T
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # Train the model
        model.fit(x_data, y_data, epochs=300, batch_size=32, validation_split=0.2)
        return self.update(para, model, x_data, y_data)

    def update(self, para, model, x_data, y_data):
        loss = -model.evaluate(x_data, y_data)[0]
        if loss < self.least_loss:
            self.minimal = para
            self.least_loss = loss
        return loss

parameter = { '0': 3,
              '1': 0,
              '2': 1}
x_min = -10
x_max = 10
whole_size = 50
loss_f = {}

one_case = value_compare(whole_size)
one_case.sample = list(generate_tem(num_samples = whole_size,
                                    x_min = x_min,
                                    para = parameter,
                                    x_max = x_max,
                                    the_seed = 7))

a_values, b_values, c_values = generate_abc(interval=1)

simu_parameter = {'0': 1,
                 '1': 0,
                 '2': 1}
one_case.simulation = list(generate_simu(num_samples = whole_size,
                                             x_min = x_min,
                                             x_max = x_max,
                                             para = simu_parameter,
                                             the_seed = 7))
loss_f[str(3)+'-'+str(0)+'-'+str(1)] = one_case.compare(simu_parameter)

print(one_case.minimal)
print(loss_f)

# Create a new figure and axis object
fig, ax = plt.subplots()
x1, y1 = zip(*one_case.sample)
x2, y2 = zip(*one_case.simulation)

# Plot the first dataset using blue circles
ax.plot(x1, y1, 'bo', label='Sample 1')

# Plot the second dataset using red squares
ax.plot(x2, y2, 'rs', label='Simulation 2')

# Add a legend to the plot
ax.legend()

# Set the axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Show the plot
plt.show()

