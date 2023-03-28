import data_generating
import simu_x_generate
import simu_xy_cal
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_model import simu_data
class value_compare:
    sampley = []
    samplex = []
    simulationx = []
    simulationy = []
    minimal = {}
    least_loss = float('-inf')
    best_simulation = []
    epochs = 0

    def __init__(self,size = 10, epochs = 300):
        self.size = size
        self.epochs = epochs
        pass

    def compare(self, para):
        x_data = np.array(self.samplex)
        y_data = np.array(self.sampley)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(1,), activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(1, activation='linear')
        ])

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

        # Train the model
        model.fit(x_data, y_data, epochs=1000, verbose=0)

        return self.update(para, model, x_data, y_data)

    def update(self, para, model, x_data, y_data):
        # we try to find the worst case
        x_test = np.array(self.simulationx)
        y_real = np.array(self.simulationy)
        y_pred = model.predict(x_test).reshape(500,)
        each_loss = np.square(np.subtract(y_pred, y_real))
        loss = np.sum(np.square(np.subtract(y_pred, y_real)))
        # loss = tf.keras.losses.mean_squared_error(y_real, y_pred).numpy()

        if loss > self.least_loss:
            self.minimal = para
            self.least_loss = loss
            self.best_simulation = self.simulationy
        return loss

parameter_name = ['0', '1', '2']
parameter = { '0': 3,
              '1': 0,
              '2': 1}
parameter_simu_range = {
    '0': [0,3,0.5],
    '1': [0,3,0.5],
    '2': [0,3,0.5]
}
x_min = -3
x_max = 3
whole_size = 500
loss_f = {}
epochs = 500
sample_para = simu_data(num_samples=whole_size,para=parameter,x_min=x_min,x_max=x_max,the_seed=7)

# 定义比较模型，存储结果
one_case = value_compare(size=whole_size, epochs=epochs)
# 存储sample数据
one_case.samplex, one_case.sampley = data_generating.generate_temxy(sample_para, noise_var = 0.5)

one_case.simulationx = simu_x_generate.generate_simu(x_min=x_min,x_max=x_max, num_samples=whole_size)

simu_parameter = {'0': 0, '1': 2.5, '2': 2}
one_case.simulationy = simu_xy_cal.generate_simuy(x_values=one_case.simulationx, para=simu_parameter)
one_case.compare(simu_parameter)

# 迭代计算开始

print(one_case.minimal)
print(one_case.least_loss)

# print(loss_all)
# Create a new figure and axis object
fig, ax = plt.subplots()
x1, y1 = one_case.samplex, one_case.sampley
x2, y2 = one_case.simulationx, one_case.best_simulation

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