import random
import simu_x_generate
import simu_xy_cal
import data_generating
from simulation_parameter_generating import generate_abc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_model import simu_data

# 我觉得我是否没有完全理解他的意思
# 最直观的办法应该是利用神经网络直接拟合函数，然后计算它与simulation之间的差，最后找到最小的并输出
# 我相信我可能还没有完全理解他的意思，如何使用classifier来完成这个目标

class value_compare:
    sample = []
    simulation = []
    minimal = {}
    least_loss = float('inf')
    best_simulation = []
    epochs = 0
    model = None

    def __init__(self,size = 10, epochs = 300):
        self.size = size
        self.epochs = epochs
        pass

    def compare(self, para):
        labels = np.concatenate([np.ones(self.size), np.zeros(self.size)])  # create labels (0 for small data, 1 for large data)
        x_data = np.concatenate((self.sample, self.simulation),axis=0)
        y_data = np.array(labels).T
        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
        # ])

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        indices = np.arange(len(x_data))
        np.random.shuffle(indices)
        x_shuffled = x_data[indices]
        y_shuffled = y_data[indices]


        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Allow early stop to save time
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
        # Train the model
        model.fit(x_shuffled, y_shuffled, epochs=self.epochs, batch_size=32, validation_split=0.2, callbacks=[early_stop])
        return self.update(para, model, x_data, y_data)

    def update(self, para, model, x_data, y_data):
        # we try to find the worst case
        y_pred = model.predict(x_data)
        y_simu_pred = y_pred[y_data == 0]
        y_sample_pred = y_pred[y_data == 1]
        loss = np.sum(np.log(y_sample_pred)) / self.size + np.sum(np.log(1-y_simu_pred)) / self.size

        if loss < self.least_loss:
            self.minimal = para
            self.least_loss = loss
            self.best_simulation = self.simulation
            self.model = model
        return loss

parameter_name = ['0', '1', '2']
parameter = { '0': 3,
              '1': 0,
              '2': 1}
parameter_simu_range = {
    '0': [0,3,1],
    '1': [0,3,1],
    '2': [0,3,1]
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
one_case.sample = np.array(data_generating.generate_temxy(sample_para,
                                    noise_var = 0.5)).T


# 抽取参数值的集合
# 全局损失函数
# 随机出生点次数
# 最大迭代限制次数
simulation_x_distribution = simu_x_generate.generate_simu(x_min=x_min,x_max=x_max, num_samples=whole_size)

# 迭代计算开始
simu_parameter = {'0': 1.3, '1': 0.8, '2': 1.6}
one_case.simulation = np.array(simu_xy_cal.generate_simuxy(x_values=simulation_x_distribution,
                                                                     para=simu_parameter)).T
loss_all = one_case.compare(simu_parameter)


print(one_case.minimal)
print(one_case.least_loss)

# print(loss_all)
# Create a new figure and axis object
fig, ax = plt.subplots()
x1, y1 = one_case.sample[:, 0].tolist(), one_case.sample[:, 1].tolist()
x2, y2 = one_case.best_simulation[:, 0].tolist(), one_case.best_simulation[:, 1].tolist()

# Plot the first dataset using blue circles
ax.plot(x1, y1, 'bo', label='Sample 1')

# Plot the second dataset using red squares
ax.plot(x2, y2, 'rs', label='Simulation 2')

# Add a legend to the plot
ax.legend()

# Set the axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')

x_min, x_max = min(min(x1),min(x2)), max(max(x1),max(x2))
y_min, y_max = min(min(y1),min(y2)), max(max(y1),max(y2))
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = one_case.model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)
ax.contour(xx, yy, Z, colors='black')
ax.legend()
plt.show()


# Show the plot
plt.show()

print(loss_all)
