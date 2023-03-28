import random
import simu_x_generate
import simu_xy_cal
from data_generating import generate_tem
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
    least_loss = float('-inf')
    best_simulation = []
    epochs = 0

    def __init__(self,size = 10, epochs = 300):
        self.size = size
        self.epochs = epochs
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
        model.fit(x_data, y_data, epochs=self.epochs, batch_size=32, validation_split=0.2)
        return self.update(para, model, x_data, y_data)

    def update(self, para, model, x_data, y_data):
        # we try to find the worst case
        loss = model.evaluate(x_data, y_data)[0]
        if loss > self.least_loss:
            self.minimal = para
            self.least_loss = loss
            self.best_simulation = self.simulation
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
one_case.sample = list(generate_tem(sample_para,
                                    noise_var = 0.5))

# 抽取参数值的集合
a_values, b_values, c_values = generate_abc(parameter_simu_range)
# 全局损失函数
loss_all = [[[0 for _ in range(len(a_values))] for _ in range(len(b_values))] for _ in range(len(c_values))]
# 随机出生点次数
repeat = 10
# 最大迭代限制次数
max_repeat = len(a_values) * len(b_values) * len(c_values)
simulation_x_distribution = simu_x_generate.generate_simu(x_min=x_min,x_max=x_max, num_samples=whole_size)

# 迭代计算开始
while repeat > 0 and max_repeat > 0:
    loop_keep = True
    max_repeat -= 1
    a_best, b_best, c_best = -1, -1, -1

    while loop_keep:
        # 如果还未出生，则随机选择出生地点
        if a_best == -1:
            a_idx = random.randrange(len(a_values))
            b_idx = random.randrange(len(b_values))
            c_idx = random.randrange(len(c_values))
        # 利用深度优先搜索，朝着loss最大化的地方进发
        else:
            a_idx, b_idx, c_idx = a_best, b_best, c_best

        # 防止数组溢出
        if loss_all[a_idx][b_idx][c_idx] == 0:
            simu_parameter = {'0': a_values[a_idx],
                              '1': b_values[b_idx],
                              '2': c_values[c_idx]}
            one_case.simulation = list(simu_xy_cal.generate_simu(x_values=simulation_x_distribution,
                                                                 para=simu_parameter))
            loss_all[a_idx][b_idx][c_idx] = one_case.compare(simu_parameter)
            # 每计算多一个，那么最大计算上限就对应减少一个
            max_repeat -= 1

        a_best, b_best, c_best = a_idx, b_idx, c_idx
        best_loss = loss_all[a_idx][b_idx][c_idx]
        for a_cur in [max(a_idx-1, 0), min(a_idx+1,len(a_values)-1)]:
            if loss_all[a_cur][b_idx][c_idx] == 0:
                simu_parameter = {'0': a_values[a_cur],
                                  '1': b_values[b_idx],
                                  '2': c_values[c_idx]}
                one_case.simulation = list(simu_xy_cal.generate_simu(x_values=simulation_x_distribution,
                                                                     para=simu_parameter))
                loss_all[a_cur][b_idx][c_idx] = one_case.compare(simu_parameter)
                # 每计算多一个，那么最大计算上限就对应减少一个
                max_repeat -= 1
            if loss_all[a_cur][b_idx][c_idx] > best_loss:
                a_best, b_best, c_best = a_cur, b_idx, c_idx

        for b_cur in [max(0, b_idx-1), min(b_idx+1,len(b_values)-1)]:
            if loss_all[a_idx][b_cur][c_idx] == 0:
                simu_parameter = {'0': a_values[a_idx],
                                  '1': b_values[b_cur],
                                  '2': c_values[c_idx]}
                one_case.simulation = list(simu_xy_cal.generate_simu(x_values=simulation_x_distribution,
                                                                     para=simu_parameter))
                loss_all[a_idx][b_cur][c_idx] = one_case.compare(simu_parameter)
                # 每计算多一个，那么最大计算上限就对应减少一个
                max_repeat -= 1
            if loss_all[a_idx][b_cur][c_idx] > best_loss:
                a_best, b_best, c_best = a_idx, b_cur, c_idx

        for c_cur in [max(0, c_idx-1), min(c_idx+1,len(c_values)-1)]:
            if loss_all[a_idx][b_idx][c_cur] == 0:
                simu_parameter = {'0': a_values[a_idx],
                                  '1': b_values[b_idx],
                                  '2': c_values[c_cur]}
                one_case.simulation = list(simu_xy_cal.generate_simu(x_values=simulation_x_distribution,
                                                                     para=simu_parameter))
                loss_all[a_idx][b_idx][c_cur] = one_case.compare(simu_parameter)
                # 每计算多一个，那么最大计算上限就对应减少一个
                max_repeat -= 1
            if loss_all[a_idx][b_idx][c_cur] > best_loss:
                a_best, b_best, c_best = a_idx, b_idx, c_cur

        if (a_best == a_idx) and (b_best == b_idx) and (c_best == c_idx):
            loop_keep = False
            repeat -= 1

print(one_case.minimal)
print(one_case.least_loss)

# print(loss_all)
# Create a new figure and axis object
fig, ax = plt.subplots()
x1, y1 = zip(*one_case.sample)
x2, y2 = zip(*one_case.best_simulation)

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