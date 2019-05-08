from matplotlib import pyplot as plt
import random

x_value = []
y_value_1 = []
y_value_2 = []
y_value_3 = []
for i in range(60):
    x_value.append(i)
    y_value_1.append(random.randint(0,99))
    y_value_2.append(random.randint(0,99))
    y_value_3.append(random.randint(0,99))
# plt.step(x_value,y_value_1,label = "step_y_value_1")
# plt.step(x_value,y_value_2,label = "step_y_value_2")
# plt.step(x_value,y_value_3,label = "step_y_value_3")
# plt.plot(x_value,y_value_1,label = "plot_y_value_1")
# plt.plot(x_value,y_value_2,label = "plot_y_value_2")
# plt.plot(x_value,y_value_3,label = "plot_y_value_3")
plt.show()

#https://matplotlib.org/gallery/index.html