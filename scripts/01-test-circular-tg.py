from collections import deque

import matplotlib.pyplot as plt
import numpy as np

steps = 30
step_size = 2 * np.pi / steps

dlen = steps + 2
data_x = deque(maxlen=dlen)
data_y = deque(maxlen=dlen)
data_x.append(0)
data_y.append(0)
alpha = np.linspace(0, 1, dlen)

a = 1
b = 1

for i in range(1, steps + 1):
    plt.clf()  # without this, the transparency doesn't work because it just overwrites

    ##### FORMULA 1
    # data_x.append(np.cos(i * 0.2))
    # data_y.append(np.sin(i * 0.2))

    ##### FORMULA 2
    # x_val = np.cos(i * step_size)
    # if x_val > 0:
    #     x_val -= 0.3 * np.cos(i * step_size)
    # data_x.append(x_val)
    # y_val = np.sin(i * step_size)
    # if y_val > 0:
    #     y_val -= 0.3 * np.sin(i * step_size)
    # data_y.append(y_val)

    ##### FORMULA 3
    # x_val = a * np.sin(i * step_size)
    # data_x.append(x_val)
    # y_val = b * np.sin(i * step_size) * np.cos(i * step_size)
    # data_y.append(y_val)

    ##### FORMULA 4
    a = (np.cos((i + steps / 4) * step_size) + 1) / 2 * 0.5 + 0.5
    x_val = a * np.sin(i * step_size)
    data_x.append(x_val)

    b = (np.sin((i + steps / 4) * step_size) + 1) / 2 * 0.7 + 0.3
    y_val = b * np.sin(i * step_size) * np.cos(i * step_size)
    data_y.append(y_val)

    for idx in range(0, len(data_x) - 1):
        plt.plot(list(data_x)[idx : idx + 2], list(data_y)[idx : idx + 2], c="magenta", alpha=alpha[-len(data_x) + idx])
    plt.axis([-1, 1, -1, 1])  # sets the limits correctly
    plt.pause(0.01)  # forces redrawing


#
plt.show()  # pauses the plot at the end and leaves it open
