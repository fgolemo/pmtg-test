import numpy as np


class TrajGen:
    def __init__(self, steps):
        self.step_size = 2 * np.pi / steps

    def get_next_point(self, step, a, b):
        # a = (np.cos((i + steps / 4) * step_size) + 1) / 2 * 0.5 + 0.5
        x_val = a * np.sin(step * self.step_size)

        # b = (np.sin((i + steps / 4) * step_size) + 1) / 2 * 0.7 + 0.3
        y_val = b * np.sin(step * self.step_size) * np.cos(step * self.step_size)
        return x_val, y_val
