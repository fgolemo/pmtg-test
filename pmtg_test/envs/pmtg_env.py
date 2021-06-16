import numpy as np
import gym

# TODO: variant with timestep being observed
import cv2
from numpy.linalg import norm

from pmtg_test.trajectory_gen import TrajGen

# reference traj
A = 1
B = 1
# resolution
RES = 84


class PmtgEnv(gym.Env):
    def __init__(self, steps=30, with_timer=True, with_pmtg=True):
        super().__init__()
        self.steps = steps
        self.step_size = 2 * np.pi / (steps - 1)
        self.with_timer = with_timer
        if with_timer:
            obslen = 3
        else:
            obslen = 2
        self.with_pmtg = with_pmtg

        self.tg = TrajGen(steps)

        # print(np.around(self.data[:], 2))
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1, 1, (obslen,), dtype=np.float32)

        self.counter = 0
        self.data_base = []
        self.data_trgt = []
        self.data_user = []

    def reset(self):
        self.counter = 0
        # self.img.fill(0)
        self.last_obs = self.tg.get_next_point(0, A, B)
        a, b = self._get_ab()
        self.last_trgt = self.tg.get_next_point(0, a, b)
        self.data_base.clear()
        self.data_base.append(self.last_obs)
        self.data_trgt.clear()
        self.data_trgt.append(self.last_trgt)
        self.data_user.clear()

        obs = self.last_obs
        if self.with_timer:
            obs = np.hstack((obs, 0))

        return obs

    def _get_ab(self):
        a = (np.cos((self.counter + self.step_size / 4) * self.step_size) + 1) / 2 * 0.5 + 0.5
        b = (np.sin((self.counter + self.step_size / 4) * self.step_size) + 1) / 2 * 0.7 + 0.3
        return a, b

    def _coord2pix(self, coord):
        x_norm = (coord[0] + 1) / 2
        y_norm = (coord[1] + 1) / 2
        x_img = int(x_norm * RES)
        y_img = int(y_norm * RES)
        return x_img, y_img

    def step(self, action):
        assert len(action) == 2

        if self.with_pmtg:
            self.action_xy = self.tg.get_next_point(self.counter, action[0], action[1])
        else:
            self.action_xy = action
        self.data_user.append(self.action_xy)

        rew = -norm((self.last_trgt[0] - self.action_xy[0], self.last_trgt[1] - self.action_xy[1]))

        self.counter += 1
        self.last_obs = self.tg.get_next_point(self.counter, A, B)
        self.data_base.append(self.last_obs)
        a, b = self._get_ab()
        self.last_trgt = self.tg.get_next_point(self.counter, a, b)
        self.data_trgt.append(self.last_trgt)

        if self.counter == self.steps:
            done = True
        else:
            done = False

        obs = self.last_obs
        if self.with_timer:
            obs = np.hstack((obs, self.counter / self.steps))

        return obs, rew, done, {}

    def _draw_circle(self, center, radius, color):
        x_min = np.clip(max(0, center[0] - radius), 0, RES)
        x_max = np.clip(min(RES, center[0] + radius), 0, RES)
        y_min = np.clip(max(0, center[1] - radius), 0, RES)
        y_max = np.clip(min(RES, center[1] + radius), 0, RES)

        self.img[y_min:y_max, x_min:x_max, 0] += color[0]
        self.img[y_min:y_max, x_min:x_max, 1] += color[1]
        self.img[y_min:y_max, x_min:x_max, 2] += color[2]

    def render(self, mode="human"):
        self.img = np.zeros((RES, RES, 3), dtype=np.uint8)
        for i in range(self.counter):
            self._draw_circle(self._coord2pix(self.data_base[i]), 2, (0, 200, 0))
            self._draw_circle(self._coord2pix(self.data_user[i]), 2, (200, 0, 0))
            self._draw_circle(self._coord2pix(self.data_trgt[i]), 2, (55, 55, 255))
        if mode == "human":
            cv2.imshow("ladeeda", self.img[:, :, ::-1])
            cv2.waitKey(100)
        else:
            return self.img

    def seed(self, seed=None):
        np.random.seed(seed)
        return super().seed(seed)


if __name__ == "__main__":
    from time import sleep

    env = PmtgEnv()

    obs = env.reset()
    print(obs)
    env.render()
    while True:
        # obs, rew, done, misc = env.step([0.5, 0.5])
        # obs, rew, done, misc = env.step([0.9, 1])
        obs, rew, done, misc = env.step([1, 1])
        print(obs, rew, done)
        env.render()
        if done:
            cv2.waitKey(-1)
            break
