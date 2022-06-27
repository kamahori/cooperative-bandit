import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Arms:
    def __init__(self, n_arm=10, n_player=2, full_feedback=False, seed=0) -> None:
        self.n_arm = n_arm
        self.n_player = n_player
        self.full_feedback = full_feedback
        self.seed = seed
        self.mean_rewards = self._set_mean()

    def _set_mean(self):
        np.random.seed(self.seed)
        return np.random.rand(self.n_arm)

    def play(self, idxs):
        assert len(idxs) == self.n_player
        rnd = self.random.rand(self.n_arm)
        reward = (rnd < self.mean_rewards).astype(float)
        if self.full_feedback:
            return np.take(reward, idxs), reward
        else:
            return np.take(reward, idxs)
