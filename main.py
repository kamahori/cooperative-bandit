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
        rnd = np.random.rand(self.n_arm)
        reward = (rnd < self.mean_rewards).astype(float)
        if self.full_feedback:
            return np.take(reward, idxs), reward
        else:
            return np.take(reward, idxs)

    def calc_optimum(self):
        # expected reward when best n_player arms are chosen
        return np.sort(self.mean_rewards)[::-1][: self.n_player].sum()


class Agent:
    def __init__(
        self,
        n_arm=10,
        n_player=2,
        max_time=10000,
        full_feedback=False,
        strategy="random",
        seed=0,
    ) -> None:
        self.n_arm = n_arm
        self.n_player = n_player
        self.max_time = max_time
        self.full_feedback = full_feedback
        self.strategy = strategy
        self.seed = seed
        self.arms = Arms(
            n_arm=n_arm, n_player=n_player, full_feedback=full_feedback, seed=seed
        )
        self.total_reward = 0.0

    def play(self):
        self.total_reward = 0.0
        for cur_time in range(self.max_time):
            if self.strategy == "random":
                idxs = np.random.randint(self.n_arm, size=self.n_player)
            else:
                raise NotImplementedError

            if self.full_feedback:
                reward, reward_all = self.arms.play(idxs)
            else:
                reward = self.arms.play(idxs)

            self.total_reward += reward.sum()

    def calc_regret(self):
        return self.max_time * self.arms.calc_optimum() - self.total_reward


if __name__ == "__main__":
    n_arm = 3  # K
    n_player = 2  # m
    max_time = 10000  # T
    rnd_agent = Agent(n_arm=n_arm, n_player=n_player, max_time=max_time)
    rnd_agent.play()
    print(rnd_agent.calc_regret())
    bound = n_player * (n_arm**5.5) * np.sqrt(max_time * np.log(max_time))
    print(f"expected regret O(m K^(11/2) sqrt(T log(T)) = {bound}")
