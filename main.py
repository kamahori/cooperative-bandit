import numpy as np


class Arms:
    """Experiment environment"""

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
        """play one timestep"""
        assert len(idxs) == self.n_player
        rnd = np.random.rand(self.n_arm)
        reward = (rnd < self.mean_rewards).astype(float)
        if self.full_feedback:
            return np.take(reward, idxs), reward
        else:
            return np.take(reward, idxs)

    def calc_optimum(self):
        """expected reward when best n_player arms are chosen"""
        return np.sort(self.mean_rewards)[::-1][: self.n_player].sum()


class DOP:
    """Doubly Ordered Partition"""

    def __init__(self, n_arm, n_player) -> None:
        self.n_arm = n_arm
        self.n_player = n_player
        assert n_arm >= n_player
        self.partition = [set(range(n_arm))]
        self.order = list()

    def calc_i(self) -> int:
        """
        return the smallest integer i >= 0 such that
        len(partition[0] | ... | partition[i]) > n_player
        i.e. we need to further partition partition[i] to
        know the top n_player acttions
        """
        tmp = 0
        for i, part in enumerate(self.partition):
            tmp += len(part)
            if tmp > self.n_player:
                return i
        assert False
        # return len(self.partition)

    def calc_a(self) -> set:
        """
        set of arms that are already identified as top
        n_player actions
        """
        tmp = 0
        res = set()
        for i, part in enumerate(self.partition):
            tmp += len(part)
            if tmp > self.n_player:
                break
            res = res.union(part)
        return res

    def calc_b(self) -> tuple[int, set]:
        """
        set of arms that needs to be partitioned further
        to identify top n_player actions
        """
        set_a = self.calc_a()
        if len(set_a) == self.n_player:
            return -1, set()

        i = self.calc_i()
        return i, self.partition[i]

    def is_leaf(self) -> bool:
        """
        return if current partition is a leaf of the T_{K,m} subtree
        TODO: condition about the order
        """
        i, set_b = self.calc_b()
        return len(set_b) == 0

    def get_parent(self) -> tuple[list, list]:
        """get the parent of current value"""
        last_partition = len(self.order) - 1
        for i, val in enumerate(self.order):
            if val == last_partition:
                par_order = self.order
                par_order.remove(val)

                par_partition = self.partition
                par_partition[i] = par_partition[i].union(par_partition[i + 1])
                par_partition.pop(i + 1)
                return par_order, par_partition
        assert False


class Agent:
    """Player agent"""

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

    def play(self) -> None:
        """play for max_time timesteps"""
        self.total_reward = 0.0
        for cur_time in range(self.max_time):
            if self.strategy == "random":
                idxs = np.random.randint(self.n_arm, size=self.n_player)
            elif self.strategy == "proposed":
                p = set(range(self.n_arm))
            else:
                raise NotImplementedError

            if self.full_feedback:
                reward, reward_all = self.arms.play(idxs)
            else:
                reward = self.arms.play(idxs)

            self.total_reward += reward.sum()

    def calc_regret(self) -> float:
        """calculate regret"""
        return self.max_time * self.arms.calc_optimum() - self.total_reward


if __name__ == "__main__":
    N_ARM = 3  # K
    N_PLAYER = 2  # m
    MAX_TIME = 10000  # T
    rnd_agent = Agent(n_arm=N_ARM, n_player=N_PLAYER, max_time=MAX_TIME)
    rnd_agent.play()
    print(rnd_agent.calc_regret())
    bound = N_PLAYER * (N_ARM**5.5) * np.sqrt(MAX_TIME * np.log(MAX_TIME))
    print(f"expected regret O(m K^(11/2) sqrt(T log(T)) = {bound}")
