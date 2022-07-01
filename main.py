import math
import random

import numpy as np


class Arms:
    """Experiment environment"""

    def __init__(self, n_arm=10, n_player=2, full_feedback=False) -> None:
        self.n_arm = n_arm
        self.n_player = n_player
        self.full_feedback = full_feedback
        self.mean_rewards = self._set_mean()

    def _set_mean(self):
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

    def set_partition(self, partition, order) -> None:
        """set arbitrary partition and order"""
        self.partition = partition.copy()
        self.order = order.copy()

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
        return i, self.partition[i].copy()

    def is_leaf(self) -> bool:
        """
        return if current partition is a leaf of the T_{K,m} subtree
        TODO: condition about the order
        """
        i, set_b = self.calc_b()
        return len(set_b) == 0

    def is_root(self) -> bool:
        """return if current partition is the root"""
        return len(self.partition) == 1 and len(self.order) == 0

    def get_parent(self) -> tuple[list, list]:
        """get the parent of current value"""
        last_partition = len(self.order) - 1
        for i, val in enumerate(self.order):
            if val == last_partition:
                par_order = self.order.copy()
                par_order.remove(val)

                par_partition = self.partition.copy()
                par_partition[i] = par_partition[i].union(par_partition[i + 1])
                par_partition.pop(i + 1)
                return par_partition, par_order
        assert False

    def calc_range(self, x) -> float:
        """calculate range function based on given vector x"""
        i, set_b = self.calc_b()
        if i == -1:
            assert False
        list_b = list(set_b)
        list_b = list(map(lambda i: x[i], list_b))
        return max(list_b) - min(list_b)

    def calc_gap(self, x) -> float:
        """calculate gap function based on given vector x"""
        last_partition = len(self.order) - 1
        for i, val in enumerate(self.order):
            if val == last_partition:
                s_1 = self.partition[i].copy()
                s_2 = self.partition[i + 1].copy()
                l_1 = list(s_1)
                l_1 = list(map(lambda i: x[i], l_1))
                l_2 = list(s_2)
                l_2 = list(map(lambda i: x[i], l_2))
                return min(l_1) - max(l_2)
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
    ) -> None:
        self.n_arm = n_arm
        self.n_player = n_player
        self.max_time = max_time
        self.full_feedback = full_feedback
        self.strategy = strategy
        self.arms = Arms(n_arm=n_arm, n_player=n_player, full_feedback=full_feedback)
        self.total_reward = 0.0

        self.list_c = np.random.rand(n_arm) / n_arm

    def play(self) -> None:
        """play for max_time timesteps"""
        self.total_reward = 0.0
        empirical_sum = np.zeros(self.n_arm)
        empirical_cnt = np.zeros(self.n_arm)
        empirical_mean = np.zeros(self.n_arm)
        for cur_time in range(self.max_time):
            if self.strategy == "random":
                idxs = np.random.randint(self.n_arm, size=self.n_player)
            elif self.strategy == "proposed" and self.full_feedback:
                # full feedback scenario for the proposed algorithm
                x = list(empirical_mean)
                eps = 10 * math.sqrt(
                    math.log(self.n_arm * self.n_player * self.max_time)
                    / (cur_time + 1)
                )
                dop = self._proposed_strategy(x, eps=eps)
                idxs = list(dop.calc_a())
                if len(idxs) < self.n_player:
                    remain = self.n_player - len(idxs)
                    _, set_b = dop.calc_b()
                    list_b = list(set_b)
                    idxs.extend(random.sample(list_b, remain))
            elif self.strategy == "proposed" and not self.full_feedback:
                raise NotImplementedError
            else:
                raise NotImplementedError

            if self.full_feedback:
                reward, reward_all = self.arms.play(idxs)
                empirical_sum += reward_all
                empirical_cnt += 1
                empirical_mean = empirical_sum / empirical_cnt
            else:
                reward = self.arms.play(idxs)

            self.total_reward += reward.sum()

    def calc_regret(self) -> float:
        """calculate regret"""
        return self.max_time * self.arms.calc_optimum() - self.total_reward

    def _func_c(self, dop) -> float:
        """
        the c function
        return i.i.d uniform variables on [0, 1/n_arm]
        """
        dist = len(dop.order)
        return self.list_c[dist]

    def _proposed_strategy(self, x, eps=0.001) -> DOP:
        """
        Algorithm 1 in the paper

        input: vector x in [0, 1]^n_player
        output: A partition which is a leaf of T_{K,m}
        """
        dop = DOP(n_arm=self.n_arm, n_player=self.n_player)
        # initialized as ROOT
        while not dop.is_leaf():
            if not dop.is_root():
                parents_list = []
                par_partition, par_order = dop.get_parent()
                tmp_dop = DOP(n_arm=self.n_arm, n_player=self.n_player)
                tmp_dop.set_partition(partition=par_partition, order=par_order)
                while True:
                    parents_list.append([par_partition, par_order])
                    if tmp_dop.is_root():
                        break
                    par_partition, par_order = tmp_dop.get_parent()
                    tmp_dop.set_partition(partition=par_partition, order=par_order)

                for dist, (par_partition, par_order) in enumerate(parents_list):
                    tmp_dop.set_partition(partition=par_partition, order=par_order)
                    i, set_b = tmp_dop.calc_b()
                    list_b = list(set_b)
                    list_b.sort(key=lambda i: -x[i])
                    # decreasing order based on the vector x
                    for j in range(1, len(list_b) + 1):
                        tmp_dop_2 = DOP(n_arm=self.n_arm, n_player=self.n_player)
                        l_1 = list_b[:j]
                        l_2 = list_b[j:]
                        new_partition = par_partition.copy()
                        new_order = par_order.copy()
                        new_partition[i] = set(l_1)
                        new_partition.insert(i + 1, set(l_2))
                        new_order.insert(i, len(par_order))
                        tmp_dop_2.set_partition(
                            partition=new_partition, order=new_order
                        )

                        v_1 = abs(
                            tmp_dop_2.calc_gap(x)
                            - self._func_c(tmp_dop) * tmp_dop.calc_range(x)
                        )
                        v_2 = (dist + 2) * 6 * eps
                        if v_1 <= v_2:
                            return dop

            i, set_b = dop.calc_b()
            list_b = list(set_b)
            list_b.sort(key=lambda i: -x[i])
            for j in range(1, len(list_b) + 1):
                l_1 = list_b[:j]
                l_2 = list_b[j:]
                new_partition = dop.partition.copy()
                new_order = dop.order.copy()
                new_partition[i] = set(l_1)
                new_partition.insert(i + 1, set(l_2))
                new_order.insert(i, len(dop.order))
                tmp_dop = DOP(n_arm=self.n_arm, n_player=self.n_player)
                tmp_dop.set_partition(partition=new_partition, order=new_order)
                if tmp_dop.calc_gap(x) >= self._func_c(dop) * dop.calc_range(x):
                    dop = tmp_dop
                    break

        return dop


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    N_ARM = 15  # K
    N_PLAYER = 3  # m
    MAX_TIME = 100000  # T
    strategy = "proposed"
    # strategy = "random"
    rnd_agent = Agent(
        n_arm=N_ARM,
        n_player=N_PLAYER,
        max_time=MAX_TIME,
        full_feedback=True,
        strategy=strategy,
    )
    rnd_agent.play()
    print(rnd_agent.calc_regret())
    bound = N_PLAYER * (N_ARM**5.5) * np.sqrt(MAX_TIME * np.log(MAX_TIME))
    print(f"expected regret O(m K^(11/2) sqrt(T log(T)) = {bound}")
