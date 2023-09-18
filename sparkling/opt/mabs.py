import time
from abc import abstractmethod, ABC
from enum import Enum
from typing import List

import numpy as np

from pyspark.sql.functions import rand, round
from pyspark.sql.types import IntegerType as Int

from .hyperopts import BaseHyperOpt, Optimal
from sparkling.data.dataframe import SparklingDF
from sparkling.util.logger import SparklingLogger


class BaseMabSolver(ABC):
    """
    Base class for switching clustering algorithms (e.g. arms).
    Reward function for arm takes into consideration its time consumption and the best reached CVI value.
    Before using specific strategy, each arm tries to initialise itself (e.g. launch some algorithm configuration).
    If initialisation is unsuccessful, the default "bad" value will be assigned.

    You should not instantiate this class yourself, :class`Sparkling` will do it automatically.
    """

    WARMUP_ATTEMPTS = 3

    def __init__(self, sparkling_df: SparklingDF, optimisers: List[BaseHyperOpt]):
        self.sparkling_df = sparkling_df
        self.optimisers = optimisers
        self.arms = len(optimisers)
        self.consumption = np.zeros(self.arms)
        self.usage = np.zeros(self.arms, dtype=int)
        self.pivot_val = self._eval_pivot()
        self.rewards = np.full(self.arms, self.pivot_val)
        self.history = list()

    def _eval_pivot(self):
        measure, measure_val = self.optimisers[0].measure, float('-inf')
        t_start = SparklingLogger.start_pivot(measure)

        origin_df = self.sparkling_df.df
        while measure_val == float('-inf'):
            random_labels = round(rand() * 2).cast(Int())
            random_labeled = origin_df.withColumn(SparklingDF.LABEL_COL, random_labels)
            random_sdf = self.sparkling_df.like(random_labeled)
            measure_val = measure.evaluate(random_sdf, minimise=False)

        SparklingLogger.finish_pivot(t_start, measure, measure_val)
        return measure_val

    def _step(self, arm):
        algo = self.optimisers[arm].algo_conf.algo_name
        arm_attempt, all_attempts = self.usage[arm] + 1, np.sum(self.usage) + 1
        t_start = SparklingLogger.start_arm(arm, arm_attempt, all_attempts, algo)

        run = self.optimisers[arm].step()
        self.history.append(run)
        arm_time = time.time() - t_start
        # optimisers are minimising reward, so need to inverse monotonicity
        run_reward = -run.value
        self._update(arm, run_reward, arm_time)

        SparklingLogger.finish_arm(arm_time, arm, arm_attempt, all_attempts, algo, run_reward)
        return arm_time

    def _warmup_if_needed(self, time_limit):
        consumed = 0
        while consumed < time_limit and np.min(self.usage) < self.WARMUP_ATTEMPTS:
            consumed += self._step(np.argmin(self.usage))
        return consumed

    def _normalised_rewards(self):
        time_rewards = 1.0 - self.consumption / np.sum(self.consumption)
        diff_max = np.max(self.rewards) - self.pivot_val
        if abs(diff_max) < 1e-12:
            return time_rewards
        quality_rewards = (self.rewards - self.pivot_val) / diff_max
        return quality_rewards + time_rewards

    def run(self, time_limit) -> Optimal:
        """
        Launches procedure of switching clustering algorithms for further bayes optimisation.
        Returns the best founded configuration (:class`Optimal`) with evaluated CVI and labeled :class`SparklingDF`

        :param time_limit: time budget (in seconds) for optimisation (not guaranteed to terminate exactly)
        """
        consumed = self._warmup_if_needed(time_limit)
        while consumed < time_limit:
            consumed += self._step(self.draw())
        opt_values = [opt.optimal.value if opt.optimal is not None else float('inf') for opt in self.optimisers]
        return self.optimisers[np.argmin(opt_values)].optimal

    @abstractmethod
    def draw(self):
        pass

    def _update(self, arm, reward, consumed):
        self.consumption[arm] += consumed
        self.usage[arm] += 1
        cur_max = self.rewards[arm]
        self.rewards[arm] = max(cur_max, reward)


class SoftmaxMab(BaseMabSolver):
    def __init__(self, sparkling_df: SparklingDF, optimisers: List[BaseHyperOpt]):
        super().__init__(sparkling_df, optimisers)

    @staticmethod
    def soft_norm(x):
        e_x = np.exp(x - np.amax(x))
        return e_x / e_x.sum()

    def draw(self):
        probs = self.soft_norm(self._normalised_rewards())
        return np.random.choice(np.arange(self.arms), p=probs)


class UcbMab(BaseMabSolver):
    def __init__(self, sparkling_df: SparklingDF, optimisers: List[BaseHyperOpt]):
        super().__init__(sparkling_df, optimisers)

    def draw(self):
        its, rewards = len(self.history), self._normalised_rewards()
        return np.argmax(rewards + np.sqrt(2 * np.log(its) / self.usage))


class MabSolver(Enum):
    """
    Enumeration of available multi-armed bandit solvers.
    Pass one of the members to :class `Sparkling` constructor.

    >>> from sparkling import Sparkling
    >>> sparkling_df = ...
    >>> optimiser = Sparkling(sparkling_df, mab_solver=MabSolver.SOFTMAX)
    """

    SOFTMAX = SoftmaxMab
    UCB = UcbMab
