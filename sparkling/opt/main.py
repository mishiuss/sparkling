import json
import random
import time
from typing import Optional, List

from sparkling.algorithms import *
from sparkling.algorithms.base import AlgoConf
from sparkling.data.dataframe import SparklingDF
from .hyperopts import HyperOpt, HistoryRun, Optimal
from .mabs import MabSolver
from .measures import Internal


class Sparkling:
    """
    The entry point of the CVI optimisation. It uses combination of multi-armed bandit solvers and
    bayes optimisation techniques for efficient time efficient hyperparameters search.
    """

    def __init__(
            self, sparkling_df: SparklingDF,
            configs: Optional[List[AlgoConf]] = None,
            mab_solver: MabSolver = MabSolver.SOFTMAX,
            hyper_opt: HyperOpt = HyperOpt.OPTUNA,
            measure: Optional[Internal] = Internal.CALINSKI_HARABASZ
    ):
        """
        Defines optimisation parameters and algorithms config spaces.

        :param configs: list of :class`AlgoConf`, which defines global search space,
        e.g. config space for each clustering algorithm and choice between different algorithms.
        If None, all supported algorithms with default config spaces will be used.
        :param mab_solver: strategy to use for choosing one of the clustering algorithms. See :class`MabSolver`
        :param hyper_opt: underlying realisation of bayes optimisation.
        One should manually install package dependencies for preferred HPO.
        :param measure: optimisation target CVI (:class`Internal`).
        """

        if measure is None:
            from sparkling.meta.cvi_predictor import CVIPredictor
            measure = CVIPredictor.predict(sparkling_df)

        self.sparkling_df, self.measure = sparkling_df, measure
        self.configs = configs if configs is not None else self._defaults()

        self.optimisers = [hyper_opt.value(conf, sparkling_df, measure) for conf in self.configs]
        self.mab_solver = mab_solver.value(sparkling_df, self.optimisers)

    def _defaults(self):
        configs = [
            KMeansConf(),
            MeanShiftConf(),
            BirchConf(),
            BisectingKMeansConf(),
            # Below algorithms showed unstable performance
            # DBSCANConf(),
            # CLIQUEConf(),
            # CUREConf(),
        ]
        random.shuffle(configs)
        return configs

    def run(self, time_limit) -> Optimal:
        """
        Actual launch of optimisation for specified time budget.
        It is not guaranteed to finish exactly in specified amount of time, because current
        clustering algorithm execution and measure evaluation are to be terminated gracefully.
        Optimiser keeps it state, so you can call this function multiple times.
        Returns :class`Optimal` with the best founded configuration according to target CVI
        and :class`SparklingDF` with label column.

        >>> sparkling_df = ...
        >>> optimiser = Sparkling(sparkling_df)
        >>> optimiser.run(150)  # Optimal(algo=MeanShift, model=MeanShiftModel, ...)

        :param time_limit: time budget in seconds (either int or float)
        """

        return self.mab_solver.run(time_limit)

    def history(self) -> List[HistoryRun]:
        """
        Obtains sequence of executed configurations with evaluated CVI (see :class`HistoryRun`).

        >>> optimiser = Sparkling(...)
        >>> optimiser.run(150)
        >>> optimiser.history()  # [HistoryRun(value=0.7, fit_time=33, ...), HistoryRun(value=0.77, ...), ...]
        """

        return self.mab_solver.history

    def history_json(self) -> str:
        """
        Obtains json-serialized sequence of executed configurations with evaluated CVI.
        >>> optimiser = Sparkling(...)
        >>> optimiser.run(150)
        >>> optimiser.history_json()  # "[{"value":0.7,"fit_time":33,...},{"value":0.77,...},...]"
        """

        return json.dumps(self.history(), indent=2, default=lambda o: o.__dict__)
