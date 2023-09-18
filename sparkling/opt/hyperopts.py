import json
from abc import abstractmethod, ABC
from enum import Enum

from py4j.protocol import Py4JError

from sparkling.algorithms.base import AlgoConf, ClusteringAlgo, ClusteringModel
from sparkling.data.dataframe import SparklingDF
from sparkling.util.logger import SparklingLogger
from .measures import Internal


class HistoryRun:
    """
    Contains information about clustering algorithm configuration, execution and evaluation.
    """

    def __init__(self, value: float, fit_time: float, eval_time: float, algo: str, params):
        self.value, self.algo, self.params = value, algo, params
        self.fit_time, self.eval_time = fit_time, eval_time

    def __str__(self):
        description = {
            'fit': self.fit_time,
            'eval': self.eval_time,
            'value': self.value,
            'algorithm': self.algo,
            'params': self.params
        }
        return json.dumps(description)


class Optimal:
    """
    Contains clustering algorithm configuration, fitted model and labeled :class `SparklingDF`.
    Represents (local) optimal and holds reference to wrappers (see :class`ClusteringModel` and
    :class`ClusteringAlgo`) of jvm objects and dataframe with label column.
    """

    def __init__(self, algo: ClusteringAlgo, model: ClusteringModel, label_sdf: SparklingDF, value: float, index: int):
        self.algo, self.model, self.index = algo, model, index
        self.label_sdf, self.value = label_sdf, value

    def to_history_run(self, fit_time: float, eval_time: float) -> HistoryRun:
        """
        Serializes wrappers and "forgets" dataframe, so it become handy for logs.
        """
        return HistoryRun(self.value, fit_time, eval_time, self.algo.algo_name, self.algo.params)

    def __str__(self):
        description = {
            'value': self.value,
            'algorithm': self.algo.algo_name,
            'params': dict(**self.algo.params)
        }
        return json.dumps(description)


class BaseHyperOpt(ABC):
    """
    Base class for algorithm's config space exploration. Provides common interface for third-party HPOs.
    It holds underlying optimiser's state, so it can be resumed without exploration history loss.

    You should not instantiate this class yourself, :class`Sparkling` will do it automatically.
    """

    INF_APPROX = int('BADBADBADBAD', 16)

    def __init__(self, algo_conf: AlgoConf, sparkling_df: SparklingDF, measure: Internal):
        self.algo_conf = algo_conf
        self.sparkling_df = sparkling_df
        self.measure = measure
        self._opt, self._runs = None, list()

    @property
    def optimal(self) -> Optimal:
        return self._opt

    def _memorise_run(self, local_opt: Optimal, fit_time: float, eval_time: float):
        if self.optimal is None or local_opt.value < self.optimal.value:
            self._opt = local_opt
        self._runs.append(local_opt.to_history_run(fit_time, eval_time))

    def _memorise_fail(self, algo: ClusteringAlgo, consumed: float):
        failed_run = HistoryRun(float('inf'), consumed, 0, algo.algo_name, algo.params)
        self._runs.append(failed_run)

    def _calc(self, param):
        return param(self.sparkling_df) if callable(param) else param

    def _parse_param(self, name, param, **kwargs):
        if isinstance(param, tuple) and len(param) == 2:
            return self._parse_range(name, param, **kwargs)
        elif isinstance(param, (set, frozenset)):
            return self._categorical_param(name, param, **kwargs)
        elif callable(param) or isinstance(param, (int, float, bool)):
            return self._const_param(name, self._calc(param), **kwargs)
        else:
            raise ValueError(f"Failed to recognize parameter '{name}': {param}")

    def _parse_range(self, name, param, **kwargs):
        lower, upper = self._calc(param[0]), self._calc(param[1])
        if isinstance(lower, int) and isinstance(upper, int):
            return self._int_param(name, lower, upper, **kwargs)
        elif isinstance(lower, float) and isinstance(upper, float):
            return self._float_param(name, lower, upper, **kwargs)
        else:
            raise ValueError(f"Expected both ints or floats (hyper parameter '{name}')")

    def _eval_algo(self, algo: ClusteringAlgo) -> float:
        algo_start = SparklingLogger.start_algo(algo)
        try:
            model, labeled_sdf = algo.fit_predict_with_model(self.sparkling_df)
            fit_time = SparklingLogger.finish_algo(algo_start, algo)
            m_start = SparklingLogger.start_measure(self.measure)
            value = self.measure.evaluate(labeled_sdf, minimise=True)
            eval_time = SparklingLogger.finish_measure(m_start, self.measure, value)
            local_opt = Optimal(algo, model, labeled_sdf, value, index=len(self._runs))
            self._memorise_run(local_opt, fit_time, eval_time)
            return local_opt.value
        except Py4JError as _:
            failed_time = SparklingLogger.failed_algo(algo_start, algo)
            self._memorise_fail(algo, failed_time)
            return self.INF_APPROX

    @abstractmethod
    def _int_param(self, name, lower, upper, **kwargs):
        pass

    @abstractmethod
    def _float_param(self, name, lower, upper, **kwargs):
        pass

    @abstractmethod
    def _categorical_param(self, name, values, **kwargs):
        pass

    @abstractmethod
    def _const_param(self, name, value, **kwargs):
        pass

    @abstractmethod
    def step(self) -> HistoryRun:
        pass


class SmacOpt(BaseHyperOpt):
    """
    Wrapper for SMAC3 optimiser (https://automl.github.io/SMAC3/master/). Using SMAC4HPO facade
    """

    def __init__(self, algorithm, sparkling_df, measure):
        BaseHyperOpt.__init__(self, algorithm, sparkling_df, measure)
        from smac.scenario.scenario import Scenario
        from smac.facade.smac_hpo_facade import SMAC4HPO
        from ConfigSpace import ConfigurationSpace, \
            UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter, Constant

        self._cs = ConfigurationSpace()
        for name, space in self.algo_conf.config_space.items():
            param = self._parse_param(
                name, space, integer=UniformIntegerHyperparameter,
                flt=UniformFloatHyperparameter, category=CategoricalHyperparameter, const=Constant
            )
            self._cs.add_hyperparameter(param)

        scenario = Scenario({
            'run_obj': 'quality',
            'deterministic': True,
            'cs': self._cs,
            'ta_run_limit': 1,
            'rand_prob': 0.3,
            'output_dir': None,
            'cost_for_crash': self.INF_APPROX
        })

        self.opt = SMAC4HPO(scenario=scenario, tae_runner=self._opt_function)

    def _int_param(self, name, lower, upper, **kwargs):
        return kwargs['integer'](name, lower, upper)

    def _float_param(self, name, lower, upper, **kwargs):
        return kwargs['flt'](name, lower, upper)

    def _categorical_param(self, name, values, **kwargs):
        return kwargs['category'](name, values)

    def _const_param(self, name, value, **kwargs):
        return kwargs['const'](name, value)

    def _opt_function(self, configuration) -> float:
        algo = self.algo_conf.build(**configuration)
        return self._eval_algo(algo)

    def step(self) -> HistoryRun:
        self.opt.optimize()
        self.opt.solver.scenario.ta_run_limit += 1
        self.opt.solver.intensifier.num_run = 0
        return self._runs[-1]


class OptunaOpt(BaseHyperOpt):
    """
    Wrapper for Optuna optimiser (https://optuna.org/)
    """

    def __init__(self, algorithm, ds, measure):
        BaseHyperOpt.__init__(self, algorithm, ds, measure)
        import optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        self.session = optuna.create_study()

    def _int_param(self, name, lower, upper, **kwargs):
        return kwargs['trial'].suggest_int(name, lower, upper)

    def _float_param(self, name, lower, upper, **kwargs):
        return kwargs['trial'].suggest_float(name, lower, upper)

    def _categorical_param(self, name, values, **kwargs):
        return kwargs['trial'].suggest_categorical(name, values)

    def _const_param(self, name, value, **kwargs):
        return value

    def objective(self, trial):
        config = dict()
        for name, space in self.algo_conf.config_space.items():
            config[name] = self._parse_param(name, space, trial=trial)
        algo = self.algo_conf.build(**config)
        return self._eval_algo(algo)

    def step(self) -> HistoryRun:
        self.session.optimize(self.objective, n_trials=1)
        return self._runs[-1]


class HyperOpt(Enum):
    """
    Enumeration of supported HPOs.
    Pass one of the members to :class `Sparkling` constructor.
    NOTE: you should manually install according package dependencies.

    >>> from sparkling import Sparkling
    >>> sparkling_df = ...
    >>> optimiser = Sparkling(sparkling_df, hyper_opt=HyperOpt.OPTUNA)
    """

    OPTUNA = OptunaOpt
    SMAC = SmacOpt
