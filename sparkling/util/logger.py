import logging
import sys
import time
from enum import IntEnum


class SparklingLogLevel(IntEnum):
    NONE = 0
    EXCERPT = 1
    INFO = 2
    VERBOSE = 3

    @staticmethod
    def configure():
        logger = logging.getLogger('SPARKLING')
        logger.setLevel('INFO')
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('|SPARKLING|>  %(message)s'))
        logger.addHandler(handler)
        return logger


class SparklingLogger:
    level = SparklingLogLevel.EXCERPT
    logger = SparklingLogLevel.configure()

    @staticmethod
    def log(message):
        if len(message) > 0:
            SparklingLogger.logger.info(message)

    @staticmethod
    def start_cvi_predictor():
        if SparklingLogger.level >= SparklingLogLevel.INFO:
            SparklingLogger.log('Started CVI prediction ...')
        return time.time()

    @staticmethod
    def finish_cvi_predictor(t_start, recommendation, meta_features):
        message, consumed = '', time.time() - t_start
        if SparklingLogger.level >= SparklingLogLevel.EXCERPT:
            message += f'CVI predictor recommended {recommendation} in {consumed}s'
        if SparklingLogger.level >= SparklingLogLevel.VERBOSE:
            message += f'. Meta-features: {meta_features}'
        SparklingLogger.log(message)

    @staticmethod
    def start_preprocessing():
        if SparklingLogger.level >= SparklingLogLevel.INFO:
            SparklingLogger.log('Started preprocessing ...')
        return time.time()

    @staticmethod
    def finish_preprocessing(t_start):
        consumed = time.time() - t_start
        if SparklingLogger.level >= SparklingLogLevel.EXCERPT:
            SparklingLogger.log(f'Finished preprocessing in {consumed}s')

    @staticmethod
    def start_stage(stage):
        if SparklingLogger.level >= SparklingLogLevel.VERBOSE:
            SparklingLogger.log(f'- Started stage {stage}')
        return time.time()

    @staticmethod
    def finish_stage(t_start, stage, updates):
        message, consumed = '', time.time() - t_start
        if SparklingLogger.level >= SparklingLogLevel.INFO:
            message += f'+ Finished stage {stage} in {consumed}s'
        if SparklingLogger.level >= SparklingLogLevel.VERBOSE:
            message += f'. Monad state updates: {updates}'
        SparklingLogger.log(message)

    @staticmethod
    def pca_not_enough_dims(name, explained, values):
        message = f"Achieve only {explained} explained variance for {name}"
        if SparklingLogger.level >= SparklingLogLevel.VERBOSE:
            message += f". PCA values: {values}"
        SparklingLogger.logger.warning(message)

    @staticmethod
    def start_pivot(measure):
        if SparklingLogger.level >= SparklingLogLevel.INFO:
            SparklingLogger.log(f'Estimating pivot value for measure {measure} ...')
        return time.time()

    @staticmethod
    def finish_pivot(t_start, measure, value):
        message, consumed = '', time.time() - t_start
        if SparklingLogger.level >= SparklingLogLevel.INFO:
            message += f'Pivot value for measure {measure.name} estimated in {consumed}s'
        if SparklingLogger.level >= SparklingLogLevel.VERBOSE:
            message += f'. Value: {value}'
        SparklingLogger.log(message)

    @staticmethod
    def start_arm(arm, arm_attempt, all_attempts, algo_name):
        if SparklingLogger.level >= SparklingLogLevel.INFO:
            SparklingLogger.log(f'Started arm #{arm} [{algo_name}], attempt {arm_attempt}/{all_attempts}')
        return time.time()

    @staticmethod
    def finish_arm(consumed, arm, arm_attempt, all_attempts, algo_name, reward):
        if SparklingLogger.level >= SparklingLogLevel.INFO:
            SparklingLogger.log(
                f'Finished arm #{arm} [{algo_name}] in {consumed} '
                f'with reward {reward}, attempt {arm_attempt}/{all_attempts}'
            )
        return time.time()

    @staticmethod
    def start_algo(algo):
        if SparklingLogger.level >= SparklingLogLevel.INFO:
            SparklingLogger.log(f'Started {algo.algo_name} with {algo.params}')
        return time.time()

    @staticmethod
    def finish_algo(algo_start, algo):
        message, consumed = '', time.time() - algo_start
        if SparklingLogger.level >= SparklingLogLevel.EXCERPT:
            message = f'Finished {algo.algo_name} in {consumed}s'
        if SparklingLogger.level >= SparklingLogLevel.INFO:
            message += f' with {algo.params}'
        SparklingLogger.log(message)
        return consumed

    @staticmethod
    def start_measure(measure):
        if SparklingLogger.level >= SparklingLogLevel.INFO:
            SparklingLogger.log(f'Started {measure.name} evaluation ...')
        return time.time()

    @staticmethod
    def finish_measure(m_start, measure, value):
        consumed = time.time() - m_start
        if SparklingLogger.level >= SparklingLogLevel.EXCERPT:
            SparklingLogger.log(f'Finished {measure.name} evaluation in {consumed}s, value: {value}')
        return consumed

    @staticmethod
    def failed_algo(algo_start, algo):
        message, consumed = '', time.time() - algo_start
        if SparklingLogger.level >= SparklingLogLevel.EXCERPT:
            message += f'{algo.algo_name} failed in {consumed}s'
        if SparklingLogger.level >= SparklingLogLevel.INFO:
            message += f' with {algo.params}'
        SparklingLogger.log(message)
