import os
import pickle

import numpy as np
import pandas as pd

from sparkling import Internal
from sparkling.data.dataframe import SparklingDF
from sparkling.util.logger import SparklingLogger


class CVIPredictor:
    """ Implements measure recommendation by means of meta-learning """

    # local path of pickled CVI Predictor and its fit dataset
    cvi_fitness, cvi_path = 'meta_fitness.csv', 'cvi_predictor.pkl'

    @staticmethod
    def _meta_features(sparkling_df: SparklingDF):
        """
        Extract meta-features of given dataframe in order to suggest recommendation by CVI Predictor.
        Calculates all pairwise distances between samples. Normalises them to upper bound 1.0.
        Extract statistical values: [mean, variance, standard deviation, skewness, kurtosis].
        Next 10 values represents percentage of distances in ranges [0.0, 0.1), [0.1, 0.2), ... [0.9, 1.0];
        To calculate the last 4 meta-features, distances values forced to obtain zero mean and unit variance;
        after that count percentage of values in ranges [0, 1), [1, 2), [2, 3) and other positives
        """
        jdf, dist, amount = sparkling_df.jdf, sparkling_df.dist, sparkling_df.amount
        evaluator = sparkling_df.jvm.ru.ifmo.rain.utils.Meta
        return list(evaluator.evaluate(jdf, dist, amount))

    @staticmethod
    def predict(sparkling_df: SparklingDF) -> Internal:
        """ Makes recommendation of internal measure based on meta-features """
        from .cvi_holder import CVI_PREDICTOR_PICKLE_STRING
        t_start = SparklingLogger.start_cvi_predictor()

        # Define integer labels for measure to match with CVI Predictor output
        cvi_measure_by_id = ['CALINSKI_HARABASZ', 'SILHOUETTE_APPROX', 'SF', 'GD41_APPROX']

        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # with open(dir_path + '/' + CVIPredictor.cvi_path, 'rb') as f:
        #     cvi_classifier = pickle.load(f)
        # NOTE: Spark has a problem reading local file, as it is packed inside zip when deploying on YARN

        cvi_classifier = pickle.loads(CVI_PREDICTOR_PICKLE_STRING)
        meta_features = CVIPredictor._meta_features(sparkling_df)
        measure_id = cvi_classifier.predict(np.array([meta_features]))[0]
        measure_name = cvi_measure_by_id[measure_id]

        SparklingLogger.finish_cvi_predictor(t_start, measure_name, meta_features)
        return Internal[measure_name]

    @staticmethod
    def _build_cvi_predictor(cvi_path):
        """
        Builds measure recommendation model from scratch based on meta-dataset and pickles it
        Labels: 1 -> CH, 2 -> SIL, 3 -> SF, 4 -> G41
        NOTE: You do not need to use this method, as fitted model is already present.
        This procedure is for demo purposes mainly
        """
        from sklearn.model_selection import GridSearchCV
        from xgboost import XGBClassifier
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cvi_fitness_path = dir_path + '/' + CVIPredictor.cvi_fitness
        meta_data = pd.read_csv(cvi_fitness_path, header=None, sep=',')
        x, y = meta_data.values[:, :-1], meta_data.values[:, -1].astype(int) - 1
        predictor = GridSearchCV(estimator=XGBClassifier(
            objective='multi:softmax', num_class=4, seed=5, use_label_encoder=False
        ), param_grid={}, scoring='balanced_accuracy', n_jobs=4, cv=4)
        predictor.fit(x, y)
        with open(cvi_path, 'wb') as f:
            pickle.dump(predictor, f)


if __name__ == '__main__':
    CVIPredictor._build_cvi_predictor('cvi_predictor.pkl')
